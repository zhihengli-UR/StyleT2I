import os
import torch
import clip
import argparse
import torch.nn.functional as F
import random
import numpy as np
import torch.nn as nn


from torch import optim
from tqdm import tqdm
from utils import differentiable_clip_preprocess_from_stylegan
from model.stylegan2.model import Generator
from dataset.celebahq import CelebAHQ
from dataset.cub import CUBZeroShotText
from torchvision import transforms
from dataset.data_utils import pad_text_seq_collate
from model.data_utils import sample_data
from model.text_encoder_cond import Sentence2DeltaLatent


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(0)

        # model
        self.generator = Generator(args.stylegan_size, 512, 8)
        stylegan_ckpt = torch.load(args.ckpt)
        g_ckpt = stylegan_ckpt["g_ema"]
        self.generator.load_state_dict(g_ckpt, strict="ffhq" not in args.ckpt)
        self.generator.eval()
        self.generator = self.generator.to(self.device)
        for p in self.generator.parameters():
            p.requires_grad = False
        self.synthesis_kwargs = dict(input_is_latent=True, randomize_noise=False)

        if args.truncation < 1:
            self.mean_latent = self.generator.mean_latent(4096)
        else:
            self.mean_latent = None

        self.clip_model_for_train = clip.load("ViT-B/32", device="cpu")[0]
        self.clip_model_for_train = self.clip_model_for_train.to(self.device)
        if args.ckpt_clip_for_train is not None:
            assert os.path.exists(args.ckpt_clip_for_train)
            ckpt = torch.load(args.ckpt_clip_for_train)
            self.clip_model_for_train.load_state_dict(ckpt["model"])
        for p in self.clip_model_for_train.parameters():
            p.requires_grad = False
        self.clip_model_for_train.eval()

        self.clip_visual_size = self.clip_model_for_train.visual.input_resolution

        if args.latent_space == "w":
            output_dim = args.latent
        elif args.latent_space == "wp":
            output_dim = args.latent * self.generator.n_latent
        else:
            raise NotImplementedError

        self.sentence2latent = Sentence2DeltaLatent(
            args.word_embed_size,
            g_latent_dim=args.latent,
            out_dim=output_dim,
            hidden_dim=args.latent,
            num_mlp_layers=args.text_encoder_num_mlp_layers,
            return_delta=True,
        ).to(self.device)
        self.sentence2latent_optimizer = optim.Adam(
            self.sentence2latent.parameters(), args.lr
        )
        self.optimizer_lst = [self.sentence2latent_optimizer]
        self.model_lst = [self.sentence2latent]

        if args.resume is not None:
            print(f"resume training from {args.resume}")
            ckpt = torch.load(args.resume)
            self.start_iter_idx = int(
                os.path.splitext(os.path.basename(args.resume))[0]
            )
            self.sentence2latent.load_state_dict(ckpt["sentence_encoder"])
            self.sentence2latent_optimizer.load_state_dict(
                ckpt["sentence_encoder_optimizer"]
            )
        else:
            self.start_iter_idx = 0

        self.ce_criterion = nn.CrossEntropyLoss()

        # dataset
        if args.dataset in ["celebahq", "ffhq"]:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(args.stylegan_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True
                    ),
                ]
            )
            train_set = CelebAHQ(
                "data",
                split="train",
                transform=transform,
            )
        elif args.dataset in ["cub", "nabirds"]:
            imsize = args.stylegan_size
            transform = transforms.Compose(
                [
                    transforms.Resize(int(imsize * 76 / 64)),
                    transforms.RandomCrop(imsize),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True
                    ),
                ]
            )
            train_set = CUBZeroShotText(
                "data",
                split="train",
                transform=transform,
            )
        else:
            raise NotImplementedError

        collate_fn = pad_text_seq_collate

        self.dataloader = torch.utils.data.DataLoader(
            train_set,
            args.batch,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=args.num_workers > 0,
        )
        self.loader = sample_data(self.dataloader)

        # for visualization
        if args.dataset in ["celebahq", "ffhq"]:
            transform = transforms.Compose(
                [
                    transforms.Resize(args.stylegan_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True
                    ),
                ]
            )
        elif args.dataset in ["cub", "nabirds"]:
            imsize = args.stylegan_size
            transform = transforms.Compose(
                [
                    transforms.Resize(int(imsize * 76 / 64)),
                    transforms.CenterCrop(imsize),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True
                    ),
                ]
            )
        else:
            raise NotImplementedError

        self.ckpt_dir = os.path.join(args.exp_dir, "ckpt")
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)

    def zero_grad_all(self):
        for o in self.optimizer_lst:
            o.zero_grad()

    def false_requires_grad_all(self):
        for m in self.model_lst:
            for p in m.parameters():
                p.requires_grad = False

    def true_requires_grad(self, model_lst):
        for m in model_lst:
            for p in m.parameters():
                p.requires_grad = True

    @torch.no_grad()
    def get_latent(self, noise):
        return self.generator(
            [noise],
            just_latent=True,
            truncation=self.args.truncation,
            truncation_latent=self.mean_latent,
        )[0]

    def forward_sentence2latent(
        self,
        text_embed,
        text_len=None,
        return_delta=False,
        noise=None,
        return_rand_latent=False,
    ):
        if noise is None:
            gaussian_noise = torch.randn(
                text_embed.shape[0], self.args.latent, device=self.device
            )
        else:
            gaussian_noise = noise
        rand_latent = self.get_latent(gaussian_noise)
        latent_code, delta = self.sentence2latent(rand_latent, text_embed, text_len)

        output_lst = [latent_code]

        if return_delta:
            output_lst.append(delta)

        if return_rand_latent:
            output_lst.append(rand_latent)

        if len(output_lst) == 1:
            return output_lst[0]
        else:
            return tuple(output_lst)

    def g_nonsaturating_loss(self, fake_pred):
        loss = F.softplus(-fake_pred).mean()
        return loss

    def d_logistic_loss(self, real_pred, fake_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def train(self):
        log_dict = {}
        self.zero_grad_all()
        self.false_requires_grad_all()
        self.sentence2latent.train()
        self.true_requires_grad([self.sentence2latent])

        loader_out = next(self.loader)
        real_img = loader_out["image"]
        clip_tokens = loader_out["clip_tokens"]
        text_embed = loader_out["word_embeds"]

        text_len = loader_out["text_len"]

        real_img = real_img.to(self.device, non_blocking=True)
        text_embed = text_embed.to(self.device, non_blocking=True)
        clip_tokens = clip_tokens.to(self.device, non_blocking=True)

        loss = 0

        latent_code, sentence_delta, rand_latent = self.forward_sentence2latent(
            text_embed, text_len, return_delta=True, return_rand_latent=True
        )

        fake_img = self.generator([latent_code], **self.synthesis_kwargs)[0]
        fake_img_for_clip = differentiable_clip_preprocess_from_stylegan(
            fake_img, self.clip_visual_size
        )

        with torch.no_grad():
            clip_text_feat = self.clip_model_for_train.encode_text(clip_tokens)
            clip_text_feat = F.normalize(clip_text_feat, dim=-1)

        fake_img_feat = self.clip_model_for_train.encode_image(fake_img_for_clip)
        fake_img_feat = F.normalize(fake_img_feat, dim=-1)

        logits_per_image_to_text = (
            self.clip_model_for_train.logit_scale
            * fake_img_feat
            @ clip_text_feat.t()
        )

        ground_truth = torch.arange(
            len(logits_per_image_to_text), device=self.device
        ).long()
        img_text_loss = self.ce_criterion(
            logits_per_image_to_text, ground_truth
        )

        loss += img_text_loss
        log_dict["clip_fake_img_text_contrastive_loss"] = img_text_loss.item()

        direction_norm = torch.norm(sentence_delta, dim=-1)
        threholded_norm = F.relu(
            direction_norm - self.args.direction_norm_penalty_threshold
        )
        threholded_norm = threholded_norm.mean()
        threholded_norm = threholded_norm * self.args.lambda_direction_norm_penalty
        log_dict["direction_norm_loss"] = threholded_norm.item()
        loss += threholded_norm

        loss.backward()
        self.sentence2latent_optimizer.step()

        log_dict["loss"] = loss.item()
        return log_dict

    def save_checkpoint(self, iteration_idx):
        state_dict = {
            "sentence_encoder": self.sentence2latent.state_dict(),
            "sentence_encoder_optimizer": self.sentence2latent_optimizer.state_dict(),
        }
        torch.save(state_dict, f"{self.ckpt_dir}/last.pt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["celebahq", "cub", "nabirds", "ffhq"],
        required=True,
    )
    parser.add_argument("--iter", type=int, default=60001)
    parser.add_argument(
        "--stylegan_size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--exp_root", type=str, default="exp/stylet2i")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text_encoder_num_mlp_layers", type=int, default=3)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--ckpt_clip_for_train", type=str)
    parser.add_argument("--truncation", type=float, default=0.5)
    parser.add_argument("--latent_space", type=str, default="wp", choices=["wp", "w"])
    parser.add_argument("--lambda_direction_norm_penalty", type=float, default=1.0)
    parser.add_argument("--direction_norm_penalty_threshold", type=float, default=10.0)

    args = parser.parse_args()

    args.latent = 512
    args.word_embed_size = 300

    if args.resume is not None:
        assert os.path.exists(args.resume)

    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    seed_all(args.seed)

    name = f"stylet2i_{args.dataset}"
    if args.name is not None:
        name += f"_{args.name}"

    if not os.path.exists(args.exp_root):
        os.mkdir(args.exp_root)

    args.exp_dir = os.path.join(args.exp_root, name)
    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)

    trainer = Trainer(args)

    pbar = tqdm(range(trainer.start_iter_idx, args.iter), dynamic_ncols=True)
    for i in pbar:
        log_dict = {}
        if i % len(trainer.dataloader) == 0:
            log_dict["epoch"] = i // len(trainer.dataloader)

        if i % 5000 == 0:
            trainer.save_checkpoint(i)

        train_log_dict = trainer.train()
        log_dict.update(train_log_dict)
        desc = ""
        for k, v in log_dict.items():
            desc += f"{k}: {v:.4f} "
        pbar.set_description(desc)


if __name__ == "__main__":
    main()
