import os
import torch
import torchvision
import argparse
import random
import numpy as np


from tqdm import tqdm
from model.stylegan2.model import Generator
from dataset.celebahq import CelebAHQ
from dataset.cub import CUBZeroShotText
from torchvision import transforms
from dataset.data_utils import pad_text_seq_collate
from model.text_encoder_cond import Sentence2DeltaLatent


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(0)

        # model
        self.generator = Generator(args.stylegan_size, 512, 8)
        g_ckpt = torch.load(args.ckpt)["g_ema"]
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
        self.model_lst = [self.sentence2latent]

        ckpt = torch.load(args.sentence2latent_ckpt)
        self.sentence2latent.load_state_dict(ckpt["sentence_encoder"])

        # dataset
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
            test_split_name = "unseen_test"
            test_set = CelebAHQ(
                "data",
                split=test_split_name,
                transform=transform,
                return_filename=True,
            )
        elif args.dataset in ["cub", "nabirds"]:
            imsize = args.stylegan_size
            test_split_name = "test_unseen"
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
            test_set = CUBZeroShotText(
                "data",
                split=test_split_name,
                transform=transform,
                return_filename=True,
            )
        else:
            raise NotImplementedError

        collate_fn = pad_text_seq_collate
        self.test_split_name = test_split_name

        exp_dir_name = args.sentence2latent_ckpt.split("/")[-3]
        exp_dir = os.path.join(args.exp_root, exp_dir_name)
        assert os.path.exists(exp_dir)
        self.vis_dir = os.path.join(exp_dir, "vis", args.name)
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)

        split_dir = os.path.join(self.vis_dir, test_split_name)
        if not os.path.exists(split_dir):
            os.mkdir(split_dir)

        self.test_loader = torch.utils.data.DataLoader(
            test_set,
            args.eval_batch,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=args.num_workers > 0,
        )

    def zero_grad_all(self):
        for o in self.optimizer_lst:
            o.zero_grad()

    def eval_all(self):
        for m in self.model_lst:
            m.eval()

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

    @torch.no_grad()
    def __vis(self, split, loader):
        dir_path = f"{self.vis_dir}/{split}"

        desc = f"visualizing {split} split"
        pbar = tqdm(loader, desc=desc, dynamic_ncols=True)

        for data_dict in pbar:
            text_embed = data_dict["word_embeds"]
            text_len = data_dict["text_len"]
            filename_lst = data_dict["filename"]

            text_embed = text_embed.to(self.device, non_blocking=True)
            gaussian_noise = torch.randn(
                text_embed.shape[0], self.args.latent, device=self.device
            )

            latent_code = self.forward_sentence2latent(
                text_embed, text_len=text_len, noise=gaussian_noise
            )
            fake_img = self.generator([latent_code], **self.synthesis_kwargs)[0]

            for idx_batch in range(fake_img.shape[0]):
                filename = filename_lst[idx_batch]

                if "/" in filename:
                    cur_dir_name = filename.split("/")[0]
                    cur_dir_path = os.path.join(dir_path, cur_dir_name)
                    if not os.path.exists(cur_dir_path):
                        os.mkdir(cur_dir_path)

                img_path = os.path.join(dir_path, filename)
                torchvision.utils.save_image(
                    fake_img[idx_batch : idx_batch + 1],
                    img_path,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                    padding=0,
                )

    def visualize(self):
        self.__vis(self.test_split_name, self.test_loader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["celebahq", "cub", "ffhq", "nabirds"],
        required=True,
    )
    parser.add_argument(
        "--stylegan_size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument("--eval_batch", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--exp_root", type=str, default="exp/stylet2i")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text_encoder_num_mlp_layers", type=int, default=3)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--sentence2latent_ckpt", type=str, required=True)
    parser.add_argument("--truncation", type=float, default=0.5)
    parser.add_argument("--latent_space", type=str, default="wp", choices=["wp", "w"])

    args = parser.parse_args()

    args.latent = 512
    args.word_embed_size = 300

    assert os.path.exists(args.sentence2latent_ckpt)

    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    seed_all(args.seed)

    if args.name is None:
        args.name = ""

    if not os.path.exists(args.exp_root):
        os.mkdir(args.exp_root)

    trainer = Trainer(args)
    trainer.visualize()


if __name__ == "__main__":
    main()
