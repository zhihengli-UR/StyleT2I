import argparse
import os
import itertools
import torch
import torch.nn as nn
import random
import numpy as np
import clip


from tqdm import tqdm
from dataset.celebahq import CelebAHQ
from dataset.cub import CUBZeroShotText
from torchvision import transforms


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(0)

        self.clip_model = clip.load(args.clip_visual_backbone, device="cpu")[
            0
        ].to(self.device)

        self.ce_criterion = nn.CrossEntropyLoss()
        self.config_dataloaders()
        self.config_optimizers()
        self.max_logit_scale = np.log(100)

    def config_dataloaders(self):
        clip_input_resolution = self.clip_model.visual.input_resolution
        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
            inplace=True,
        )

        def convert_to_rgb(image):
            return image.convert("RGB")

        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    clip_input_resolution,
                    scale=(0.9, 1.0),
                    interpolation=transforms.functional.InterpolationMode.BICUBIC,
                ),
                convert_to_rgb,
                transforms.ToTensor(),
                normalize,
            ]
        )

        if self.args.dataset == "celebahq":
            train_set = CelebAHQ(
                "data",
                split=self.args.train_split,
                transform=train_transform,
                not_return_word_embed=True,
            )
        elif self.args.dataset == "cub":
            train_set = CUBZeroShotText(
                "data",
                split=self.args.train_split,
                transform=train_transform,
                not_return_word_embed=True,
            )
        else:
            raise NotImplementedError

        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            self.args.batch,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=True,
            persistent_workers=self.args.num_workers > 0,
        )

    def config_optimizers(self):
        for p in self.clip_model.parameters():
            p.requires_grad = False

        ft_parameters = []

        if self.args.clip_visual_backbone.startswith("RN"):
            ft_parameters.append(self.clip_model.visual.layer4.parameters())
            ft_parameters.append(self.clip_model.visual.attnpool.parameters())
        elif self.args.clip_visual_backbone.startswith("ViT"):
            ft_parameters.append(
                self.clip_model.visual.transformer.resblocks[
                    -self.args.vit_ft_layers :
                ].parameters()
            )
            ft_parameters.append(self.clip_model.visual.ln_post.parameters())
            ft_parameters.append([self.clip_model.visual.proj])
        else:
            raise NotImplementedError

        ft_parameters.append(
            self.clip_model.transformer.resblocks[-1].parameters()
        )

        ft_parameters.append(self.clip_model.ln_final.parameters())
        ft_parameters.append([self.clip_model.text_projection])
        ft_parameters.append([self.clip_model.logit_scale])

        chained_ft_parameters = itertools.chain(*ft_parameters)
        for p in chained_ft_parameters:
            p.requires_grad = True

        self.optimizer = torch.optim.AdamW(
            itertools.chain(*ft_parameters), lr=self.args.lr
        )

    def train(self, epoch):
        total_loss = 0
        total_sent_img_loss = 0

        self.clip_model.train()
        pbar = tqdm(
            enumerate(self.train_loader),
            dynamic_ncols=True,
            total=len(self.train_loader),
        )
        for i, data_dict in pbar:
            loss = 0
            self.optimizer.zero_grad()

            img = data_dict["image"]
            clip_tokens = data_dict["clip_tokens"]
            img = img.to(self.device, non_blocking=True)
            clip_tokens = clip_tokens.to(self.device, non_blocking=True)

            logits_per_image, logits_per_text = self.clip_model(
                img, clip_tokens
            )

            gt = torch.arange(len(logits_per_image), device=self.device).long()

            sent_img_loss = (
                self.ce_criterion(logits_per_image, gt)
                + self.ce_criterion(logits_per_text, gt)
            ) / 2
            loss += sent_img_loss
            total_sent_img_loss += sent_img_loss.item()

            loss.backward()
            self.optimizer.step()

            self.clip_model.logit_scale.data.clamp_(0, self.max_logit_scale)

            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)
            desc = f"[{epoch}/{self.args.epoch}] loss: {avg_loss:.3f}"
            pbar.set_description(desc)

        avg_loss = total_loss / len(self.train_loader)
        avg_sent_img_loss = total_sent_img_loss / len(self.train_loader)
        log_dict = {"loss": avg_loss, "sent_img_loss": avg_sent_img_loss}

        return log_dict

    def save(self, epoch):
        state_dict = {
            "model": self.clip_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
        }
        fpath = os.path.join(self.args.ckpt_dir, "ckpt.pth")
        torch.save(state_dict, fpath)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["cub", "celebahq"]
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--exp_root", type=str, default="exp/ft_clip_text")
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        choices=["train", "trainval", "all"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--clip_visual_backbone",
        type=str,
        default="ViT-B/32",
        choices=list(clip.clip._MODELS.keys()),
    )
    parser.add_argument("--vit_ft_layers", type=int, default=1)
    parser.add_argument("--name", type=str)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--ckpt", type=str)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert args.vit_ft_layers >= 1

    if args.name is None:
        args.name = ""
    else:
        args.name = f"{args.name}_"

    args.name = f"ft_clip_{args.name}"

    if args.clip_visual_backbone.startswith("ViT"):
        clip_visual_backbone_name = args.clip_visual_backbone.replace("/", "_")
    elif args.clip_visual_backbone.startswith("RN"):
        clip_visual_backbone_name = args.clip_visual_backbone
    else:
        raise NotImplementedError

    args.name += (
        f"{clip_visual_backbone_name}_{args.dataset}_{args.train_split}"
    )

    args.ckpt_dir = os.path.join(args.exp_root, args.name)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    return args


def main():
    args = parse_args()
    trainer = Trainer(args)

    for e in range(args.epoch):
        trainer.train(e)
        trainer.save(e)


if __name__ == "__main__":
    main()
