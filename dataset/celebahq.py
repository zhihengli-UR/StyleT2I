import os
import torch
import pandas as pd
import torchtext
import clip
import pickle
import torchvision.transforms.functional as tvf


from functools import partial
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset


class CelebAHQ(Dataset):
    base_folder = "celebahq"
    image_base_folder = "CelebAMask-HQ"

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        loader=default_loader,
        image_only=False,
        not_return_word_embed=False,
        return_original_img=False,
        return_filename=False,
    ):
        assert split in ["train", "valid", "test", "all", "unseen_test", "unseen_valid"]
        split_str = split
        root = os.path.join(root, self.base_folder)
        self.root = root
        self.transform = transform
        self.img_loader = loader

        self.return_original_img = return_original_img
        self.return_filename = return_filename

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "unseen_valid": 1,
            "unseen_test": 2,
            "all": None,
        }
        split = split_map[split]
        self.split = split

        fn = partial(os.path.join, root, self.image_base_folder)

        splits = pd.read_csv(fn("list_eval_partition.csv"))
        mask = slice(None) if split is None else (splits["split"] == split)

        # filtering out "attractive" due to ethical concerns
        filter_mask_df = pd.read_csv(os.path.join(root, "filter_mask.csv"))
        filter_mask = filter_mask_df["filter_mask"].astype(bool)
        if isinstance(mask, pd.Series):
            mask = mask & filter_mask
        else:
            mask = filter_mask

        if split_str.startswith("unseen"):
            unseen_mask = pd.read_csv(os.path.join(root, "unseen_split.csv"))[
                split_str
            ].values.astype(bool)
            mask = mask & unseen_mask
            pkl_file_prefix = "novel_composition_caption"
        else:
            pkl_file_prefix = "filtered_celeba_caption"

        self.filename = splits[mask]["idx"].values
        self.image_only = image_only

        self.vocab = torchtext.vocab.GloVe()

        self.not_return_word_embed = not_return_word_embed

        cached_caption_pickle_fpath = os.path.join(
            root, f"{pkl_file_prefix}_{split_str}_cached.pkl"
        )

        with open(cached_caption_pickle_fpath, "rb") as f:
            self.cached_caption_lst = pickle.load(f)

    def __getitem__(self, index):
        original_img = img = self.get_img(index)
        if self.transform is not None:
            img = self.transform(img)
        if self.image_only:
            return img

        caption, caption_index = self.get_caption(
            index, deterministic=self.split != 0, return_index=True
        )
        clip_tokens = clip.tokenize(caption)[0]

        out_dict = {
            "image": img,
            "clip_tokens": clip_tokens,
        }

        if self.return_filename:
            out_dict["filename"] = f"{self.filename[index]}.jpg"

        if self.return_original_img:
            out_dict["original_image"] = tvf.to_tensor(
                tvf.resize(original_img, size=256).convert("RGB")
            )

        if not self.not_return_word_embed:
            word_embeds = self.get_word_embed_by_caption(caption)
            out_dict["word_embeds"] = word_embeds

        return out_dict

    def __len__(self):
        return len(self.filename)

    def get_caption(
        self,
        index,
        deterministic=False,
        return_index=False,
    ):
        captions = self.cached_caption_lst[index]

        if deterministic:
            caption_index = 0
        else:
            caption_index = torch.randint(low=0, high=len(captions), size=(1,))[
                0
            ].item()
        caption = captions[caption_index]
        # caption = caption.strip().lower()
        if return_index:
            return caption, caption_index
        return caption

    def get_word_embed_by_caption(self, caption):
        words = caption.split()
        word_embeds = self.vocab.get_vecs_by_tokens(words)
        return word_embeds

    def get_img(self, index):
        img_fpath = os.path.join(
            self.root,
            self.image_base_folder,
            "CelebA-HQ-img",
            f"{self.filename[index]}.jpg",
        )
        img = self.img_loader(img_fpath)
        return img
