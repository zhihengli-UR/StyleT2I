import os
import torch
import clip
import numpy as np
import pickle
import torchtext
import torchvision.transforms.functional as tvf

from PIL import Image
from torch.utils.data import Dataset


class CUBZeroShot(Dataset):
    base_folder = "CUB_200_2011/images"
    folder_name = "cub"

    def __init__(self, root, split="train", transform=None):
        assert split in ["train", "val", "trainval", "test_seen", "test_unseen"]
        super(CUBZeroShot, self).__init__()
        self.root = os.path.expanduser(os.path.join(root, self.folder_name))
        self.transform = transform
        self.split = split
        self._load_metadata()

    def _load_metadata(self):
        metadata = torch.load(os.path.join(self.root, "split.pth"))
        self.file_list = metadata["file_list"]
        self.bbox = metadata["bbox"]
        self.indices = metadata[self.split]

    def __len__(self):
        return len(self.indices)

    def crop_img_by_bbox(self, img, file_idx):
        bbox = self.bbox[file_idx]
        width, height = img.size
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
        return img

    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        path = os.path.join(self.root, self.base_folder, self.file_list[file_idx])
        img = Image.open(path).convert("RGB")
        img = self.crop_img_by_bbox(img, file_idx)

        if self.transform is not None:
            img = self.transform(img)

        return img


class CUBZeroShotText(Dataset):
    image_base_folder = "CUB_200_2011/images"
    folder_name = "cub"

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        not_return_word_embed=False,
        return_original_img=False,
        return_filename=True,
    ):
        assert split in ["train", "val", "trainval", "test_seen", "test_unseen", "all"]

        super(CUBZeroShotText, self).__init__()
        root = os.path.expanduser(os.path.join(root, self.folder_name))
        self.root = root
        self.transform = transform
        self.split = split

        self.vocab = torchtext.vocab.GloVe()
        self._load_metadata()
        self.return_filename = return_filename
        self.not_return_word_embed = not_return_word_embed
        self.deterministic = split in ["val", "test_seen", "test_unseen"]
        self.return_original_img = return_original_img

        cached_caption_pickle_fpath = os.path.join(
            root, f"preprocessed_text_{split}_cached.pkl"
        )
        assert os.path.exists(cached_caption_pickle_fpath), f"{cached_caption_pickle_fpath} not found."
        with open(cached_caption_pickle_fpath, "rb") as f:
            self.cached_caption_lst = pickle.load(f)

    def _load_metadata(self):
        metadata = torch.load(os.path.join(self.root, "split.pth"))
        self.file_list = metadata["file_list"]
        self.bbox = metadata["bbox"]
        if self.split == "all":
            split_indices_lst = []
            for key, ids in metadata.items():
                if key in {"train", "val", "test_seen", "test_unseen"}:
                    split_indices_lst.append(ids)
            self.indices = np.concatenate(split_indices_lst)
        else:
            self.indices = metadata[self.split]

    def __len__(self):
        return len(self.indices)

    def crop_img_by_bbox(self, img, idx):
        file_idx = self.indices[idx]
        bbox = self.bbox[file_idx]
        width, height = img.size
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
        return img

    def __getitem__(self, idx):
        img = self.get_img(idx)
        original_img = img = self.crop_img_by_bbox(img, idx)

        if self.transform is not None:
            img = self.transform(img)

        caption, caption_index = self.get_caption(
            idx, deterministic=self.split != 0, return_index=True
        )

        clip_tokens = clip.tokenize(caption)[0]

        out_dict = {
            "image": img,
            "clip_tokens": clip_tokens,
        }

        if self.return_filename:
            file_idx = self.indices[idx]
            img_fname = self.file_list[file_idx]
            out_dict["filename"] = img_fname

        if self.return_original_img:
            out_dict["original_image"] = tvf.to_tensor(
                tvf.resize(original_img, size=256).convert("RGB")
            )

        if not self.not_return_word_embed:
            word_embeds = self.get_word_embed_by_caption(caption)
            out_dict["word_embeds"] = word_embeds

        return out_dict

    def get_img(self, idx):
        file_idx = self.indices[idx]
        img_fname = self.file_list[file_idx]
        path = os.path.join(self.root, self.image_base_folder, img_fname)
        img = Image.open(path).convert("RGB")
        return img

    def get_caption(
        self,
        idx,
        deterministic=False,
        return_index=False,
    ):
        captions = self.cached_caption_lst[idx]
        if deterministic:
            caption_index = 0
        else:
            caption_index = torch.randint(low=0, high=len(captions), size=(1,))[
                0
            ].item()
        caption = captions[caption_index]
        if return_index:
            return caption, caption_index
        return caption

    def get_word_embed_by_caption(self, caption):
        words = caption.split()
        word_embeds = self.vocab.get_vecs_by_tokens(words)
        return word_embeds
