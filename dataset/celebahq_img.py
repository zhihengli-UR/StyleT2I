import os
import pandas as pd

from functools import partial
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset


class CelebAHQ(Dataset):
    image_base_folder = "celebahq/CelebAMask-HQ"

    def __init__(self, root, split="train", transform=None, loader=default_loader):
        assert split in ["train", "valid", "test", "all"]
        self.root = root
        self.transform = transform
        self.img_loader = loader

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split = split_map[split]
        self.split = split

        fn = partial(os.path.join, root, self.image_base_folder)
        splits = pd.read_csv(fn("list_eval_partition.csv"))
        # attr = pd.read_csv(fn("CelebAMask-HQ-attribute-anno.txt"), delim_whitespace=True, header=1)
        mask = slice(None) if split is None else (splits["split"] == split)
        filter_mask_df = pd.read_csv(os.path.join(root, "celebahq/filter_mask.csv"))
        filter_mask = filter_mask_df["filter_mask"].astype(bool)
        if isinstance(mask, pd.Series):
            mask = mask & filter_mask
        else:
            mask = filter_mask

        self.filename = splits[mask]["idx"].values

    def __getitem__(self, index):
        img = self.get_img(index)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.filename)

    def get_img(self, index):
        img_fpath = os.path.join(
            self.root,
            self.image_base_folder,
            "CelebA-HQ-img",
            f"{self.filename[index]}.jpg",
        )
        img = self.img_loader(img_fpath)
        return img


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset.data_utils import pad_text_seq_collate
    import torchvision

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = CelebAHQ(root="data", split="train", transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_text_seq_collate,
    )
    for i, (img, clip_tokens, text_embed, text_len) in enumerate(loader):
        pass
