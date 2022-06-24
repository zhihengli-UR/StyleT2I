import os
import numpy as np
import pandas as pd


from PIL import Image
from torch.utils.data import Dataset


class NABirds(Dataset):
    base_folder = "nabirds"

    def __init__(self, root, split="train", transform=None):
        assert split in ["train", "val", "test_unseen"]
        super(NABirds, self).__init__()
        self.root = os.path.expanduser(os.path.join(root, self.base_folder))
        self.transform = transform
        self.split = split
        self._load_metadata()

    def _load_metadata(self):
        split_df = pd.read_csv(
            os.path.join(self.root, "cub_zeroshot_split.txt"), sep=" ", header=None
        )
        split_str_to_int = {"train": 0, "val": 1, "test_unseen": 2}
        split_int = split_str_to_int[self.split]
        split_mask = split_df[1] == split_int
        self.img_ids = split_df[0][split_mask].values
        self.file_id_to_bbox = {}
        with open(os.path.join(self.root, "bounding_boxes.txt")) as f:
            lines = f.readlines()
        for line in lines:
            pieces = line.strip().split()
            image_id = pieces[0]
            bbox = [int(x) for x in pieces[1:]]
            self.file_id_to_bbox[image_id] = bbox

        self.file_id_to_path = {}
        with open(os.path.join(self.root, "images.txt")) as f:
            for line in f:
                pieces = line.strip().split()
                image_id = pieces[0]
                self.file_id_to_path[image_id] = pieces[1]

    def __len__(self):
        return len(self.img_ids)

    def crop_img_by_bbox(self, img, file_id):
        bbox = self.file_id_to_bbox[file_id]
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
        file_id = self.img_ids[idx]
        path = os.path.join(self.root, "images", self.file_id_to_path[file_id])
        img = Image.open(path).convert("RGB")
        img = self.crop_img_by_bbox(img, file_id)

        if self.transform is not None:
            img = self.transform(img)

        return img
