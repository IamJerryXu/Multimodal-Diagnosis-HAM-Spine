import json
import os
import re
from glob import glob

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer


def _load_text_map(json_path):
    if json_path is None:
        return {}
    with open(json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        text_map = {}
        for item in data:
            name = item.get("image_name") or item.get("image_info")
            if name is None and item.get("image_path"):
                name = os.path.basename(item.get("image_path"))
            if name is None:
                continue
            text = item.get("description", item.get("response", ""))
            text_map[name] = text
        return text_map
    return {os.path.basename(k): v for k, v in data.items()}


def _clean_text(text):
    return re.sub(r"[\u4e00-\u9fa5\u3000-\u303F\uff00-\uffef]", "", text or "")


class SpineTextImageDataset(Dataset):
    def __init__(
        self,
        image_root,
        csv_path,
        json_path,
        bert_path="/data/QLI/BERT_pretain",
        is_train=True,
    ):
        self.image_root = image_root
        df = pd.read_csv(csv_path)
        self.image_names = df["image"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.text_map = _load_text_map(json_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        if is_train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        label = int(self.labels[idx])
        path = os.path.join(self.image_root, name)
        image = Image.open(path)
        if image.mode == "L":
            image = image.convert("RGB")
        image = self.transform(image)

        text = _clean_text(self.text_map.get(name, ""))
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        return {
            "transformed_image": image,
            "label": label,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "image_id": name,
        }


class SpinePredictDataset(Dataset):
    def __init__(
        self,
        image_root,
        json_path,
        bert_path="/data/QLI/BERT_pretain",
    ):
        self.image_paths = sorted(
            glob(os.path.join(image_root, "*.png"))
            + glob(os.path.join(image_root, "*.jpg"))
        )
        self.image_names = [os.path.basename(p) for p in self.image_paths]
        self.text_map = _load_text_map(json_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        name = self.image_names[idx]
        image = Image.open(path)
        if image.mode == "L":
            image = image.convert("RGB")
        image = self.transform(image)

        text = _clean_text(self.text_map.get(name, ""))
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        return {
            "transformed_image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "image_id": name,
        }
