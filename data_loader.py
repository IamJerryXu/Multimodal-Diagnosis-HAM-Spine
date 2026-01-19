import os
import json
import re
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer
from PIL import Image
import torchvision.transforms as transforms

try:
    import cv2
except Exception:
    cv2 = None


class StainNormalizer:
    def __init__(self, target_mean, target_std):
        if cv2 is None:
            raise ImportError("Stain normalization requires opencv-python.")
        self.target_mean = np.array(target_mean, dtype=np.float32)
        self.target_std = np.array(target_std, dtype=np.float32)

    def __call__(self, img):
        arr = np.array(img)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
        mean = lab.reshape(-1, 3).mean(axis=0)
        std = lab.reshape(-1, 3).std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        lab = (lab - mean) / std * self.target_std + self.target_mean
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)


def _build_tabular_map(metadata_csv, fields, normalize="zscore"):
    df = pd.read_csv(metadata_csv)
    df["image_id"] = df["image_id"].astype(str)
    df["image_id_base"] = df["image_id"].apply(lambda x: os.path.splitext(x)[0])

    numeric_fields = []
    categorical_fields = []
    for field in fields:
        if field not in df.columns:
            continue
        if field == "age" or pd.api.types.is_numeric_dtype(df[field]):
            numeric_fields.append(field)
        else:
            categorical_fields.append(field)

    numeric_stats = {}
    for field in numeric_fields:
        values = pd.to_numeric(df[field], errors="coerce")
        mean = float(values.mean()) if values.notna().any() else 0.0
        std = float(values.std()) if values.notna().any() else 1.0
        if std == 0.0:
            std = 1.0
        numeric_stats[field] = (mean, std)

    category_maps = {}
    for field in categorical_fields:
        values = df[field].dropna().astype(str).unique().tolist()
        values = sorted(set(values))
        if "unknown" not in values:
            values.append("unknown")
        category_maps[field] = values

    total_dim = len(numeric_fields) + sum(
        len(category_maps[field]) for field in categorical_fields
    )

    tabular_map = {}
    for _, row in df.iterrows():
        feats = []
        for field in numeric_fields:
            val = pd.to_numeric(row.get(field), errors="coerce")
            mean, std = numeric_stats[field]
            if np.isnan(val):
                val = mean
            if normalize == "zscore":
                feats.append((val - mean) / std)
            else:
                feats.append(val)

        for field in categorical_fields:
            categories = category_maps[field]
            val = row.get(field)
            val = "unknown" if pd.isna(val) else str(val)
            if val not in categories:
                val = "unknown"
            one_hot = [0.0] * len(categories)
            one_hot[categories.index(val)] = 1.0
            feats.extend(one_hot)

        tabular_map[row["image_id_base"]] = torch.tensor(
            feats, dtype=torch.float32
        )

    return tabular_map, total_dim

class MultimodalDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        image_transform,
        image_dir,
        json_path,
        csv_path,
        max_length,
        metadata_csv=None,
        tabular_enabled=False,
        tabular_fields=None,
        tabular_normalize="zscore",
        extra_image_dirs=None,
        pseudo_2p5d=None,
        sequence_cfg=None,
        multi_view_cfg=None,
    ):
        
        self.image_dirs = [image_dir]
        if extra_image_dirs:
            self.image_dirs.extend(extra_image_dirs)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_transform = image_transform
        self.tabular_enabled = tabular_enabled
        self.tabular_dim = 0
        self.tabular_map = None
        self.pseudo_2p5d = pseudo_2p5d or {}
        self.pseudo_enabled = bool(self.pseudo_2p5d.get("enabled", False))
        self.pseudo_offsets = self.pseudo_2p5d.get("offsets", [-1, 0, 1])
        self.sequence_cfg = sequence_cfg or {}
        self.sequence_enabled = bool(self.sequence_cfg.get("enabled", False))
        self.sequence_offsets = self.sequence_cfg.get("offsets", [-2, -1, 0, 1, 2])
        self.multi_view_cfg = multi_view_cfg or {}
        self.multi_view_enabled = bool(self.multi_view_cfg.get("enabled", False))
        self.multi_view_count = int(self.multi_view_cfg.get("num_views", 2))
        
        
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        
        
        description_map = {}
        for item in json_data:
            image_key = None
            if "image_info" in item:
                image_key = os.path.basename(item["image_info"])
            elif "image_name" in item:
                image_key = os.path.basename(item["image_name"])
            elif "image_path" in item:
                image_key = os.path.basename(item["image_path"])
            if not image_key:
                continue

            description = item.get("description") or item.get("response") or item.get("caption")
            if description is None:
                continue
            description_map[image_key] = description

        
        label_df = pd.read_csv(csv_path)
        
        image_col = [col for col in label_df.columns if 'image' in col][0]
        label_col = [col for col in label_df.columns if 'label' in col][0]
        
        label_map = pd.Series(label_df[label_col].values, index=label_df[image_col]).to_dict()

        
        self.metadata = []
        missing_desc = 0
        for image_id, label in label_map.items():
            description = description_map.get(image_id, "")
            if not description:
                missing_desc += 1
            self.metadata.append({
                'image_id': image_id,
                'description': description,
                'label': label
            })

        print(
            f"成功从 {os.path.basename(json_path)} 和 {os.path.basename(csv_path)} "
            f"中加载 {len(self.metadata)} 条数据。"
        )
        if missing_desc > 0:
            print(f"提示: {missing_desc} 张图片未在JSON中找到描述，已填空文本。")

        if self.tabular_enabled:
            if not metadata_csv:
                raise ValueError("tabular_enabled requires metadata_csv.")
            if not tabular_fields:
                tabular_fields = ["age", "sex", "localization"]
            self.tabular_map, self.tabular_dim = _build_tabular_map(
                metadata_csv, tabular_fields, normalize=tabular_normalize
            )

    def __len__(self):
        return len(self.metadata)

    def _find_image_path(self, image_id):
        for base_dir in self.image_dirs:
            image_path = os.path.join(base_dir, image_id)
            if os.path.exists(image_path):
                return image_path
        return None

    def _neighbor_name(self, image_id, offset):
        if offset == 0:
            return image_id
        match = re.match(r"^(.*_)(\\d+)(\\.[^.]+)$", image_id)
        if not match:
            match = re.match(r"^(.*?)(\\d+)(\\.[^.]+)$", image_id)
        if not match:
            return image_id
        prefix, idx_str, suffix = match.groups()
        idx = int(idx_str) + offset
        if idx < 0:
            idx = 0
        return f"{prefix}{idx}{suffix}"

    def _load_pseudo_2p5d(self, image_id):
        neighbor_ids = [self._neighbor_name(image_id, o) for o in self.pseudo_offsets]
        images = []
        base_size = None
        for nid in neighbor_ids:
            image_path = self._find_image_path(nid)
            if image_path is None:
                image_path = self._find_image_path(image_id)
            if image_path is None:
                raise FileNotFoundError(f"Image not found in any dir: {image_id}")
            img = Image.open(image_path).convert('L')
            if base_size is None:
                base_size = img.size
            elif img.size != base_size:
                img = img.resize(base_size)
            images.append(np.array(img))

        if len(images) != 3:
            raise ValueError(f"pseudo_2p5d expects 3 slices, got {len(images)}")

        stacked = np.stack(images, axis=2).astype(np.uint8)
        image = Image.fromarray(stacked, mode='RGB')
        return self.image_transform(image)

    def _load_sequence(self, image_id):
        neighbor_ids = [self._neighbor_name(image_id, o) for o in self.sequence_offsets]
        images = []
        for nid in neighbor_ids:
            image_path = self._find_image_path(nid)
            if image_path is None:
                image_path = self._find_image_path(image_id)
            if image_path is None:
                raise FileNotFoundError(f"Image not found in any dir: {image_id}")
            img = Image.open(image_path).convert('RGB')
            images.append(self.image_transform(img))
        if not images:
            raise ValueError("sequence expects at least 1 slice")
        return torch.stack(images, dim=0)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_id = item['image_id']
        
        
        try:
            if self.multi_view_enabled:
                image_path = self._find_image_path(image_id)
                if image_path is None:
                    raise FileNotFoundError(f"Image not found in any dir: {image_id}")
                img = Image.open(image_path).convert('RGB')
                views = [self.image_transform(img) for _ in range(self.multi_view_count)]
                image = torch.stack(views, dim=0)
            elif self.sequence_enabled:
                image = self._load_sequence(image_id)
            elif self.pseudo_enabled:
                image = self._load_pseudo_2p5d(image_id)
            else:
                image_path = self._find_image_path(image_id)
                if image_path is None:
                    raise FileNotFoundError(f"Image not found in any dir: {image_id}")
                image = Image.open(image_path).convert('RGB')
                image = self.image_transform(image)
        except Exception as e:
            print(f"加载图片时出错 {image_id}: {e}")
            
            image = torch.zeros((3, 224, 224)) 
        
        
        text = item['description']
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        
        
        label = torch.tensor(item['label'], dtype=torch.long)
        
        if self.tabular_enabled:
            base_id = os.path.splitext(image_id)[0]
            tabular = self.tabular_map.get(
                base_id, torch.zeros(self.tabular_dim, dtype=torch.float32)
            )
        else:
            tabular = torch.zeros(0, dtype=torch.float32)

        return image, input_ids, attention_mask, tabular, label, image_id

def create_data_loader(config, split='train', test_image_dir=None, test_json_path=None, num_workers=4):
    
    
    if split == 'train':
        image_dir = config['data']['train_image_dir']
        json_path = config['data']['train_json_path']
        csv_path = config['data']['train_label_csv'] 
    elif split == 'val':
        image_dir = config['data']['val_image_dir']
        json_path = config['data']['val_json_path']
        csv_path = config['data']['val_label_csv'] 
    elif split == 'test':
        image_dir = test_image_dir if test_image_dir else config['data']['test_image_dir']
        json_path = test_json_path if test_json_path else config['data']['test_json_path']
        
        csv_path = config['data'].get('test_label_csv', None)
    else:
        raise ValueError(f"不支持的数据集split: {split}")

    tokenizer = BertTokenizer.from_pretrained(config['model']['text_encoder']['model_name'])
    max_length = config['tokenizer']['max_length']
    stain_cfg = config.get("data", {}).get("stain_normalization", {})
    stain_enabled = bool(stain_cfg.get("enabled", False))
    stain_mean = stain_cfg.get("target_mean", [150.0, 140.0, 140.0])
    stain_std = stain_cfg.get("target_std", [20.0, 20.0, 20.0])

    if split == 'train':
        transform_steps = []
        if stain_enabled:
            transform_steps.append(StainNormalizer(stain_mean, stain_std))
        transform_steps.extend([
            # 加强版数据增强：
            # 1. RandomResizedCrop: 随机裁剪，强迫模型看局部细节，scale=(0.2, 1.0) 避免切得太碎
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            # 2. 几何变换：翻转和旋转，打破位置偏见（如角落特征）
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            # 3. 颜色变换：防止过拟合特定光照或色调
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image_transform = transforms.Compose(transform_steps)
    else: 
        transform_steps = []
        if stain_enabled:
            transform_steps.append(StainNormalizer(stain_mean, stain_std))
        transform_steps.extend([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image_transform = transforms.Compose(transform_steps)

    
    tabular_cfg = config.get("model", {}).get("tabular", {})
    tabular_enabled = bool(tabular_cfg.get("enabled", False))
    tabular_fields = tabular_cfg.get("fields")
    tabular_normalize = tabular_cfg.get("normalize", "zscore")
    metadata_csv = config.get("data", {}).get("metadata_csv")

    extra_image_dirs = config.get("data", {}).get("extra_image_dirs", [])
    pseudo_2p5d = config.get("data", {}).get("pseudo_2p5d", {})
    sequence_cfg = config.get("data", {}).get("sequence", {})
    multi_view_cfg = config.get("data", {}).get("multi_view", {})
    dataset = MultimodalDataset(
        tokenizer,
        image_transform,
        image_dir,
        json_path,
        csv_path,
        max_length,
        metadata_csv=metadata_csv,
        tabular_enabled=tabular_enabled,
        tabular_fields=tabular_fields,
        tabular_normalize=tabular_normalize,
        extra_image_dirs=extra_image_dirs,
        pseudo_2p5d=pseudo_2p5d,
        sequence_cfg=sequence_cfg,
        multi_view_cfg=multi_view_cfg,
    )

    
    if len(dataset) == 0:
        raise ValueError(f"创建 '{split}' 数据集失败：没有从JSON和CSV中匹配到任何有效数据。请检查文件路径和内容。")

    is_train = split == 'train'
    sampler = None
    if is_train and config.get("training", {}).get("sampler") == "weighted":
        labels = [int(item["label"]) for item in dataset.metadata]
        num_classes = int(config["model"]["num_classes"])
        counts = [0] * num_classes
        for label in labels:
            if 0 <= label < num_classes:
                counts[label] += 1
        total = max(1, len(labels))
        weights_per_class = [
            (total / (num_classes * count)) if count > 0 else 0.0 for count in counts
        ]
        sample_weights = [weights_per_class[label] for label in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=total, replacement=True)
    data_loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=is_train and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return data_loader
