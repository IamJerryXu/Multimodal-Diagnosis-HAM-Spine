import re
import cv2
import pdb
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from glob import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from PIL import Image
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence

import re
import os
import json
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl
from transformers import BertTokenizer

import json
from collections import Counter
from torch.utils.data import WeightedRandomSampler
class MedDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, label_paths, des_path, hidden_json_path=None,
                 des_path_fallback=None, hidden_json_path_fallback=None, image_transform=None):
        super().__init__()
        print(f"DEBUG: MedDataset.__init__ called with img_path: {img_path}")
        # self.img_paths = glob(img_path + "/images/*")
        self.img_paths = sorted(
            glob(os.path.join(img_path, "*.jpg")) + glob(os.path.join(img_path, "*.png"))
        )
        print(f"DEBUG: Found {len(self.img_paths)} image files")
        self.labels = {}
        for label_path in label_paths:
            with open(label_path, 'r') as f:
                reader = f.readlines()
                for line in reader:
                    parts = line.strip().split(',')
                    if len(parts) != 2:
                        continue
                    img_name, label = parts
                    try:
                        label = int(label)
                    except ValueError:
                        continue
                    self.labels[img_name] = label
    
        # ✅ 加载 response (description)，支持备用路径
        self.des = {}
        print(f"DEBUG: Loading descriptions from {des_path}")
        # 先加载主路径
        if des_path is not None:
            try:
                with open(des_path, 'r') as f:
                    desc_data = json.load(f)
                for item in desc_data:
                    name = item.get('image_info') or item.get('image_name')
                    if name is None:
                        continue
                    self.des[name] = item.get('description', "")
                print(f"DEBUG: Loaded {len(desc_data)} descriptions")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"警告: 无法加载主 response 文件 {des_path}: {e}")
        else:
            print("DEBUG: des_path is None, skipping description loading")
        
        # 再加载备用路径（不覆盖已有的）
        if des_path_fallback is not None:
            try:
                with open(des_path_fallback, 'r') as f:
                    desc_data_fallback = json.load(f)
                for item in desc_data_fallback:
                    name = item.get('image_info') or item.get('image_name')
                    if name is None:
                        continue
                    # 只在主路径中没有时才添加
                    if name not in self.des:
                        self.des[name] = item.get('description', "")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"警告: 无法加载备用 response 文件 {des_path_fallback}: {e}")

        # ✅ 加载 first/last hidden_state，支持备用路径
        self.first_hidden_states = {}
        self.last_hidden_states = {}
        self.has_hidden_states = hidden_json_path is not None or hidden_json_path_fallback is not None
        
        # 先加载主路径
        if hidden_json_path is not None:
            try:
                with open(hidden_json_path, 'r') as f:
                    hidden_data = json.load(f)
                for item in hidden_data:
                    name = item['image_name']
                    self.first_hidden_states[name] = torch.tensor(item['first_hidden_state'], dtype=torch.float)
                    self.last_hidden_states[name] = torch.tensor(item['last_hidden_state'], dtype=torch.float)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"警告: 无法加载主 hidden_state 文件 {hidden_json_path}: {e}")
        
        # 再加载备用路径（不覆盖已有的）
        if hidden_json_path_fallback is not None:
            try:
                with open(hidden_json_path_fallback, 'r') as f:
                    hidden_data_fallback = json.load(f)
                for item in hidden_data_fallback:
                    name = item['image_name']
                    # 只在主路径中没有时才添加
                    if name not in self.first_hidden_states:
                        self.first_hidden_states[name] = torch.tensor(item['first_hidden_state'], dtype=torch.float)
                        self.last_hidden_states[name] = torch.tensor(item['last_hidden_state'], dtype=torch.float)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"警告: 无法加载备用 hidden_state 文件 {hidden_json_path_fallback}: {e}")

        self.image_transform = image_transform
    def __len__(self):
        return len(self.img_paths)  
    def __getitem__(self, index):
        subject_name = os.path.basename(self.img_paths[index])
        
        # 校验数据完整性，让错误暴露出来
        if subject_name not in self.labels:
            raise KeyError(f"在标签文件中找不到图片 '{subject_name}' 的标签。")
        
        result = {
            'img': self.image_transform(Image.open(self.img_paths[index])),
            'label': self.labels[subject_name],
            'text_raw': self.des.get(subject_name, ""),
        }

        # result['first_hidden'] = torch.zeros(5120)
        # result['last_hidden'] = torch.zeros(5120)

        # ✅ 获取 hidden_state，如果找不到则使用 0 填充
        if subject_name in self.first_hidden_states:
            result['first_hidden'] = self.first_hidden_states[subject_name]
            result['last_hidden'] = self.last_hidden_states[subject_name]
        else:
            # 如果找不到，使用 0 填充
            # 需要先检查维度，如果已有数据则使用相同维度，否则使用默认维度 5120
            if len(self.first_hidden_states) > 0:
                # 从已有数据中获取维度
                sample_hidden = next(iter(self.first_hidden_states.values()))
                hidden_dim = sample_hidden.shape[0]
            else:
                hidden_dim = 3584
            result['first_hidden'] = torch.zeros(hidden_dim)
            result['last_hidden'] = torch.zeros(hidden_dim)

        return result


# Tokenizer
tokenizer = BertTokenizer.from_pretrained('/data/QLI/BERT_pretain')


# Collate function
def collate_fn(batch):
    imgs = [item['img'] for item in batch]
    labels = [item['label'] for item in batch]
    texts = [item['text_raw'] for item in batch]
    first_hidden = [item['first_hidden'] for item in batch]  # 新增
    last_hidden = [item['last_hidden'] for item in batch]    # 新增

    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels)

    text_encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt',
    )

    first_hidden = torch.stack(first_hidden, dim=0)
    last_hidden = torch.stack(last_hidden, dim=0)

    return {
        'imgs': imgs,
        'labels': labels,
        'texts': text_encodings,
        'first_hidden': first_hidden,
        'last_hidden': last_hidden
    }


# Lightning DataModule
class DataModule(pl.LightningDataModule):
    def __init__(self, config, collate_fn=None):
        super().__init__()
        self.batch_size = config['train']['batch_size']
        self.num_workers = config['train']['num_workers']
        self.config = config
        self.collate_fn = collate_fn

        self.transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    
    def setup(self, stage=None):
        print("DEBUG: DataModule.setup() called")
        image_transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        image_transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        print("DEBUG: Image transforms created")


        
        # print("-" * 50)
        # print("DEBUG: 正在设置 DataModule...")
        
        # # 打印验证/测试集使用的路径
        # test_img_path = self.config["data"]['test_img_path']
        # label_glob_pattern = f'{self.config["data"]["label_path"]}/*.csv'
        
        # print(f"验证集图片路径 (img_path): {test_img_path}")
        # print(f"标签文件搜索模式 (label_paths glob): {label_glob_pattern}")
        
        # # 实际搜索到的标签文件
        # actual_label_paths = glob(label_glob_pattern)
        # print(f"实际找到的标签文件数量: {len(actual_label_paths)}")
        # for path in actual_label_paths:
        #     print(f"  - {path}")
        # print("-" * 50)




        print(f"DEBUG: Creating train dataset with img_path: {self.config['data']['train_img_path']}")
        print(f"DEBUG: Train label path: {self.config['data']['train_label_path']}")
        print(f"DEBUG: Train text desc path: {self.config['data']['train_text_desc_path']}")
        print(f"DEBUG: Train hidden path: {self.config['data'].get('train_hidden_path')}")

        self.train_dataset = MedDataset(
            img_path=self.config["data"]['train_img_path'],
            # 不再使用 glob，而是传入一个只包含训练标签路径的列表
            label_paths=[self.config["data"]["train_label_path"]],
            des_path=self.config['data']['train_text_desc_path'],
            hidden_json_path=self.config['data'].get('train_hidden_path'),
            # 添加备用路径：训练集找不到时，去测试集找
            des_path_fallback=self.config['data'].get('test_text_desc_path'),
            hidden_json_path_fallback=self.config['data'].get('test_hidden_path'),
            image_transform=image_transform_train,
        )
        print(f"DEBUG: Train dataset created with {len(self.train_dataset)} samples")

        # --- START: Oversampling Logic ---
        # 1. Get all labels from the training dataset
        labels = [self.train_dataset.labels[os.path.basename(path)] for path in self.train_dataset.img_paths]
        
        # 2. Count class occurrences
        class_counts = Counter(labels)
        
        # 3. Calculate weights for each class (inverse of frequency)
        class_weights = {c: 1.0 / count for c, count in class_counts.items()}
        
        # 4. Assign a weight to each sample in the dataset
        sample_weights = [class_weights[self.train_dataset.labels[os.path.basename(path)]] for path in self.train_dataset.img_paths]

        # 5. Create the sampler
        self.sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        # --- END: Oversampling Logic ---

        self.test_dataset = MedDataset(
            img_path=self.config["data"]['test_img_path'],
            # 同样，这里只传入测试标签路径
            label_paths=[self.config["data"]["test_label_path"]],
            des_path=self.config['data']['test_text_desc_path'],
            hidden_json_path=self.config['data'].get('test_hidden_path'),
            # 添加备用路径：测试集找不到时，去训练集找
            des_path_fallback=self.config['data'].get('train_text_desc_path'),
            hidden_json_path_fallback=self.config['data'].get('train_hidden_path'),
            image_transform=image_transform_test,
        )

    def train_dataloader(self):
        if len(self.train_dataset) == 0:
            raise ValueError("训练数据集为空！请检查数据路径和加载逻辑。")
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          sampler=self.sampler, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn or collate_fn)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn or collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn or collate_fn)


if __name__ == "__main__":
    pass


# class MedDataset(torch.utils.data.Dataset):
#     def __init__(self, img_path, label_paths, des_path, image_transform=None):
#         super().__init__()
#         self.img_paths = glob(img_path+"/images/*.jpg")
#         self.labels = {} # key是图片名称，values是图片类别

#         # 
#         for label_path in label_paths:
#             with open(label_path, 'r') as f:
#                 reader = f.readlines()
#                 for line in reader:
#                     img_name, lable = line.replace("\n", "").split(',')
#                     self.labels[img_name] = int(lable)
#         self.des = {}
#         with open(des_path, 'r') as f:
#             desc_data = json.load(f)

#         for k, v in desc_data.items():
#             data_name = os.path.basename(k)
#             self.des[data_name] = v

#         self.image_transform = image_transform


#     def __len__(self, ):
#         return len(self.img_paths)

#     def remove_chinese_and_punctuation(self, text):
#         return re.sub(r'[\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef]', '', text)

#     def __getitem__(self, index):
#         subject_name = os.path.basename(self.img_paths[index])
#         result = {
#             'img': self.image_transform(Image.open(self.img_paths[index])),
#             'label': self.labels[subject_name],
#             'text_raw': self.des[subject_name]  # 返回原始文本，collate_fn中统一tokenize
#         }
#         return result



# tokenizer = BertTokenizer.from_pretrained('/mnt/data2/zzixuantang/classfier_convNext/model/BERT_pretain')
# def collate_fn(batch):
#     # 分离不同字段
#     imgs = [item['img'] for item in batch]
#     labels = [item['label'] for item in batch]
#     texts = [item['text_raw'] for item in batch]  # 假设返回的是原始文本

#     imgs = torch.stack(imgs, dim=0)
#     labels = torch.tensor(labels)

#     # 处理文本：统一tokenize并填充
#     text_encodings = tokenizer(
#         texts,
#         padding=True,          # 自动填充到批次内最长长度
#         truncation=True,       # 截断超长文本
#         max_length=512,       # 设置最大长度
#         return_tensors='pt',   # 返回PyTorch张量
#     )

#     # 返回批处理结果
#     return {
#         'imgs': imgs,
#         'labels': labels,
#         'texts': text_encodings  # 包含input_ids, attention_mask等
#     }

# # 1. 定义 LightningDataModule
# class DataModule(pl.LightningDataModule):
#     def __init__(self, config):
#         super().__init__()


#         self.batch_size = config['train']['batch_size']
#         self.num_workers = config['train']['num_workers']
        
#         self.config = config
#         # 定义数据增强和转换
#         self.transform = transforms.Compose([
#             transforms.RandomRotation(10),  # 随机旋转
#             transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
#             transforms.ToTensor(),
#         ])
        
#         # 测试集不需要数据增强
#         self.test_transform = transforms.Compose([
#             transforms.ToTensor(),
#         ])
    

#     def setup(self, stage=None):

#         image_transform = transforms.Compose([
#             transforms.RandomResizedCrop(224),  # 随机裁剪到224x224大小
#             transforms.RandomHorizontalFlip(),  # 随机水平翻转
#             transforms.ToTensor(),              # 转换为Tensor格式
#         ])
#         self.train_dataset = MedDataset(
#             self.config["data"]['train_img_path'], 
#             glob(f'{self.config["data"]["label_path"]}/*.csv'), 
#             self.config['data']['train_text_desc_path'],
#             image_transform = image_transform,
#         )
    
#         image_transform = transforms.Compose([
#             transforms.RandomResizedCrop(224),  # 随机裁剪到224x224大小
#             transforms.ToTensor(),              # 转换为Tensor格式
#         ])
#         self.test_dataset = MedDataset(
#             self.config["data"]['test_img_path'], 
#             glob(f'{self.config["data"]["label_path"]}/*.csv'), 
#             self.config['data']['test_text_desc_path'],
#             image_transform = image_transform,
#         )
    
#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, 
#                          shuffle=True, num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn)

#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size, 
#                          num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn)

#     def val_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size, 
#                          num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn)
    

# if __name__ == "__main__":
#     pass
