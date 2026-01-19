import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet34,
    ResNet34_Weights,
)
from transformers import BertModel, BertConfig
import os


class ImageEncoder(nn.Module):
    """
    ResNet backbone that returns multi-scale patch tokens (layer2/3/4),
    each projected to a shared feature dimension.
    """

    def __init__(
        self,
        feature_dim=512,
        pretrained=True,
        weights_path=None,
        backbone="resnet18",
        multi_scale=False,
    ):
        super(ImageEncoder, self).__init__()

        self.multi_scale = multi_scale

        backbone = backbone.lower()
        if backbone not in ["resnet18", "resnet34"]:
            raise ValueError(f"Unsupported backbone: {backbone}. Use resnet18 or resnet34.")

        if backbone == "resnet18":
            build_model = resnet18
            default_weights = ResNet18_Weights.IMAGENET1K_V1
            channels = {"layer2": 128, "layer3": 256, "layer4": 512}
        else:
            build_model = resnet34
            default_weights = ResNet34_Weights.IMAGENET1K_V1
            channels = {"layer2": 128, "layer3": 256, "layer4": 512}

        if weights_path:
            print(f"正在从本地路径加载{backbone}权重: {weights_path}")
            self.model = build_model(weights=None)

            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"权重文件未找到: {weights_path}")

            state_dict = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)
        elif pretrained:
            print("正在从torchvision在线下载ImageNet预训练权重...")
            self.model = build_model(weights=default_weights)
        else:
            print(f"正在初始化一个未经预训练的{backbone}模型...")
            self.model = build_model(weights=None)

        # 不使用分类头，去掉多余参数
        self.model.fc = nn.Identity()
        # Keep blocks to tap intermediate features
        self.stem = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
        )
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4

        # Linear projections to shared feature_dim for each scale
        if self.multi_scale:
            self.proj2 = nn.Linear(channels["layer2"], feature_dim)
            self.proj3 = nn.Linear(channels["layer3"], feature_dim)
        self.proj4 = nn.Linear(channels["layer4"], feature_dim)

    def _flatten_and_project(self, feat_map, proj):
        """
        feat_map: (B, C, H, W)
        returns: (B, H*W, feature_dim)
        """
        tokens = feat_map.flatten(2).transpose(1, 2)
        return proj(tokens)

    def forward(self, x):
        """
        Returns:
          - single scale: layer4 tokens (B, N, feature_dim)
          - multi-scale: dict with layer2/3/4 tokens
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        f4 = self.layer4(x)

        if self.multi_scale:
            t2 = self._flatten_and_project(f2, self.proj2)
            t3 = self._flatten_and_project(f3, self.proj3)
            t4 = self._flatten_and_project(f4, self.proj4)
            return {"layer2": t2, "layer3": t3, "layer4": t4}

        t4 = self._flatten_and_project(f4, self.proj4)
        return t4


class TextEncoder(nn.Module):
    
    def __init__(self, model_path='bert-base-uncased', feature_dim=768):
        
        super(TextEncoder, self).__init__()
        
        if os.path.isdir(model_path):
            print(f"正在从本地文件夹加载BERT模型: {model_path}")
        else:
            print(f"正在从HuggingFace Hub在线下载BERT模型: {model_path}")

        
        try:
            self.model = BertModel.from_pretrained(model_path)
        except Exception as e:
            print(f"加载BERT模型失败: {e}")
            raise
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # return token-level features (B, seq_len, hidden)
        token_features = outputs.last_hidden_state
        return token_features
