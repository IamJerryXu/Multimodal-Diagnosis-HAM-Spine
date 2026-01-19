import torch
from torch import nn
from torchvision import models
from transformers import ConvNextForImageClassification

from .BERT import BertEncoder


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.key_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.value_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        query = self.query_conv(x)
        key = self.key_conv(y)
        value = self.value_conv(y)
        attention = self.softmax(
            torch.matmul(
                query.view(query.size(0), query.size(1), -1).permute(0, 2, 1),
                key.view(key.size(0), key.size(1), -1),
            )
        )
        out = torch.matmul(
            attention,
            value.view(value.size(0), value.size(1), -1).permute(0, 2, 1),
        )
        return out.permute(0, 2, 1).view(x.size())


class OurClassfierConvnextV2(nn.Module):
    # 使用 image/text 作为 kqv 的双向融合版本
    def __init__(self, num_labels=2, pretrained=True, pretrained_path="/data/QLI/ConNexT/convnext-base-224"):
        super().__init__()
        self.text_encoder = BertEncoder()
        self._use_hf = False

        if pretrained and pretrained_path:
            try:
                hf_model = ConvNextForImageClassification.from_pretrained(pretrained_path)
                self.image_encoder = hf_model.convnext
                self.conv = nn.Conv2d(in_channels=1024, out_channels=768, kernel_size=1)
                self._use_hf = True
            except Exception:
                self._use_hf = False

        if not self._use_hf:
            weights = None
            if pretrained:
                try:
                    weights = models.ConvNeXt_Base_Weights.DEFAULT
                except Exception:
                    weights = None
            try:
                convnext_model = models.convnext_base(weights=weights)
            except Exception:
                convnext_model = models.convnext_base(weights=None)

            self.image_encoder = convnext_model.features
            self.conv = nn.Conv2d(in_channels=1024, out_channels=768, kernel_size=1)
        self.textbased_cross_attention = CrossAttention(dim=768)
        self.imagbased_cross_attention = CrossAttention(dim=768)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, num_labels)

    def forward(self, batch_data):
        text_embedding = self.text_encoder(
            batch_data["input_ids"], batch_data["attention_mask"]
        )
        if self._use_hf:
            image_embedding = self.image_encoder(batch_data["transformed_image"]).last_hidden_state
        else:
            image_embedding = self.image_encoder(batch_data["transformed_image"])
        image_embedding_reduced = self.conv(image_embedding)
        text_embedding_expanded = text_embedding.unsqueeze(-1).unsqueeze(-1)

        text_fused_features = self.textbased_cross_attention(
            image_embedding_reduced, text_embedding_expanded
        )
        pooled_features_1 = self.avg_pool(text_fused_features).view(
            batch_data["transformed_image"].shape[0], 768
        )

        imag_fused_features = self.imagbased_cross_attention(
            text_embedding_expanded, image_embedding_reduced
        )
        pooled_features_2 = self.avg_pool(imag_fused_features).view(
            batch_data["transformed_image"].shape[0], 768
        )

        output = self.fc(pooled_features_1 + pooled_features_2)
        return output
