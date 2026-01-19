import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .bert import BertEncoder
from .attention import MultiHeadCrossAttention_v2, compute_kl_divergence, SelfAttention


class Resnet50WithOurs(nn.Module):
    def __init__(self, num_labels=6, loss_class="KL_loss", bert_path="/data/QLI/BERT_pretain"):
        super().__init__()
        self.text_encoder = BertEncoder(model_path=bert_path)

        backbone = models.resnet50(pretrained=True)
        backbone.fc = nn.Linear(backbone.fc.in_features, 768)
        self.image_encoder = backbone

        self.textbased_cross_attention = MultiHeadCrossAttention_v2(dim=768, num_heads=1)
        self.imagbased_cross_attention = MultiHeadCrossAttention_v2(dim=768, num_heads=1)
        self.I2Iattention = SelfAttention(input_dim=768)
        self.fc = nn.Linear(768 * 2, num_labels)
        self.fc_image = self._build_mlp(768, num_labels)
        self.fc_text = self._build_mlp(768, num_labels)
        self.loss_class = loss_class
        self.loss = nn.CrossEntropyLoss()

    def _build_mlp(self, input_dim, num_labels):
        return nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels),
        )

    def forward(self, batch_data):
        text_embedding = self.text_encoder(
            batch_data["input_ids"], batch_data["attention_mask"]
        )
        image_embedding = self.image_encoder(batch_data["transformed_image"])
        text_embedding_expanded = text_embedding.unsqueeze(dim=1)
        image_embedding_pooled = image_embedding.unsqueeze(dim=1)

        text_fused_features = self.textbased_cross_attention(
            image_embedding_pooled, text_embedding_expanded
        )
        pooled_features_1 = text_fused_features.view(
            batch_data["transformed_image"].shape[0], 768
        )

        imag_fused_features = self.imagbased_cross_attention(
            text_embedding_expanded, image_embedding_pooled
        )
        pooled_features_2 = imag_fused_features.view(
            batch_data["transformed_image"].shape[0], 768
        )

        output = {
            "image_text": self.fc(torch.cat([pooled_features_1, pooled_features_2], dim=1)),
            "text": self.fc_text(text_fused_features),
            "image": self.fc_image(imag_fused_features),
        }
        return output

    def cal_loss(self, output, labels):
        if self.loss_class == "textimage_loss":
            return self.loss(output["image_text"], labels)
        if self.loss_class == "text_image_textimage_loss":
            return (
                self.loss(output["image"], labels)
                + self.loss(output["text"], labels)
                + self.loss(output["image_text"], labels)
            )
        return self.compute_kl_loss(output, labels)

    def compute_kl_loss(self, output, labels):
        image_logits = output["image"]
        text_logits = output["text"]
        image_text_logits = output["image_text"]
        image_prob = F.softmax(image_logits, dim=-1)
        text_prob = F.softmax(text_logits, dim=-1)

        kl = (
            compute_kl_divergence(image_prob, text_prob)
            + compute_kl_divergence(text_prob, image_prob)
        ) / 2
        kl = torch.nan_to_num(kl, nan=0.0, posinf=10.0, neginf=0.0)
        kl = torch.clamp(kl, min=0.0, max=10.0)
        image_loss = F.cross_entropy(image_logits, labels)
        text_loss = F.cross_entropy(text_logits, labels)
        image_text_loss = F.cross_entropy(image_text_logits, labels)
        weight_factor = torch.exp(kl)
        weighted_image_text_loss = torch.mean(weight_factor * image_text_loss)
        return 0.3 * image_loss + 0.6 * text_loss + 1.1 * weighted_image_text_loss
