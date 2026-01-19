import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from .BERT import BertEncoder
from .block.kan1 import KAN1
from .block.moe import MoE
from .ourmodel import OurClassfierConvnextV2
try:
    from .block.len4mamba import MultimodalMamba, MultimodalMambaWithKANAttention
except ImportError:
    MultimodalMamba = None
    MultimodalMambaWithKANAttention = None
# from .block.mamba_vision import mamba_vision_S, mamba_vision_T, MambaVisionEncoder # MambaVision is no longer needed
import math

from torchmetrics import MetricCollection, F1Score, AUROC, Accuracy, Precision, Recall

# ==============================================================================
# 1. 新增：定义 ConvNeXt 图像编码器
# ==============================================================================
class ConvNeXtEncoder(nn.Module):
    """
    使用 torchvision 的 ConvNeXt 作为图像编码器。
    - 加载预训练的 convnext_large 模型。
    - 移除原始的分类头，只使用特征提取部分。
    - 将输出的特征图 [B, C, H, W] 展平为 [B, C, H*W] 以匹配后续模块。
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载预训练的 ConvNeXt-Large 模型
        weights = models.ConvNeXt_Large_Weights.DEFAULT if pretrained else None
        convnext_model = models.convnext_large(weights=weights)

        # 只保留特征提取部分，移除分类器
        self.features = convnext_model.features
        # ConvNeXt-Large 的输出特征维度通常为 1536
        self.output_dim = 1536

    def forward(self, x):
        # 1. 提取特征图，输出形状: [B, 768, 7, 7]
        feature_map = self.features(x)
        
        # 2. 将空间维度 (H, W) 展平，输出形状: [B, 768, 49]
        #    这与原 MambaVision 编码器输出 [B, C, N] 的格式保持一致
        return feature_map.flatten(2)


# ==============================================================================
# 2. 修改：使用 ConvNeXt 替换 MambaVision
# ==============================================================================
class BaseLineConvNeXt_KAN_mamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_classes = config['model']['num_classes']
        self.net = OurClassfierConvnextV2(
            num_labels=num_classes,
            pretrained=True,
            pretrained_path="/data/QLI/ConNexT/convnext-base-224",
        )

    def forward(self, x):
        batch = {
            "transformed_image": x["imgs"],
            "input_ids": x["texts"]["input_ids"],
            "attention_mask": x["texts"]["attention_mask"],
        }
        logits = self.net(batch)
        balance_loss = torch.zeros((), device=logits.device)
        return logits, balance_loss


class SimpleFusion(nn.Module):
    def __init__(self, text_dim, img_dim, hidden_dim, proj_dim=256):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.img_proj = nn.Linear(img_dim, proj_dim)
        self.hidden_proj = nn.Linear(hidden_dim * 2, proj_dim)

    def forward(self, text_embedding, image_embedding, first_hidden, last_hidden):
        img_global = image_embedding.mean(dim=2)
        text_token = self.text_proj(text_embedding)
        img_token = self.img_proj(img_global)
        hidden_token = self.hidden_proj(torch.cat([first_hidden, last_hidden], dim=1))
        return torch.stack([text_token, img_token, hidden_token], dim=1)


class Model4AAAI_MoE(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, config=None):
        super().__init__()
        self.save_hyperparameters()

        class_weights_list = config['train'].get('class_weights', None)
        if class_weights_list:
            print("使用加权交叉熵损失...")
            weights = torch.tensor(class_weights_list, dtype=torch.float)
            self.register_buffer('class_weights', weights)
            self.loss = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            print("使用标准交叉熵损失...")
            self.loss = nn.CrossEntropyLoss()
        
        # --- 核心修改：实例化新的模型 ---
        self.net = BaseLineConvNeXt_KAN_mamba(config)
        # -------------------------------
        
        self.config = config
        self.balance_weight = config['model']['moe'].get('balance_weight', 0.01)
        
        num_classes = config['model']['num_classes']
        task = "multiclass"
        
        macro_metrics = MetricCollection({
            'Accuracy': Accuracy(task=task, num_classes=num_classes),
            'Precision': Precision(task=task, num_classes=num_classes, average='macro'),
            'Recall': Recall(task=task, num_classes=num_classes, average='macro'),
            'F1_macro': F1Score(task=task, num_classes=num_classes, average='macro'),
        })
        self.validation_macro_metrics = macro_metrics.clone(prefix='val_')
        self.val_auroc = AUROC(task=task, num_classes=num_classes)

        per_class_metrics = MetricCollection({
            'Accuracy': Accuracy(task=task, num_classes=num_classes, average=None),
            'Precision': Precision(task=task, num_classes=num_classes, average=None),
            'Recall': Recall(task=task, num_classes=num_classes, average=None),
            'F1_score': F1Score(task=task, num_classes=num_classes, average=None),
        })
        self.validation_per_class_metrics = per_class_metrics.clone(prefix='val_per_class_')

        self.class_names = config.get('data', {}).get('class_names', [f'class_{i}' for i in range(num_classes)])
        if len(self.class_names) != num_classes:
            raise ValueError("The number of class_names provided does not match num_classes.")

    def forward(self, x):
        return self.net(x)[0]

    def training_step(self, batch, batch_idx):
        pred, balance_loss = self.net(batch)
        cls_loss = self.loss(pred, batch["labels"])
        total_loss = cls_loss + self.balance_weight * balance_loss
        
        accuracy = (pred.argmax(dim=-1) == batch["labels"]).float().mean()
        
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        pred, _ = self.net(batch)
        loss = self.loss(pred, batch["labels"])
        pred_labels = torch.argmax(pred, dim=1)

        self.validation_macro_metrics.update(pred_labels, batch["labels"])
        self.val_auroc.update(pred, batch["labels"])
        self.validation_per_class_metrics.update(pred_labels, batch["labels"])

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        macro_results = self.validation_macro_metrics.compute()
        auroc_result = self.val_auroc.compute()
        per_class_results = self.validation_per_class_metrics.compute()

        val_acc = macro_results['val_Accuracy']
        val_precision = macro_results['val_Precision']
        val_recall = macro_results['val_Recall']
        val_f1 = macro_results['val_F1_macro']
        
        self.log('val_Accuracy', val_acc, prog_bar=True)
        self.log('val_Precision_macro', val_precision, prog_bar=False)
        self.log('val_Recall_macro', val_recall, prog_bar=False)
        self.log('val_F1_macro', val_f1, prog_bar=True)
        self.log('val_AUROC', auroc_result, prog_bar=False)

        for metric_name, values in per_class_results.items():
            base_metric_name = metric_name.replace('val_per_class_', '')
            for i, value in enumerate(values):
                class_name = self.class_names[i]
                log_tag = f"per_class/{base_metric_name}_{class_name}"
                self.log(log_tag, value, prog_bar=False)

        self.validation_macro_metrics.reset()
        self.val_auroc.reset()
        self.validation_per_class_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6),
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
