#!/usr/bin/env python3
"""
ConvNeXt encoder wrapper modeled after the MambaVision encoder style.
Provides a simple factory `create_convnext_encoder` and a nn.Module `ConvNeXtEncoder`.
"""
import torch
import torch.nn as nn
import warnings

try:
    import timm
except Exception:
    timm = None


class ConvNeXtEncoder(nn.Module):
    """
    ConvNeXt 编码器包装：
    - 使用 timm 加载 ConvNeXt（若未安装 timm 则抛错）
    - 移除分类头并暴露特征到投影层
    - forward 返回投影后的特征向量
    """
    def __init__(self, output_dim=768, pretrained=True, model_variant='large', model_paths=None, **kwargs):
        super().__init__()
        if timm is None:
            raise ImportError("ConvNeXt encoder requires `timm` package. Please install timm.")
        model_name = model_variant if model_variant.startswith("convnext") else f"convnext_{model_variant}"
        # 使用 num_classes=0 尝试返回不带 head 的骨干（timm 的行为会因版本而异）
        try:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        except Exception:
            # 回退：创建带分类头的模型然后替换为 Identity
            self.backbone = timm.create_model(model_name, pretrained=pretrained)
            if hasattr(self.backbone, "head"):
                self.backbone.head = nn.Identity()
            elif hasattr(self.backbone, "classifier"):
                self.backbone.classifier = nn.Identity()

        # 尝试读取特征维度
        feature_dim = None
        if hasattr(self.backbone, "head") and hasattr(self.backbone.head, "in_features"):
            feature_dim = self.backbone.head.in_features
        elif hasattr(self.backbone, "fc") and hasattr(self.backbone.fc, "in_features"):
            feature_dim = self.backbone.fc.in_features

        # 如果仍未知，使用一次前向推理来推断特征维度（安全地在 no_grad 下）
        if feature_dim is None:
            try:
                self.backbone.eval()
                with torch.no_grad():
                    sample = torch.randn(1, 3, 224, 224)
                    if hasattr(self.backbone, "forward_features"):
                        feat = self.backbone.forward_features(sample)
                    else:
                        feat = self.backbone(sample)
                feature_dim = feat.shape[1]
            except Exception:
                warnings.warn("无法推断 backbone feature dim，使用 1024 作为默认值。")
                feature_dim = 1024

        self.projection = nn.Linear(feature_dim, output_dim)

    def forward(self, x):
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)
        return self.projection(features)


def create_convnext_encoder(output_dim=768, pretrained=True, model_variant='large', model_paths=None, **kwargs):
    """
    工厂函数，返回 ConvNeXt 编码器实例。
    """
    # model_paths is accepted for API compatibility (may contain local checkpoint paths); currently ignored.
    return ConvNeXtEncoder(output_dim=output_dim, pretrained=pretrained, model_variant=model_variant, model_paths=model_paths, **kwargs)


