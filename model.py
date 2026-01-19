import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import ImageEncoder, TextEncoder
from modules.fusion_blocks import (
    FusionModule,
    ConcatFusionModule,
    MultiScaleFusionModule,
    WeightedConcatFusionModule,
    HadamardFusionModule,
    BilinearFusionModule,
    SSMFusionModule,
    VMambaFusionModule,
)
from modules.heads import ResidualClassifier, AttentionPoolingClassifier, build_kan_head
from modules.tabular import TabularEncoder
from modules.gating import DualExpertGate
from modules.sequence_blocks import SequenceEncoder


class MultimodalBaselineModel(nn.Module):
    def __init__(
        self,
        num_classes,
        image_feature_dim=512,
        text_feature_dim=768,
        hidden_dim=256,
        dropout=0.2,
        pretrained_image=True,
        image_weights_path="/home/medteam/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth",
        text_model_name="bert-base-uncased",
        num_heads=8,
        image_backbone="resnet18",
        classifier_type="mlp",
        fusion_type="basic",
        text_pool="cls",
        kan_num_groups=8,
        kan_act_mode="gelu",
        tabular_enabled=False,
        tabular_input_dim=0,
        tabular_hidden_dim=128,
        tabular_dropout=0.1,
        gate_enabled=False,
        gate_hidden_dim=128,
        gate_use_entropy=True,
        gate_local_mode="image_only",
        gate_context_mode="full",
        sequence_enabled=False,
        sequence_type="lstm",
        sequence_hidden_dim=256,
        sequence_num_layers=1,
        sequence_bidirectional=True,
        sequence_dropout=0.1,
        sequence_num_heads=4,
        global_local_enabled=False,
        global_local_crop_ratio=0.6,
        global_local_combine="avg",
    ):
        super(MultimodalBaselineModel, self).__init__()

        # Clamp dropout to a lighter value (suggested 0.1) to avoid over-regularizing.
        fusion_dropout = min(dropout, 0.1)
        head_dropout = min(dropout, 0.1)

        self.fusion_type = fusion_type

        self.tabular_enabled = tabular_enabled
        self.sequence_enabled = sequence_enabled
        self.global_local_enabled = global_local_enabled
        self.global_local_crop_ratio = global_local_crop_ratio
        self.global_local_combine = global_local_combine

        self.image_encoder = ImageEncoder(
            feature_dim=hidden_dim,
            pretrained=pretrained_image,
            weights_path=image_weights_path,
            backbone=image_backbone,
            multi_scale=(fusion_type == "multiscale"),
        )

        if self.sequence_enabled:
            self.sequence_encoder = SequenceEncoder(
                input_dim=hidden_dim,
                hidden_dim=sequence_hidden_dim,
                encoder_type=sequence_type,
                num_layers=sequence_num_layers,
                bidirectional=sequence_bidirectional,
                dropout=sequence_dropout,
                num_heads=sequence_num_heads,
            )
            self.sequence_proj = (
                nn.Linear(sequence_hidden_dim, hidden_dim)
                if sequence_hidden_dim != hidden_dim
                else nn.Identity()
            )

        self.global_local_proj = None
        if self.global_local_enabled and self.global_local_combine == "concat":
            self.global_local_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.text_encoder = TextEncoder(
            model_path=text_model_name, feature_dim=text_feature_dim
        )

        if fusion_type == "multiscale":
            self.fusion = MultiScaleFusionModule(
                text_dim=text_feature_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=fusion_dropout,
            )
        elif fusion_type == "hadamard":
            self.fusion = HadamardFusionModule(
                text_dim=text_feature_dim,
                hidden_dim=hidden_dim,
                text_pool=text_pool,
            )
        elif fusion_type == "bilinear":
            self.fusion = BilinearFusionModule(
                text_dim=text_feature_dim,
                hidden_dim=hidden_dim,
                text_pool=text_pool,
            )
        elif fusion_type == "mamba":
            self.fusion = SSMFusionModule(
                text_dim=text_feature_dim,
                hidden_dim=hidden_dim,
                text_pool=text_pool,
            )
        elif fusion_type == "vmamba":
            self.fusion = VMambaFusionModule(
                text_dim=text_feature_dim,
                hidden_dim=hidden_dim,
                text_pool=text_pool,
            )
        elif fusion_type == "weighted_concat":
            self.fusion = WeightedConcatFusionModule(
                text_dim=text_feature_dim,
                hidden_dim=hidden_dim,
                text_pool=text_pool,
            )
        elif fusion_type == "concat":
            self.fusion = ConcatFusionModule(
                text_dim=text_feature_dim,
                hidden_dim=hidden_dim,
                text_pool=text_pool,
            )
        else:
            self.fusion = FusionModule(
                text_dim=text_feature_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=fusion_dropout,
            )

        if self.tabular_enabled:
            if tabular_input_dim <= 0:
                raise ValueError("tabular_input_dim must be > 0 when tabular is enabled.")
            self.tabular_encoder = TabularEncoder(
                tabular_input_dim,
                hidden_dim=tabular_hidden_dim,
                dropout=tabular_dropout,
            )
            self.tabular_fusion = nn.Sequential(
                nn.Linear(hidden_dim + tabular_hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(head_dropout),
            )

        self.gate_enabled = gate_enabled
        self.gate_local_mode = gate_local_mode
        self.gate_context_mode = gate_context_mode
        if self.gate_enabled:
            self.gate = DualExpertGate(
                lesion_dim=hidden_dim,
                context_dim=hidden_dim,
                hidden_dim=gate_hidden_dim,
                use_entropy=gate_use_entropy,
            )
        
        self.classifier_type = classifier_type
        if classifier_type == "kan":
            self.classifier = build_kan_head(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=head_dropout,
                num_groups=kan_num_groups,
                act_mode=kan_act_mode,
            )
        elif classifier_type == "residual":
            self.classifier = ResidualClassifier(hidden_dim, hidden_dim, num_classes, head_dropout)
        elif classifier_type == "attention_pooling":
            self.classifier = AttentionPoolingClassifier(hidden_dim, hidden_dim, num_classes, num_heads, head_dropout)
        else:
            # Linear(H -> H -> num_classes)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward_features(
        self,
        image_input,
        text_input_ids,
        text_attention_mask,
        tabular_input=None,
        ablation_mode=None,
    ):
        image_tokens, pooled_image = self._encode_image_tokens(image_input)

        if ablation_mode == "image_only":
            return pooled_image

        text_tokens = self.text_encoder(text_input_ids, text_attention_mask)

        if ablation_mode == "text_off":
            text_tokens = torch.zeros_like(text_tokens)

        if self.sequence_enabled and isinstance(self.fusion, MultiScaleFusionModule):
            image_tokens = {
                "layer2": image_tokens,
                "layer3": image_tokens,
                "layer4": image_tokens,
            }

        fused_features = self.fusion(image_tokens, text_tokens, text_attention_mask)

        if self.tabular_enabled:
            if tabular_input is None:
                raise ValueError("tabular_input is required when tabular is enabled.")
            tabular_feat = self.tabular_encoder(tabular_input)
            fused_features = self.tabular_fusion(
                torch.cat([fused_features, tabular_feat], dim=1)
            )

        return fused_features

    def forward(
        self,
        image_input,
        text_input_ids,
        text_attention_mask,
        tabular_input=None,
        ablation_mode=None,
    ):
        if ablation_mode is not None or not self.gate_enabled:
            fused_features = self.forward_features(
                image_input,
                text_input_ids,
                text_attention_mask,
                tabular_input=tabular_input,
                ablation_mode=ablation_mode,
            )
            return self.classifier(fused_features)

        context_mode = None if self.gate_context_mode == "full" else self.gate_context_mode
        context_feat = self.forward_features(
            image_input,
            text_input_ids,
            text_attention_mask,
            tabular_input=tabular_input,
            ablation_mode=context_mode,
        )
        local_feat = self.forward_features(
            image_input,
            text_input_ids,
            text_attention_mask,
            tabular_input=tabular_input,
            ablation_mode=self.gate_local_mode,
        )

        logits_context = self.classifier(context_feat)
        logits_local = self.classifier(local_feat)
        entropy = None
        if self.gate.use_entropy:
            probs = torch.softmax(logits_local, dim=1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=1, keepdim=True)

        alpha = self.gate(local_feat, context_feat, entropy)
        return alpha * logits_local + (1 - alpha) * logits_context

    def _pool_image_tokens(self, image_tokens):
        if isinstance(image_tokens, dict):
            pooled = []
            for key in ("layer2", "layer3", "layer4"):
                tokens = image_tokens[key]
                pooled.append(tokens.mean(dim=1))
            return sum(pooled) / float(len(pooled))
        return image_tokens.mean(dim=1)

    def _center_crop(self, x, ratio):
        _, _, h, w = x.shape
        ch = max(1, int(h * ratio))
        cw = max(1, int(w * ratio))
        y0 = max(0, (h - ch) // 2)
        x0 = max(0, (w - cw) // 2)
        cropped = x[:, :, y0:y0 + ch, x0:x0 + cw]
        if cropped.shape[-2:] != (h, w):
            cropped = F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)
        return cropped

    def _combine_tokens(self, global_tokens, local_tokens):
        if isinstance(global_tokens, dict) or isinstance(local_tokens, dict):
            if not (isinstance(global_tokens, dict) and isinstance(local_tokens, dict)):
                raise ValueError("global/local token types must match.")
            combined = {}
            for key in global_tokens.keys():
                combined[key] = 0.5 * (global_tokens[key] + local_tokens[key])
            return combined
        if self.global_local_combine == "concat":
            combined = torch.cat([global_tokens, local_tokens], dim=-1)
            return self.global_local_proj(combined)
        return 0.5 * (global_tokens + local_tokens)

    def _encode_image_tokens(self, image_input):
        if image_input.dim() == 5:
            if not self.sequence_enabled:
                raise ValueError("Sequence input provided but sequence encoder is disabled.")
            batch_size, seq_len = image_input.size(0), image_input.size(1)
            flat = image_input.view(batch_size * seq_len, *image_input.shape[2:])
            tokens = self.image_encoder(flat)
            if self.global_local_enabled:
                local_flat = self._center_crop(flat, self.global_local_crop_ratio)
                local_tokens = self.image_encoder(local_flat)
                tokens = self._combine_tokens(tokens, local_tokens)
            pooled = self._pool_image_tokens(tokens)
            seq_feats = pooled.view(batch_size, seq_len, -1)
            seq_encoded = self.sequence_encoder(seq_feats)
            seq_encoded = self.sequence_proj(seq_encoded)
            return seq_encoded.unsqueeze(1), seq_encoded

        tokens = self.image_encoder(image_input)
        if self.global_local_enabled:
            local_input = self._center_crop(image_input, self.global_local_crop_ratio)
            local_tokens = self.image_encoder(local_input)
            tokens = self._combine_tokens(tokens, local_tokens)
        pooled = self._pool_image_tokens(tokens)
        return tokens, pooled

    def freeze_encoders(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
