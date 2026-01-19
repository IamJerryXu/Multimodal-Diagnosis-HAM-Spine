import argparse
import os
import sys
import yaml
import torch
from tqdm import tqdm
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model import MultimodalBaselineModel
from data_loader import create_data_loader


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(BASE_DIR, 'config.yml')
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def _apply_tta(images, tta_transforms):
    variants = [images]
    for name in tta_transforms:
        if name == "hflip":
            variants.append(images.flip(-1))
        elif name == "vflip":
            variants.append(images.flip(-2))
        elif name == "rot90":
            variants.append(torch.rot90(images, k=1, dims=(-2, -1)))
    return variants


def evaluate(model, data_loader, device, ablation_mode=None, tta_cfg=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating [{ablation_mode or 'full'}]"):
            images, input_ids, attention_mask, tabular, labels, _ = batch
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if tabular is not None:
                tabular = tabular.to(device)
            labels = labels.to(device)

            if tta_cfg and tta_cfg.get("enabled"):
                tta_transforms = tta_cfg.get("transforms", ["hflip"])
                logits_list = []
                for aug_images in _apply_tta(images, tta_transforms):
                    logits_list.append(
                        model(
                            aug_images,
                            input_ids,
                            attention_mask,
                            tabular_input=tabular,
                            ablation_mode=ablation_mode,
                        )
                    )
                logits = torch.stack(logits_list, dim=0).mean(dim=0)
            else:
                logits = model(
                    images,
                    input_ids,
                    attention_mask,
                    tabular_input=tabular,
                    ablation_mode=ablation_mode,
                )
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total if total else 0.0


def main(args):
    config = load_config(args.config)
    device = torch.device(config['training']['device'])

    print("正在创建数据加载器...")
    test_loader = create_data_loader(
        config,
        split='test',
        test_image_dir=args.image_dir,
        test_json_path=args.json_path,
    )

    print("正在初始化模型...")
    model_config = config['model']
    tabular_cfg = model_config.get("tabular", {})
    seq_cfg = model_config.get("sequence_encoder", {})
    gl_cfg = model_config.get("global_local", {})
    if tabular_cfg.get("enabled"):
        tabular_cfg["input_dim"] = test_loader.dataset.tabular_dim

    gate_cfg = model_config.get("gate", {})
    model = MultimodalBaselineModel(
        num_classes=model_config['num_classes'],
        image_feature_dim=model_config['image_encoder']['feature_dim'],
        text_feature_dim=model_config['text_encoder']['feature_dim'],
        hidden_dim=model_config['mlp_head']['hidden_dim'],
        dropout=model_config['mlp_head']['dropout'],
        pretrained_image=model_config['image_encoder']['pretrained'],
        image_weights_path=model_config['image_encoder'].get('weights_path', None),
        image_backbone=model_config['image_encoder'].get('backbone', 'resnet18'),
        text_model_name=model_config['text_encoder']['model_name'],
        classifier_type=model_config.get('classifier_type', 'mlp'),
        fusion_type=model_config.get('fusion_type', 'basic'),
        text_pool=model_config.get('text_pool', 'cls'),
        tabular_enabled=tabular_cfg.get("enabled", False),
        tabular_input_dim=tabular_cfg.get("input_dim", 0),
        tabular_hidden_dim=tabular_cfg.get("hidden_dim", 128),
        tabular_dropout=tabular_cfg.get("dropout", 0.1),
        gate_enabled=gate_cfg.get("enabled", False),
        gate_hidden_dim=gate_cfg.get("hidden_dim", 128),
        gate_use_entropy=gate_cfg.get("use_entropy", True),
        gate_local_mode=gate_cfg.get("local_mode", "image_only"),
        gate_context_mode=gate_cfg.get("context_mode", "full"),
        sequence_enabled=seq_cfg.get("enabled", False),
        sequence_type=seq_cfg.get("type", "lstm"),
        sequence_hidden_dim=seq_cfg.get("hidden_dim", model_config['mlp_head']['hidden_dim']),
        sequence_num_layers=seq_cfg.get("num_layers", 1),
        sequence_bidirectional=seq_cfg.get("bidirectional", True),
        sequence_dropout=seq_cfg.get("dropout", 0.1),
        sequence_num_heads=seq_cfg.get("num_heads", 4),
        global_local_enabled=gl_cfg.get("enabled", False),
        global_local_crop_ratio=gl_cfg.get("crop_ratio", 0.6),
        global_local_combine=gl_cfg.get("combine", "avg"),
    ).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型权重不存在: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    tta_cfg = config.get("inference", {}).get("tta", {})
    modes = [
        ("full_fusion", None),
        ("image_only", "image_only"),
        ("text_off", "text_off"),
    ]

    results = {
        "model_path": args.model_path,
        "image_dir": args.image_dir,
        "json_path": args.json_path,
        "config": args.config,
        "metrics": {},
    }
    for name, mode in modes:
        acc = evaluate(model, test_loader, device, ablation_mode=mode, tta_cfg=tta_cfg)
        print(f"[{name}] accuracy: {acc:.2f}%")
        results["metrics"][name] = acc

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"ablation_{stamp}.yml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(results, f, sort_keys=False, allow_unicode=True)
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模态消融评估脚本")
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径 (.pth)')
    parser.add_argument('--image_dir', type=str, default='', help='测试集图像文件夹路径')
    parser.add_argument('--json_path', type=str, default='', help='测试集JSON路径')
    parser.add_argument('--config', type=str, default='config.yml', help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default=os.path.join(BASE_DIR, 'results', 'ablation'), help='结果输出目录')
    args = parser.parse_args()
    main(args)
