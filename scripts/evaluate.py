import os
import sys
import yaml
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model import MultimodalBaselineModel
from data_loader import create_data_loader

def load_config(config_path=None):
    """加载YAML配置文件"""
    if config_path is None:
        config_path = os.path.join(BASE_DIR, 'config.yml')
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def evaluate(model, data_loader, device):
    """在测试集上评估模型并返回准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            images, input_ids, attention_mask, tabular, labels, _ = batch
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if tabular is not None:
                tabular = tabular.to(device)
            labels = labels.to(device)
            
            logits = model(images, input_ids, attention_mask, tabular_input=tabular)
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy

def main(args):
    # 1. 加载指定的配置文件
    config = load_config(args.config)

    # 2. 设置设备
    device = torch.device(config['training']['device'])

    # 3. 创建测试数据加载器
    #    直接将命令行传入的路径传递给data_loader
    print("正在创建测试数据加载器...")
    test_loader = create_data_loader(
        config, 
        split='test', 
        test_image_dir=args.test_image_dir, 
        test_json_path=args.test_json_path
    )

    # 4. 初始化模型
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
    
    # 5. 加载学生提交的模型权重
    print(f"正在加载模型权重从: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # 6. 在隐藏测试集上评估
    print("开始在隐藏测试集上评估...")
    accuracy = evaluate(model, test_loader, device)
    
    print(f"\n评估完成！")
    print(f"模型在隐藏测试集上的准确率: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在隐藏测试集上评估学生模型")
    parser.add_argument('--model_path', type=str, required=True, help='学生提交的模型权重文件路径 (.pth)')
    parser.add_argument('--test_image_dir', type=str, required=True, help='隐藏测试集的图像文件夹路径')
    parser.add_argument('--test_json_path', type=str, required=True, help='隐藏测试集的JSON元数据文件路径')
    parser.add_argument('--config', type=str, default='config.yml', help='用于模型架构的配置文件路径')
    
    args = parser.parse_args()
    main(args)
