import argparse
import os
import sys
import yaml
import torch
import pandas as pd
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from model import MultimodalBaselineModel
    from data_loader import create_data_loader
except ImportError:
    print("错误：无法从 model.py 或 data_loader.py 导入所需的类或函数。")
    print("请确保 predict.py 与 model.py, data_loader.py 在同一目录下，并且包含了正确的类/函数定义。")
    exit()


def load_config(config_path=None):
    """加载YAML配置文件。评估时将使用固定的配置文件来构建模型。"""
    if config_path is None:
        config_path = os.path.join(BASE_DIR, 'config.yml')
    if not os.path.exists(config_path):
        print(f"错误：配置文件 {config_path} 不存在。")
        exit()
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

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

def predict(model, data_loader, device, tta_cfg=None):
    
    model.eval()  
    predictions = []
    image_ids = []
    
    with torch.no_grad(): 
        for batch in tqdm(data_loader, desc="正在推理..."):
            
            images, input_ids, attention_mask, tabular, _, batch_image_ids = batch
            
            
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if tabular is not None:
                tabular = tabular.to(device)
            
            
            if tta_cfg and tta_cfg.get("enabled"):
                tta_transforms = tta_cfg.get("transforms", ["hflip"])
                logits_list = []
                for aug_images in _apply_tta(images, tta_transforms):
                    logits_list.append(
                        model(aug_images, input_ids, attention_mask, tabular_input=tabular)
                    )
                logits = torch.stack(logits_list, dim=0).mean(dim=0)
            else:
                logits = model(images, input_ids, attention_mask, tabular_input=tabular)
            
            
            _, predicted_labels = torch.max(logits.data, 1)
            
            
            predictions.extend(predicted_labels.cpu().numpy())
            image_ids.extend(batch_image_ids)
            
    return image_ids, predictions

def main(args):
    """主执行函数"""
    
    
    
    print(f"正在从 {args.config} 加载模型配置...")
    config = load_config(args.config)

    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")

    
    print("正在创建测试数据加载器...")
    
    test_loader = create_data_loader(
        config, 
        split='test',
        test_image_dir=args.image_dir, 
        test_json_path=args.json_path
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
        print(f"错误：找不到指定的模型权重文件: {args.model_path}")
        exit()
    print(f"正在加载模型权重: {args.model_path}")
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        print("请确保您的模型权重文件与 config.yml 中定义的模型结构匹配。")
        exit()

    
    tta_cfg = config.get("inference", {}).get("tta", {})
    print("模型加载完毕，开始执行预测。")
    image_ids, predicted_labels = predict(model, test_loader, device, tta_cfg=tta_cfg)
    
    
    print("预测完成，正在生成提交文件...")
    submission_df = pd.DataFrame({
        'image_id': image_ids,
        'predicted_label': predicted_labels
    })
    
    
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    submission_df.to_csv(args.output_path, index=False)
    
    print("-" * 50)
    print(f"成功！提交文件已保存至: {args.output_path}")
    print(f"总共预测了 {len(submission_df)} 条数据。")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="课程大作业模型预测脚本",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--image_dir', type=str, required=True, help='【必需】测试集图像文件夹的路径。')
    parser.add_argument('--json_path', type=str, required=True, help='【必需】测试集JSON元数据文件的路径。')
    parser.add_argument('--model_path', type=str, required=True, help='【必需】您训练好的模型权重文件（.pth文件）的路径。')
    parser.add_argument('--output_path', type=str, required=True, help='【必需】预测结果的输出路径（例如：./submission.csv）。')
    parser.add_argument('--config', type=str, default='config.yml', help='【可选】模型配置文件路径，默认为 "config.yml"。')
    
    args = parser.parse_args()
    main(args)
