import argparse
import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model import MultimodalBaselineModel
from data_loader import create_data_loader
from analysis_tools import GradCAM, visualize_cam, FeatureRankAnalyzer

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(BASE_DIR, 'config.yml')
    if not os.path.exists(config_path):
        print(f"错误：配置文件 {config_path} 不存在。")
        exit()
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def main(args):
    # 1. Load Config
    print(f"正在从 {args.config} 加载配置...")
    config = load_config(args.config)
    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. Create Data Loader (Test set)
    print("正在创建数据加载器...")
    # Use a smaller batch size for analysis to avoid OOM with hooks if necessary, 
    # but standard batch size should be fine.
    test_loader = create_data_loader(
        config,
        split='test',
        test_image_dir=args.image_dir,
        test_json_path=args.json_path,
        num_workers=0,
    )

    # 3. Initialize Model (detect fusion type from checkpoint)
    print("正在初始化模型...")
    model_config = config['model']

    if not os.path.exists(args.model_path):
        print(f"错误：找不到模型权重文件: {args.model_path}")
        exit()
    print(f"正在加载权重: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location="cpu")
    use_multiscale = any(k.startswith("fusion.cross_l2") for k in state_dict.keys())
    if use_multiscale:
        print("检测到多层融合权重，启用 multiscale 模式")

    ablation_mode = args.ablation_mode
    if not ablation_mode:
        ablation_mode = model_config.get('ablation_mode', 'none')
    if ablation_mode not in ("none", "image_only", "text_off"):
        print(f"未知 ablation_mode: {ablation_mode}，将使用 none")
        ablation_mode = "none"

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
        fusion_type="multiscale" if use_multiscale else "basic",
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

    # 4. Load Weights
    model.load_state_dict(state_dict)
    model.eval()

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    cam_dir = os.path.join(args.output_dir, "grad_cam")
    cam_layer_dir = os.path.join(args.output_dir, "grad_cam_layers")
    os.makedirs(cam_dir, exist_ok=True)
    os.makedirs(cam_layer_dir, exist_ok=True)

    # --- Feature Rank Analysis Setup ---
    rank_analyzer = FeatureRankAnalyzer(model)
    
    # --- Grad-CAM Setup ---
    # Target multiple layers for multi-scale analysis
    # We use the last block of layer2, layer3, and layer4
    target_layers = [
        ("stem", model.image_encoder.stem),
        ("layer1", model.image_encoder.layer1[-1]),
        ("layer2", model.image_encoder.layer2[-1]),
        ("layer3", model.image_encoder.layer3[-1]),
        ("layer4", model.image_encoder.layer4[-1]),
    ]
    grad_cam = GradCAM(model, target_layers)

    print("开始分析...")
    
    # Limit Grad-CAM to first N images to save time/space
    num_cam_images = 20
    cam_count = 0
    
    with torch.enable_grad(): # Grad-CAM requires gradients even in eval mode
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Running Analysis")):
            images, input_ids, attention_mask, tabular, labels, image_ids = batch
            
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if tabular is not None:
                tabular = tabular.to(device)
            labels = labels.to(device)
            
            # 1. Feature Rank Analysis (Forward pass is enough, hooks capture features)
            # We need to run a forward pass. 
            # Note: GradCAM also runs a forward pass.
            # If we run GradCAM, it does forward+backward.
            # If we just want features, we do forward.
            
            # To avoid double computation, we can let GradCAM's forward trigger the feature hook.
            # But GradCAM might zero_grad and mess things up if we are not careful?
            # Actually GradCAM calls model(images...), which triggers the hooks.
            # So running GradCAM will also populate rank_analyzer.features.
            
            # However, we only run GradCAM on a few images.
            # We need features for ALL images for robust rank analysis.
            
            if cam_count < num_cam_images:
                # Run Grad-CAM (includes forward pass)
                avg_cams, layer_cams, predicted_classes = grad_cam(
                    images,
                    input_ids,
                    attention_mask,
                    return_layer_cams=True,
                    return_avg=True,
                    ablation_mode=None if ablation_mode == "none" else ablation_mode,
                    tabular_input=tabular,
                )
                
                # Save CAM images
                for i in range(len(avg_cams)):
                    if cam_count >= num_cam_images:
                        break
                    
                    img_id = image_ids[i] if isinstance(image_ids, list) else f"batch{batch_idx}_img{i}"
                    save_path = os.path.join(cam_dir, f"{img_id}_class{predicted_classes[i].item()}_avg.jpg")
                    visualize_cam(images[i], avg_cams[i], save_path)

                    # Save per-layer CAMs
                    layer_cam_map = layer_cams[i]
                    for layer_name, cam in layer_cam_map.items():
                        layer_path = os.path.join(
                            cam_layer_dir,
                            f"{img_id}_class{predicted_classes[i].item()}_{layer_name}.jpg",
                        )
                        visualize_cam(images[i], cam, layer_path)
                    cam_count += 1
            else:
                # Just run forward pass for Feature Rank Analysis
                with torch.no_grad():
                    _ = model(
                        images,
                        input_ids,
                        attention_mask,
                        tabular_input=tabular,
                        ablation_mode=None if ablation_mode == "none" else ablation_mode,
                    )

    # --- Process Feature Rank ---
    print("正在计算特征秩...")
    features, singular_values = rank_analyzer.compute_rank()
    
    if singular_values is not None:
        rank_plot_path = os.path.join(args.output_dir, "feature_rank_distribution.png")
        rank_analyzer.plot_singular_values(singular_values, rank_plot_path)
        print(f"特征秩分析图已保存至: {rank_plot_path}")
        
        # Calculate effective rank (e.g. number of SVs needed to explain 99% variance)
        # SVs are already normalized to max=1, but for variance we need squared SVs.
        # Let's just print the number of SVs > 0.01 (1% of max)
        effective_dim = np.sum(singular_values > 0.01)
        print(f"有效特征维度 (SV > 0.01 * max): {effective_dim} / {len(singular_values)}")
    
    print(f"Grad-CAM 可视化已保存至: {cam_dir}")
    print(f"分层 Grad-CAM 已保存至: {cam_layer_dir}")
    print("分析完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型分析工具：Grad-CAM 和 特征秩分析")
    parser.add_argument('--image_dir', type=str, required=True, help='测试集图像文件夹路径')
    parser.add_argument('--json_path', type=str, required=True, help='测试集JSON路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='分析结果输出目录')
    parser.add_argument('--config', type=str, default='config.yml', help='配置文件路径')
    parser.add_argument(
        '--ablation_mode',
        type=str,
        default='',
        help='Ablation: none | image_only | text_off (default: none)',
    )
    
    args = parser.parse_args()
    main(args)
