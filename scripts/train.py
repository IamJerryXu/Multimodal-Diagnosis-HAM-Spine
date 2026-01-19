import os
import sys
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import shutil
import logging 
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model import MultimodalBaselineModel
from data_loader import create_data_loader

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (B, D), labels: (B,)
        features = F.normalize(features, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - torch.max(logits, dim=1, keepdim=True).values

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        mask.fill_diagonal_(0)

        exp_logits = torch.exp(logits) * (
            1 - torch.eye(logits.size(0), device=features.device)
        )
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return -mean_log_prob_pos.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def _compute_class_weights(dataset, num_classes, device):
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for item in dataset.metadata:
        label = int(item["label"])
        if 0 <= label < num_classes:
            counts[label] += 1
    total = counts.sum().clamp(min=1.0)
    weights = total / (counts.clamp(min=1.0) * num_classes)
    return weights.to(device)

def setup_logging(output_dir):
    log_path = os.path.join(output_dir, 'training.log')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def load_config(config_path=None):
    """加载YAML配置文件"""
    if config_path is None:
        config_path = os.path.join(BASE_DIR, 'config.yml')

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def validate(model, data_loader, criterion, device):
    """在验证集上评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            images, input_ids, attention_mask, tabular, labels, _ = batch
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if tabular is not None:
                tabular = tabular.to(device)
            labels = labels.to(device)
            
            logits = model(images, input_ids, attention_mask, tabular_input=tabular)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def main(config_path=None):
    config_path = config_path or os.path.join(BASE_DIR, 'config.yml')
    config = load_config(config_path)

    
    run_name = config['output'].get('run_name', 'unnamed_run')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(config['output']['log_dir'], f"{run_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    
    setup_logging(output_dir)
    
    logging.info(f"所有输出将保存到: {output_dir}")
    
    # 可选：只使用指定 GPU（例如 [0,1]）。未配置则默认可见所有 GPU。
    gpu_ids = config.get('training', {}).get('gpu_ids')
    if gpu_ids:
        if isinstance(gpu_ids, str):
            gpu_ids = [int(x) for x in gpu_ids.split(',') if x.strip() != '']
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids)
        logging.info(f"限制可见 GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

    
    shutil.copy(config_path, os.path.join(output_dir, 'config.yml'))

    
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
    
    
    device = torch.device(config['training']['device'])
    use_multi_gpu = device.type == "cuda" and torch.cuda.device_count() > 1

    
    logging.info("正在创建数据加载器...")
    train_loader = create_data_loader(config, split='train')
    val_loader = create_data_loader(config, split='val')

    
    logging.info("正在初始化模型...")
    model_config = config['model']
    tabular_cfg = model_config.get("tabular", {})
    if tabular_cfg.get("enabled"):
        tabular_cfg["input_dim"] = train_loader.dataset.tabular_dim

    gate_cfg = model_config.get("gate", {})
    seq_cfg = model_config.get("sequence_encoder", {})
    gl_cfg = model_config.get("global_local", {})
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
    )
    
    
    if model_config['image_encoder']['freeze']:
        for param in model.image_encoder.parameters():
            param.requires_grad = False
    if model_config['text_encoder']['freeze']:
        for param in model.text_encoder.parameters():
            param.requires_grad = False

    if use_multi_gpu:
        logging.info(f"使用 DataParallel，在 {torch.cuda.device_count()} 张 GPU 上训练")
        model = nn.DataParallel(model)

    model = model.to(device)

    resume_path = config.get('training', {}).get('resume_from')
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"resume_from not found: {resume_path}")
        state = torch.load(resume_path, map_location=device)
        if hasattr(model, "module"):
            model.module.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state, strict=False)
        logging.info(f"已从权重恢复: {resume_path}")
    
    loss_cfg = config.get("training", {}).get("loss", {})
    loss_type = loss_cfg.get("type", "ce").lower()
    label_smoothing = loss_cfg.get("label_smoothing", 0.02)
    class_weight_cfg = config.get("training", {}).get("class_weight")
    class_weights = None
    if class_weight_cfg == "balanced":
        class_weights = _compute_class_weights(
            train_loader.dataset, model_config["num_classes"], device
        )

    if loss_type == "focal":
        gamma = loss_cfg.get("focal_gamma", 2.0)
        criterion = FocalLoss(gamma=gamma, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=label_smoothing
        )
    params_to_update = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer_name = config['training']['optimizer']
    if optimizer_name == "Adam":
        optimizer = optim.Adam(params_to_update, lr=config['training']['learning_rate'])
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(params_to_update, lr=config['training']['learning_rate'])
    elif optimizer_name == "Muon":
        try:
            from muon import MuonWithAuxAdam
        except Exception as exc:
            raise ImportError(
                "Muon optimizer not available. Install with "
                "`pip install git+https://github.com/KellerJordan/Muon`."
            ) from exc
        import torch.distributed as dist
        import tempfile

        if dist.is_available() and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            with tempfile.NamedTemporaryFile() as tmp:
                dist.init_process_group(
                    backend=backend,
                    init_method=f"file://{tmp.name}",
                    rank=0,
                    world_size=1,
                )

        muon_lr = config['training'].get('muon_lr', 0.02)
        muon_weight_decay = config['training'].get('muon_weight_decay', 0.01)
        aux_lr = config['training'].get('muon_aux_lr', 3e-4)
        aux_betas = config['training'].get('muon_aux_betas', (0.9, 0.95))
        aux_weight_decay = config['training'].get('muon_aux_weight_decay', 0.01)

        muon_params = []
        aux_params = []
        for param in params_to_update:
            if param.ndim >= 2:
                muon_params.append(param)
            else:
                aux_params.append(param)

        param_groups = [
            dict(params=muon_params, use_muon=True, lr=muon_lr, weight_decay=muon_weight_decay),
            dict(
                params=aux_params,
                use_muon=False,
                lr=aux_lr,
                betas=aux_betas,
                weight_decay=aux_weight_decay,
            ),
        ]
        optimizer = MuonWithAuxAdam(param_groups)
    else:
        optimizer = optim.SGD(params_to_update, lr=config['training']['learning_rate'])

    # 可选学习率调度器
    scheduler = None
    lr_sched_name = config['training'].get('lr_scheduler', None)
    if lr_sched_name:
        name = lr_sched_name.lower()
        if name == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['training']['num_epochs']
            )
            logging.info("启用学习率调度器: CosineAnnealingLR")
        elif name in ("warmup_cosine", "warmup-cosine"):
            warmup_epochs = config['training'].get('warmup_epochs', 5)
            total_steps = config['training']['num_epochs'] * len(train_loader)
            warmup_steps = min(int(warmup_epochs * len(train_loader)), total_steps)

            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step + 1) / float(max(1, warmup_steps))
                progress = step - warmup_steps
                cosine_steps = max(1, total_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress / cosine_steps))

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
            logging.info(f"启用学习率调度器: Warmup + Cosine (warmup_epochs={warmup_epochs})")
        else:
            logging.info(f"未识别的 lr_scheduler: {lr_sched_name}，将不使用调度器")
        
    
    logging.info("开始训练...")
    top_3_checkpoints = []
    supcon_cfg = config.get("training", {}).get("supcon", {})
    supcon_enabled = bool(supcon_cfg.get("enabled", False))
    supcon_stage = supcon_cfg.get("stage", "finetune")
    supcon_temp = supcon_cfg.get("temperature", 0.07)
    supcon_weight = supcon_cfg.get("weight", 0.1)
    supcon_loss = SupConLoss(temperature=supcon_temp).to(device) if supcon_enabled else None

    ablation_mode = config.get("model", {}).get("ablation_mode")
    for epoch in range(config['training']['num_epochs']):
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} [Training]")
        for i, batch in enumerate(train_pbar):
            images, input_ids, attention_mask, tabular, labels, _ = batch
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if tabular is not None:
                tabular = tabular.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            if supcon_enabled and supcon_stage == "pretrain":
                features = model.forward_features(
                    images,
                    input_ids,
                    attention_mask,
                    tabular_input=tabular,
                    ablation_mode=ablation_mode,
                )
                loss = supcon_loss(features, labels)
            else:
                features = model.forward_features(
                    images,
                    input_ids,
                    attention_mask,
                    tabular_input=tabular,
                    ablation_mode=ablation_mode,
                )
                logits = model.classifier(features)
                loss = criterion(logits, labels)
                if supcon_enabled and supcon_stage == "finetune":
                    loss = loss + supcon_weight * supcon_loss(features, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None and isinstance(scheduler, lr_scheduler.LambdaLR):
                scheduler.step()
            
            total_train_loss += loss.item()
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if (i + 1) % 100 == 0:
                writer.add_scalar('Loss/Train_Batch', loss.item(), epoch * len(train_loader) + i)
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        logging.info(f"Epoch {epoch+1}/{config['training']['num_epochs']} Summary -> Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        
        writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch + 1)
        writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch + 1)
        if scheduler is not None and not isinstance(scheduler, lr_scheduler.LambdaLR):
            scheduler.step()
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch + 1)

        
        is_top_3 = len(top_3_checkpoints) < 3 or val_acc > min(top_3_checkpoints, key=lambda x: x[0])[0]
        if is_top_3:
            checkpoint_path = os.path.join(output_dir, f"epoch_{epoch+1}_val_acc_{val_acc:.2f}.pth")
            # 兼容 DataParallel 保存无前缀权重
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, checkpoint_path)
            logging.info(f"  -> 新的高分模型已保存: {os.path.basename(checkpoint_path)}")

            if len(top_3_checkpoints) == 3:
                worst_checkpoint = min(top_3_checkpoints, key=lambda x: x[0])
                if os.path.exists(worst_checkpoint[1]):
                    os.remove(worst_checkpoint[1])
                    logging.info(f"  -> 已删除旧的低分模型: {os.path.basename(worst_checkpoint[1])}")
                top_3_checkpoints.remove(worst_checkpoint)
            
            top_3_checkpoints.append((val_acc, checkpoint_path))
            top_3_checkpoints.sort(key=lambda x: x[0], reverse=True)

        logging.info(f"  -> 当前最佳Top-3准确率: {[f'{acc:.2f}%' for acc, path in top_3_checkpoints]}")

    writer.close()
    logging.info("\n训练完成！")
    logging.info(f"最佳的3个模型已保存在: {output_dir}")

if __name__ == "__main__":
    main()
