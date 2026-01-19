"""
python /data/sb/aaa_final_isic/models/test.py \
  --ckpt "/data/sb/MOE_exp_isic/baseLineMambaVision_KAN_mamba vertebra 0.00003536 CosineAnnealingLR 8heads_20251209_201051/checkpoints/epoch=189-acc_val_Accuracy=0.9202-f1_val_F1_macro=0.9040-auc_val_AUROC=0.9910.ckpt" \
  --config /data/sb/aaa_final_isic/config.yaml \
  --label_csv /data/sb/ISIC_2019_output_split/val.csv \
  --img_root /data/sb/ISIC_2019_output_split/val \
  --num_classes 9
  """
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
import argparse
import yaml
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

# ==============================================================================
# 1. 关键：设置路径并导入模型
# ==============================================================================
# 确保这个路径下包含 'models' 和 'dataset' 文件夹
# PROJECT_ROOT = '/mnt/sda1/algorithom_code_summary/ToolsLearning/02-pytorch_lighting训练框架'
# sys.path.append(PROJECT_ROOT)
try:
    from models.pl_model_MOE2 import Model4AAAI_MoE
except ImportError:
    # 方式 B: 或者直接导入 (如果 sys.path 包含了 models 目录)
    from pl_model_MOE2 import Model4AAAI_MoE
# try:
#     from pl_model_MOE2 import Model4AAAI_MoE
    # print("Success: Imported Model4AAAI_MoE")
# except ImportError as e:
#     print(f"Error: Could not import model. Check the path: {PROJECT_ROOT}")
#     print(f"Details: {e}")
#     sys.exit(1)

# ==============================================================================
# 2. 工具函数
# ==============================================================================
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        # 确保 label 是 int
        try:
            self.img_labels["label"] = self.img_labels["label"].astype(int)
        except:
            print("Error: CSV 'label' column must be integers.")
            sys.exit(1)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels.iloc[idx]
        img_name = str(row["image"])
        label = int(row["label"])
        
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path) and os.path.exists(img_name):
            img_path = img_name

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (224, 224)) # 坏图补黑
            
        if self.transform:
            image = self.transform(image)
        return image, label

@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"Processing {len(loader.dataset)} images...")
    
    for x, y in loader:
        x = x.to(device)
        
        # --- 复杂模型的 Forward 处理 ---
        # 你的模型可能返回 (logits, loss) 或 {logits: ...} 或 直接 logits
        # 这里假设返回 logits，如果报错请根据 Model4AAAI_MoE 的 forward 修改
        outputs = model(x)
        
        if isinstance(outputs, dict):
            logits = outputs['logits'] # 或者是其他 key
        elif isinstance(outputs, (list, tuple)):
            logits = outputs[0]
        else:
            logits = outputs

        _, pred = torch.max(logits, 1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 强制包含所有类别
    labels_range = list(range(num_classes))

    # --- 计算指标 ---
    overall_acc = accuracy_score(all_labels, all_preds)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    p_class, r_class, f1_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=labels_range, zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_preds, labels=labels_range)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_class = cm.diagonal() / cm.sum(axis=1)
        acc_class = np.nan_to_num(acc_class)

    # --- 打印 ---
    print(f"\nUsing device: {device}")
    
    print("\n=== Overall (weighted) ===")
    print("{")
    print(f'  "accuracy": {overall_acc:.4f},')
    print(f'  "precision_weighted": {w_prec:.4f},')
    print(f'  "recall_weighted": {w_rec:.4f},')
    print(f'  "f1_weighted": {w_f1:.4f}')
    print("}")

    print("\n=== Per-class ===")
    for i in range(num_classes):
        if support[i] > 0: # 只打印有数据的类，想看所有就把 if 去掉
            print(f"{i}: acc={acc_class[i]:.4f}, prec={p_class[i]:.4f}, rec={r_class[i]:.4f}, f1={f1_class[i]:.4f}")


# ==============================================================================
# 3. Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml used for training")
    parser.add_argument("--label_csv", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    # 你可以手动指定类别数，或者让脚本从 config 里读
    parser.add_argument("--num_classes", type=int, default=None) 
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载 Config
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)

    # 2. 确定类别数
    # 优先使用命令行参数，其次尝试从 config 读取
    num_classes = args.num_classes
    if num_classes is None:
        # 假设 config 结构里有 ['data']['num_classes'] 或类似字段
        # 请根据你的 config.yaml 结构修改这里！
        try:
            num_classes = config['model']['num_classes'] 
        except KeyError:
            try:
                num_classes = config['data']['num_classes']
            except:
                print("Warning: Could not find 'num_classes' in config. Defaulting to 9 (ISIC).")
                num_classes = 9 # Fallback
    print(f"Evaluating with {num_classes} classes.")

    # 3. 加载 Lightning 模型 (核心步骤)
    print(f"Loading Model from {args.ckpt}...")
    
    # load_from_checkpoint 会自动处理 .ckpt 里的 state_dict 和超参
    # 我们必须传入 config，因为 __init__ 需要它
    model = Model4AAAI_MoE.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        config=config, 
        learning_rate=config.get('train', {}).get('learning_rate', 1e-4), # 防止 __init__ 报错
        map_location='cpu'
    )
    
    model.to(device)
    model.eval()

    # 4. 数据加载
    # 假设测试时统一 resize 到 224，如果你的模型需要 256 或其他，请修改这里
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CustomImageDataset(args.label_csv, args.img_root, transform=test_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # 5. 评估
    evaluate(model, loader, num_classes, device)

if __name__ == "__main__":
    main()