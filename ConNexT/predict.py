"""
ConNexT prediction script.

Usage:
  python predict.py --checkpoint /path/to/ckpt --output /path/to/preds.csv --config config.yaml
"""

import argparse
import os
import yaml
import torch
import pandas as pd
from tqdm import tqdm

from dataset.pl_datset import DataModule
from models.pl_model_MOE2 import Model4AAAI_MoE


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def move_batch_to_device(batch: dict, device: torch.device):
    batch["imgs"] = batch["imgs"].to(device)
    batch["labels"] = batch["labels"].to(device)
    batch["first_hidden"] = batch["first_hidden"].to(device)
    batch["last_hidden"] = batch["last_hidden"].to(device)
    batch["texts"] = {k: v.to(device) for k, v in batch["texts"].items()}
    return batch


def main():
    parser = argparse.ArgumentParser(description="ConNexT prediction")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--checkpoint", required=True, help="Lightning .ckpt 权重路径")
    parser.add_argument("--output", required=True, help="输出 CSV 路径")
    parser.add_argument("--device", default=None, help="cuda:0 / cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    dm = DataModule(config)
    dm.setup("test")
    loader = dm.test_dataloader()
    dataset = dm.test_dataset

    model = Model4AAAI_MoE.load_from_checkpoint(
        args.checkpoint, config=config, map_location=device
    )
    model.eval()
    model.to(device)

    preds = []
    image_ids = []
    offset = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", ncols=100):
            batch = move_batch_to_device(batch, device)
            logits = model(batch)
            batch_preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            preds.extend(batch_preds.tolist())

            batch_size = batch["labels"].shape[0]
            batch_paths = dataset.img_paths[offset : offset + batch_size]
            image_ids.extend([os.path.basename(p) for p in batch_paths])
            offset += batch_size

    df = pd.DataFrame({"image_id": image_ids, "predicted_label": preds})
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
