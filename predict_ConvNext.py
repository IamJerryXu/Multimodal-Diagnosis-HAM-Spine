import argparse
import os
import sys

import pandas as pd
import torch
import yaml
from tqdm import tqdm

CONNEXT_DIR = "/data/QLI/Assignment_for_dl/lib_dl/ConNexT"
if CONNEXT_DIR not in sys.path:
    sys.path.insert(0, CONNEXT_DIR)

from dataset.pl_datset import DataModule
from models.pl_model_MOE2 import Model4AAAI_MoE


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def move_batch_to_device(batch, device):
    batch["imgs"] = batch["imgs"].to(device)
    batch["labels"] = batch["labels"].to(device)
    batch["first_hidden"] = batch["first_hidden"].to(device)
    batch["last_hidden"] = batch["last_hidden"].to(device)
    batch["texts"] = {k: v.to(device) for k, v in batch["texts"].items()}
    return batch


def main():
    parser = argparse.ArgumentParser(description="ConNexT prediction wrapper")
    parser.add_argument("--image_dir", required=True, help="Test image directory")
    parser.add_argument("--json_path", required=True, help="Text json path")
    parser.add_argument("--model_path", required=True, help="Checkpoint (.ckpt)")
    parser.add_argument("--output_path", required=True, help="Output CSV path")
    parser.add_argument("--config", default="/data/QLI/Assignment_for_dl/lib_dl/ConNexT/config.yaml")
    parser.add_argument("--label_csv", default=None, help="Optional label csv for test")
    parser.add_argument("--device", default=None, help="cuda:0 / cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    config["data"]["test_img_path"] = args.image_dir
    config["data"]["test_text_desc_path"] = args.json_path
    if args.label_csv:
        config["data"]["test_label_path"] = args.label_csv

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
        args.model_path, config=config, map_location=device
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
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    df.to_csv(args.output_path, index=False)
    print(f"Saved predictions to {args.output_path}")


if __name__ == "__main__":
    main()
