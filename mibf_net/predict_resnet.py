import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset_spine import SpinePredictDataset
from .model_resnet import Resnet50WithOurs


def _load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")


def predict(model, loader, device):
    model.eval()
    preds = []
    image_ids = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", ncols=100):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch)
            logits = outputs["image_text"]
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            preds.extend(pred)
            image_ids.extend(batch["image_id"])
    return image_ids, preds


def main():
    parser = argparse.ArgumentParser(description="MIBF ResNet prediction (lib_dl integration)")
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--bert_path", type=str, default="/data/QLI/BERT_pretain")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--loss_type", type=str, default="KL_loss")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SpinePredictDataset(
        image_root=args.image_root,
        json_path=args.json_path,
        bert_path=args.bert_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = Resnet50WithOurs(
        num_labels=args.num_classes,
        loss_class=args.loss_type,
        bert_path=args.bert_path,
    ).to(device)
    _load_checkpoint(model, args.model_path, device)

    image_ids, preds = predict(model, loader, device)
    df = pd.DataFrame({"image_id": image_ids, "predicted_label": preds})
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    df.to_csv(args.output_path, index=False)
    print(f"Saved predictions to {args.output_path}")


if __name__ == "__main__":
    main()
