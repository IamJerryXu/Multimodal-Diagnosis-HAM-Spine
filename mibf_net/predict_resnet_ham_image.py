import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset_spine import SpineTextImageDataset
from .model_resnet import Resnet50WithOurs
from .predict_resnet import _load_checkpoint


def _softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / (exp.sum(axis=1, keepdims=True) + 1e-12)


def predict(model, loader, device):
    model.eval()
    preds = []
    probs = []
    image_ids = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", ncols=100):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch)
            logits = outputs["image_text"].detach().cpu().numpy()
            prob = _softmax(logits)
            pred = prob.argmax(axis=1)
            preds.extend(pred.tolist())
            probs.extend(prob.tolist())
            image_ids.extend(batch["image_id"])
            labels.extend(batch["label"].detach().cpu().numpy().tolist())
    return image_ids, preds, probs, labels


def main():
    parser = argparse.ArgumentParser(description="MIBF ResNet prediction for HAM image.csv")
    parser.add_argument(
        "--image_root",
        type=str,
        default="/data/QLI/Assignment_for_dl/Data/data_for_dl/HAM10000_dataset/image",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/data/QLI/Assignment_for_dl/Data/data_for_dl/HAM10000_dataset/image.csv",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="/data/QLI/Assignment_for_dl/Data/data_for_dl/HAM10000_dataset/response/train/responses.json",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--output_path",
        type=str,
        default="/data/QLI/Assignment_for_dl/lib_dl/results/mibf_net/preds_ham_image.csv",
    )
    parser.add_argument("--bert_path", type=str, default="/data/QLI/BERT_pretain")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--loss_type", type=str, default="KL_loss")
    parser.add_argument("--compute_auc", action="store_true", help="Compute macro AUC")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SpineTextImageDataset(
        image_root=args.image_root,
        csv_path=args.csv_path,
        json_path=args.json_path,
        bert_path=args.bert_path,
        is_train=False,
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

    image_ids, preds, probs, labels = predict(model, loader, device)
    df = pd.DataFrame({"image_id": image_ids, "predicted_label": preds})
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    df.to_csv(args.output_path, index=False)
    print(f"Saved predictions to {args.output_path}")

    if args.compute_auc:
        try:
            from sklearn.metrics import roc_auc_score

            y_true = np.array(labels)
            y_score = np.array(probs)
            auc = roc_auc_score(
                y_true,
                y_score,
                multi_class="ovr",
                average="macro",
                labels=list(range(args.num_classes)),
            )
            print(f"Macro AUC: {auc:.4f}")
        except Exception as exc:
            print(f"AUC computation failed: {exc}")


if __name__ == "__main__":
    main()
