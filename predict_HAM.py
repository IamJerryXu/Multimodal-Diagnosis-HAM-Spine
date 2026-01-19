import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mibf_net.dataset_spine import SpinePredictDataset
from mibf_net.model_resnet import Resnet50WithOurs
from mibf_net.predict_resnet import _load_checkpoint


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
    parser = argparse.ArgumentParser(description="MIBF ResNet HAM prediction")
    parser.add_argument("--image_dir", required=True, help="Test image directory")
    parser.add_argument("--json_path", required=True, help="Text json path")
    parser.add_argument("--model_path", required=True, help="Model checkpoint (.pth)")
    parser.add_argument("--output_path", required=True, help="Output CSV path")
    parser.add_argument("--bert_path", default="/data/QLI/BERT_pretain")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--loss_type", default="KL_loss")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SpinePredictDataset(
        image_root=args.image_dir,
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
