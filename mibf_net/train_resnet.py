import argparse
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset_spine import SpineTextImageDataset
from .model_resnet import Resnet50WithOurs

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    base_model = model.module if hasattr(model, "module") else model
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in tqdm(loader, desc="Training", ncols=100):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(batch)
        loss = base_model.cal_loss(outputs, batch["label"])
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        preds = outputs["image_text"].argmax(dim=1).detach().cpu().numpy()
        labels = batch["label"].detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


def eval_one_epoch(model, loader, device):
    model.eval()
    base_model = model.module if hasattr(model, "module") else model
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", ncols=100):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch)
            loss = base_model.cal_loss(outputs, batch["label"])
            total_loss += loss.item()
            preds = outputs["image_text"].argmax(dim=1).detach().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


def main():
    parser = argparse.ArgumentParser(description="MIBF ResNet training (lib_dl integration)")
    parser.add_argument("--train_image_root", type=str, default="/data/QLI/Assignment_for_dl/Data/data_for_dl/Spine_dataset/image/images")
    parser.add_argument("--train_csv", type=str, default="/data/QLI/Assignment_for_dl/Data/data_for_dl/Spine_dataset/image.csv")
    parser.add_argument("--train_json", type=str, default="/data/QLI/Assignment_for_dl/Data/data_for_dl/Spine_dataset/all_responses_1232.json")
    parser.add_argument("--val_image_root", type=str, default="/data/QLI/Assignment_for_dl/Data/data_for_dl/Spine_dataset/image/images")
    parser.add_argument("--val_csv", type=str, default="/data/QLI/Assignment_for_dl/Data/data_for_dl/Spine_dataset/image.csv")
    parser.add_argument("--val_json", type=str, default="/data/QLI/Assignment_for_dl/Data/data_for_dl/Spine_dataset/all_responses_1232.json")
    parser.add_argument("--bert_path", type=str, default="/data/QLI/BERT_pretain")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam")
    parser.add_argument("--loss_type", type=str, default="KL_loss")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--expname", type=str, default="mibf_spine_resnet")
    parser.add_argument("--output_dir", type=str, default="results/mibf_net")
    args = parser.parse_args()

    use_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if use_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = f"{args.expname}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    train_dataset = SpineTextImageDataset(
        image_root=args.train_image_root,
        csv_path=args.train_csv,
        json_path=args.train_json,
        bert_path=args.bert_path,
        is_train=True,
    )
    val_dataset = SpineTextImageDataset(
        image_root=args.val_image_root,
        csv_path=args.val_csv,
        json_path=args.val_json,
        bert_path=args.bert_path,
        is_train=False,
    )

    train_sampler = DistributedSampler(train_dataset) if use_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False if use_ddp else True,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_dataset,
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
    if use_ddp:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=True)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "last.pth"))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "best.pth"))


if __name__ == "__main__":
    main()
