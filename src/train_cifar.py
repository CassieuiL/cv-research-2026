import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


@dataclass
class Cfg:
    seed: int = 42
    epochs: int = 3
    batch_size: int = 128
    lr: float = 0.1
    num_workers: int = 2
    log_dir: str = "outputs/runs/cifar_resnet18"
    ckpt_dir: str = "outputs/checkpoints/cifar_resnet18"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cfg(path: str) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        d: Dict[str, Any] = yaml.safe_load(f)
    return Cfg(**d)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    writer = SummaryWriter(log_dir=cfg.log_dir)
    best_acc = 0.0
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        seen, correct, loss_sum = 0, 0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", ncols=100)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            seen += bs
            loss_sum += loss.item() * bs
            correct += (logits.argmax(dim=1) == y).sum().item()
            global_step += 1

            if global_step % 50 == 0:
                train_loss = loss_sum / max(seen, 1)
                train_acc = correct / max(seen, 1)
                writer.add_scalar("train/loss", train_loss, global_step)
                writer.add_scalar("train/acc", train_acc, global_step)
                pbar.set_postfix(loss=f"{train_loss:.4f}", acc=f"{train_acc:.4f}")

        scheduler.step()

        val_acc = evaluate(model, test_loader, device)
        writer.add_scalar("val/acc", val_acc, epoch)
        print(f"[Epoch {epoch}] val_acc = {val_acc:.4f}")

        torch.save({"epoch": epoch, "model": model.state_dict()}, os.path.join(cfg.ckpt_dir, "last.pt"))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch": epoch, "model": model.state_dict(), "best_acc": best_acc},
                       os.path.join(cfg.ckpt_dir, "best.pt"))
            print(f"[Info] New best: {best_acc:.4f}")

    writer.close()
    print(f"[Done] best_acc = {best_acc:.4f}")


if __name__ == "__main__":
    main()