#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sensenova_drone.bc_data import ACTION_VOCAB

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object


class TransitionDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], *, image_width: int, image_height: int):
        if torch is None:
            raise RuntimeError("torch is required for TransitionDataset.")
        self.records = records
        self.image_width = int(image_width)
        self.image_height = int(image_height)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = load_image(record["image_path"], self.image_width, self.image_height)
        next_image = load_image(record["next_image_path"], self.image_width, self.image_height)
        action_index = int(record["action_index"])
        return {
            "image": image,
            "next_image": next_image,
            "action_index": torch.tensor(action_index, dtype=torch.long),
        }


class ActionConditionedWorldModel(nn.Module if nn is not None else object):
    def __init__(
        self,
        *,
        num_actions: int,
        image_width: int,
        image_height: int,
        latent_dim: int = 128,
        action_dim: int = 32,
    ):
        if nn is None:
            raise RuntimeError("torch is required for ActionConditionedWorldModel.")
        super().__init__()
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.latent_dim = int(latent_dim)
        self.start_h = self.image_height // 8
        self.start_w = self.image_width // 8
        if self.start_h * 8 != self.image_height or self.start_w * 8 != self.image_width:
            raise ValueError("image_width and image_height must be divisible by 8.")

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, latent_dim),
        )
        self.action_embed = nn.Embedding(num_actions, action_dim)
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.start_h * self.start_w),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 96, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, image):
        return self.encoder(image)

    def predict_latent(self, image, action_index):
        latent = self.encode(image)
        action = self.action_embed(action_index)
        return self.transition(torch.cat([latent, action], dim=1))

    def decode(self, latent):
        features = self.decoder_input(latent)
        features = features.view(latent.shape[0], 128, self.start_h, self.start_w)
        return self.decoder(features)

    def forward(self, image, action_index):
        next_latent_pred = self.predict_latent(image, action_index)
        next_image_pred = self.decode(next_latent_pred)
        return {
            "next_latent_pred": next_latent_pred,
            "next_image_pred": next_image_pred,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an action-conditioned pixel world model on Gym drone-game transitions.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-width", type=int, default=64)
    parser.add_argument("--image-height", type=int, default=48)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--latent-loss-weight", type=float, default=0.1)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--contact-sheet-samples", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    if torch is None or nn is None or F is None or DataLoader is None:
        raise RuntimeError("torch is required for world-model training.")

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    records = load_manifest(args.manifest)
    train_records = [record for record in records if record.get("split", "train") == "train"]
    val_records = [record for record in records if record.get("split") == "val"]
    if not train_records:
        raise RuntimeError("Manifest has no train records.")
    if not val_records:
        val_records = train_records[: min(512, len(train_records))]

    train_dataset = TransitionDataset(train_records, image_width=args.image_width, image_height=args.image_height)
    val_dataset = TransitionDataset(val_records, image_width=args.image_width, image_height=args.image_height)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = ActionConditionedWorldModel(
        num_actions=len(ACTION_VOCAB),
        image_width=args.image_width,
        image_height=args.image_height,
        latent_dim=args.latent_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            latent_loss_weight=args.latent_loss_weight,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device=device,
            latent_loss_weight=args.latent_loss_weight,
            optimizer=None,
        )
        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_metrics)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
            "epoch": epoch,
            "metrics": epoch_metrics,
            "action_vocab": ACTION_VOCAB,
            "model_type": "action_conditioned_pixel_world_model",
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            torch.save(checkpoint, out_dir / "best.pt")
        print(json.dumps(epoch_metrics), flush=True)

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["model_state_dict"])
    make_prediction_contact_sheet(
        model,
        val_dataset,
        out_dir / "prediction_contact_sheet.png",
        device=device,
        num_samples=args.contact_sheet_samples,
    )
    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "out_dir": str(out_dir.resolve()),
        "device": device,
        "num_train": len(train_records),
        "num_val": len(val_records),
        "image_width": args.image_width,
        "image_height": args.image_height,
        "latent_dim": args.latent_dim,
        "latent_loss_weight": args.latent_loss_weight,
        "best_val_loss": best_val_loss,
        "history": history,
        "best_checkpoint": str((out_dir / "best.pt").resolve()),
        "last_checkpoint": str((out_dir / "last.pt").resolve()),
        "prediction_contact_sheet": str((out_dir / "prediction_contact_sheet.png").resolve()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out_dir, summary)
    print(json.dumps(summary, indent=2))
    return 0


def resolve_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(path)
    records = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_image(path: str | Path, width: int, height: int):
    image = Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def run_epoch(
    model: ActionConditionedWorldModel,
    loader,
    *,
    device: str,
    latent_loss_weight: float,
    optimizer,
) -> dict[str, Any]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    total_examples = 0
    total_loss = 0.0
    total_image_loss = 0.0
    total_latent_loss = 0.0

    for batch in loader:
        image = batch["image"].to(device)
        next_image = batch["next_image"].to(device)
        action_index = batch["action_index"].to(device)

        with torch.set_grad_enabled(train_mode):
            outputs = model(image, action_index)
            image_loss = F.mse_loss(outputs["next_image_pred"], next_image)
            with torch.no_grad():
                next_latent_target = model.encode(next_image)
            latent_loss = F.mse_loss(outputs["next_latent_pred"], next_latent_target)
            loss = image_loss + float(latent_loss_weight) * latent_loss

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        batch_size = int(image.shape[0])
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_image_loss += float(image_loss.item()) * batch_size
        total_latent_loss += float(latent_loss.item()) * batch_size

    denom = max(total_examples, 1)
    return {
        "loss": total_loss / denom,
        "image_mse": total_image_loss / denom,
        "latent_mse": total_latent_loss / denom,
        "num_examples": total_examples,
    }


def make_prediction_contact_sheet(
    model: ActionConditionedWorldModel,
    dataset: TransitionDataset,
    out_path: Path,
    *,
    device: str,
    num_samples: int,
) -> None:
    model.eval()
    count = min(num_samples, len(dataset))
    if count <= 0:
        return
    indices = np.linspace(0, len(dataset) - 1, count).astype(int).tolist()
    rows = []
    with torch.no_grad():
        for index in indices:
            item = dataset[index]
            image = item["image"].unsqueeze(0).to(device)
            action_index = item["action_index"].unsqueeze(0).to(device)
            pred = model(image, action_index)["next_image_pred"][0].detach().cpu()
            rows.append(
                (
                    ACTION_VOCAB[int(item["action_index"].item())],
                    tensor_to_image(item["image"]),
                    tensor_to_image(item["next_image"]),
                    tensor_to_image(pred),
                )
            )

    cell_w, cell_h = rows[0][1].size
    label_h = 18
    label_w = 120
    sheet = Image.new("RGB", (label_w + cell_w * 3, label_h + cell_h * len(rows)), color=(28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(["frame_t", "target_t+1", "pred_t+1"]):
        draw.text((label_w + col * cell_w + 4, 2), label, fill=(240, 240, 240))
    for row_idx, (action, current, target, pred) in enumerate(rows):
        y = label_h + row_idx * cell_h
        draw.text((4, y + 4), action, fill=(240, 240, 240))
        for col, image in enumerate([current, target, pred]):
            sheet.paste(image, (label_w + col * cell_w, y))
    sheet.save(out_path)


def tensor_to_image(tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0.0, 1.0).numpy()
    array = np.transpose(array, (1, 2, 0))
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    first = summary["history"][0]
    last = summary["history"][-1]
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Gym Drone Pixel World Model</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f4f1e8; color: #202020; }}
    table {{ border-collapse: collapse; background: white; }}
    th, td {{ border: 1px solid #c9c1ad; padding: 6px 8px; text-align: right; }}
    th {{ background: #292f25; color: white; }}
    img {{ max-width: 100%; border: 1px solid #9f967f; background: white; }}
    code {{ background: #ebe4d4; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Gym Drone Pixel World Model</h1>
  <p>Task: <code>(frame_t, action_t) -> frame_t+1</code></p>
  <table>
    <tr><th>Metric</th><th>First Epoch Val</th><th>Last Epoch Val</th><th>Best</th></tr>
    <tr><td>Loss</td><td>{first['val']['loss']:.6f}</td><td>{last['val']['loss']:.6f}</td><td>{summary['best_val_loss']:.6f}</td></tr>
    <tr><td>Image MSE</td><td>{first['val']['image_mse']:.6f}</td><td>{last['val']['image_mse']:.6f}</td><td></td></tr>
    <tr><td>Latent MSE</td><td>{first['val']['latent_mse']:.6f}</td><td>{last['val']['latent_mse']:.6f}</td><td></td></tr>
  </table>
  <h2>Prediction Contact Sheet</h2>
  <img src="prediction_contact_sheet.png" />
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
