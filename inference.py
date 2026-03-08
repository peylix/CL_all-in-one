"""Inference script for a single task.

Usage:
    # With ground truth saving
    uv run inference.py --checkpoint ./checkpoints/haze_rain_snow/rain/ffa_best.pk
        --input_dir /path/to/raindrop/test_a/data --gt_dir /path/to/raindrop/test_a/gt \
        --gt_name_fn raindrop --output_dir ./results/rain --device cuda:0
"""

import os
import argparse
import shutil
import torch
import torchvision.transforms as tfs
from torchvision.utils import save_image
from PIL import Image
from models.FFA import FFA


GT_NAME_FNS = {
    "raindrop": lambda f: f.replace("_rain", "_clean"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to ffa_best.pk or net_stepXXX.pth",
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory of degraded images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save restored images",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Directory of ground truth images (optional)",
    )
    parser.add_argument(
        "--gt_name_fn",
        type=str,
        default=None,
        choices=list(GT_NAME_FNS.keys()),
        help="Filename mapping from input to gt (e.g. raindrop: _rain -> _clean)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gps", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=20)
    args = parser.parse_args()

    pred_dir = os.path.join(args.output_dir, "pred")
    os.makedirs(pred_dir, exist_ok=True)
    if args.gt_dir:
        gt_save_dir = os.path.join(args.output_dir, "gt")
        os.makedirs(gt_save_dir, exist_ok=True)

    gt_name_fn = GT_NAME_FNS.get(args.gt_name_fn) if args.gt_name_fn else None

    device = torch.device(args.device)
    model = FFA(gps=args.gps, blocks=args.blocks).to(device)

    # Load checkpoint (supports both ffa_best.pk and net_stepXXX.pth formats)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    transform = tfs.Compose(
        [
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152]),
        ]
    )

    files = sorted(
        [
            f
            for f in os.listdir(args.input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))
        ]
    )
    print(f"Found {len(files)} images in {args.input_dir}")

    with torch.no_grad():
        for filename in files:
            img = Image.open(os.path.join(args.input_dir, filename)).convert("RGB")
            inp = transform(img).unsqueeze(0).to(device)
            pred = model(inp)
            pred = pred.clamp(0, 1)
            save_image(pred, os.path.join(pred_dir, filename))

            if args.gt_dir:
                gt_filename = gt_name_fn(filename) if gt_name_fn else filename
                shutil.copy2(
                    os.path.join(args.gt_dir, gt_filename),
                    os.path.join(gt_save_dir, filename),
                )

            print(f"Saved: {filename}")

    print(f"Done. Predictions saved to {pred_dir}")
    if args.gt_dir:
        print(f"Ground truths saved to {gt_save_dir}")


if __name__ == "__main__":
    main()
