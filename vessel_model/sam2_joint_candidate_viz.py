import argparse
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .data_manager import VolumeManager
from .sam2_baseline.predict_utils import map_local_point


COLORS = [
    (1.0, 0.1, 0.1),
    (0.1, 1.0, 0.1),
    (0.1, 0.4, 1.0),
]


def get_args():
    parser = argparse.ArgumentParser(
        description="Visualize 3 SAM2 candidate masks on nearby slices around a seed."
    )
    parser.add_argument("--sam2_checkpoint", required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2_model_cfg", required=True, help="Path to SAM2 model config")
    parser.add_argument("--volume_path", required=True, help="Path to .h5 or .nii/.nii.gz file")
    parser.add_argument("--dataset_key", default="main", help="Key name in h5 file")
    parser.add_argument("--seed", required=True, help="Seed in z,y,x format, e.g. 120,300,280")
    parser.add_argument(
        "--axis",
        type=int,
        default=-1,
        choices=[-1, 0, 1, 2],
        help="Tracking axis. -1 means auto-select on seed slice by min-area mask.",
    )
    parser.add_argument(
        "--half_window",
        type=int,
        default=8,
        help="Number of slices above and below seed index to visualize.",
    )
    parser.add_argument("--output_dir", default="./candidate_mask_viz", help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.35, help="Mask overlay alpha")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_seed(seed_text):
    parts = [int(p.strip()) for p in seed_text.split(",")]
    if len(parts) != 3:
        raise ValueError("--seed must be z,y,x")
    return tuple(parts)


def get_slice_rgb_and_box(vol_man, axis, idx):
    if axis == 0:
        img = vol_man.vol[idx, :, :]
        box = (idx, 0, vol_man.shape[1], 0, vol_man.shape[2])
    elif axis == 1:
        img = vol_man.vol[:, idx, :]
        box = (idx, 0, vol_man.shape[0], 0, vol_man.shape[2])
    else:
        img = vol_man.vol[:, :, idx]
        box = (idx, 0, vol_man.shape[0], 0, vol_man.shape[1])
    return np.stack([img] * 3, axis=-1), box


def predict_three_masks(img_predictor, image, local_point):
    img_predictor.set_image(image)
    masks, scores, _ = img_predictor.predict(
        point_coords=np.array([local_point]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    order = np.argsort(scores)[::-1]
    masks = masks[order]
    scores = scores[order]
    return masks.astype(np.uint8), scores


def auto_choose_axis(img_predictor, vol_man, seed):
    best_axis = -1
    best_area = float("inf")

    for axis in [0, 1, 2]:
        idx = seed[axis]
        rgb, box = get_slice_rgb_and_box(vol_man, axis, idx)
        local_pt = map_local_point(seed, axis, box)
        masks, _ = predict_three_masks(img_predictor, rgb, local_pt)
        if len(masks) == 0:
            continue
        min_area = min(int(m.sum()) for m in masks)
        if min_area < best_area:
            best_area = min_area
            best_axis = axis

    if best_axis < 0:
        raise RuntimeError("Failed to auto-select axis; SAM2 returned no masks.")
    return best_axis


def overlay_and_save(image, masks, scores, seed_xy, out_path, title, alpha):
    base = image.astype(np.float32) / 255.0
    canvas = base.copy()

    for i, mask in enumerate(masks[:3]):
        color = np.array(COLORS[i], dtype=np.float32)
        m = mask.astype(bool)
        if not np.any(m):
            continue
        canvas[m] = canvas[m] * (1 - alpha) + color * alpha

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(canvas)
    ax.scatter([seed_xy[0]], [seed_xy[1]], c="yellow", s=24, marker="x", linewidths=1.0)
    score_text = " | ".join([f"m{i}:{scores[i]:.3f}" for i in range(min(3, len(scores)))])
    ax.set_title(f"{title}\n{score_text}")
    ax.axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = get_args()
    seed = parse_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    vol_man = VolumeManager(args.volume_path, key=args.dataset_key)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module("use_sam2", version_base="1.2")

    sam2_img_model = build_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device=device)
    img_predictor = SAM2ImagePredictor(sam2_img_model)

    with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16):
        axis = args.axis
        if axis == -1:
            axis = auto_choose_axis(img_predictor, vol_man, seed)
            print(f"Auto-selected axis: {axis}")

        max_idx = vol_man.shape[axis] - 1
        start = max(0, seed[axis] - args.half_window)
        end = min(max_idx, seed[axis] + args.half_window)

        for idx in range(start, end + 1):
            rgb, box = get_slice_rgb_and_box(vol_man, axis, idx)
            local_pt = map_local_point(seed, axis, box)
            masks, scores = predict_three_masks(img_predictor, rgb, local_pt)

            out_name = f"axis{axis}_slice{idx:04d}.png"
            out_path = os.path.join(args.output_dir, out_name)
            overlay_and_save(
                image=rgb,
                masks=masks,
                scores=scores,
                seed_xy=local_pt,
                out_path=out_path,
                title=f"axis={axis}, slice={idx}, seed={seed}",
                alpha=args.alpha,
            )

    print(f"Done. Saved visualizations to: {args.output_dir}")


if __name__ == "__main__":
    main()
