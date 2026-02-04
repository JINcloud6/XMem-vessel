#main2的原版，即只使用视频追踪分割，不去判别新种子
#可视化memory
import argparse
import os
import shutil
import tempfile

import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm
import hydra
from PIL import Image

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .data_manager import VolumeManager
from .preprocessing import get_multi_axis_init_seg, get_seeds_from_init_seg, get_seg
from .utils import select_masks
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import math
import matplotlib.pyplot as plt
import numpy as np
import torch

import matplotlib.pyplot as plt

def plot_time_value_heatmap(values_per_frame, title, savepath,
                            n_time_bins=40, n_val_bins=40, val_range=None,
                            log_scale=True):
    """
    values_per_frame: list[np.ndarray|None]，每帧若干个样本值
    生成 heatmap：x=time(frame)，y=value bins，颜色=频次
    """
    T = len(values_per_frame)
    if T == 0:
        return

    xs = []
    ys = []
    for t, arr in enumerate(values_per_frame):
        if arr is None:
            continue
        arr = np.asarray(arr).reshape(-1)
        if arr.size == 0:
            continue
        # 用 frame idx 作为 x（也可归一化 t/(T-1)）
        xs.append(np.full(arr.shape, t, dtype=np.float32))
        ys.append(arr.astype(np.float32))

    if len(xs) == 0:
        return

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)

    # bins
    x_bins = np.linspace(0, T, n_time_bins + 1)
    if val_range is None:
        y_min, y_max = float(np.nanpercentile(y, 1)), float(np.nanpercentile(y, 99))
        if y_max <= y_min + 1e-6:
            y_min, y_max = float(np.min(y)), float(np.max(y) + 1e-6)
        y_range = (y_min, y_max)
    else:
        y_range = val_range
    y_bins = np.linspace(y_range[0], y_range[1], n_val_bins + 1)

    H, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])  # H shape: [n_time_bins, n_val_bins]
    H = H.T  # -> [n_val_bins, n_time_bins]，方便 y 轴向上

    if log_scale:
        H_show = np.log1p(H)
    else:
        H_show = H

    plt.figure(figsize=(7, 4))
    plt.imshow(
        H_show,
        origin="lower",
        aspect="auto",
        extent=[0, T, y_bins[0], y_bins[-1]],
    )
    plt.title(title)
    plt.xlabel("frame idx")
    plt.ylabel("value")
    plt.colorbar(label="log count" if log_scale else "count")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()

def sample_query_metrics_from_attn(attn_bhqs: torch.Tensor, num_ptr: int, max_q: int = 512):
    """
    attn_bhqs: [B,H,Sq,Sk] (CPU tensor ok)
    返回：entropy/top1/ptrmass 的一维样本数组（长度<=max_q）
    """
    # -> [Sq,Sk]：先平均 heads，再取 batch=0
    if attn_bhqs.dim() != 4:
        raise ValueError(f"Unexpected attn shape: {attn_bhqs.shape}")
    p = attn_bhqs.mean(dim=1)[0]  # [Sq,Sk]

    eps = 1e-8
    p2 = p.clamp_min(eps)

    # per-query entropy / top1
    ent_q = -(p2 * p2.log()).sum(dim=-1)      # [Sq]
    top1_q = p.max(dim=-1).values             # [Sq]

    # per-query pointer mass（假设 pointer keys 在最后 num_ptr 个）
    if num_ptr > 0:
        ptr_q = p[:, -num_ptr:].sum(dim=-1)   # [Sq]
    else:
        ptr_q = torch.zeros_like(ent_q)

    # 随机采样 max_q 个 query token
    sq = ent_q.numel()
    if sq > max_q:
        idx = torch.randint(0, sq, (max_q,), device=ent_q.device)
        ent_q = ent_q[idx]
        top1_q = top1_q[idx]
        ptr_q = ptr_q[idx]

    return ent_q.detach().cpu().numpy(), top1_q.detach().cpu().numpy(), ptr_q.detach().cpu().numpy()

def mean_attn_entropy(attn_bhqs: torch.Tensor, eps=1e-8) -> float:
    # attn_bhqs: [B,H,Sq,Sk] on CPU
    p = attn_bhqs.clamp_min(eps)
    ent = -(p * p.log()).sum(dim=-1)  # [B,H,Sq]
    return ent.mean().item()

def mean_attn_top1(attn_bhqs: torch.Tensor) -> float:
    top1 = attn_bhqs.max(dim=-1).values  # [B,H,Sq]
    return top1.mean().item()

def mean_pointer_mass(attn_bhqs: torch.Tensor, num_ptr: int) -> float:
    if num_ptr <= 0:
        return 0.0
    sk = attn_bhqs.shape[-1]
    ptr_mass = attn_bhqs[..., sk - num_ptr: sk].sum(dim=-1)  # [B,H,Sq]
    return ptr_mass.mean().item()

def _recover_missing_components_sam2(
    img_predictor,
    vol_man,
    prev_components,
    curr_mask,
    curr_rgb,          # 当前帧的 RGB (H,W,3)
    axis,
    vol_idx,           # 当前帧在 volume 的真实索引（不是 f_idx）
    box,
    overlap_threshold=0.5,
):
    """
    若上一帧连通域在当前帧消失，则在当前帧以其 center 重新点提示分割，恢复缺失部分。
    返回 recovered_mask (uint8) 或 None。
    """
    if not prev_components:
        return None

    if curr_mask is None:
        curr_mask = np.zeros_like(prev_components[0]["mask"], dtype=np.uint8)

    # 找出缺失连通域：与 curr_mask 无重叠
    missing = []
    for comp in prev_components:
        if (curr_mask > 0).any() and (curr_mask & comp["mask"]).any():
            continue
        missing.append(comp)
    if not missing:
        return None

    # 取 crop 范围内已经落到 global_mask 的区域，避免重复/串段
    if axis == 0:
        existing = vol_man.global_mask[vol_idx, box[1]:box[2], box[3]:box[4]]
    elif axis == 1:
        existing = vol_man.global_mask[box[1]:box[2], vol_idx, box[3]:box[4]]
    else:
        existing = vol_man.global_mask[box[1]:box[2], box[3]:box[4], vol_idx]

    img_predictor.set_image(curr_rgb)

    recovered = np.zeros_like(curr_mask, dtype=np.uint8)

    for comp in missing:
        cy, cx = comp["center"]
        cy_i = int(round(cy))
        cx_i = int(round(cx))

        masks, scores, _ = img_predictor.predict(
            point_coords=np.array([[cx_i, cy_i]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )

        # 这里别太严，恢复阶段建议 thr=0 或较低，并用后面的 overlap 来抑制错误
        cand_mask, _ = select_masks(
            masks, scores,
            thr=0.0, crit="max",
            max_size=5000,
            min_circularity=0.0
        )
        if cand_mask is None or cand_mask.sum() == 0:
            continue

        # 若与已有 global mask 重叠过大，说明可能串到已分割血管/其它段，跳过
        cand_b = np.asarray(cand_mask).astype(bool)
        exist_b = np.asarray(existing).astype(bool)
        overlap = np.logical_and(cand_b, exist_b).sum()

        # overlap = (cand_mask & (existing > 0)).sum()
        ratio = overlap / (cand_mask.sum() + 1e-6)
        if ratio > overlap_threshold:
            continue

        recovered = np.logical_or(recovered, cand_mask)

    if recovered.sum() == 0:
        return None

    return recovered.astype(np.uint8)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_checkpoint", required=True, help="Path to SAM2 checkpoint",
                        default="/home/jiangshuai/code/sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_model_cfg", required=True, help="Path to SAM2 model config",
                        default="/home/jiangshuai/code/sam2/sam2.1_hiera_l.yaml")
    parser.add_argument("--volume_path", required=True, help="Path to .h5 or .nii/.nii.gz file")
    parser.add_argument("--output_dir", default="./bv_seg_output", help="Directory to save outputs")
    parser.add_argument("--output_filename", default="segmentation.nii.gz", help="Output filename")
    parser.add_argument("--dataset_key", default="main", help="Key name in h5 file")
    parser.add_argument("--seed_file", default=None, help="Optional path to seed list (z,y,x per line)")
    parser.add_argument("--axis", type=int, default=3, help="Axis to generate init seg (0=Z,1=Y,2=X), 3=all")
    parser.add_argument("--stride", type=int, default=5, help="Stride for init seg generation")
    parser.add_argument("--gaussian_kernel", type=int, default=5)
    parser.add_argument("--min_bright", type=int, default=40)
    parser.add_argument("--remove_portion", type=float, default=0.98)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--need_transpose", default="False")
    parser.add_argument("--max_track_distance", type=int, default=2000)
    parser.add_argument("--recover_overlap_threshold", type=float, default=0.5)

    # video predictor / io knobs
    parser.add_argument("--vos_offload_video_to_cpu", action="store_true",
                        help="Offload video frames to CPU inside SAM2 video predictor (save GPU mem)")
    parser.add_argument("--keep_tmp_vos_frames", action="store_true",
                        help="Do not delete tmp JPEG frames dirs (debug)")

    return parser.parse_args()


def _extract_components(binary_mask):
    labeled, num = ndimage.label(binary_mask > 0)
    components = []
    if num == 0:
        return components
    for label_id in range(1, num + 1):
        comp_mask = labeled == label_id
        area = int(comp_mask.sum())
        if area == 0:
            continue
        cy, cx = ndimage.center_of_mass(comp_mask)
        if np.isnan(cy) or np.isnan(cx):
            continue
        components.append({"mask": comp_mask, "center": (float(cy), float(cx)), "area": area})
    return components


def _predict_from_point(predictor, image, point):
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([point]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    print(scores)
    mask, _ = select_masks(masks, scores, thr=0.6, crit="max", max_size=10000, min_circularity=0.0)
    print("mask is ",mask)
    if mask is None:
        return None
    return mask.astype(np.uint8)


def _map_local_point(seed, axis, box):
    if axis == 0:
        return [seed[2] - box[3], seed[1] - box[1]]
    if axis == 1:
        return [seed[2] - box[3], seed[0] - box[1]]
    return [seed[1] - box[3], seed[0] - box[1]]


def _get_slice(volume, axis, curr_idx, box):
    if axis == 0:
        sl = volume[curr_idx, box[1]:box[2], box[3]:box[4]]
    elif axis == 1:
        sl = volume[box[1]:box[2], curr_idx, box[3]:box[4]]
    else:
        sl = volume[box[1]:box[2], box[3]:box[4], curr_idx]
    return sl


def _to_uint8_rgb(slice_2d: np.ndarray) -> np.ndarray:
    """
    Convert a 2D slice to uint8 RGB for SAM2.
    """
    sl = slice_2d
    if sl.size == 0:
        return None
    if sl.dtype != np.uint8:
        sl = sl.astype(np.float32)
        mn = float(np.min(sl))
        mx = float(np.max(sl))
        if mx <= mn + 1e-6:
            sl = np.zeros_like(sl, dtype=np.uint8)
        else:
            sl = (sl - mn) / (mx - mn + 1e-6) * 255.0
            sl = np.clip(sl, 0, 255).astype(np.uint8)
    rgb = np.stack([sl] * 3, axis=-1)
    return rgb


def _write_jpeg_frames(frames_rgb: list[np.ndarray], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # ensure empty
    for fn in os.listdir(out_dir):
        fp = os.path.join(out_dir, fn)
        if os.path.isfile(fp):
            os.remove(fp)

    for i, fr in enumerate(frames_rgb):
        # SAM2 loader commonly expects names like 00000.jpg ...
        path = os.path.join(out_dir, f"{i:05d}.jpg")
        Image.fromarray(fr).save(path, quality=95, subsampling=0)


def _vos_track_one_direction(
    video_predictor,
    vol_man: VolumeManager,
    axis: int,
    box: tuple,
    idx_list: list[int],          # volume indices (already in desired order)
    init_mask_2d: np.ndarray,     # mask on the FIRST frame of idx_list (crop coords)
    global_update_axis: int,
    vos_tmp_root: str,
    offload_video_to_cpu: bool,
    rope_attn,
    log_prefix="human"
):
    """
    Use SAM2 video predictor to propagate init_mask_2d across frames in idx_list.
    Updates vol_man.global_mask via vol_man.update_global_mask().
    """
    # 1) build RGB frames
    frames_rgb = []
    for vidx in idx_list:
        sl = _get_slice(vol_man.vol, axis, vidx, box)
        if sl.size == 0:
            break
        rgb = _to_uint8_rgb(sl)
        if rgb is None:
            break
        frames_rgb.append(rgb)

    if len(frames_rgb) == 0:
        return

    # 2) write to temp JPEG dir for init_state
    tmp_dir = tempfile.mkdtemp(prefix="sam2_vos_", dir=vos_tmp_root)
    _write_jpeg_frames(frames_rgb, tmp_dir)

    try:
        ent_curve = []
        top1_curve = []
        ptr_curve = []
        ent_samples_per_frame = []
        top1_samples_per_frame = []
        ptr_samples_per_frame = []


        # 3) init state, add init mask on frame 0, then propagate
        with torch.inference_mode(), torch.autocast(str(vol_man.device), dtype=torch.bfloat16):
            state = video_predictor.init_state(
                video_path=tmp_dir,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=False,
                async_loading_frames=False,
            )

            # add init mask as conditioning signal on frame 0
            # NOTE: init_mask_2d is in crop resolution; video predictor will resize internally if needed.
            _, obj_ids, masks0 = video_predictor.add_new_mask(
                state,
                frame_idx=0,
                obj_id=1,
                mask=init_mask_2d.astype(bool),
            )

            # update frame 0
            # masks0: (num_obj, 1, H, W) scores/logits at video resolution
            m0 = masks0[0, 0]
            if torch.is_tensor(m0):
                m0 = (m0 > 0).to(torch.uint8).cpu().numpy()
            else:
                m0 = (m0 > 0).astype(np.uint8)
            vol_man.update_global_mask(m0, global_update_axis, (idx_list[0], *box[1:]))

            # propagate forward through this clip
            for f_idx, obj_ids, masks in video_predictor.propagate_in_video(state):
                # f_idx corresponds to frames_rgb index => idx_list[f_idx]
                if f_idx < 0 or f_idx >= len(idx_list):
                    continue
                mm = masks[0, 0]
                if torch.is_tensor(mm):
                    mm = (mm > 0).to(torch.uint8).cpu().numpy()
                else:
                    mm = (mm > 0).astype(np.uint8)

                if mm.sum() == 0:
                    # 可选：空了就跳过，但 propagate 仍会继续产出；这里直接 continue
                    continue
                attn = getattr(rope_attn, "last_attn", None)
                if attn is None:
                    ent_curve.append(np.nan)
                    top1_curve.append(np.nan)
                    ptr_curve.append(np.nan)
                else:
                    ent_curve.append(mean_attn_entropy(attn))
                    top1_curve.append(mean_attn_top1(attn))
                    ptr_curve.append(mean_pointer_mass(attn, getattr(rope_attn, "last_num_k_exclude_rope", 0)))
                attn = getattr(rope_attn, "last_attn", None)
                num_ptr = int(getattr(rope_attn, "last_num_k_exclude_rope", 0))

                if attn is None:
                    ent_samples_per_frame.append(None)
                    top1_samples_per_frame.append(None)
                    ptr_samples_per_frame.append(None)
                else:
                    ent_s, top1_s, ptr_s = sample_query_metrics_from_attn(attn, num_ptr=num_ptr, max_q=512)
                    ent_samples_per_frame.append(ent_s)
                    top1_samples_per_frame.append(top1_s)
                    ptr_samples_per_frame.append(ptr_s)

                vol_man.update_global_mask(mm, global_update_axis, (idx_list[f_idx], *box[1:]))
                        
    finally:
        # cleanup tmp frames
        import os
        os.makedirs(os.path.join("./data/macaque", "attn_viz"), exist_ok=True)
        out_dir = os.path.join("./data/human", "attn_viz")

        plt.figure()
        plt.plot(ent_curve)
        plt.title(f"{log_prefix} entropy (mean)")
        plt.xlabel("frame idx")
        plt.ylabel("entropy")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{log_prefix}_entropy.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.plot(top1_curve)
        plt.title(f"{log_prefix} top1 mass (mean)")
        plt.xlabel("frame idx")
        plt.ylabel("top1 mass")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{log_prefix}_top1.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.plot(ptr_curve)
        plt.title(f"{log_prefix} pointer mass (mean)")
        plt.xlabel("frame idx")
        plt.ylabel("pointer mass")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{log_prefix}_ptrmass.png"), dpi=200)
        plt.close()


        out_dir = os.path.join(vos_tmp_root, "attn_viz")
        os.makedirs(out_dir, exist_ok=True)

        plot_time_value_heatmap(
            ent_samples_per_frame,
            title=f"{log_prefix} entropy dist",
            savepath=os.path.join(out_dir, f"{log_prefix}_entropy_heat.png"),
            n_time_bins=50, n_val_bins=50, log_scale=True,
        )

        plot_time_value_heatmap(
            top1_samples_per_frame,
            title=f"{log_prefix} top1 dist",
            savepath=os.path.join(out_dir, f"{log_prefix}_top1_heat.png"),
            n_time_bins=50, n_val_bins=50, log_scale=True,
        )

        plot_time_value_heatmap(
            ptr_samples_per_frame,
            title=f"{log_prefix} ptrmass dist",
            savepath=os.path.join(out_dir, f"{log_prefix}_ptrmass_heat.png"),
            n_time_bins=50, n_val_bins=50, log_scale=True,
        )

        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_segmentation():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    vol_man = VolumeManager(args.volume_path, key=args.dataset_key)
    # 如果你的 VolumeManager 里没有 device 字段，给它补一个
    if not hasattr(vol_man, "device"):
        vol_man.device = device

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('use_sam2', version_base='1.2')

    # --- build SAM2 image predictor (for best_axis & initial mask) ---
    sam2_img_model = build_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device=device)
    img_predictor = SAM2ImagePredictor(sam2_img_model)

    # --- build SAM2 video predictor (for VOS tracking) ---
    video_predictor = build_sam2_video_predictor(args.sam2_model_cfg, args.sam2_checkpoint, device=device)

    from sam2.modeling.sam.transformer import RoPEAttention

    def find_cross_rope_attention(model):
        candidates = []
        for name, m in model.named_modules():
            if isinstance(m, RoPEAttention):
                candidates.append((name, m))
        # 优先挑 cross_attn_image（更接近 memory cross-attn）
        for name, m in reversed(candidates):
            if "cross_attn_image" in name or "memory_attention" in name:
                return name, m
        # 退化：取最后一个 RoPEAttention（通常更靠后、可解释性更强）
        return candidates[-1] if candidates else (None, None)

    rope_name, rope_attn = find_cross_rope_attention(video_predictor)
    print("Hook RoPEAttention:", rope_name)
    if rope_attn is None:
        raise RuntimeError("Cannot find RoPEAttention in video_predictor.")
    rope_attn.save_attention = True


    seeds = []
    if args.seed_file:
        with open(args.seed_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = [int(p) for p in line.strip().split(",")]
                if len(parts) != 3:
                    raise ValueError(f"Invalid seed line: {line}")
                seeds.append(tuple(parts))
    else:
        if args.axis in (0, 1, 2):
            axis = args.axis
            init_seg = np.zeros_like(vol_man.vol, dtype=np.uint8)
            if axis == 0:
                for m in tqdm(range(0, vol_man.shape[0], args.stride), desc="Axis 0 (Z)"):
                    image = vol_man.vol[m, :, :]
                    temp_seg = get_seg(image, remove_portion=args.remove_portion,
                                       gaussian_kernel=args.gaussian_kernel, min_bright=args.min_bright)
                    init_seg[m, :, :][temp_seg > 0] = 1
            elif axis == 1:
                for m in tqdm(range(0, vol_man.shape[1], args.stride), desc="Axis 1 (Y)"):
                    image = vol_man.vol[:, m, :]
                    temp_seg = get_seg(image, remove_portion=args.remove_portion,
                                       gaussian_kernel=args.gaussian_kernel, min_bright=args.min_bright)
                    init_seg[:, m, :][temp_seg > 0] = 1
            else:
                for m in tqdm(range(0, vol_man.shape[2], args.stride), desc="Axis 2 (X)"):
                    image = vol_man.vol[:, :, m]
                    temp_seg = get_seg(image, remove_portion=args.remove_portion,
                                       gaussian_kernel=args.gaussian_kernel, min_bright=args.min_bright)
                    init_seg[:, :, m][temp_seg > 0] = 1
        else:
            init_seg = get_multi_axis_init_seg(
                vol_man.vol,
                stride=args.stride,
                thr=args.remove_portion,
                gaussian_kernel=args.gaussian_kernel,
                min_bright=args.min_bright,
            )
        seeds = get_seeds_from_init_seg(init_seg)

    if not seeds:
        raise RuntimeError("No seeds available for SAM2 tracking.")

    print(f"Starting SAM2 segmentation with {len(seeds)} seeds...")

    # tmp root for VOS frames
    vos_tmp_root = os.path.join(args.output_dir, "_tmp_sam2_vos")
    os.makedirs(vos_tmp_root, exist_ok=True)

    with tqdm(total=len(seeds), desc="Tracking (SAM2 VOS)") as pbar:
        for seed_index, seed in enumerate(seeds):
            pbar.update(1)
            z, y, x = seed
            if vol_man.global_mask[z, y, x] > 0:
                continue

            crops = vol_man.get_triplane_crops(seed)
            best_axis = -1
            best_mask = None
            min_area = float("inf")

            # 1) per-axis init segmentation on the seed slice (still image)
            with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16):
                for axis in [0, 1, 2]:
                    img, box = crops[axis]
                    local_pt = _map_local_point(seed, axis, box)
                    mask = _predict_from_point(img_predictor, img, local_pt)
                    if mask is None:
                        continue
                    area = int(mask.sum())
                    if area < min_area:
                        min_area = area
                        best_axis = axis
                        best_mask = mask

            if best_axis == -1 or best_mask is None or best_mask.sum() == 0:
                continue

            # 2) build index lists (forward/backward) around the seed on the chosen axis
            track_dim = [0, 1, 2][best_axis]
            start_idx = seed[track_dim]
            max_dist = args.max_track_distance

            # IMPORTANT: idx_list[0] corresponds to the frame where best_mask was predicted
            forward_idxs = list(range(start_idx, min(start_idx + max_dist, vol_man.shape[track_dim])))
            backward_idxs = list(range(start_idx, max(start_idx - max_dist, -1), -1))

            _, box = crops[best_axis]

            # 3) VOS tracking using SAM2VideoPredictor
            # forward
            _vos_track_one_direction(
                video_predictor=video_predictor,
                vol_man=vol_man,
                axis=best_axis,
                box=box,
                idx_list=forward_idxs,
                init_mask_2d=best_mask,
                global_update_axis=best_axis,
                vos_tmp_root=vos_tmp_root,
                offload_video_to_cpu=args.vos_offload_video_to_cpu,
                rope_attn=rope_attn,
                log_prefix=f"seed{seed_index}_axis{best_axis}_fw"
            )
            # backward
            _vos_track_one_direction(
                video_predictor=video_predictor,
                vol_man=vol_man,
                axis=best_axis,
                box=box,
                idx_list=backward_idxs,
                init_mask_2d=best_mask,
                global_update_axis=best_axis,
                vos_tmp_root=vos_tmp_root,
                offload_video_to_cpu=args.vos_offload_video_to_cpu,
                rope_attn=rope_attn,
                log_prefix=f"seed{seed_index}_axis{best_axis}_bw"
            )

    if (not args.keep_tmp_vos_frames) and os.path.isdir(vos_tmp_root):
        shutil.rmtree(vos_tmp_root, ignore_errors=True)

    print("Cleanup...")
    final_path = os.path.join(args.output_dir, args.output_filename)
    flag = (args.need_transpose != "False")
    vol_man.save(final_path, flag)
    print(f"Done! Saved to {final_path}")


if __name__ == "__main__":
    run_segmentation()
