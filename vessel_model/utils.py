import torch
import gc

def print_gpu_memory(stage_name=""):
    """打印当前显存占用情况及主要张量"""
    if not torch.cuda.is_available():
        return
        
    torch.cuda.synchronize() # 确保同步
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"\n--- GPU Memory at {stage_name} ---")
    print(f"Allocated: {allocated:.2f} MB")
    print(f"Reserved: {reserved:.2f} MB")

    # 打印显存中占用前 5 的张量
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:
                    tensors.append(obj)
        except:
            pass
    
    tensors.sort(key=lambda x: x.element_size() * x.nelement(), reverse=True)
    print("Top 5 Tensors in GPU:")
    for i, t in enumerate(tensors[:5]):
        size_mb = (t.element_size() * t.nelement()) / (1024**2)
        print(f"  {i+1}: Shape {list(t.shape)} | {size_mb:.2f} MB | {t.dtype}")
    print("---------------------------------\n")

# def select_masks(masks, scores, thr=0.85, crit='min', max_size=1000):
#     # 你的筛选逻辑：优选面积小的（血管截面）
#     if len(masks) == 0: return None, -1.0
#     valid = []
#     for i, (m, s) in enumerate(zip(masks, scores)):
#         area = m.sum()
#         if s > thr and area > 10 and area < max_size:
#             valid.append((i, area, s))
#     if not valid: return None, -1.0
    
#     if crit == 'min': valid.sort(key=lambda x: x[1])
#     else: valid.sort(key=lambda x: x[2], reverse=True)
    
#     idx = valid[0][0]
#     return masks[idx], scores[idx]

import numpy as np
from skimage.measure import perimeter

def compute_circularity(mask):
    area = mask.sum()
    if area == 0:
        return 0.0
    perim = perimeter(mask, neighborhood=8)
    if perim == 0:
        return 0.0
    return 4 * np.pi * area / (perim ** 2)


def select_masks(
    masks,
    scores,
    thr=0.85,
    crit='min',
    max_size=1000,
    min_circularity=0.6
):
    """
    Select vessel-like masks based on confidence, size, and circularity.
    """
    if len(masks) == 0:
        return None, -1.0

    valid = []
    for i, (m, s) in enumerate(zip(masks, scores)):
        area = m.sum()
        if s <= thr or area <= 10 or area >= max_size:
            continue

        circ = compute_circularity(m)
        if circ < min_circularity:
            continue

        valid.append((i, area, s, circ))

    if not valid:
        return None, -1.0

    # 优先选择“更像血管截面”的 mask
    if crit == 'min':
        # 面积小 + 圆度高
        valid.sort(key=lambda x: (x[1], -x[3]))
    else:
        # score 高 + 圆度高
        valid.sort(key=lambda x: (x[2], x[3]), reverse=True)

    idx = valid[0][0]
    return masks[idx], scores[idx]
