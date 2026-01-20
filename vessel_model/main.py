import os
import sys



# --- Path Setup ---
# 确保可以引用父目录下的模块 (model, inference)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)



import time
import numpy as np
import torch
import h5py
from tqdm import tqdm

# --- Local Imports ---
from .config import get_args, xmem_config
from .utils import select_masks
from .preprocessing import get_multi_axis_init_seg, get_seeds_from_init_seg
from .data_manager import VolumeManager

# --- Parent Imports ---
# 假设父目录结构包含这些模块
try:
    from model.network import XMem
    from inference.inference_core import InferenceCore
    from inference.kv_memory_store import KeyValueMemoryStore
    from segment_anything import sam_model_registry, SamPredictor
except ImportError as e:
    print("Error importing model/inference modules. Make sure the script is running with access to the parent directory.")
    raise e

def run_segmentation():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading volume from {args.volume_path}...")
    vol_man = VolumeManager(args.volume_path, key=args.dataset_key, crop_size=args.crop_size, need_transpose=args.need_transpose)
    
    # 2. Get Init Seg (The "Map")
    init_seg_name = f"init_seg_axis{args.axis}_s{args.stride}_t{args.remove_portion}.h5"
    init_seg_path = os.path.join(args.output_dir, init_seg_name)
    
    if os.path.exists(init_seg_path):
        print(f"Loading existing init_seg from {init_seg_path}...")
        with h5py.File(init_seg_path, 'r') as f:
            init_seg = f['main'][:]
    else:
        # init_seg = get_init_seg(
        #     vol_man.vol, 
        #     axis=args.axis, 
        #     stride=args.stride, 
        #     thr=args.remove_portion,
        #     gaussian_kernel=args.gaussian_kernel,
        #     min_bright=args.min_bright
        # )
        init_seg = get_multi_axis_init_seg(
            vol_man.vol, 
            stride=args.stride, 
            thr=args.remove_portion,
            gaussian_kernel=args.gaussian_kernel,
            min_bright=args.min_bright
            # ... 其他参数
        )
        print(f"Saving init_seg to {init_seg_path}...")
        with h5py.File(init_seg_path, 'w') as f:
            f.create_dataset('main', data=init_seg, compression='gzip')
        # 生成合并了三个轴向的初始分割图
        

    # 3. Extract Seeds
    seeds = get_seeds_from_init_seg(init_seg)
    if not seeds:
        print("No seeds found! Check parameters.")
        return

    # 4. Load Models
    print("Loading Models...")
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_checkpoint).to(args.device)
    sam_predictor = SamPredictor(sam)
    
    xmem = XMem(xmem_config, args.xmem_checkpoint).to(args.device).eval()
    if args.xmem_checkpoint:
        xmem.load_weights(torch.load(args.xmem_checkpoint), init_as_zero_if_needed=True)

    # 5. Iterative Segmentation Loop
    print(f"Starting segmentation with {len(seeds)} seeds...")
    
    processed_count = 0
    useonevos = args.useonevos
    if useonevos:
        print('use one vos')
        processor = InferenceCore(xmem, config=xmem_config)
        processor.set_all_labels([1])
    else:
        print('use different vos')
    global_mem = KeyValueMemoryStore(count_usage=False)
    for seed in tqdm(seeds, desc="Tracking"):
        z, y, x = seed
        
        # --- Check overlap (Critical for efficiency) ---
        if vol_man.global_mask[z, y, x] > 0:
            continue
        
        vol_man.global_mask[z, y, x] = 0 # Temporarily clear seed point
        
        # --- A. Tri-plane SAM Initialization ---
        crops = vol_man.get_triplane_crops(seed)
        best_axis = -1
        best_mask = None
        min_area = float('inf')

        for axis in [0, 1, 2]:
            img, box = crops[axis]
            # Map seed to local crop coords
            if axis == 0: local_pt = [seed[2]-box[3], seed[1]-box[1]] 
            elif axis == 1: local_pt = [seed[2]-box[3], seed[0]-box[1]]
            else: local_pt = [seed[1]-box[3], seed[0]-box[1]]
            
            sam_predictor.set_image(img)
            masks, scores, _ = sam_predictor.predict(
                point_coords=np.array([local_pt]), 
                point_labels=np.array([1]), 
                multimask_output=True
            )
            
            mask, score = select_masks(masks, scores, thr=0.9, crit='max', max_size=5000,min_circularity=0.6)
            
            if mask is not None:
                area = mask.sum()
                if area < min_area:
                    min_area = area
                    best_axis = axis
                    best_mask = mask.astype(np.uint8)
        
        if best_axis == -1: continue

        # --- B. XMem Propagation ---
        track_dim = [0, 1, 2][best_axis]
        start_idx = seed[track_dim]
        max_dist = 2000
        sequences = [
            range(start_idx, min(start_idx + max_dist, vol_man.shape[track_dim])),
            range(start_idx, max(start_idx - max_dist, -1), -1)
        ]
        
        if not useonevos:
            processor = InferenceCore(xmem, config=xmem_config)
            processor.set_all_labels([1])
        _, box = crops[best_axis] 
        
        for seq in sequences:
            if not seq: continue
            first_frame = True
            for curr_idx in seq:
                # Dynamic slicing
                if best_axis==0: sl = vol_man.vol[curr_idx, box[1]:box[2], box[3]:box[4]]
                elif best_axis==1: sl = vol_man.vol[box[1]:box[2], curr_idx, box[3]:box[4]]
                else: sl = vol_man.vol[box[1]:box[2], box[3]:box[4], curr_idx]
                
                if sl.size == 0: break
                
                rgb = torch.from_numpy(np.stack([sl]*3, -1)).permute(2,0,1).float().to(args.device)/255.0
                
                msk = None
                if first_frame:
                    msk = torch.from_numpy(best_mask).long().to(args.device).unsqueeze(0)
                    first_frame = False
                
                with torch.no_grad():
                    prob = processor.step(
                        rgb,
                        msk,
                        valid_labels=[1] if msk is not None else None,
                        query_pos=curr_idx,
                    )
                    pred = torch.argmax(prob, dim=0).cpu().numpy().astype(np.uint8)
                
                if pred.sum() > 5000: break 
                
                if pred.sum() > 0:
                    vol_man.update_global_mask(pred, best_axis, (curr_idx, *box[1:]))
                else:
                    break

        processed_count += 1
        if not useonevos:
            del processor

    print('Cleanup...')
    vol_man.clean_up() # 可选，根据需要取消注释

    # 6. Save Final
    file_name = args.output_filename
    final_path = os.path.join(args.output_dir, file_name)
    need_transpose = args.need_transpose
    flag = True
    if need_transpose =='False':
        flag = False
    vol_man.save(final_path,flag)
    print(f"Done! Saved to {final_path}")

if __name__ == '__main__':
    run_segmentation()
