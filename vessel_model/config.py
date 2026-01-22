import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--xmem_checkpoint', default='./saves/XMem.pth')
    parser.add_argument('--sam_checkpoint', default='./saves/sam_vit_h_4b8939.pth')
    parser.add_argument('--sam_type', default='vit_h')
    parser.add_argument('--volume_path', required=True, help='Path to .h5 file')
    parser.add_argument('--output_dir', default='./bv_seg_output', help='Directory to save outputs')
    parser.add_argument('--dataset_key', default='main', help='Key name in h5 file')
    parser.add_argument('--gpu_id', default='0', help='指定使用的GPU ID')
    parser.add_argument('--output_filename', default='segmentation.nii.gz', help='输出文件名')

    # Seed Generation Params
    parser.add_argument('--axis', type=int, default=3, help='Axis to generate init seg (0=Z, 1=Y, 2=X),3 is use all')
    parser.add_argument('--stride', type=int, default=5, help='Stride for init seg generation')
    parser.add_argument('--gaussian_kernel', type=int, default=5) 
    parser.add_argument('--min_bright', type=int, default=40)
    parser.add_argument('--remove_portion', type=float, default=0.98)
    parser.add_argument('--crop_size', type=int, default=384)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--need_transpose', default='False')
    parser.add_argument('--useonevos',type=bool,default=False)

    return parser.parse_args()

# XMem Configuration
xmem_config = {
    'enable_long_term': False,
    'enable_long_term_count_usage': True,
    'max_mid_term_frames': 10,
    'min_mid_term_frames': 5,
    'max_long_term_elements': 10000,
    'num_prototypes': 128,
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'save_scores': False,
    'temporal_decay': 5.0,
    'global_mem_max_elements': 500000,
}
