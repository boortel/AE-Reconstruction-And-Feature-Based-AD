import argparse
import json
import os
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from hardnet_updated import extract_descriptors, load_hardnet_model
from scipy.spatial import distance
from tqdm import tqdm
from ad_utils import (
    build_pyramid,
    compute_iou,
    extract_patches,
    get_image_pairs,
    load_and_crop_pair,
    rebuild_similarity_map,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./IndustryBiscuit_Folders')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--weights', type=str, default='HardNetPS.pth')
    parser.add_argument('--cov_path', type=str, default='./hardnet_checkpoint.pickle')
    parser.add_argument('--num_images', type=int, default=1070)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=0.5643743872642517)
    parser.add_argument('--pooling', type=str, choices=['max', 'min'], default='min')
    return parser.parse_args()

def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(script_dir, args.data_dir))
    output_dir = os.path.normpath(os.path.join(script_dir, args.output_dir))
    weights_path = os.path.normpath(os.path.join(script_dir, args.weights))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        pairs = get_image_pairs(data_dir, limit=args.num_images)
        if not pairs:
            print(f"No image pairs found in {data_dir}.")
            return
    except Exception as e:
        print(f"Error accessing data directory: {e}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_hardnet_model(weights_path, device)

    if os.path.exists(args.cov_path):
        print(f"Loading Covariance Matrix from {args.cov_path} ...")
        with open(args.cov_path, 'rb') as f:
            cov_data = pickle.load(f)
        
        if isinstance(cov_data, dict):
            inv_cov = list(cov_data.values())[0]
        else:
            inv_cov = cov_data
    else:
        print(f"Error: No se encontró el archivo {args.cov_path}")
        return

    scales = [1.0, 0.5, 0.25]
    metrics = {}
    aggregated_iou = []

    print(f"Processing {len(pairs)} image pairs...")

    for pair in tqdm(pairs):
        base_name = pair['base_name']
        try:
            ref_img, eval_img, mask_img, meta, roi_offset = load_and_crop_pair(
                pair['ref_path'], 
                pair['eval_path']
            )
        except Exception as e:
            print(f"Skipping {base_name}: {e}")
            continue

        base_shape = ref_img.shape[:2]
        ref_pyramid = build_pyramid(ref_img, scales)
        eval_pyramid = build_pyramid(eval_img, scales)
        scale_maps = []

        for scale_idx, scale in enumerate(scales):
            r_img = ref_pyramid[scale_idx]
            e_img = eval_pyramid[scale_idx]

            if r_img.shape[0] < args.patch_size or r_img.shape[1] < args.patch_size:
                continue

            r_patches, r_coords = extract_patches(r_img, args.patch_size, args.stride)
            e_patches, e_coords = extract_patches(e_img, args.patch_size, args.stride)

            if len(r_patches) == 0:
                continue

            r_descs = extract_descriptors(model, r_patches, device)
            e_descs = extract_descriptors(model, e_patches, device)

            diffs = r_descs - e_descs
            m_dists = np.array([distance.mahalanobis(r, e, inv_cov) for r, e in zip(r_descs, e_descs)])
            similarities = 1 / (1 + m_dists)

            sim_map_scale = rebuild_similarity_map(similarities, r_coords, r_img.shape, args.patch_size)

            if scale != 1.0:
                sim_map_base = cv2.resize(sim_map_scale, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_LINEAR)
            else:
                sim_map_base = sim_map_scale

            scale_maps.append(sim_map_base)

        if not scale_maps:
            continue

        if args.pooling == 'min':
            pooled_sim_map = np.min(np.stack(scale_maps, axis=0), axis=0)
        else:
            pooled_sim_map = np.max(np.stack(scale_maps, axis=0), axis=0)

        min_sim = float(np.min(pooled_sim_map))
        mean_sim = float(np.mean(pooled_sim_map))
        anomaly_map = (pooled_sim_map < args.threshold).astype(np.uint8)

        iou = compute_iou(anomaly_map, mask_img)
        aggregated_iou.append(iou)
        metrics[base_name] = {'iou': iou, 'min_sim': min_sim, 'mean_sim': mean_sim}

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        axes[0].imshow(ref_img, cmap='gray'); axes[0].set_title('Ref ROI')
        axes[1].imshow(eval_img, cmap='gray'); axes[1].set_title('Eval ROI')
        axes[2].imshow(mask_img, cmap='gray'); axes[2].set_title('True Mask')
        sm_plot = axes[3].imshow(pooled_sim_map, cmap='jet_r', vmin=0.0, vmax=1.0)
        axes[3].set_title('Similarity Map'); fig.colorbar(sm_plot, ax=axes[3])
        axes[4].imshow(anomaly_map, cmap='gray'); axes[4].set_title('Anomaly Detect')
        
        plt.savefig(os.path.join(output_dir, f"{base_name}_viz.png"))
        plt.close(fig)

    metrics['aggregate'] = {'mean_iou': float(np.mean(aggregated_iou)) if aggregated_iou else 0.0}
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluación completada. Mean IoU: {metrics['aggregate']['mean_iou']:.4f}")

if __name__ == '__main__':
    main()