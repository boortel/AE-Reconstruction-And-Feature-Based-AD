import glob
import os

import numpy as np
import cv2


def extract_patches(img, patch_size=32, stride=16):
    """
    Slides a window over the image and extracts patches.
    Returns the patches and their (Y, X) coordinates.
    """
    h, w = img.shape[:2]
    patches = []
    coords = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))
            
    return np.array(patches), coords

def rebuild_similarity_map(similarities, coords, base_shape, patch_size):
    """
    Takes the similarity values of each patch and reassembles them
    into an image of the original size, averaging the overlapping areas.
    """
    h, w = base_shape
    sim_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    for sim, (y, x) in zip(similarities, coords):
        sim_map[y:y+patch_size, x:x+patch_size] += sim
        count_map[y:y+patch_size, x:x+patch_size] += 1
        
    count_map[count_map == 0] = 1
    
    return sim_map / count_map

def build_pyramid(image, scales=[1.0, 0.5, 0.25]):
    """
    Generates versions of the image at different scales.
    """
    pyramid = []
    for scale in scales:
        if scale == 1.0:
            pyramid.append(image)
        else:
            new_w = int(image.shape[1] * scale)
            new_h = int(image.shape[0] * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            pyramid.append(resized)
    return pyramid

def compute_iou(mask_true, mask_pred):
    """
    Computes the IoU (Intersection over Union) metric between two binary masks.
    """
    m_true = mask_true > 0
    m_pred = mask_pred > 0
    
    intersection = np.logical_and(m_true, m_pred)
    union = np.logical_or(m_true, m_pred)
    
    if np.sum(union) == 0:
        return 0.0
    return np.sum(intersection) / np.sum(union)

def load_and_crop_pair(ref_path, eval_path):
    """
    Loads two images and ensures they have the same size by cropping to the minimum dimensions.
    """
    img_ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    img_eval = cv2.imread(eval_path, cv2.IMREAD_GRAYSCALE)
    
    if img_ref is None or img_eval is None:
        return None, None
        
    h, w = min(img_ref.shape[0], img_eval.shape[0]), min(img_ref.shape[1], img_eval.shape[1])
    
    return img_ref[:h, :w], img_eval[:h, :w]


def get_image_pairs(data_dir, limit=None):
    pairs = []
    
    ref_path = os.path.join(data_dir, 'valid', 'ok', '000.jpg')
    
    if not os.path.exists(ref_path):
        print(f"Advertencia: No se encontró la imagen de referencia en {ref_path}")
        return pairs

    eval_images = glob.glob(os.path.join(data_dir, 'valid', '**', '*.jpg'), recursive=True)
    
    for eval_path in eval_images:
        base_name = os.path.splitext(os.path.basename(eval_path))[0]
        folder_name = os.path.basename(os.path.dirname(eval_path))
        
        if eval_path == ref_path:
            continue
            
        mask_path = os.path.join(data_dir, 'ground_truth', folder_name, f"{base_name}.png")
        
        if not os.path.exists(mask_path):
            mask_path = None
            
        pair_info = {
            'base_name': f"{folder_name}_{base_name}",
            'ref_path': ref_path,
            'eval_path': eval_path,
            'mask_path': mask_path,
            'meta': {}, 
            'roi_offset': (0, 0)
        }
        pairs.append(pair_info)
    
    if limit is not None and limit > 0:
        pairs = pairs[:limit]
        
    return pairs