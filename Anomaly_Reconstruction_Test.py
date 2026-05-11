import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from ModelSaved import ModelSaved
from hardnet_updated import load_hardnet_model, extract_descriptors
from ad_utils import extract_patches, rebuild_similarity_map

base_dir = os.path.abspath(os.getcwd())

dataset = 'cookie'
img_classes = ['ok', 'nok']

convm_list = ['ConvM1', 'ConvM2', 'ConvM3', 'ConvM4', 'ConvM5', 'ConvM6'] 
ae_list = ['BAE1', 'BAE2', 'VAE1', 'VAE2', 'VQVAE1', 'DAE', 'SAE', 'AttnAE']

imageDim = (256, 256, 3)

patch_size = 32
stride = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = os.path.join(base_dir, 'data', 'anomaly_results')
os.makedirs(output_dir, exist_ok=True)

hardnet_path = os.path.join(base_dir, 'HardNetPS.pth')
hardnet_model = load_hardnet_model(
    #weights_path=hardnet_path,
)
hardnet_model.eval()

transform = transforms.Compose([
    transforms.Resize((imageDim[0], imageDim[1])),
    transforms.ToTensor(),
])

def compute_anomaly_map(orig_gray, recon_gray):
    orig_patches, coords = extract_patches(orig_gray, patch_size, stride)
    recon_patches, _ = extract_patches(recon_gray, patch_size, stride)

    if len(orig_patches) == 0:
        return np.zeros(orig_gray.shape)

    orig_patches = orig_patches[:, None, :, :]
    recon_patches = recon_patches[:, None, :, :]

    orig_patches = torch.from_numpy(orig_patches).float().to(device) / 255.0
    recon_patches = torch.from_numpy(recon_patches).float().to(device) / 255.0

    with torch.no_grad():
        orig_desc = hardnet_model(orig_patches).cpu().numpy()
        recon_desc = hardnet_model(recon_patches).cpu().numpy()

    diff = orig_desc - recon_desc
    dists = np.linalg.norm(diff, axis=1)

    sim_map = rebuild_similarity_map(dists, coords, orig_gray.shape, patch_size)
    return sim_map

for img_class in img_classes:
    img_path = os.path.join(
        base_dir,
        'IndustryBiscuit_Folders',
        'valid',
        img_class,
        '000.jpg'
    )

    if not os.path.exists(img_path):
        print(f"Missing image: {img_path}")
        continue

    print(f"\n--- Processing Image Class: {img_class} ---")

    original_img = Image.open(img_path).convert('RGB')
    input_tensor = transform(original_img).unsqueeze(0).to(device) # type: ignore

    orig_gray_np = np.array(original_img.resize((256, 256)).convert('L'))
    orig_gray_blurred = cv2.GaussianBlur(orig_gray_np, (21, 21), 0)
    
    _, orig_mask = cv2.threshold(orig_gray_np, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    orig_mask = cv2.morphologyEx(orig_mask, cv2.MORPH_OPEN, kernel)

    for convm in convm_list:
        for ae in ae_list:
            print(f"  -> Testing combination: {convm} + {ae}")

            combo_dir = os.path.join(output_dir, f"{convm}_{ae}")
            os.makedirs(combo_dir, exist_ok=True)

            weights_path = os.path.join(
                base_dir,
                'data',
                f'{dataset.capitalize()}_OCC',
                f'{convm}_{dataset.capitalize()}_OCC',
                ae,
                'model.weights.pt'
            )

            if not os.path.exists(weights_path):
                print(f"     [!] Missing model weights: {weights_path}")
                continue

            model = ModelSaved(
                modelSel=ae,
                layerSel=convm,
                imageDim=imageDim,
                dataVariance=0.5,
                intermediateDim=64,
                latentDim=32,
                num_embeddings=32
            ).get_model()

            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.to(device)
            model.eval()

            with torch.no_grad():
                out = model(input_tensor)
                recon = out[0] if isinstance(out, tuple) else out

            recon = recon.squeeze(0).cpu()
            recon = torch.clamp(recon, 0, 1)
            recon_img = transforms.ToPILImage()(recon)

            recon_gray_np = np.array(recon_img.convert('L'))
            _, recon_mask = cv2.threshold(recon_gray_np, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            recon_mask = cv2.morphologyEx(recon_mask, cv2.MORPH_OPEN, kernel)
            
            combined_mask = np.logical_or(orig_mask, recon_mask).astype(np.float32)
            combined_mask = recon_mask.astype(np.float32)

            sim_map_hardnet = compute_anomaly_map(orig_gray_blurred, recon_gray_np)
            
            sim_map_hardnet = (sim_map_hardnet - sim_map_hardnet.min()) / (sim_map_hardnet.max() - sim_map_hardnet.min() + 1e-8)

            pixel_diff = cv2.absdiff(orig_gray_blurred, recon_gray_np)
            pixel_diff_norm = pixel_diff / 255.0

            sim_map = sim_map_hardnet + pixel_diff_norm

            sim_map = gaussian_filter(sim_map, sigma=4)
            sim_map = sim_map * combined_mask

            cookie_errors = sim_map[combined_mask > 0]
            
            if len(cookie_errors) > 0:
                dynamic_threshold = np.percentile(cookie_errors, 95) 
            else:
                dynamic_threshold = 0.5 

            raw_mask = (sim_map > dynamic_threshold).astype(np.uint8)

            kernel_close = np.ones((9, 9), np.uint8)
            solid_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_close)

            contours, _ = cv2.findContours(solid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            anomaly_mask = np.zeros_like(solid_mask)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 150:  
                    cv2.drawContours(anomaly_mask, [cnt], -1, 1, thickness=cv2.FILLED)

            fig, axes = plt.subplots(1, 4, figsize=(18, 5))

            axes[0].imshow(original_img)
            axes[0].set_title(f"Original ({img_class})")
            axes[0].axis('off')

            axes[1].imshow(recon_img)
            axes[1].set_title(f"Recon ({convm}+{ae})")
            axes[1].axis('off')

            im = axes[2].imshow(sim_map, cmap='jet')
            axes[2].set_title("Anomaly Map")
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

            # Overlay
            axes[3].imshow(original_img.resize((256, 256)))
            red_mask = np.zeros((256, 256, 4))
            red_mask[anomaly_mask == 1] = [1, 0, 0, 0.6] 
            axes[3].imshow(red_mask)
            axes[3].set_title("Mask Overlay")
            axes[3].axis('off')

            plt.tight_layout()

            save_path = os.path.join(combo_dir, f"{img_class}_{convm}_{ae}_result.png")
            plt.savefig(save_path, dpi=150)
            plt.close()

            print(f"     -[OK] Saved: {convm}_{ae}/{img_class}_{convm}_{ae}_result.png")

print("\nDone. All combinations tested and sorted.")