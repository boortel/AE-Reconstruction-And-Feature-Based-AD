import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from ModelSaved import ModelSaved

base_dir = os.path.abspath(os.getcwd())

datasets = ['cookie']
convm_list = ['ConvM1', 'ConvM2', 'ConvM3', 'ConvM4', 'ConvM5', 'ConvM6']
ae_list = ['BAE1', 'BAE2', 'VAE1', 'VAE2', 'VQVAE1', 'DAE', 'SAE', 'AttnAE']
img_classes = ['ok', 'nok']

imageDim = (256, 256, 3) 

output_dir = os.path.join(base_dir, 'data', 'reconstructions')

# Create subdirectories for ok and nok
for img_class in img_classes:
    os.makedirs(os.path.join(output_dir, img_class), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Running on device: {device} ---")

transform = transforms.Compose([
    transforms.Resize((imageDim[0], imageDim[1]), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

for dataset in datasets:
    dataset_cap = dataset.capitalize() 
    
    for img_class in img_classes:
        print(f"Processing dataset: {dataset} | Class: {img_class.upper()}")
        
        img_path = os.path.join(base_dir, 'IndustryBiscuit_Folders', 'valid', img_class, '000.jpg')
        
        if not os.path.exists(img_path):
            print(f"Error: Image not found {img_path}")
            continue
        
        try:
            original_img = Image.open(img_path).convert('RGB')
            input_tensor = transform(original_img).unsqueeze(0).to(device) # type: ignore
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

        # Setup plot grid
        fig, axes = plt.subplots(nrows=len(convm_list), ncols=len(ae_list) + 1, figsize=(20, 20), squeeze=False)
        fig.suptitle(f"Reconstructions - Dataset: {dataset_cap} ({img_class.upper()})", fontsize=24, y=0.98)

        for i, convm in enumerate(convm_list):
            ax_orig = axes[i, 0]
            ax_orig.imshow(original_img)
            ax_orig.axis('off')
            
            ax_orig.text(-0.1, 0.5, convm, transform=ax_orig.transAxes, 
                         fontsize=14, fontweight='bold', va='center', ha='right', rotation=90)
            
            if i == 0:
                ax_orig.set_title("Original", fontsize=16, fontweight='bold', pad=10)

            for j, ae in enumerate(ae_list):
                ax_recon = axes[i, j + 1]
                weights_path = os.path.join(
                    base_dir, 'data', f'{dataset_cap}_OCC', f'{convm}_{dataset_cap}_OCC', ae, 'model.weights.pt'
                )
                
                recon_img = None

                if os.path.exists(weights_path):
                    try:
                        modelObj = ModelSaved(
                            modelSel=ae,            
                            layerSel=convm,         
                            imageDim=imageDim,      
                            dataVariance=0.5,       
                            intermediateDim=64,     
                            latentDim=32,           
                            num_embeddings=32
                        )
                        model = modelObj.get_model()
                        model.load_state_dict(torch.load(weights_path, map_location=device))
                        model.to(device)
                        model.eval()

                        with torch.no_grad():
                            output = model(input_tensor)
                            recon_tensor = output[0] if isinstance(output, tuple) else output

                        recon_tensor = recon_tensor.squeeze(0).cpu()
                        recon_tensor = torch.clamp(recon_tensor, 0, 1) 
                        recon_img = transforms.ToPILImage()(recon_tensor)

                    except Exception as e:
                        print(f"Error processing {convm}_{ae}: {e}")
                else:
                    print(f"Warning: Weights not found - {weights_path}")

                if recon_img is None:
                    recon_img = Image.new('RGB', (imageDim[0], imageDim[1]), color=(200, 200, 200))
                    ax_recon.text(0.5, 0.5, 'N/A', color='red', fontsize=12,
                                  ha='center', va='center', transform=ax_recon.transAxes)

                ax_recon.imshow(recon_img)
                ax_recon.axis('off')
                
                if i == 0:
                    ax_recon.set_title(ae, fontsize=16, fontweight='bold', pad=10)

        plt.tight_layout()
        fig.subplots_adjust(top=0.93, left=0.08) 
        
        # Save dynamically into the ok or nok folder
        save_path = os.path.join(output_dir, img_class, f"cookie_reconstructions.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved grid to: {save_path}")

print("Process completed successfully.")