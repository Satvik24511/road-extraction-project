import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import skimage.morphology as sk

from model import UNetWithPretrainedEncoder
from utils import load_checkpoint
import config

def get_test_images(test_dir, num_images=5):
    all_images = sorted(os.listdir(test_dir))
    
    if len(all_images) > num_images:
        test_images_names = all_images[num_images:num_images+num_images]
    else:
        test_images_names = all_images
        
    test_images = []
    for img_name in test_images_names:
        img_path = os.path.join(test_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        test_images.append((image, img_name))
        
    return test_images

def predict_mask(model, original_image, transform, device):
    transformed = transform(image=original_image)
    input_image_tensor = transformed['image'].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_image_tensor)
        
    predicted_mask_tensor = torch.sigmoid(logits)
    predicted_mask_tensor = (predicted_mask_tensor > 0.5).float()

    predicted_mask_np = predicted_mask_tensor.squeeze().cpu().numpy()
    
    return predicted_mask_np


def skel(mask):    
    mask_bool = mask.astype(bool)
    skeleton = sk.skeletonize(mask_bool)
    skeleton_float = skeleton.astype(np.float32)
    return skeleton_float
    

def visualize_results(original_image, predicted_mask, skeleton, title="Result"):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)

    ax1.imshow(original_image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(predicted_mask, cmap='gray')
    ax2.set_title('Predicted Mask (Resized)')
    ax2.axis('off')
    
    ax3.imshow(skeleton, cmap='gray')
    ax3.set_title('Skeleton')
    ax3.axis('off')

    overlay_image = original_image.copy()
    overlay_image[predicted_mask == 1] = (
        overlay_image[predicted_mask == 1] * 0.5 + np.array([255, 0, 0]) * 0.5
    ).astype(np.uint8)
    ax4.imshow(overlay_image)
    ax4.set_title('Mask Overlay')
    ax4.axis('off')
    
    overlay_skel_image = original_image.copy()
    overlay_skel_image[skeleton == 1] = (
        overlay_skel_image[skeleton == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
    ).astype(np.uint8)
    ax5.imshow(overlay_skel_image)
    ax5.set_title('Skeleton Overlay')
    ax5.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    model = UNetWithPretrainedEncoder(encoder_name='resnet34', num_classes=1)
    
    checkpoint_path = os.path.join(os.getcwd(), config.CHECKPOINT)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
        
    optimizer = torch.optim.Adam(model.parameters()) 
    load_checkpoint(checkpoint_path, model, optimizer)
    
    device = config.DEVICE
    model.to(device)
    model.eval()
    
    test_transform = config.val_transforms
    
    print("Loading test images...")
    test_images_data = get_test_images(test_dir=config.TEST_DIR, num_images=5)
    
    if not test_images_data:
        print("No images found in the test directory.")
        return

    print("Generating and visualizing masks...")
    for original_image, filename in tqdm(test_images_data, desc="Processing images"):
        mask_256 = predict_mask(model, original_image, test_transform, device)
        skeleton = skel(mask_256)
        
        mask_resized = cv2.resize(mask_256, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        skel_resized = cv2.resize(skeleton, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        visualize_results(original_image, mask_resized, skel_resized, title=f"Prediction for: {filename}")

if __name__ == "__main__":
    main()