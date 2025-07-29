import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import skimage.morphology as sk
from shapely.geometry import LineString

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

def skel(model, original_image, transform, device):
    transformed = transform(image=original_image)
    input_image_tensor = transformed['image'].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_image_tensor)
        
    predicted_mask_tensor = torch.sigmoid(logits)
    predicted_mask_tensor = (predicted_mask_tensor > 0.5).float()

    predicted_mask_np = predicted_mask_tensor.squeeze().cpu().numpy()
    
    mask_bool = predicted_mask_np.astype(bool)
    skeleton = sk.skeletonize(mask_bool)
    skeleton_float = skeleton.astype(np.float32)
    print(skeleton_float.shape)
    return skeleton_float    

def vectorize_mask(skeleton):
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    contours, _ = cv2.findContours(skeleton_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
    vector_lines = []
    for contour in contours:
        if len(contour) >= 2:
            xy_coords = contour.squeeze().tolist()

            line = LineString(xy_coords)
            simplified_line = line.simplify(tolerance=1.5, preserve_topology=True) 
            vector_lines.append(simplified_line)
            
    return vector_lines
    
def visualize_results(original_image, skeleton, vector_lines, title="Result"):
    
    original_height, original_width, _ = original_image.shape
    model_size = 256

    scale_x = original_width / model_size
    scale_y = original_height / model_size
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 7))
    fig.suptitle(title, fontsize=18)

    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(skeleton, cmap='gray')
    axes[1].set_title('Skeleton')
    axes[1].axis('off')
    
    overlay_skel_image = original_image.copy()
    overlay_skel_image[skeleton == 1] = (
        overlay_skel_image[skeleton == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
    ).astype(np.uint8)
    axes[2].imshow(overlay_skel_image)
    axes[2].set_title('Skeleton Overlay')
    axes[2].axis('off')
    
    height, width = skeleton.shape 
    
    axes[3].set_title('Vectorized Lines')
    axes[3].set_facecolor('black')
    
    axes[3].set_ylim(height, 0) 
    axes[3].set_xlim(0, width)
    for line in vector_lines:
        if line.is_empty: 
            continue
        coords = np.array(line.coords)
        scaled_x_coords = coords[:, 0] * scale_x
        scaled_y_coords = coords[:, 1] * scale_y
        axes[3].plot(scaled_x_coords, scaled_y_coords, color='red', linewidth=1.5)
    axes[3].set_aspect('equal') 
    axes[3].axis('off')
    
    axes[4].imshow(original_image)
    axes[4].set_title('Vector Overlay')
    for line in vector_lines:
        if line.is_empty: 
            continue
        coords = np.array(line.coords)
        scaled_x_coords = coords[:, 0] * scale_x
        scaled_y_coords = coords[:, 1] * scale_y
        axes[4].plot(scaled_x_coords, scaled_y_coords, color='red', linewidth=1) 
    axes[4].set_aspect('equal')
    axes[4].axis('off')
    
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
        mask_256 = skel(model, original_image, test_transform, device)
        vector_lines = vectorize_mask(mask_256)
        
        print(f"Number of vectorized lines generated: {len(vector_lines)}")
        if len(vector_lines) > 0:
            print(f"Is the first line empty? {vector_lines[0].is_empty}")
            print(f"Length of the first line: {vector_lines[0].length}") 
        
        mask_resized = cv2.resize(mask_256, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        visualize_results(original_image, mask_resized, vector_lines, title=f"Result for {filename}")

if __name__ == "__main__":
    main()