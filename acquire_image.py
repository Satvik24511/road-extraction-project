import ee
import os
import requests
import numpy as np
import cv2
from datetime import datetime

# --- GEE Authentication & Initialization ---
# You need to run 'ee.Authenticate()' once, typically from your terminal or a Jupyter notebook.
# It will open a browser window for you to log in to your Google account.
# After authentication, you can initialize the Earth Engine API.
try:
    ee.Initialize()
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print(f"Failed to initialize GEE. Please ensure you've authenticated and have an active project. Error: {e}")
    print("Try running 'ee.Authenticate()' in your terminal or a separate script if you haven't yet.")
    exit() # Exit if GEE isn't initialized, as the rest of the script won't work

# --- Configuration ---
# Approximate coordinates for a small area within Rohini, Delhi
# You can find exact coordinates using Google Maps or a mapping tool.
# Example: a bounding box [min_lon, min_lat, max_lon, max_lat]
# This is a small area to ensure direct download is feasible.
AOI_COORDINATES = [77.10, 28.70, 77.13, 28.73] 
# (This is roughly part of Rohini Sector 16, 17 area based on a quick check)

OUTPUT_DIR = "downloaded_gee_images"
OUTPUT_FILENAME = "rohini_gee_image.png" # Using PNG as it's lossless

# --- 1. Define Area of Interest (AoI) ---
aoi = ee.Geometry.Rectangle(AOI_COORDINATES)
print(f"Defined Area of Interest (AoI): {AOI_COORDINATES}")

# --- 2. Access GEE Image Collections (Sentinel-2 as an example) ---
# Sentinel-2 Level-2A: Surface Reflectance (atmospherically corrected)
# Check GEE data catalog for specific collection IDs and band names.
collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(aoi) \
    .filterDate('2024-01-01', '2024-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) # Filter for less than 10% cloud cover

print(f"Found {collection.size().getInfo()} images in the collection for the AoI.")

# --- 3. Filter and Mosaic Images ---
# Get the median composite of all images in the collection for the AoI
# Use B4 (Red), B3 (Green), B2 (Blue) bands for RGB composite
image = collection.median().clip(aoi).select(['B4', 'B3', 'B2'])

if image.getInfo() is None:
    print("No suitable images found after filtering. Try adjusting date range or cloud filter.")
    exit()

print("Selected and clipped image for export.")

# --- 4. Export/Download the Image ---
# For small images, you can get a direct thumbnail URL.
# For larger images, you'd use ee.batch.Export.image.toDrive() or .toCloudStorage()
# and then download them from there.
# Adjust dimensions as needed. Max 4096x4096 for getDownloadURL.
scale = 10 # Sentinel-2 native resolution is 10m for RGB bands
# Calculate dimensions if you want a specific output pixel size, otherwise GEE uses default
# For 256x256 pixel output that matches your model input size
# Make sure the scale aligns with the desired resolution in meters/pixel
# If you want 256x256, you need to calculate the bounding box span in meters
# and divide by 256 to get the effective pixel resolution.
# Or simpler: set 'dimensions' directly if you want a fixed output pixel size.

# Example to get a 256x256 pixel thumbnail. This is a common way to get small images directly.
download_url = image.getThumbUrl({
    'min': 0, 'max': 3000, # Min/Max pixel values for visualization (adjust if image is too dark/bright)
    'dimensions': '256x256', # Specify output pixel dimensions
    'region': aoi.getInfo()['coordinates'] # Must pass coordinates here
})

print(f"Generated download URL: {download_url}")

# Download the image using requests
try:
    response = requests.get(download_url, stream=True)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Image downloaded successfully to {output_path}")

except requests.exceptions.RequestException as e:
    print(f"Error downloading image: {e}")
    print("This often happens if the image area is too large for a direct thumbnail export,")
    print("or if there's no data for the filters applied.")
    print("Consider adjusting AOI_COORDINATES, date range, cloud filter, or output dimensions.")
    exit()

# --- 5. Integration with your test.py workflow ---
# Now, in your test.py, you would modify get_test_images() or main() to:
# - Load this downloaded image (e.g., image = cv2.imread(output_path)).
# - Ensure it's converted to RGB if needed (cv2.cvtColor).
# - Pass it to your predict_mask function.
# Your predict_mask already handles the normalization and ToTensorV2().

print("\n--- Next Steps for Integration ---")
print(f"1. Run this script to download '{OUTPUT_FILENAME}'.")
print(f"2. Modify your 'test.py' to load '{output_path}' as one of the test images.")
print("   - You can replace a call to get_test_images with direct loading for this specific image.")
print("   - Make sure your test_transform (A.Compose) handles the image correctly.")
print("3. Run 'test.py' to see the prediction for the GEE image.")