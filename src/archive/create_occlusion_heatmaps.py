import argparse
import numpy as np
import matplotlib.pyplot as plt

import random
import glob
from tqdm import tqdm

from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed

from huggingface_hub import hf_hub_download

import network
from datasets import SegmentationDataset

import utils

def main():

     # Define the argument parser
    parser = argparse.ArgumentParser(description="Create occlusion heatmaps for a segmentation model")

    parser.add_argument("--data_path",type=str,default="../../data/LICS/test",help="Path to the training data",)
    parser.add_argument("--save_path",type=str,default="../maps",help="Path template for saving the model",)
    parser.add_argument("--device",type=str,default="cuda",choices=["cuda", "cpu", "mps"],help="Device to use for training",)
    parser.add_argument("--test",type=bool,default=False,help="Whether to run a test")
    args = parser.parse_args()

    # Load paths
    data_path = args.data_path
    paths = glob.glob(data_path + "/*")

    # Create dataset object
    lics_dataset = SegmentationDataset(paths) 

    # Download the model directly from Hugging Face
    model_path = hf_hub_download(
        repo_id="a-data-odyssey/coastal-image-segmentation", 
        filename="LICS_UNET_12JUL2024.pth")

    # Load the model
    model = torch.load(model_path, 
                       weights_only=False,
                       map_location=torch.device('cpu'))

    # Set the model to evaluation mode
    device = torch.device(args.device)
    model.to(device)
    model.eval()  

    # Sense check model 
    #test_model(model, lics_dataset, device)

    # Run the occlusion experiment
    do_occlusion_experiment(model, lics_dataset, device, args)

def test_model(model, dataset, device):
    print("Testing model on dataset")
    print("# Instances: {}".format(dataset.__len__()))

    # Calculate the accuracy of the model on the dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    accuracy_list = []
    for bands, target, edge in iter(dataloader):
        input = bands.to(device)
        output = model(input)

        # Get the water mask 
        target = target.squeeze()
        target_water = np.argmax(target, axis=0)

        # Get the predicted water mask
        output = output.cpu().detach().numpy().squeeze()
        output = np.argmax(output, axis=0)

        accuracy = np.mean(np.asarray(target_water == output))
        accuracy_list.append(accuracy)

    print("Mean accuracy: {:.3f}".format(np.mean(accuracy_list)))


def do_occlusion_experiment(model, dataset, device, args):

    n_segments = 1024
    patch_size = 8
    stride = 1

    n = dataset.__len__()

    for i in range(n):
        bands, target, edge = dataset.__getitem__(i)
        name = dataset.__getname__(i)
        name = name.split(".")[0]

        water_mask = target[1].numpy()
        land_mask = target[0].numpy()
        coastline_mask = edge.numpy()

        water_pixels = get_random_pixels(water_mask, 10)
        land_pixels = get_random_pixels(land_mask, 10)
        coastline_pixels = get_random_pixels(coastline_mask, 10)

        segments = get_segments(bands, method='slic', n_segments=n_segments, compactness=10)

        # Do occlusion
        map_water = generate_occlusion_map(model, device, bands, mask=water_mask, patch_size=patch_size, stride=stride)
        np.save(args.save_path + f"/{name}_regular_water_mask_{patch_size}_{stride}.npy", map_water)
        map_land = generate_occlusion_map(model, device, bands, mask=land_mask, patch_size=patch_size, stride=stride)
        np.save(args.save_path + f"/{name}_regular_land_mask_{patch_size}_{stride}.npy", map_land)
        map_coastline = generate_occlusion_map(model, device, bands, mask=coastline_mask, patch_size=patch_size, stride=stride)
        np.save(args.save_path + f"/{name}_regular_coastline_mask_{patch_size}_{stride}.npy", map_coastline)

        sp_map_water = generate_superpixel_occlusion_map(model, device, bands, mask=water_mask, superpixel_list=segments)
        np.save(args.save_path + f"/{name}_sp_water_mask_{n_segments}.npy", sp_map_water)     
        sp_map_land = generate_superpixel_occlusion_map(model, device, bands, mask=land_mask, superpixel_list=segments)
        np.save(args.save_path + f"/{name}_sp_land_mask_{n_segments}.npy", sp_map_land)
        sp_map_coastline = generate_superpixel_occlusion_map(model, device, bands, mask=coastline_mask, superpixel_list=segments)
        np.save(args.save_path + f"/{name}_sp_coastline_mask_{n_segments}.npy", sp_map_coastline)
       
        for pixel in water_pixels:
            mask = mask_from_pixel(water_mask, pixel)
            map = generate_occlusion_map(model, device, bands, mask=mask, patch_size=patch_size, stride=stride)
            np.save(args.save_path + f"/{name}_regular_water_pixel_{pixel[0]}_{pixel[1]}_{patch_size}_{stride}.npy", map)

            sp_map = generate_superpixel_occlusion_map(model, device, bands, mask=mask, superpixel_list=segments)
            np.save(args.save_path + f"/{name}_sp_water_pixel_{pixel[0]}_{pixel[1]}_{n_segments}.npy", sp_map)
        
        for pixel in land_pixels:
            mask = mask_from_pixel(land_mask, pixel)
            map = generate_occlusion_map(model, device, bands, mask=mask, patch_size=patch_size, stride=stride)
            np.save(args.save_path + f"/{name}_regular_land_pixel_{pixel[0]}_{pixel[1]}_{patch_size}_{stride}.npy", map)

            sp_map = generate_superpixel_occlusion_map(model, device, bands, mask=mask, superpixel_list=segments)
            np.save(args.save_path + f"/{name}_sp_land_pixel_{pixel[0]}_{pixel[1]}_{n_segments}.npy", sp_map)

        for pixel in coastline_pixels:
            mask = mask_from_pixel(coastline_mask, pixel)
            map = generate_occlusion_map(model, device, bands, mask=mask, patch_size=patch_size, stride=stride)
            np.save(args.save_path + f"/{name}_regular_coastline_pixel_{pixel[0]}_{pixel[1]}_{patch_size}_{stride}.npy", map)

            sp_map = generate_superpixel_occlusion_map(model, device, bands, mask=mask, superpixel_list=segments)
            np.save(args.save_path + f"/{name}_sp_coastline_pixel_{pixel[0]}_{pixel[1]}_{n_segments}.npy", sp_map)

        # Display results
        """fig, ax = plt.subplots(2, 3, figsize=(15, 10))

        ax[0, 0].imshow(map_water, cmap='hot')
        ax[0, 0].set_title("Water")
        ax[0, 0].axis("off")

        ax[0, 1].imshow(map_land, cmap='hot')
        ax[0, 1].set_title("Land")
        ax[0, 1].axis("off")

        ax[0, 2].imshow(map_coastline, cmap='hot')
        ax[0, 2].set_title("Coastline")
        ax[0, 2].axis("off")

        ax[1, 0].imshow(sp_map_water, cmap='hot')
        ax[1, 0].set_title("Superpixel Water")
        ax[1, 0].axis("off")

        ax[1, 1].imshow(sp_map_land, cmap='hot')
        ax[1, 1].set_title("Superpixel Land")
        ax[1, 1].axis("off")

        ax[1, 2].imshow(sp_map_coastline, cmap='hot')
        ax[1, 2].set_title("Superpixel Coastline")
        ax[1, 2].axis("off")

        fig,ax = plt.subplots(1,4)
        ax[0].imshow(water_mask, cmap='gray')
        ax[1].imshow(land_mask, cmap='gray')
        ax[2].imshow(coastline_mask, cmap='gray')

        img_processed = utils.get_rgb(np.asarray(bands),contrast=0.3)
        ax[3].imshow(utils.mark_boundaries(img_processed, segments))

        for a in ax:
            a.set_xticks([])
            a.set_yticks([])

        plt.show()"""

        # Generate superpixel segments

        print(name)
        if args.test:
            break

def get_random_pixels(mask, n_pixels):
    """Get random pixels from a binary mask"""
    pixels = np.argwhere(mask)
    pixels = random.sample(list(pixels), n_pixels)

    return pixels

def mask_from_pixel(mask, pixel):
    """Create a mask from a single pixel"""
    mask = np.zeros_like(mask)
    mask[pixel[0], pixel[1]] = 1

    return mask

def generate_occlusion_map(model, device, input_image, mask=None, patch_size=5, stride=1):
    """
    Generate an occlusion map for a given image input and specified mask region.

    Args:
    - model (torch.nn.Module): Pre-trained segmentation model.
    - input_image (torch.Tensor): Input image tensor of shape (C, H, W).
    - mask (torch.Tensor): Mask of shape (H, W), with 1s for pixels to analyze, 0s elsewhere.
    - patch_size (int): The size of the occlusion patch.
    - stride (int): Stride for moving the occlusion patch across the image.

    Returns:
    - occlusion_map (np.array): Heatmap of occlusion influence for the masked region.
    """

    # Ensure input image and model are on the same device
    input_image = input_image.to(device).unsqueeze(0)  # add batch dimension
    original_pred = model(input_image).squeeze(0)  # model prediction on original image
    original_pred = original_pred.cpu().detach().numpy()
    original_masked_pred = original_pred * mask  # Apply mask to the output

    # Initialize occlusion map
    occlusion_map = np.zeros(input_image.shape[2:])

    # Iterate over image in patches
    for y in tqdm(range(0, input_image.shape[2] - patch_size + 1, stride)):
        for x in range(0, input_image.shape[3] - patch_size + 1, stride):
            # Clone and occlude a patch in the input image
            occluded_image = deepcopy(input_image)
            occluded_image[:, :, y:y + patch_size, x:x + patch_size] = 0  # occlude with zeros

            # Get prediction for the occluded image
            with torch.no_grad():
                occluded_pred = model(occluded_image).squeeze(0)
                occluded_pred = occluded_pred.cpu().detach().numpy()

            # Calculate difference in predictions for the masked region
            occlusion_influence = np.abs(original_masked_pred - (occluded_pred * mask)).sum().item()
            occlusion_map[y:y + patch_size, x:x + patch_size] += occlusion_influence

    # Normalize occlusion map to [0, 1]
    occlusion_map = (occlusion_map - np.min(occlusion_map)) / (np.max(occlusion_map) - np.min(occlusion_map))

    return occlusion_map


def get_segments(bands, method='slic', **kwargs):
    # Copy the input image to avoid modifying the original
    img = np.asarray(bands)
    img = utils.get_rgb(img)

    # Define a dictionary to map method names to their respective function calls
    segmentation_methods = {
        'felzenszwalb': lambda: felzenszwalb(img, **kwargs),
        'slic': lambda: slic(img, **kwargs),
        'quickshift': lambda: quickshift(img, **kwargs),
        'watershed': lambda: watershed(sobel(rgb2gray(img)), **kwargs),
    }

    # Validate the method
    if method not in segmentation_methods:
        raise ValueError(f"Unsupported segmentation method: {method}. Choose from {list(segmentation_methods.keys())}.")

    # Try to execute the selected method, handle potential errors gracefully
    try:
        segments = segmentation_methods[method]()
    except Exception as e:
        raise ValueError(f"An error occurred during segmentation with {method}: {str(e)}")

    return segments


if __name__ == "__main__":

    main()