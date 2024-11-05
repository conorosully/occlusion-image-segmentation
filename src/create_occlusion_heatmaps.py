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

    parser.add_argument("--patch_size",type=int,default=8,help="Patch size for occlusion",)
    parser.add_argument("--stride",type=int,default=1,help="Stride for occlusion",)
    parser.add_argument("--n_pixels",type=int,default=10,help="Number of pixels to sample",)

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
    test_model(model, lics_dataset, device)

    # Run the occlusion experiment
    do_occlusion_experiment(model, lics_dataset, device, args)

def get_predicted_mask(model,device, bands):

    """Get the predicted mask from the model"""

    if len(bands.shape) == 3:
        bands = bands.unsqueeze(0)

    input = bands.to(device)
    output = model(input)

    # Get the predicted water mask
    output = output.cpu().detach().numpy().squeeze()
    output = np.argmax(output, axis=0)

    return output

def test_model(model, dataset, device):
    print("Testing model on dataset")
    print("# Instances: {}".format(dataset.__len__()))

    # Calculate the accuracy of the model on the dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    accuracy_list = []
    for bands, target, edge in iter(dataloader):

         # Get the water mask 
        target = target.squeeze()
        target_water = np.argmax(target, axis=0)
        
        # Get the predicted mask
        output = get_predicted_mask(model, device, bands)

        accuracy = np.mean(np.asarray(target_water == output))
        accuracy_list.append(accuracy)

    print("Mean accuracy: {:.3f}".format(np.mean(accuracy_list)))

def do_occlusion_experiment(model, dataset, device, args):

    patch_size = args.patch_size
    stride = args.stride
    n_pixels = args.n_pixels
    n = dataset.__len__()

    print("\nRunning occlusion experiment")
    print("# Instances: {}".format(n))
    print("Patch size: {}".format(patch_size))
    print("Stride: {}".format(stride))
    print("Number of pixels: {}".format(n_pixels))

    for i in tqdm(range(n)):
        bands, target, edge = dataset.__getitem__(i)
        name = dataset.__getname__(i)
        name = name.split(".")[0]

        pred = get_predicted_mask(model, device, bands)
        water_mask = target[1].numpy()
        land_mask = target[0].numpy()
        coastline_mask = edge.numpy()

        # get false negatives and false positives
        fp_mask = np.logical_and(np.logical_not(water_mask), pred) # predicted water, actual land
        fn_mask = np.logical_and(water_mask, np.logical_not(pred)) # predicted land, actual water
        
        # Get random pixels
        water_pixels = get_random_pixels(water_mask, n_pixels)
        land_pixels = get_random_pixels(land_mask, n_pixels)
        coastline_pixels = get_random_pixels(coastline_mask, n_pixels)
        fp_pixels = get_random_pixels(fp_mask, n_pixels)
        fn_pixels = get_random_pixels(fn_mask, n_pixels)

        # Do occlusion for masks
        map_water = generate_occlusion_map(model, device, bands, mask=water_mask, patch_size=patch_size, stride=stride)
        np.save(args.save_path + f"/{name}_mask_water_{patch_size}_{stride}.npy", map_water)
        map_land = generate_occlusion_map(model, device, bands, mask=land_mask, patch_size=patch_size, stride=stride)
        np.save(args.save_path + f"/{name}_mask_land_{patch_size}_{stride}.npy", map_land)
        map_coastline = generate_occlusion_map(model, device, bands, mask=coastline_mask, patch_size=patch_size, stride=stride)
        np.save(args.save_path + f"/{name}_mask_coastline_{patch_size}_{stride}.npy", map_coastline)
        map_fp = generate_occlusion_map(model, device, bands, mask=fp_mask, patch_size=patch_size, stride=stride)
        np.save(args.save_path + f"/{name}_mask_fp_{patch_size}_{stride}.npy", map_fp)
        map_fn = generate_occlusion_map(model, device, bands, mask=fn_mask, patch_size=patch_size, stride=stride)
        np.save(args.save_path + f"/{name}_mask_fn_{patch_size}_{stride}.npy", map_fn)

        # Do occlusion for pixels
        for pixel in water_pixels:
            mask = mask_from_pixel(water_mask, pixel)
            map = generate_occlusion_map(model, device, bands, mask=mask, patch_size=patch_size, stride=stride)
            np.save(args.save_path + f"/{name}_pixel_water_{pixel[0]}_{pixel[1]}_{patch_size}_{stride}.npy", map)

        for pixel in land_pixels:
            mask = mask_from_pixel(land_mask, pixel)
            map = generate_occlusion_map(model, device, bands, mask=mask, patch_size=patch_size, stride=stride)
            np.save(args.save_path + f"/{name}_pixel_land_{pixel[0]}_{pixel[1]}_{patch_size}_{stride}.npy", map)

        for pixel in coastline_pixels:
            mask = mask_from_pixel(coastline_mask, pixel)
            map = generate_occlusion_map(model, device, bands, mask=mask, patch_size=patch_size, stride=stride)
            np.save(args.save_path + f"/{name}_pixel_coastline_pixel_{pixel[0]}_{pixel[1]}_{patch_size}_{stride}.npy", map)

        for pixel in fp_pixels:
            mask = mask_from_pixel(fp_mask, pixel)
            map = generate_occlusion_map(model, device, bands, mask=mask, patch_size=patch_size, stride=stride)
            np.save(args.save_path + f"/{name}_pixel_fp_{pixel[0]}_{pixel[1]}_{patch_size}_{stride}.npy", map)
        
        for pixel in fn_pixels:
            mask = mask_from_pixel(fn_mask, pixel)
            map = generate_occlusion_map(model, device, bands, mask=mask, patch_size=patch_size, stride=stride)
            np.save(args.save_path + f"/{name}_pixel_fn_{pixel[0]}_{pixel[1]}_{patch_size}_{stride}.npy", map)

        print(i,name)
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
    for y in range(0, input_image.shape[2] - patch_size + 1, stride):
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


if __name__ == "__main__":

    main()