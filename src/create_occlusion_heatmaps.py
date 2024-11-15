import argparse
import numpy as np
import matplotlib.pyplot as plt

import random
import glob
from tqdm import tqdm
import pickle
import os

from copy import deepcopy

import torch
from torch.utils.data import DataLoader


from huggingface_hub import hf_hub_download

import network
from datasets import SegmentationDataset

import utils

def main():

     # Define the argument parser
    parser = argparse.ArgumentParser(description="Create occlusion heatmaps for a segmentation model")

    parser.add_argument("--patch_size",type=int,default=8,help="Patch size for occlusion",)
    parser.add_argument("--stride",type=int,default=1,help="Stride for occlusion",)

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

    # load pixel list
    pixel_list = pickle.load(open("pixels.pkl", "rb"))

    # Sense check model 
    #utils.test_model(model, lics_dataset, device)

    # Run the occlusion experiment
    do_occlusion_experiment(model, lics_dataset, pixel_list, device, args)


def do_occlusion_experiment(model, dataset, pixel_list, device, args):

    patch_size = args.patch_size
    stride = args.stride
    n = dataset.__len__()

    # Load list of maps already created
    maps = glob.glob(args.save_path + "/*")
    maps = [m.split("/")[-1] for m in maps]

    print("\nRunning occlusion experiment")
    print("# Instances: {}".format(n))
    print("Patch size: {}".format(patch_size))
    print("Stride: {}".format(stride))
    print("Number of maps: {}".format(len(maps)))

    for i in range(n):

        bands, target, edge = dataset.__getitem__(i)
        name = dataset.__getname__(i)
        name = name.split(".")[0]
        
        pred = utils.get_predicted_mask(model, device, bands)
        water_mask = target[1].numpy()
        land_mask = target[0].numpy()
        coastline_mask = edge.numpy()

        # get false negatives and false positives
        fp_mask = np.logical_and(np.logical_not(water_mask), pred) # predicted water, actual land
        fn_mask = np.logical_and(water_mask, np.logical_not(pred)) # predicted land, actual water
        
        # Get random pixels
        water_pixels = pixel_list[name]["water"]
        land_pixels = pixel_list[name]["land"]
        coastline_pixels = pixel_list[name]["coastline"]
        fp_pixels = pixel_list[name]["fp"]
        fn_pixels = pixel_list[name]["fn"]

        print(i,name,len(water_pixels),len(land_pixels),len(coastline_pixels),len(fp_pixels),len(fn_pixels))

        # Do occlusion for masks
        masks = [water_mask, land_mask, coastline_mask, fp_mask, fn_mask]
        pixel_lists = [water_pixels, land_pixels, coastline_pixels, fp_pixels, fn_pixels]
        types = ["water", "land", "coastline", "fp", "fn"]

        for mask, mask_type in tqdm(zip(masks, types)):
            save_name = f"{name}_mask_{mask_type}_{patch_size}_{stride}.npy"
            if save_name in maps:
                continue

            map = generate_occlusion_map(model, device, bands, mask=mask, patch_size=patch_size, stride=stride)
            np.save(os.path.join(args.save_path, save_name), map)

        # Do occlusion for pixels
        for pixel_list, pixel_type in tqdm(zip(pixel_lists, types)):
            for pixel in pixel_list:
                save_name = f"{name}_pixel_{pixel_type}_{pixel[0]}_{pixel[1]}_{patch_size}_{stride}.npy"
                if save_name in maps:
                    continue

                mask = mask_from_pixel(water_mask, pixel) # only the dimensions of mask matter
                map = generate_occlusion_map(model, device, bands, mask=mask, patch_size=patch_size, stride=stride)
                np.save(os.path.join(args.save_path, save_name), map)

        if args.test:
            break

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