import argparse
import numpy as np
import matplotlib.pyplot as plt

import random
import glob
from tqdm import tqdm

import torch

from huggingface_hub import hf_hub_download

import network
from datasets import SegmentationDataset
import pickle

import utils

def main():

     # Define the argument parser
    parser = argparse.ArgumentParser(description="Select random pixels for occlusion experiment")

    parser.add_argument("--n_pixels",type=int,default=10,help="Number of pixels to sample",)

    parser.add_argument("--data_path",type=str,default="../../data/LICS/test",help="Path to the training data",)
    parser.add_argument("--device",type=str,default="cuda",choices=["cuda", "cpu", "mps"],help="Device to use for training",)
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
    #utils.test_model(model, lics_dataset, device)

    # Get random pixels
    save_pixel_list(model, lics_dataset, device, args)


def save_pixel_list(model, dataset, device, args):

    n_pixels = args.n_pixels
    n = dataset.__len__()

    print("\nGetting random pixels for occlusion experiment")
    print("# Instances: {}".format(n))
    print("Number of pixels: {}".format(n_pixels))

    pixel_list = {}
    for i in range(n):

        bands, target, edge = dataset.__getitem__(i)
        name = dataset.__getname__(i)
        name = name.split(".")[0]
        print(i,name)

        pred = utils.get_predicted_mask(model, device, bands)
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

        pixel_list[name]={"water":water_pixels, "land":land_pixels, "coastline":coastline_pixels, "fp":fp_pixels, "fn":fn_pixels}
    
    # Save the pixel list
    pickle.dump(pixel_list, open("pixels.pkl", "wb"))


def get_random_pixels(mask, n_pixels):
    """Get random pixels from a binary mask.
    
    Args:
        mask (np.ndarray): A binary mask.
        n_pixels (int): Number of random pixels to retrieve.
        
    Returns:
        list: List of pixel coordinates, empty if no pixels are found.
    """
    pixels = np.argwhere(mask)
    
    # Return an empty list if there are no pixels in the mask
    if len(pixels) == 0:
        return []

    # If there are fewer pixels than requested, return all of them
    if len(pixels) < n_pixels:
        n_pixels = len(pixels)

    return random.sample(list(pixels), n_pixels)





if __name__ == "__main__":

    main()