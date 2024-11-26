# Class containing utility functions for the project

#import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed

from skimage.filters import threshold_otsu

from torch.utils.data import DataLoader
import cv2

base_path = "../data/"
save_path = "/Users/conorosullivan/Google Drive/My Drive/UCD/research/Journal Paper 2 - superpixels/figures/"

# dictionary of band names for each satellite
band_dic = {"sentinel":{"coastal":0,"blue":1,"green":2,"red":3,"rededge1":4,"rededge2":5,"rededge3":6,"nir":7,"narrownir":8,"watervapour":9,"swir1":10,"swir2":11},
         "landsat":{"blue":0,"green":1,"red":2,"nir":3,"swir1":4,"swir2":5,"thermal":6}}

def scale_bands(img,satellite="landsat"):
    """Scale bands to 0-1"""
    img = img.astype("float32")
    if satellite == "landsat":
        img = np.clip(img * 0.0000275 - 0.2, 0, 1)
    elif satellite == "sentinel":
        img = np.clip(img/10000, 0, 1)
    return img

def save_fig(fig, name):
    """Save figure to figures folder"""
    fig.savefig(save_path + f"/{name}.png", dpi=300, bbox_inches="tight")


def display_bands(img, satellite="landsat"):
    """Visualize all SR bands of a satellite image."""

    if satellite == "landsat":
        n = img.shape[2]
        band_names = ["Blue","Green","Red","NIR","SWIR1","SWIR2","Thermal"]
        if n == 8:
            band_names.append("Mask")
        else:
            # Bands for LICS test set 
            band_names.extend(["QA","Train","Mask","Edge"])

    elif satellite == "sentinel":
        n = 13
        band_names = [ "Coastal","Blue","Green","Red","Red Edge 1","Red Edge 2","Red Edge 3","NIR","Red Edge 4","Water Vapour","SWIR1","SWIR2","Mask"]

    fig, axs = plt.subplots(1, n, figsize=(20, 5))

    for i in range(n):
        if np.unique(img[:, :, i]).size == 1:
            axs[i].imshow(img[:, :, i], cmap="gray", vmin=0, vmax=1)
        else:
            axs[i].imshow(img[:, :, i], cmap="gray")
        axs[i].set_title(band_names[i])
        axs[i].axis("off")


def get_rgb(img, bands=['red','green',"blue"],satellite ='landsat', contrast=1):
    """Convert a stacked array of bands to RGB"""

    r = band_dic[satellite][bands[0]]
    g = band_dic[satellite][bands[1]]
    b = band_dic[satellite][bands[2]]

    r = img[:,:,r]
    g = img[:,:,g]
    b = img[:,:,b]
    rgb = np.stack([r,g,b],axis=0)

    rgb = scale_bands(rgb, satellite)
    rgb = rgb.astype(np.float32)
    
    #rgb = scale_bands(rgb, satellite)
    rgb = np.clip(rgb, 0, contrast) / contrast

    rgb = rgb.transpose(1,2,0)

    return rgb


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

def calc_weighted_context(pixel,heatmap):

    """Calculate the weighted context of a pixel based on the occlusion map."""

    # Flatten the heatmap
    pixel_cords = np.indices(heatmap.shape).reshape(2, -1).T
    flat_heatmap = heatmap.flatten()
    flat_heatmap = np.array(flat_heatmap)

    # Compute the euclidian distance between the pixel and all other pixels
    pixel_cords = np.array(pixel_cords)
    pixel = np.array(pixel)
    x, y = pixel

    distances = np.sqrt((pixel_cords[:, 0] - x)**2 + (pixel_cords[:, 1] - y)**2)

    # Compute the weighted distance
    weighted_context = np.sum(distances * flat_heatmap)

    normalized_weighted_context = weighted_context / np.sum(distances)

    return weighted_context, normalized_weighted_context

def occlusion_visualisation(background, map, alpha=0.5, use_rgb=True):
    """Overlay the occlusion map on the background image.
    
    Args:
        background (np.array): The background image.
        map (np.array): The occlusion map.
        alpha (float): The transparency of the occlusion map.
        
    Returns:
        np.array: The visualization.
    """
    # Normalize the occlusion map to the range [0, 1]
    map = (map - np.min(map)) / (np.max(map) - np.min(map) + 1e-8)

     # Convert the occlusion map to a heatmap
    heatmap = cv2.applyColorMap((map * 255).astype(np.uint8), cv2.COLORMAP_JET)

    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Ensure the background is 3-channel (RGB)
    if len(background.shape) == 2 or background.shape[2] == 1:  # Grayscale or binary segmentation
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
    
    # Convert both background and heatmap to the same data type (uint8)
    if background.dtype != np.uint8:
        background = (255 * (background - np.min(background)) / (np.max(background) - np.min(background) + 1e-8)).astype(np.uint8)
    
    # Overlay the heatmap on the background
    overlay = cv2.addWeighted(heatmap, alpha, background, 1 - alpha, 0)
    
    return overlay

def plot_occlusion_map(img, overlay, pixel, colour=(255,0,0),length=5, weighted_context= None):

    # Add cross at pixel location
    img = img.copy()
    # add diagonal lines
    cv2.line(img, (pixel[1]-length, pixel[0]-length), (pixel[1]+length, pixel[0]+length), colour, 2)
    cv2.line(img, (pixel[1]+length, pixel[0]-length), (pixel[1]-length, pixel[0]+length), colour, 2)


    # Plotting the occlusion map
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].imshow(img)
    ax[0].set_title('Image')

    ax[1].imshow(overlay)
    if weighted_context is None:
        ax[1].set_title('Occlusion Map')
    else:
        ax[1].set_title(f'Occlusion Map ({weighted_context})')
    
    for a in ax:
        a.axis('off')