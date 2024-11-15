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

    rgb = img[[r,g,b]]
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