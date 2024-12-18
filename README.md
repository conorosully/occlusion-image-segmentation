# weighted-context-interpretation

Weighted Context: A Global Interpretation Method for Image Segmentation Models

This repository contains the code required to reproduce the results in the conference paper:

> To be updated

This code is only for academic and research purposes. Please cite the above paper if you intend to use whole/part of the code. 

## Data Files

We have used the following dataset in our analysis: 

1. The Landsat Irish Coastal Segmentation (LICS) Dataset found [here](https://zenodo.org/records/8414665).

 The data is available under the Creative Commons Attribution 4.0 International license.

## Code Files
You can find the following files in the src folder:

- `get_results.ipynb` Used to calculate weighted context values and get all results in the paper
- `select_random_pixels.py` Select random pixels for the occlusion experiment
- `create_occlusion_heatmaps.py` Create occlusion heatmaps for a segmentation model
- `datasets.py` Contains the PyTorch dataset class necessary to load and format the LICS dataset
- `network.py` Contains the U-Net architecture used to load the LICS model
- `utils.py` Contains additional helper functions for the project

## Additional Files

- `pixels.json` The sample of randomly selected pixels 

