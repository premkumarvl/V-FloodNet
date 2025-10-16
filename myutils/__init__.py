from .data import *
from .plot_depth import Visualizer
from .system import *

import numpy as np
from PIL import Image

def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (PIL Image) to a one-hot format.
    The palette is used to identify the class of each pixel.

    Args:
        mask (PIL.Image): The segmentation mask.
        palette (list): The color palette where the index corresponds to the class ID.

    Returns:
        np.array: A one-hot encoded NumPy array of shape (Height, Width, Num_Classes).
    """
    # Convert the PIL image to a NumPy array
    mask_np = np.array(mask, dtype=np.uint8)
    
    # Get the number of classes from the palette length
    num_classes = len(palette)
    
    # Create an empty NumPy array for the one-hot encoding
    # Shape: (Height, Width, Num_Classes)
    one_hot = np.zeros((mask_np.shape[0], mask_np.shape[1], num_classes), dtype=np.uint8)
    
    # Iterate through each class and set the corresponding channel to 1
    for class_id in range(num_classes):
        # Find all pixels in the mask that have the value of the current class_id
        one_hot[:, :, class_id] = (mask_np == class_id).astype(np.uint8)
        
    return one_hot