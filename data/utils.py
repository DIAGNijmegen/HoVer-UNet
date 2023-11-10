import numpy as np


def remap_labels(mask):
    nidx = 1
    new_mask = np.zeros(mask.shape[:2])
    for ch in range(1, mask.shape[-1]):
        for idx in np.unique(mask[..., ch]):
            if idx != 0:
                new_mask[mask[..., ch] == idx] = nidx
                nidx += 1
    return new_mask


def unpatchify(patches, patch_size, overlap, image_size):
    """
    Unpatchify an image using overlapping patches, with median in the overlapped points.

    Parameters:
    patches: a numpy array of shape (num_patches, patch_size[0], patch_size[1], num_channels)
             containing the patches to be unpatchified
    patch_size: a tuple (patch_height, patch_width) specifying the size of each patch
    overlap: an integer specifying the overlap between patches
    image_size: a tuple (image_height, image_width) specifying the size of the image to be
                reconstructed from the patches

    Returns:
    image: a numpy array of shape (image_height, image_width, num_channels) containing the
           reconstructed image
    """
    # Compute the step size for moving the patch
    step = patch_size[0] - overlap
    # Initialize the image to be reconstructed
    image = np.zeros(image_size + (patches.shape[-1],))
    # Initialize the current position in the image
    pos = [0, 0]
    # Iterate over the patches and place them in the image
    for patch in patches:
        image[pos[0]:pos[0] + patch_size[0], pos[1]:pos[1] + patch_size[1], :] += patch
        pos[1] += step
        # If we reached the end of the row, move to the next row
        if pos[1] + patch_size[1] > image_size[1]:
            pos[0] += step
            pos[1] = 0
    # Compute the number of overlapping pixels for each pixel
    overlap_count = np.zeros(image_size + (patches.shape[-1],))
    pos = [0, 0]
    for patch in patches:
        overlap_count[pos[0]:pos[0] + patch_size[0], pos[1]:pos[1] + patch_size[1], :] += 1
        pos[1] += step
        if pos[1] + patch_size[1] > image_size[1]:
            pos[0] += step
            pos[1] = 0
    # Divide the sum of the overlapping pixels by the number of overlapping pixels to get the median
    image = image / overlap_count
    return image
