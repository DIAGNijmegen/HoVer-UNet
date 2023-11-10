import kornia.augmentation as au


def get_augmentation_gpu():
    return au.AugmentationSequential(
        au.ColorJiggle(brightness=0.1, contrast=0.1, hue=0.1, saturation=0.1, p=0.5),
        au.RandomGaussianBlur((3, 3), (1, 2), p=0.5),
        data_keys=['input', 'mask'],
    )