import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode


class XRayDataset(Dataset):
    """
    A custom PyTorch Dataset class for chest X-Ray images.

    This class is designed to handle X-Ray image datasets, supporting both grayscale
    and RGB image modes. It allows for on-the-fly transformations of the images and labels,
    facilitating data augmentation and preprocessing steps.

    Parameters:
    - metadata (pd.DataFrame): DataFrame containing image metadata (e.g., filenames, labels).
    - img_dir (str): Directory path where images are stored.
    - classes (list): List of column names in `metadata` representing the label(s) for each image.
    - img_mode (str, optional): The mode of the images, either "RGB" or "GRAY". Default is "RGB".
    - transform (callable, optional): A function/transform that takes in an image and returns a transformed version. E.g., data augmentation procedures.
    - target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    - filename_prefix (str, optional): Prefix to add to filenames from `metadata` before loading images. Useful if `metadata` filenames do not include a common path prefix that is present in `img_dir`.

    Usage:
    dataset = XRayDataset(metadata=df, img_dir="/path/to/images", classes=['Normal', 'Pneumonia'], img_mode='RGB')
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        img_dir: str,
        classes: list,
        img_mode: str = "RGB",
        transform=None,
        target_transform=None,
    ):
        self.img_metadata = metadata
        self.img_dir = img_dir
        self.classes = classes
        if img_mode.lower() == "rgb":
            self.img_mode = ImageReadMode.RGB
        elif img_mode.lower() == "gray":
            self.img_mode = ImageReadMode.GRAY
        else:
            raise ValueError("Unknown image mode (img_mode), must be GRAY or RGB.")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.img_metadata)

    def __getitem__(self, idx):
        """Retrieves the image and its label at the specified index `idx`.

        Parameters:
        - idx (int): Index of the item to retrieve.

        Returns:
        - tuple: (image, label) where image is the transformed image tensor, and label is the corresponding label tensor.
        """
        # Construct the full path to the image file
        img_path = os.path.join(self.img_dir, self.img_metadata.iloc[idx, 0])

        # Read the image file
        image = read_image(img_path, mode=self.img_mode)

        # Extract label(s) for the current image
        label = torch.tensor(self.img_metadata[self.classes].iloc[idx, :], dtype=torch.float32)

        # Apply transformations to the image and label if any
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
