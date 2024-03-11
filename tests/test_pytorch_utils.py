import pandas as pd
import pytest
import torch
from unittest.mock import patch
from ai_smart_health.pytorch_utils import XRayDataset


sample_metadata = pd.DataFrame(
    {
        "filename": ["image1.png", "image2.png"],
        "class1": [0, 1],
        "class2": [1, 0],
        "class3": [1, 1],
    }
)

# Here: Fake directory where test images are located
img_dir = "/path/to/test/images"

# Mock response for read_image.
mock_image = torch.rand((3, 224, 224))


@pytest.fixture
def dataset():
    return XRayDataset(metadata=sample_metadata, img_dir=img_dir, classes=["class1", "class3"])


@patch("ai_smart_health.pytorch_utils.read_image", return_value=mock_image)
def test_dataset_len(mock_read_image, dataset):
    """Test that the dataset reports correct length."""
    assert len(dataset) == 2, "Dataset length should match the number of entries in metadata"


@patch("ai_smart_health.pytorch_utils.read_image", return_value=mock_image)
def test_getitem(mock_read_image, dataset):
    """Test that __getitem__ returns a data sample correctly."""
    image, label = dataset[0]
    assert torch.is_tensor(image), "The returned image should be a tensor"
    assert torch.is_tensor(label), "The returned label should be a tensor"
    assert label.shape == torch.Size([2]), "Label tensor should have the correct shape"
    assert image.size() == mock_image.size(), "Image tensor should match the mock image size"


def test_invalid_img_mode():
    """Test initializing dataset with invalid image mode raises ValueError."""
    with pytest.raises(ValueError):
        XRayDataset(
            metadata=sample_metadata,
            img_dir=img_dir,
            classes=["class1"],
            img_mode="INVALID_MODE",
        )
