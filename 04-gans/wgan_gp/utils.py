from PIL import Image
import numpy as np
from pathlib import Path


def ensure_exists(directory: str | Path) -> Path:
    if isinstance(directory, str):
        directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve()


def grid_image_from_batch(batch, num_rows, norm="center"):
    """
    Generate a grid image from a batch of images.
    Assumes input has shape (B, H, W, C).
    """

    B, H, W, _ = batch.shape

    num_cols = B // num_rows

    # Calculate the size of the output grid image
    grid_height = num_rows * H
    grid_width = num_cols * W

    # Normalize and convert to the desired data type
    if norm == "center":
        batch = np.array(batch * 127.5 + 127.5, dtype=np.uint8)
    elif norm == "minmax":
        batch = np.array(batch * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown normalization '{norm}'")

    # Reshape the batch of images into a 2D grid
    grid_image = batch.reshape(num_rows, num_cols, H, W, -1)
    grid_image = grid_image.swapaxes(1, 2)
    grid_image = grid_image.reshape(grid_height, grid_width, -1)

    # Convert the grid to a PIL Image
    return Image.fromarray(grid_image.squeeze())
