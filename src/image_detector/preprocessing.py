"""
preprocessing.py — Robust Image Ingestion, Validation, and Preprocessing Pipeline

Sprint 2: Baseline Detection Service (Member 1: Divisha Manak Bohra - 23ESKCA038)
Task 1: Developing an image inference service with a defined input/output interface
"""

import base64
import io
import os
from pathlib import Path
from typing import Union, Tuple

from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as T

from src.image_detector.schemas import ImageMetadata

# --------------------------------------------------------------------------
# Constants & Defaults
# --------------------------------------------------------------------------
SUPPORTED_FORMATS = ["PNG", "JPEG", "JPG", "WEBP", "BMP"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_SIZE = (224, 224)


# --------------------------------------------------------------------------
# Domain Exceptions
# --------------------------------------------------------------------------
class ImageProcessingError(Exception):
    """Base exception for image processing errors."""
    pass


class EmptyImageError(ImageProcessingError):
    """Raised when the provided image payload is empty or zero-byte."""
    pass


class CorruptedImageError(ImageProcessingError):
    """Raised when the image bytes cannot be decoded or verified."""
    pass


class UnsupportedFormatError(ImageProcessingError):
    """Raised when an image format is outside the supported set."""
    pass


# --------------------------------------------------------------------------
# Transformation Pipeline
# --------------------------------------------------------------------------
def get_inference_transform(target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> T.Compose:
    """Build deterministic standard torchvision transformation pipeline."""
    return T.Compose([
        T.Resize(target_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# --------------------------------------------------------------------------
# Image Decoding & Validation
# --------------------------------------------------------------------------
def decode_image(
    image_input: Union[bytes, str, Path, Image.Image]
) -> Tuple[Image.Image, ImageMetadata]:
    """
    Decodes diverse input types (raw bytes, base64 data URI, filesystem path,
    or PIL Image) into a normalized RGB PIL Image alongside its structural metadata.

    Raises:
        EmptyImageError: If the payload is empty.
        CorruptedImageError: If bytes fail verification.
        UnsupportedFormatError: If the image format is unsupported.
    """
    if image_input is None:
        raise EmptyImageError("Image input cannot be None.")

    raw_format = "PNG"

    if isinstance(image_input, Image.Image):
        pil_img = image_input
        raw_format = pil_img.format or "PNG"
    elif isinstance(image_input, (str, Path)):
        path_or_str = str(image_input).strip()
        if not path_or_str:
            raise EmptyImageError("Received empty image path or string.")

        # Check for Base64 Data URI or raw base64 string
        if path_or_str.startswith("data:image/") or ";base64," in path_or_str:
            try:
                base64_data = path_or_str.split(";base64,")[-1]
                image_bytes = base64.b64decode(base64_data)
                return decode_image(image_bytes)
            except Exception as e:
                raise CorruptedImageError(f"Failed to decode base64 data URI: {e}") from e
        elif os.path.exists(path_or_str):
            if os.path.getsize(path_or_str) == 0:
                raise EmptyImageError(f"Image file is empty (0 bytes): {path_or_str}")
            try:
                with open(path_or_str, "rb") as f:
                    image_bytes = f.read()
                return decode_image(image_bytes)
            except ImageProcessingError:
                raise
            except Exception as e:
                raise CorruptedImageError(f"Failed to read image file from disk: {e}") from e
        else:
            # Attempt plain base64 decode if string is not a valid file path
            try:
                image_bytes = base64.b64decode(path_or_str, validate=True)
                return decode_image(image_bytes)
            except Exception as e:
                raise CorruptedImageError(f"Input is neither an existing file path nor valid base64: {e}") from e
    elif isinstance(image_input, bytes):
        if len(image_input) == 0:
            raise EmptyImageError("Image payload contains 0 bytes.")
        try:
            stream = io.BytesIO(image_input)
            pil_img = Image.open(stream)
            pil_img.load()  # Force decode to catch truncated/corrupted files
            raw_format = pil_img.format or "PNG"
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            raise CorruptedImageError(f"Corrupted or unrecognizable image payload: {e}") from e
    else:
        raise ImageProcessingError(f"Unsupported image input type: {type(image_input)}")

    orig_width, orig_height = pil_img.size
    orig_mode = pil_img.mode
    channels = len(orig_mode) if orig_mode in ("RGB", "RGBA", "CMYK") else (1 if orig_mode in ("L", "1") else 3)

    if raw_format.upper() not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Image format '{raw_format}' is not supported. Supported: {SUPPORTED_FORMATS}"
        )

    # Standardize to RGB color space
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    metadata = ImageMetadata(
        width=orig_width,
        height=orig_height,
        channels=channels,
        format=raw_format.upper(),
    )

    return pil_img, metadata


def preprocess_image_tensor(
    pil_image: Image.Image,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    unsqueeze: bool = True,
) -> torch.Tensor:
    """
    Transforms a standardized PIL Image into a normalized PyTorch FloatTensor.

    Args:
        pil_image: 3-channel RGB PIL Image.
        target_size: (H, W) spatial dimensions for network input.
        unsqueeze: When True, prepends batch dimension (1, 3, H, W).
    """
    transform = get_inference_transform(target_size)
    tensor = transform(pil_image)
    if unsqueeze:
        tensor = tensor.unsqueeze(0)
    return tensor
