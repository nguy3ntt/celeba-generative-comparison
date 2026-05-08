from PIL import Image, ImageOps


def resize_and_center_crop(image: Image.Image, image_size: int = 64) -> Image.Image:
    """
    Resize image to a square image using center crop behaviour.

    CelebA aligned images are usually portrait-style face images.
    This function creates a consistent square 64x64 image.
    """
    image = image.convert("RGB")

    return ImageOps.fit(
        image,
        (image_size, image_size),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )