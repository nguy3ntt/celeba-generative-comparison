from typing import Dict
from diffusers import UNet2DModel


def create_diffusion_unet(config: Dict) -> UNet2DModel:
    """
    Create a small U-Net model for 64x64 DDPM-style image generation
    """
    data_config = config["data"]
    diffusion_config = config["diffusion"]

    model = UNet2DModel(
        sample_size=diffusion_config["image_size"],
        in_channels=data_config["channels"],
        out_channels=data_config["channels"],
        layers_per_block=diffusion_config["layers_per_block"],
        block_out_channels=tuple(diffusion_config["block_out_channels"]),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    return model