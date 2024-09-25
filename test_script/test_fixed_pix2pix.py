import os
import sys
import random
from dataclasses import dataclass
from typing import Union
from PIL import Image, ImageOps
import torch
import math
from torch import Tensor, nn
from jaxtyping import Float
from diffusers import (
    DDIMScheduler,
    StableDiffusionInstructPix2PixPipeline,
)
from fixed_latent_ipix2pix import FixedLatentInstructPix2Pix
from pathlib import Path
import argparse
import datetime

IMG_DIM = 512
CONST_SCALE = 0.18215


DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
SD_SOURCE = "runwayml/stable-diffusion-v1-5"
CLIP_SOURCE = "openai/clip-vit-large-patch14"
IP2P_SOURCE = "timbrooks/instruct-pix2pix"


def edit_image(
    pipe: FixedLatentInstructPix2Pix,
    input_image: Image.Image,
    instruction: str,
    steps: int,
    generator: torch.Generator,
    guidance_scale: float,
    image_guidance_scale: float,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = True,
):

    width, height = input_image.size
    factor = 512 / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(
        input_image, (width, height), method=Image.Resampling.LANCZOS
    )

    input_image = pipe.image_processor.preprocess(input_image)

    prompt_embeds = pipe._encode_prompt(
        prompt=instruction,
        device=pipe.device,
        do_classifier_free_guidance=do_classifier_free_guidance,
        num_images_per_prompt=num_images_per_prompt,
    )

    image_cond_latents = pipe.prepare_image_latents(
        image=input_image,
        device=pipe.device,
        num_images_per_prompt=num_images_per_prompt,
        batch_size=1,
        dtype=prompt_embeds.dtype,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    edited_image = pipe(
        # instruction,
        image=input_image,
        batch_size=1,
        guidance_scale=guidance_scale,
        prompt_embeds=prompt_embeds,
        image_cond_latents=image_cond_latents,
        image_guidance_scale=image_guidance_scale,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]
    return [guidance_scale, image_guidance_scale, edited_image]


def main(args):

    device = args.device
    text_prompt = args.text_prompt
    guidance_scale = args.guidance_scale
    image_guidance_scale = args.image_guidance_scale
    seed = args.seed
    diffusion_steps = args.diffusion_steps
    output_path = Path(args.output_path)
    root_dir = Path(args.root_dir)

    image_root_path = root_dir

    run_id = "_".join(
        [
            root_dir.name,
            "ipix2pix",
            text_prompt,
            str(guidance_scale),
            str(image_guidance_scale),
            datetime.datetime.now().strftime("%m-%d-%H:%M"),
        ]
    )

    output_path = output_path / (run_id + "_" + str(seed))
    os.makedirs(output_path, exist_ok=True)

    images = {
        image_path.name.split(".")[0]: Image.open(image_path)
        for image_path in image_root_path.glob("*.[jpJP][npNP]*[gG$]")
    }

    pipe = FixedLatentInstructPix2Pix.from_pretrained(
        IP2P_SOURCE, torch_dtype=torch.float16, safety_checker=None
    ).to(device)

    generator = torch.manual_seed(seed)

    for name, image in images.items():
        text_cfg_scale, image_cfg_scale, edited_image = edit_image(
            pipe=pipe,
            input_image=image,
            instruction=text_prompt,
            steps=diffusion_steps,
            generator=generator,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
        )
        edited_image.save(output_path / f"{name}.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Prepare stylized targets by using InstructPix2Pix"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/dataset/cv_people/20220629_sven/resized_images",
        help="Path to the inputs' root directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/lucky/Desktop/test_fixed_pix2pix/",
        help="Path to save the results",
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        default="Turn him into a stone statue",
        help="Text prompt to guide the editing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=33,
        help="Seed for the random number generator",
    )

    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    parser.add_argument(
        "--guidance_scale", type=float, default=5, help="Weight of the text prompt"
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=2,
        help="Weight of the reference image",
    )

    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=20,
        help="Number of diffusion steps to use for image editing",
    )

    args = parser.parse_args()

    main(args)
