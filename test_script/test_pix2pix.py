from custom_ipix2pix_pipe import CustomInstructPix2Pix
from pathlib import Path
from PIL import Image

import numpy as np
import common_image_preparation as cip
import argparse
import datetime
import os
import torch
from torchvision.transforms import v2


def main(args):
    # Your code here

    device = args.device
    text_prompt = args.text_prompt
    guidance_scale = args.guidance_scale
    image_guidance_scale = args.image_guidance_scale
    output_path = Path(args.output_path)
    root_dir = Path(args.root_dir)
    print(device)
    image_root_path = root_dir
    torch.cuda.device(0)

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

    output_path = output_path / run_id
    os.makedirs(output_path, exist_ok=True)

    images = []

    for image_path in image_root_path.glob("*.[jpJP][npNP]*[gG$]"):
        print(image_path)
        i = Image.open(image_path)
        image_tensor = cip.img2tensor(image=i, device=device)

        images.append((image_path.name, image_tensor))

    ip2p = CustomInstructPix2Pix(device=device, ip2p_use_full_precision=False)

    text_embeddings = ip2p.pipe._encode_prompt(
        text_prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt="",
    )

    for data in images:
        torch.cuda.empty_cache()
        img = data[1].to(text_embeddings.dtype).to(device)
        name = data[0]

        edited_image = ip2p.edit_image(
            text_embeddings=text_embeddings,
            image=img,
            image_cond=img,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
        )
        edited_image = v2.ToPILImage()(edited_image.cpu().detach().squeeze(0))
        edited_image.save(output_path / name, quality=100)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Prepare stylized targets by using InstructPix2Pix"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/lucky/Desktop/render/stone/epoch_0/",
        help="Path to the inputs' root directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/lucky/Desktop/test_pix2pix/",
        help="Path to save the results",
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        default="Turn him into a stone statue",
        help="Text prompt to guide the editing",
    )

    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    parser.add_argument(
        "--guidance_scale", type=float, default=12, help="Weight of the text prompt"
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=0.5,
        help="Weight of the reference image",
    )

    args = parser.parse_args()

    main(args)
