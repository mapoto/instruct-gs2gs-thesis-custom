import datetime
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
import torch

from diffusers.utils import load_image
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from controlnet_aux import OpenposeDetector

from transformers import pipeline


def create_canny(image, low_threshold=100, high_threshold=200):
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


def detect_poses(image):
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    openpose_image = openpose(
        image, include_face=True, include_hand=False, include_body=False
    )
    return openpose_image


def estimate_depth(image):
    depth_estimator = pipeline("depth-estimation")
    depth_image = depth_estimator(image)["depth"]
    depth_image = np.array(depth_image)
    depth_image = depth_image[:, :, None]
    depth_image = np.concatenate([depth_image, depth_image, depth_image], axis=2)
    return Image.fromarray(depth_image)


if __name__ == "__main__":
    image_path = "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/dataset/cv_people/20220629_sven/resized_images"
    mask_path = "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/dataset/cv_people/20220629_sven/resized_masks/"

    controlnets = [
        ControlNetModel.from_pretrained(
            "lllyasviel/control_v11e_sd15_ip2p", torch_dtype=torch.float16
        ),
        ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
        ),
        ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
        ),
        ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
        ),
    ]

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnets,
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")

    prompt = "Turn him into a stone statue"

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    output_path = Path(
        "./controlnet_output/" + datetime.datetime.now().strftime("%m%d_%H%M%S")
    )
    output_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    canny_path = output_path / "canny"
    openpose_path = output_path / "openpose"
    depth_path = output_path / "depth"

    canny_path.mkdir(parents=True, exist_ok=True)
    openpose_path.mkdir(parents=True, exist_ok=True)
    depth_path.mkdir(parents=True, exist_ok=True)

    for filename in Path(image_path).rglob("*.JPG"):
        print(filename.as_posix())

        image = load_image(filename.as_posix())

        canny_image = create_canny(image)
        openpose_image = detect_poses(image)
        depth_image = estimate_depth(image)

        canny_image.save(
            Path(
                canny_path
                / str(filename.name.removesuffix(filename.suffix) + "_canny.png")
            )
        )
        openpose_image.save(
            Path(
                openpose_path
                / str(filename.name.removesuffix(filename.suffix) + "_openpose.png")
            )
        )
        depth_image.save(
            Path(
                depth_path
                / str(filename.name.removesuffix(filename.suffix) + "_depth.png")
            )
        )

        conditioner = [image, canny_image, openpose_image, depth_image]
        generator = torch.Generator(device="cpu").manual_seed(33)

        image = pipe(
            prompt,
            image=conditioner,
            num_inference_steps=20,
            generator=generator,
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            controlnet_conditioning_scale=[1, 0.6, 0.7, 0.5],
        ).images[0]

        image.save(
            Path(
                output_path / str(filename.name.removesuffix(filename.suffix) + ".png")
            )
        )

        pass
