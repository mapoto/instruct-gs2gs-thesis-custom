"""
Nerfstudio InstructGS2GS Pipeline
"""

import csv
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import random
import os
import torch
import torch.distributed as dist

from dataclasses import dataclass, field
from itertools import cycle
from typing import Literal, Optional, Type
from PIL import Image

from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

from igs2gs.igs2gs import InstructGS2GSModel, InstructGS2GSModelConfig
from igs2gs.igs2gs_datamanager import InstructGS2GSDataManagerConfig
from igs2gs.igs2gs_metrics import clip_metrics_batch as cm
from igs2gs.ip2p import InstructPix2Pix

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig, FullImageDatamanager

from rembg import remove, new_session


@dataclass
class InstructGS2GSPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: InstructGS2GSPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = InstructGS2GSDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = InstructGS2GSModelConfig()
    """specifies the model config"""
    prompt: str = "don't change the image"
    """prompt for InstructPix2Pix"""
    guidance_scale: float = 12.5
    """(text) guidance scale for InstructPix2Pix"""
    image_guidance_scale: float = 1.5
    """image guidance scale for InstructPix2Pix"""
    incremental_guidance_scale: float = 1.5
    """incremental guidance scale for InstructPix2Pix"""
    incremental_image_guidance_scale: float = 0
    """incremental image guidance scale for InstructPix2Pix"""
    gs_steps: int = 5000
    """how many GS steps between dataset updates"""
    diffusion_steps: int = 20
    """Number of diffusion steps to take for InstructPix2Pix"""
    lower_bound: float = 0.7
    """Lower bound for diffusion timesteps to use for image editing"""
    upper_bound: float = 0.98
    """Upper bound for diffusion timesteps to use for image editing"""
    ip2p_device: Optional[str] = None
    """Second device to place InstructPix2Pix on. If None, will use the same device as the pipeline"""
    ip2p_use_full_precision: bool = False
    """Whether to use full precision for InstructPix2Pix"""
    seed: int = 42
    """Random seed for reproducibility"""


class InstructGS2GSPipeline(VanillaPipeline):
    """InstructGS2GS Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: InstructGS2GSPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

        self.img_outpath = Path(
            "/home/lucky/Desktop/ig2g/"
            + datetime.datetime.now().strftime("%d_%H-%M")
            + "_"
            + str(self.config.prompt).replace(" ", "_")
            + "_"
            + str(self.config.guidance_scale)
            + "_"
            + str(self.config.image_guidance_scale)
        )
        self.img_outpath.mkdir(parents=True, exist_ok=True)

        # select device for InstructPix2Pix
        self.ip2p_device = (
            torch.device(device) if self.config.ip2p_device is None else torch.device(self.config.ip2p_device)
        )

        self.ip2p = InstructPix2Pix(self.ip2p_device, ip2p_use_full_precision=self.config.ip2p_use_full_precision)

        # load base text embedding using classifier free guidance
        self.text_embedding = self.ip2p.pipe._encode_prompt(
            self.config.prompt,
            device=self.ip2p_device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="deformed",
        )

        # which image index we are editing
        self.curr_edit_idx = 0
        # whether we are doing regular GS updates or editing images
        self.makeSquentialEdits = False

        self.set_seed(self.config.seed)

    def set_seed(self, seed: int = 42) -> None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}")

    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if step == 30000:
            with open(self.img_outpath / "losses.csv", "w") as f:
                f.write("step,main_loss,scale_reg,psnr,gaussian_count\n")

        if ((step - 1) % self.config.gs_steps) == 0:
            self.makeSquentialEdits = True

        if not self.makeSquentialEdits:
            camera, data = self.datamanager.next_train(step)
            model_outputs = self.model(camera)
            metrics_dict = self.model.get_metrics_dict(model_outputs, data)

            loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

            main_loss = loss_dict["main_loss"].detach().cpu().item()
            scale_reg = loss_dict["scale_reg"].detach().cpu().item()
            metric = metrics_dict["psnr"].detach().cpu().item()

        else:

            Path(self.img_outpath / str(step)).mkdir(parents=True, exist_ok=True)

            # compute new losses
            main_loss_list = []
            scale_reg_list = []
            metric_list = []

            rgb_stack = []
            acc_stack = []
            background_stack = []

            # session = new_session()

            # do all the editing stuff for each image
            for idx in range(0, len(self.datamanager.original_cached_train)):

                camera, data = self.datamanager.next_train_idx(idx)
                model_outputs = self.model(camera)
                metrics_dict = self.model.get_metrics_dict(model_outputs, data)

                original_image = (
                    self.datamanager.original_cached_train[idx]["image"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
                )
                conditioning_image = original_image[:, :3, :, :]
                rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

                edited_image = self.ip2p.edit_image(
                    self.text_embedding.to(self.ip2p_device),
                    rendered_image.to(self.ip2p_device),
                    conditioning_image.to(self.ip2p_device),
                    guidance_scale=self.config.guidance_scale,
                    image_guidance_scale=self.config.image_guidance_scale,
                    diffusion_steps=self.config.diffusion_steps,
                    lower_bound=self.config.lower_bound,
                    upper_bound=self.config.upper_bound,
                )

                # resize to original image size (often not necessary)
                if edited_image.size() != rendered_image.size():
                    edited_image = torch.nn.functional.interpolate(
                        edited_image, size=rendered_image.size()[2:], mode="bilinear"
                    )

                # write edited image to dataloader

                edited_image = edited_image.to(original_image.dtype)
                edited_image = edited_image.squeeze().permute(1, 2, 0)

                original_mask = (
                    original_image[:, 3, :, :].permute(1, 2, 0).to(device=edited_image.device, dtype=edited_image.dtype)
                )

                # masked = remove(self.to_image(edited_image), session=session)

                # mask = (
                #     self.to_tensor(masked.getchannel("A"))
                #     .permute(1, 2, 0)
                #     .to(device=edited_image.device, dtype=edited_image.dtype)
                # )

                masked_as_tensor = torch.cat([edited_image, original_mask], dim=2)
                self.datamanager.cached_train[idx]["image"] = masked_as_tensor

                data["image"] = masked_as_tensor
                data["mask"] = original_mask

                # save current render
                rendered_image = self.to_image(rendered_image.squeeze(0).permute(1, 2, 0))
                rendered_image.save(Path(self.img_outpath / str(step) / f"{str(self.curr_edit_idx)}_render.png"))

                # save original image
                original_image = self.to_image(original_image.squeeze(0).permute(1, 2, 0))
                original_image.save(Path(self.img_outpath / str(step) / f"{str(self.curr_edit_idx)}_original.png"))

                # save edited image
                result = self.to_image(masked_as_tensor)
                result.save(Path(self.img_outpath / str(step) / f"{str(self.curr_edit_idx)}_edited.png"))

                rgb_stack.append(model_outputs["rgb"])
                acc_stack.append(model_outputs["accumulation"])
                background_stack.append(model_outputs["background"])
                loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

                main_loss = loss_dict["main_loss"]
                scale_reg = loss_dict["scale_reg"]
                metric = metrics_dict["psnr"]

                main_loss_list.append(main_loss)
                scale_reg_list.append(scale_reg)
                metric_list.append(metric)

                gaussian = metrics_dict["gaussian_count"]

                # increment curr edit idx
                self.curr_edit_idx += 1

            # least_similar_pairs = self.do_similarity_check(
            #     self.datamanager.cached_train, len(self.datamanager.cached_train)
            # )
            # csv_filename = Path(self.img_outpath / (str(step) + "_least_similar.csv"))

            # store_similarity_matrix(csv_filename, least_similar_pairs)
            if self.curr_edit_idx >= len(self.datamanager.cached_train):
                self.curr_edit_idx = 0
                self.makeSquentialEdits = False
                self.config.guidance_scale += self.config.incremental_guidance_scale
                self.config.image_guidance_scale += self.config.incremental_image_guidance_scale

            model_outputs = {
                "rgb": torch.stack(rgb_stack, dim=0),
                "accumulation": torch.stack(acc_stack, dim=0),
                "background": torch.stack(background_stack, dim=0),
                "depth": None,
            }

            average_main_loss = sum(main_loss_list) / len(main_loss_list)

            average_scale_reg = sum(scale_reg_list) / len(scale_reg_list)

            average_metric = sum(metric_list) / len(metric_list)

            loss_dict["main_loss"] = average_main_loss
            loss_dict["scale_reg"] = average_scale_reg
            metrics_dict["psnr"] = average_metric

            main_loss = average_main_loss.detach().cpu().item()
            scale_reg = average_scale_reg.detach().cpu().item()
            metric = average_metric.detach().cpu().item()

        gaussian = metrics_dict["gaussian_count"]

        with open(self.img_outpath / "losses.csv", "a+") as f:
            f.write(f"{step},{main_loss},{scale_reg},{metric},{gaussian}\n")

        return model_outputs, loss_dict, metrics_dict

    def to_image(self, tensor: torch.Tensor) -> Image:
        """Convert a tensor to an image"""
        return Image.fromarray((tensor.cpu().numpy() * 255).astype("uint8"))

    def to_tensor(self, image: Image) -> torch.Tensor:
        """Convert an image to a tensor"""
        return transforms.PILToTensor()(image)

    def do_similarity_check(self, trained, length, model_name="ViT-L/14", top_n=100):
        """Do similarity check for InstructPix2Pix"""
        print("------------ ", __file__, " do_similarity_check InstructGS2GSPipeline")
        images = [trained[idx]["image"].permute(2, 0, 1).unsqueeze(0) for idx in range(length)]
        # images = torch.cat(images).to(self.ip2p_device)

        clip_model = cm.ClipSimilarity(model_name).to(self.ip2p_device)
        image_features = cm.process_images_in_batches(clip_model, images, 4)

        similarity_matrix = clip_model.compute_all_similarities(image_features)
        idx = [f"{idx}.png" for idx in range(length)]
        least_similar_pairs = cm.find_least_similar(similarity_matrix, idx, top_n=top_n)
        torch.cuda.empty_cache()
        for (file1, file2), score in least_similar_pairs:
            print(f"Images: {file1} and {file2} have a similarity score of {score:.4f}")

        return least_similar_pairs

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError


def store_similarity_matrix(csv_filename, least_similar_pairs):
    print(least_similar_pairs)

    # Open the file in write mode
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(["image 1", "image 2", "similarity_score"])

        # Write the data rows
        for (file1, file2), score in least_similar_pairs:
            writer.writerow([file1, file2, f"{score:.4f}"])

    print(f"Data has been written to {csv_filename}")
