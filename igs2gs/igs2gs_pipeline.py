"""
Nerfstudio InstructGS2GS Pipeline
"""

import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pdb
import typing
from dataclasses import dataclass, field
from itertools import cycle
from typing import Literal, Optional, Type

import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

# eventually add the igs2gs datamanager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig, FullImageDatamanager
from igs2gs.igs2gs import InstructGS2GSModel, InstructGS2GSModelConfig
from igs2gs.igs2gs_datamanager import InstructGS2GSDataManagerConfig

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from igs2gs.ip2p import InstructPix2Pix
from PIL import Image


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
    gs_steps: int = 2500
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
        print("initialize InstructGS2GSPipeline with test_mode " + str(test_mode))
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
            negative_prompt="",
        )

        # which image index we are editing
        self.curr_edit_idx = 0
        # whether we are doing regular GS updates or editing images
        self.makeSquentialEdits = False

    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """

        print("get_train_loss_dict InstructGS2GSPipeline at step", step)

        if ((step - 1) % self.config.gs_steps) == 0:
            self.makeSquentialEdits = True
            print("makeSquentialEdits set to True")

        # image_output_path = Path("/home/lucky/Desktop/ig2g/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        # image_output_path.mkdir(parents=True, exist_ok=True)
        # loss_output_path = Path(image_output_path / str("losses"))
        # loss_output_path.mkdir(parents=True, exist_ok=True)

        if not self.makeSquentialEdits:
            camera, data = self.datamanager.next_train(step)
            model_outputs = self.model(camera)
            metrics_dict = self.model.get_metrics_dict(model_outputs, data)
            print("not makeSquentialEdits setting metrics_dict" + metrics_dict.keys())
        else:

            print("makeSquentialEdits set to True. Starts editing images")
            # get index
            idx = self.curr_edit_idx
            camera, data = self.datamanager.next_train_idx(idx)
            model_outputs = self.model(camera)
            metrics_dict = self.model.get_metrics_dict(model_outputs, data)

            original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
            rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

            # Path(image_output_path / str(step)).mkdir(parents=True, exist_ok=True)

            edited_image = self.ip2p.edit_image(
                self.text_embedding.to(self.ip2p_device),
                rendered_image.to(self.ip2p_device),
                original_image.to(self.ip2p_device),
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
            self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1, 2, 0)
            data["image"] = edited_image.squeeze().permute(1, 2, 0)

            # save current render
            rendered_image = self.to_image(rendered_image.squeeze(0).permute(1, 2, 0))
            # rendered_image.save(Path(image_output_path / str(step) / f"{str(self.curr_edit_idx)}_render.png"))

            # save original image
            original_image = self.to_image(original_image.squeeze(0).permute(1, 2, 0))
            # original_image.save(Path(image_output_path / str(step) / f"{str(self.curr_edit_idx)}_original.png"))

            # save edited image
            result = self.to_image(data["image"])
            # result.save(Path(image_output_path / str(step) / f"{str(self.curr_edit_idx)}_edited.png"))

            # increment curr edit idx
            self.curr_edit_idx += 1
            if self.curr_edit_idx >= len(self.datamanager.cached_train):
                self.curr_edit_idx = 0
                self.makeSquentialEdits = False

        loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def to_image(self, tensor: torch.Tensor) -> Image:
        """Convert a tensor to an image"""
        return Image.fromarray((tensor.cpu().numpy() * 255).astype("uint8"))

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
