"""
Nerfstudio InstructGS2GS Pipeline
"""

import csv
import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
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

from torchmetrics.image.fid import FrechetInceptionDistance

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

# from igs2gs.matching.depth_matching import gs_matching


IMG_OUT_PATH = Path("/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/igs2gs/")


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
    guidance_scale: float = 5
    """(text) guidance scale for InstructPix2Pix"""
    image_guidance_scale: float = 2
    """image guidance scale for InstructPix2Pix"""
    incremental_guidance_scale: float = 0.5
    """incremental guidance scale for InstructPix2Pix"""
    incremental_image_guidance_scale: float = 0.2
    """incremental image guidance scale for InstructPix2Pix"""
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
    seed: int = 42
    """Random seed for reproducibility"""
    apply_original_masking: bool = True
    """Whether to apply the original mask to the edited image"""
    multiview_loss_num_samples: int = 20
    """Number of samples to use for multiview loss"""
    dataset_name: Literal["Simon", "Dora", "Ephra", "Irene"] = "Simon"
    """Name of the dataset"""
    batch_size: int = 1
    """Batch size for the pipeline"""


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
        self.apply_original_masking = config.apply_original_masking

        dir_name = "_".join(
            [
                datetime.datetime.now().strftime("%m-%d-%H-%M"),
                str(self.config.dataset_name),
                str(self.config.prompt).replace(" ", "-"),
                str(self.config.seed),
                str(self.config.guidance_scale),
                str(self.config.incremental_guidance_scale),
                str(self.config.image_guidance_scale),
                str(self.config.incremental_image_guidance_scale),
            ]
        )
        self.img_outpath = IMG_OUT_PATH / Path(dir_name)

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

        CAM_ADJ_MATRICES = {
            "Simon": torch.tensor(pd.read_csv(Path("./igs2gs/adj_matrices/simon.csv"), header=0).values),
            "Dora": torch.tensor(pd.read_csv(Path("./igs2gs/adj_matrices/dora.csv"), header=0).values),
            "Ephra": torch.tensor(pd.read_csv(Path("./igs2gs/adj_matrices/ephra.csv"), header=0).values),
            "Irene": torch.tensor(pd.read_csv(Path("./igs2gs/adj_matrices/irene.csv"), header=0).values),
        }

        self.camera_adj_matrix = CAM_ADJ_MATRICES[self.config.dataset_name]
        print(f"Camera adjacency matrix: {self.camera_adj_matrix}")

        # which image index we are editing
        # self.curr_edit_num = 0
        # whether we are doing regular GS updates or editing images
        self.makeSquentialEdits = False
        self.image_guidance_scale = self.config.image_guidance_scale
        self.guidance_scale = self.config.guidance_scale
        self.seed = self.config.seed

        self.set_seed(self.seed)

    def set_seed(self, seed: int = 42) -> None:
        self.seed = seed
        # np.random.seed(seed)
        # random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}")

    def get_camera_batches(self, step: int):
        """Get camera batches based on adjacency matrix"""
        batch = []
        for i, row in enumerate(self.camera_adj_matrix[step]):
            if row == 1:
                batch.append(i)
        print(f"Camera batch: {batch}")
        return batch

    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        training_camera_length = len(self.datamanager.original_cached_train)
        print(f"Training camera length: {str(training_camera_length)}")

        if step == 30000:
            with open(self.img_outpath / "losses.csv", "w") as f:
                f.write("step,main_loss,scale_reg,psnr,clip,fid,gaussian_count\n")

        self.makeSquentialEdits = ((step - 1) % self.config.gs_steps) == 0

        if self.makeSquentialEdits:
            curr_edit_num = 0
            print(
                f"Editing start with guidance scale {self.guidance_scale} and image guidance scale {self.image_guidance_scale}"
            )

            for camera_idx in range(0, len(self.datamanager.original_cached_train)):
                print(f"Editing image {camera_idx}")
                camera, data = self.datamanager.next_train_idx(camera_idx)
                model_outputs = self.model(camera)

                original_image = (
                    self.datamanager.original_cached_train[camera_idx]["image"]
                    .detach()
                    .unsqueeze(dim=0)
                    .permute(0, 3, 1, 2)
                )
                conditioning_image = original_image[:, :3, :, :]
                rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
                depth_image = model_outputs["depth"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

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

                # apply original mask to the alpha channel of the edited image
                alpha_channel = original_image[:, 3, :, :].unsqueeze(1).to(edited_image.device)
                edited_image = torch.cat([edited_image, alpha_channel], dim=1)

                # store camera as json for debug
                self.store_images(
                    step=step,
                    camera_index=camera_idx,
                    rendered=rendered_image,
                    original=original_image,
                    edited=edited_image,
                    depth=depth_image,
                )
                self.store_camera_json(camera, step, camera_idx)

                self.update_dataset(data, camera_idx, edited_image, original_image.dtype)

                # increment curr edit idx
                curr_edit_num += 1

            print(f"Editing finished with {curr_edit_num} images")

            if curr_edit_num >= len(self.datamanager.cached_train):
                self.guidance_scale += self.config.incremental_guidance_scale
                self.image_guidance_scale += self.config.incremental_image_guidance_scale

        # compute losses according to camera batches
        (model_outputs, loss_dict, metrics_dict, rgb_stack) = self.compute_regional_losses(step)

        main_loss = loss_dict["main_loss"].detach().cpu().item()
        scale_reg = loss_dict["scale_reg"].detach().cpu().item()
        metric = metrics_dict["psnr"].detach().cpu().item()
        gaussian = metrics_dict["gaussian_count"]

        clip_average = 0
        fid = 0

        with open(self.img_outpath / "losses.csv", "a+") as f:
            f.write(f"{step},{main_loss},{scale_reg},{metric},{clip_average},{fid},{gaussian}\n")

        if step % 10 == 0 or self.makeSquentialEdits:
            Path(self.img_outpath / str(step)).mkdir(parents=True, exist_ok=True)

            # Store images for debugging
            for camera_idx in range(0, len(self.datamanager.original_cached_train)):
                camera, _ = self.datamanager.next_train_idx(camera_idx)
                model_outputs = self.model(camera)

                rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
                depth_image = model_outputs["depth"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

                self.store_images(
                    step=step,
                    camera_index=camera_idx,
                    rendered=rendered_image,
                    original=None,
                    edited=None,
                    depth=depth_image,
                )
                self.store_camera_json(camera, step, camera_idx)

        return model_outputs, loss_dict, metrics_dict

    def update_dataset(self, data, camera_index: int, edited_image: torch.Tensor, dtype: torch.dtype = torch.float32):
        """Update dataset ith edited image as H x W x C tensor"""
        edited_image = edited_image.squeeze().permute(1, 2, 0).to(dtype)
        self.datamanager.update_image_data(camera_index, edited_image)

    def store_camera_json(self, camera, step, camera_idx):
        camera_json = camera.to_json(0)
        camera_json["camera_index"] = camera_idx

        save_path = Path(self.img_outpath / str(step))
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / f"camera_{str(camera_idx)}.json", "w", encoding="UTF-8") as file:
            json.dump(camera_json, file, indent=4)

    def compute_regional_losses(self, step: int):

        camera, _ = self.datamanager.next_train(step)
        anchor_idx = camera.metadata["cam_idx"]

        # compute new losses
        main_loss_list = []
        scale_reg_list = []
        metric_list = []

        rgb_stack = []
        acc_stack = []
        background_stack = []
        depth_stack = []

        camera_batch = self.get_camera_batches(anchor_idx)
        message = ",".join([str(i) for i in camera_batch])
        print(f"Computing Loss for Camera batch: {message}")

        for camera_idx in camera_batch:

            camera, data = self.datamanager.next_train_idx(camera_idx)
            # self.store_camera_json(camera, step, camera_idx)

            model_outputs = self.model(camera)
            metrics_dict = self.model.get_metrics_dict(model_outputs, data)

            rgb_stack.append(model_outputs["rgb"])
            acc_stack.append(model_outputs["accumulation"])
            background_stack.append(model_outputs["background"])
            depth_stack.append(model_outputs["depth"])
            loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

            main_loss = loss_dict["main_loss"]
            scale_reg = loss_dict["scale_reg"]
            metric = metrics_dict["psnr"]

            main_loss_list.append(main_loss)
            scale_reg_list.append(scale_reg)
            metric_list.append(metric)

            # self.store_images(
            #     step=step,
            #     camera_index=camera_idx,
            #     rendered=model_outputs["rgb"],
            #     original=self.datamanager.original_cached_train[camera_idx]["image"],
            #     edited=None,
            #     depth=model_outputs["depth"],
            # )

        model_outputs = {
            "rgb": torch.stack(rgb_stack, dim=0),
            "accumulation": torch.stack(acc_stack, dim=0),
            "background": torch.stack(background_stack, dim=0),
            "depth": torch.stack(depth_stack, dim=0),
        }

        average_main_loss = sum(main_loss_list) / len(main_loss_list)

        average_scale_reg = sum(scale_reg_list) / len(scale_reg_list)

        average_metric = sum(metric_list) / len(metric_list)

        loss_dict = {
            "main_loss": average_main_loss,
            "scale_reg": average_scale_reg,
        }

        metrics_dict["psnr"] = average_metric

        return (model_outputs, loss_dict, metrics_dict, rgb_stack)

    def store_images(
        self,
        step: int,
        camera_index: int,
        rendered: Optional[torch.Tensor] = None,
        original: Optional[torch.Tensor] = None,
        edited: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        height: int = 765,
        width: int = 512,
    ):

        save_path = Path(self.img_outpath / str(step))
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving images for camera index {camera_index}")

        with torch.no_grad():
            if rendered is not None:
                # save rendered image
                rendered_image = self.to_image(
                    rendered.squeeze(0).permute(1, 2, 0) if len(rendered.shape) == 4 else rendered
                )
                rendered_image.save(save_path / f"{str(camera_index)}_render.png")
                print("Rendered image saved")

            if original is not None:
                # save original image
                original_image = self.to_image(
                    original.squeeze(0).permute(1, 2, 0) if len(original.shape) == 4 else original
                )
                original_image.save(save_path / f"{str(camera_index)}_original.png")
                print("Original image saved")

            if edited is not None:
                # save edited image
                edited = self.to_image(edited.squeeze(0).permute(1, 2, 0) if len(edited.shape) == 4 else edited)
                edited.save(save_path / f"{str(camera_index)}_edited.png")
                print("Edited image saved")

            if depth is not None:
                # save depth image
                depth = depth.squeeze(0).permute(1, 2, 0) if len(depth.shape) == 4 else depth
                torch.save(depth.reshape(height, width), save_path / f"{str(camera_index)}_depth.pt")

                depth_image = self.to_image(depth)
                depth_image.save(save_path / f"{str(camera_index)}_depth.png")
                print("Depth image saved")

    def to_image(self, tensor: torch.Tensor) -> Image:
        """Convert a tensor to an image"""

        array = (tensor.cpu().numpy() * 255).astype("uint8")
        if array.shape[-1] == 1:  # handle grayscale images
            array = array.squeeze(-1)
        return Image.fromarray(array)

    def to_tensor(self, image: Image) -> torch.Tensor:
        """Convert an image to a tensor"""
        return transforms.PILToTensor()(image)

    def reedits_voting(self, reedit_candidates):
        """Re-edit images based on voting"""
        votes = {}
        for (idx1, idx2), _ in reedit_candidates:
            if idx1 not in votes:
                votes[idx1] = 0
            if idx2 not in votes:
                votes[idx2] = 0
            votes[idx1] += 1
            votes[idx2] += 1

        reedits_idx = [int(idx) for idx, vote in votes.items() if vote > 1]
        return reedits_idx

    def compute_losses(self, num_samples: int = 20):

        # compute new losses
        main_loss_list = []
        scale_reg_list = []
        metric_list = []

        rgb_stack = []
        acc_stack = []
        background_stack = []
        depth_stack = []

        random_indices = random.sample(range(len(self.datamanager.original_cached_train)), num_samples)

        print(f"random loss indices{random_indices}")

        for idx in random_indices:

            camera, data = self.datamanager.next_train_idx(idx)
            model_outputs = self.model(camera)
            metrics_dict = self.model.get_metrics_dict(model_outputs, data)

            rgb_stack.append(model_outputs["rgb"])
            acc_stack.append(model_outputs["accumulation"])
            background_stack.append(model_outputs["background"])
            depth_stack.append(model_outputs["depth"])
            loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

            main_loss = loss_dict["main_loss"]
            scale_reg = loss_dict["scale_reg"]
            metric = metrics_dict["psnr"]

            main_loss_list.append(main_loss)
            scale_reg_list.append(scale_reg)
            metric_list.append(metric)

        model_outputs = {
            "rgb": torch.stack(rgb_stack, dim=0),
            "accumulation": torch.stack(acc_stack, dim=0),
            "background": torch.stack(background_stack, dim=0),
            "depth": torch.stack(depth_stack, dim=0),
        }

        average_main_loss = sum(main_loss_list) / len(main_loss_list)

        average_scale_reg = sum(scale_reg_list) / len(scale_reg_list)

        average_metric = sum(metric_list) / len(metric_list)

        loss_dict = {
            "main_loss": average_main_loss,
            "scale_reg": average_scale_reg,
        }

        metrics_dict["psnr"] = average_metric

        return (model_outputs, loss_dict, metrics_dict, rgb_stack)

    def do_csmv(self, rendered, model_name="ViT-L/14"):
        """Do similarity check for InstructPix2Pix"""
        # images = [rgb.detach().unsqueeze(dim=0).permute(0, 3, 1, 2) for rgb in rendered]
        images = rendered

        clip_model = cm.ClipSimilarity(model_name).to(self.ip2p_device)
        image_features = cm.process_images_in_batches(clip_model, images, 4)

        similarity_matrix = clip_model.compute_all_similarities(image_features)

        average = torch.mean(similarity_matrix)

        return average.detach().cpu().item(), similarity_matrix

    def do_fid_batch(self, rendered, original):
        """Do FID calculation for InstructPix2Pix"""
        new_dataset = torch.cat([rgb.detach().unsqueeze(dim=0).permute(0, 3, 1, 2) for rgb in rendered])
        new_dataset = (new_dataset * 255).byte().to(self.ip2p_device)
        original_dataset = torch.cat(
            [rgb.detach().unsqueeze(dim=0).permute(0, 3, 1, 2)[:, :3, :, :] for rgb in original]
        )
        original_dataset = (original_dataset * 255).byte().to(self.ip2p_device)

        fid = FrechetInceptionDistance(feature=64).to(self.ip2p_device)

        fid.update(new_dataset, real=False)
        fid.update(original_dataset, real=True)
        score = fid.compute()

        return score.item()

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
