"""
Nerfstudio InstructGS2GS Pipeline
"""

import csv
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
from torchvision import transforms

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
from igs2gs.fixed_latent_ipix2pix import FixedLatentInstructPix2Pix
from igs2gs.ip2p import InstructPix2Pix
from PIL import Image

from igs2gs.igs2gs_metrics import clip_metrics_batch as cm


IP2P_SOURCE = "timbrooks/instruct-pix2pix"

# LOSS_OUT_PATH = Path(IMG_OUT_PATH / str("losses"))
# LOSS_OUT_PATH.mkdir(parents=True, exist_ok=True)

print("################## Executing script " + __file__ + " ##################")


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
    max_num_iterations: int = 7500
    """Maximum number of iterations to run the pipeline"""


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
        seed: int = 302,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        print("------------ ", __file__, "initialize InstructGS2GSPipeline with test_mode " + str(test_mode))

        self.img_outpath = Path("/home/lucky/Desktop/ig2g/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.img_outpath.mkdir(parents=True, exist_ok=True)

        # select device for InstructPix2Pix
        self.ip2p_device = torch.device(device)
        # self.ip2p = InstructPix2Pix(self.ip2p_device, ip2p_use_full_precision=self.config.ip2p_use_full_precision)

        self.ip2p = FixedLatentInstructPix2Pix.from_pretrained(
            IP2P_SOURCE, torch_dtype=torch.float16, safety_checker=None
        ).to(device)

        # load base text embedding using classifier free guidance
        self.text_embedding = self.ip2p._encode_prompt(
            self.config.prompt,
            device=self.ip2p_device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="",
        )

        self.generator = torch.manual_seed(seed)

        # which image index we are editing
        self.curr_edit_idx = 0
        # whether we are doing regular GS updates or editing images
        self.makeSquentialEdits = False

    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """

        # print("------------ ", __file__, " get_train_loss_dict InstructGS2GSPipeline at step", step)
        if step == 30000:
            with open(self.img_outpath / "losses.csv", "w") as f:
                f.write("step,main_loss,scale_reg,psnr,gaussian_count\n")

        if ((step - 1) % self.config.gs_steps) == 0:
            self.makeSquentialEdits = True
            # print("------------ ", __file__, " makeSquentialEdits set to True")

        if not self.makeSquentialEdits:
            camera, data = self.datamanager.next_train(step)
            model_outputs = self.model(camera)
            metrics_dict = self.model.get_metrics_dict(model_outputs, data)
            # print("------------ ", __file__, " not makeSquentialEdits setting metrics_dict", metrics_dict.keys())
            # print(
            #     "------------ ",
            #     __file__,
            #     " makeSquentialEdits setting metrics_dict psnr",
            #     list(metrics_dict["psnr"].size()),
            # )
            # print(
            #     "------------ ",
            #     __file__,
            #     " makeSquentialEdits setting metrics_dict gaussian_count",
            #     metrics_dict["gaussian_count"],
            # )
            # loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

        else:
            print(
                "------------ ",
                __file__,
                f"------------ makeSquentialEdits set to True at {step}. Starting image editing with guidance scale {self.config.guidance_scale}",
            )

            # do all the editing stuff for each image
            for idx in range(0, len(self.datamanager.original_cached_train)):

                # get index
                # idx = self.curr_edit_idx

                camera, data = self.datamanager.next_train_idx(idx)
                model_outputs = self.model(camera)
                metrics_dict = self.model.get_metrics_dict(model_outputs, data)

                Path(self.img_outpath / str(step)).mkdir(parents=True, exist_ok=True)

                self.to_image(data["image"]).save(
                    Path(self.img_outpath / str(step) / (str(data["image_idx"]) + "_data.png"))
                )

                # print("------------ ", __file__, " makeSquentialEdits setting metrics_dict", metrics_dict.keys())
                # print(
                #     "------------ ",
                #     __file__,
                #     " makeSquentialEdits setting metrics_dict psnr",
                #     list(metrics_dict["psnr"].size()),
                # )
                # print(
                #     "------------ ",
                #     __file__,
                #     " makeSquentialEdits setting metrics_dict gaussian_count",
                #     metrics_dict["gaussian_count"],
                # )

                original_image = (
                    self.datamanager.original_cached_train[idx]["image"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
                )
                rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

                image_cond_latents = self.ip2p.prepare_image_latents(
                    image=original_image,
                    device=self.ip2p_device,
                    num_images_per_prompt=1,
                    batch_size=1,
                    dtype=self.text_embedding.dtype,
                    do_classifier_free_guidance=True,
                )
                edited_image = self.ip2p(
                    # instruction,
                    image=rendered_image.to(self.ip2p_device),
                    batch_size=1,
                    guidance_scale=self.config.guidance_scale,
                    prompt_embeds=self.text_embedding.to(self.ip2p_device),
                    image_cond_latents=image_cond_latents,
                    image_guidance_scale=self.config.image_guidance_scale,
                    num_inference_steps=self.config.diffusion_steps,
                    generator=self.generator,
                ).images[0]

                edited_image = transforms.ToTensor()(edited_image).unsqueeze_(0)

                # resize to original image size (often not necessary)
                if edited_image.size() != rendered_image.size():
                    edited_image = torch.nn.functional.interpolate(
                        edited_image, size=rendered_image.size()[2:], mode="bilinear"
                    )

                # write edited image to dataloader
                edited_image = edited_image.to(original_image.dtype)
                edited_image = edited_image.squeeze().permute(1, 2, 0)
                self.datamanager.cached_train[idx]["image"] = edited_image
                data["image"] = edited_image

                # save current render
                rendered_image = self.to_image(rendered_image.squeeze(0).permute(1, 2, 0))
                rendered_image.save(Path(self.img_outpath / str(step) / f"{str(self.curr_edit_idx)}_render.png"))

                # save original image
                original_image = self.to_image(original_image.squeeze(0).permute(1, 2, 0))
                original_image.save(Path(self.img_outpath / str(step) / f"{str(self.curr_edit_idx)}_original.png"))

                # save edited image
                result = self.to_image(edited_image)
                result.save(Path(self.img_outpath / str(step) / f"{str(self.curr_edit_idx)}_edited.png"))

                # increment curr edit idx
                self.curr_edit_idx += 1

            least_similar_pairs = self.do_similarity_check(
                self.datamanager.cached_train, len(self.datamanager.cached_train)
            )
            csv_filename = Path(self.img_outpath / (str(step) + "_least_similar.csv"))

            store_similarity_matrix(csv_filename, least_similar_pairs)
            if self.curr_edit_idx >= len(self.datamanager.cached_train):
                self.curr_edit_idx = 0
                self.makeSquentialEdits = False
                self.config.guidance_scale += 1

        # if (step + 1) == self.config.max_num_iterations:
        #     last_path = Path(self.img_outpath / str(step))
        #     last_path.mkdir(parents=True, exist_ok=True)

        #     for idx in range(0, len(self.datamanager.original_cached_train)):

        #         camera, data = self.datamanager.next_train_idx(idx)
        #         model_outputs = self.model(camera)
        #         rendered_image = model_outputs["rgb"]
        #         rendered_image = self.to_image(rendered_image)
        #         rendered_image.save(last_path / f"{str(self.curr_edit_idx)}_render.png")

        loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

        main_loss = loss_dict["main_loss"].detach().cpu().item()
        scale_reg = loss_dict["scale_reg"].detach().cpu().item()
        metric = metrics_dict["psnr"].detach().cpu().item()
        gaussian = metrics_dict["gaussian_count"]

        with open(self.img_outpath / "losses.csv", "a+") as f:
            f.write(f"{step},{main_loss},{scale_reg},{metric},{gaussian}\n")

        return model_outputs, loss_dict, metrics_dict

    def aggregate_data(self, data: typing.List[typing.Dict]) -> typing.Dict:
        """Aggregate data"""
        return data[0]

    def aggregate_model_outputs(self, model_outputs: typing.List[torch.Tensor]) -> torch.Tensor:
        """Aggregate model outputs"""
        return torch.stack(model_outputs).mean(dim=0)

    def to_image(self, tensor: torch.Tensor) -> Image:
        """Convert a tensor to an image"""
        return Image.fromarray((tensor.cpu().numpy() * 255).astype("uint8"))

    def do_similarity_check(self, trained, length, model_name="ViT-L/14", top_n=100):
        """Do similarity check for InstructPix2Pix"""
        # print("------------ ", __file__, " do_similarity_check InstructGS2GSPipeline")
        images = [trained[idx]["image"].permute(2, 0, 1).unsqueeze(0) for idx in range(length)]
        # images = torch.cat(images).to(self.ip2p_device)

        clip_model = cm.ClipSimilarity(model_name).to(self.ip2p_device)
        image_features = cm.process_images_in_batches(clip_model, images, 4)

        similarity_matrix = clip_model.compute_all_similarities(image_features)
        idx = [f"{idx}.png" for idx in range(length)]
        least_similar_pairs = cm.find_least_similar(similarity_matrix, idx, top_n=top_n)
        torch.cuda.empty_cache()
        # for (file1, file2), score in least_similar_pairs:
        #     print(f"Images: {file1} and {file2} have a similarity score of {score:.4f}")

        return least_similar_pairs

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError


def store_similarity_matrix(csv_filename, least_similar_pairs):
    """Store the similarity matrix in a CSV file"""

    # Open the file in write mode
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(["image 1", "image 2", "similarity_score"])

        # Write the data rows
        for (file1, file2), score in least_similar_pairs:
            writer.writerow([file1, file2, f"{score:.4f}"])

    print(f"Data has been written to {csv_filename}")
