import argparse
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T


def image_center_crop_torch(image_tensor: torch.Tensor, target_width: int, target_height: int) -> torch.Tensor:
    """Center crops the image tensor using PyTorch."""
    _, _, h, w = image_tensor.shape

    # Calculate the coordinates for center cropping
    top = (h - target_height) // 2
    left = (w - target_width) // 2
    bottom = top + target_height
    right = left + target_width

    # Crop the image
    cropped_image_tensor = image_tensor[:, :, top:bottom, left:right]
    return cropped_image_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Prepare stylized targets by using InstructPix2Pix")
    parser.add_argument(
        "--input_path",
        type=str,
        default="/data_storage/Lucky_Thesis_Data/dataset/fangzhou-small/",
        help="Path to the images root directory",
    )

    parser.add_argument("--resized", type=bool, default=False, help="Minimum size for center crop")

    parser.add_argument("--min_size", type=int, default=512, help="Minimum size for center crop")

    parser.add_argument(
        "--output_path",
        const="arg_was_not_given",
        nargs="?",
        help="Path to the output directory",
    )

    args = parser.parse_args()

    images_path = Path(args.input_path)
    output_path = args.output_path

    if output_path is None:
        output_path = images_path
    else:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    print(f"Output path: {output_path}")

    # Define PyTorch transforms
    transform = T.ToTensor()  # Convert PIL image to PyTorch tensor
    to_pil = T.ToPILImage()  # Convert PyTorch tensor back to PIL image
    resized_height = 3060
    resized_width = 2048
    if args.resized:
        resized_height = 512
        resized_width = int(2048 / 3060 * resized_height)

    # Iterate through all images. Matches JPG, jpg, PNG, png
    for path in images_path.glob("*.[jJpP][pPnN][gG]"):
        image = Image.open(path).convert("RGBA")  # Open image and ensure it's in RGB format
        image_tensor = transform(image).unsqueeze(0).to("cuda")  # Convert to tensor and move to GPU

        # Perform center crop on GPU
        image_tensor_cropped = image_center_crop_torch(image_tensor, resized_width, resized_height)

        # Convert tensor back to PIL image and save
        cropped_image_pil = to_pil(image_tensor_cropped.squeeze(0).cpu())
        cropped_image_pil.save(output_path / path.name.replace("*.[jJpP][pPnN][gG]", ".png"), quality=100)

        if not args.resized:
            continue

    print("Done!")

    # create empty image
    # image = Image.new("RGB", (2048, 3060), (255, 255, 255))
    # image.save("background.jpg")
