from transparent_background import Remover
from pathlib import Path
from PIL import Image

if __name__ == "__main__":
    remover = Remover()  # default setting

    image_root_path = Path("/home/lucky/dataset/gauss-dataset/lucky_2/center_crop/images")

    root_output_path = image_root_path.parent / "masked"
    root_output_path.mkdir(exist_ok=True)
    model_choices = ["u2net", "u2net_human_seg", "u2netp"]

    for image_path in image_root_path.glob("*.[jpJP][npNP]*[gG$]"):
        if not ("render" in image_path.stem):
            continue
        input_path = str(image_path)
        output_path = str(root_output_path / (image_path.stem + ".png"))
        img = Image.open(input_path).convert("RGB")  # read image

        with open(input_path, "rb") as i:
            with open(output_path, "wb") as o:
                out = remover.process(img)
                out.save(output_path)  # save result
