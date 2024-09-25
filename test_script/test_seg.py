from pathlib import Path
from rembg import remove, new_session
from PIL import Image

import argparse
import torch

from transformers import (
    OneFormerProcessor,
    OneFormerForUniversalSegmentation,
    AutoProcessor,
    AutoModelForUniversalSegmentation,
)
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm


def draw_segmentation(name, model, segmentation, segments_info):
    # get the used color map
    viridis = cm.get_cmap("viridis", torch.max(segmentation))
    fig, ax = plt.subplots()
    ax.imshow(segmentation)
    instances_counter = defaultdict(int)
    handles = []
    # for each segment, draw its legend
    for segment in segments_info:
        segment_id = segment["id"]
        segment_label_id = segment["label_id"]
        segment_label = model.config.id2label[segment_label_id]
        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        instances_counter[segment_label_id] += 1
        color = viridis(segment_id)
        handles.append(mpatches.Patch(color=color, label=label))

    ax.legend(handles=handles)
    plt.savefig(name)


def segments(input_path, output_path, suffix=".png"):
    processor = AutoProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")

    model = AutoModelForUniversalSegmentation.from_pretrained(
        "shi-labs/oneformer_coco_swin_large"
    )

    output_path = Path(output_path) / "onetransseg07"
    output_path.mkdir(parents=True, exist_ok=True)

    for file in Path(input_path).glob("*" + suffix):
        input = file.as_posix()

        output = str(output_path / (file.stem + ".png"))

        img = Image.open(input)
        # o1 = remove(img, session=u2net_human_seg)
        # o2 = remove(img, session=isnet)
        panoptic_inputs = processor(
            images=img, task_inputs=["panoptic"], return_tensors="pt"
        )
        for k, v in panoptic_inputs.items():
            print(k, v.shape)

        with torch.no_grad():
            outputs = model(**panoptic_inputs)

        panoptic_segmentation = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[img.size[::-1]]
        )[0]
        print(panoptic_segmentation.keys())

        index = 0
        for segment in panoptic_segmentation["segments_info"]:
            print(segment)

            segment_label_id = segment["label_id"]

            if segment_label_id == 0:
                index = segment["id"]
            segment_label = model.config.id2label[segment_label_id]
            print("Segment label:", segment_label + ":", segment_label_id)

        print(panoptic_segmentation["segmentation"].shape)

        person = panoptic_segmentation["segmentation"] == index
        person = person.type(torch.uint8)

        print("Converted to uint8", person.dtype)
        # print(person)
        out = Image.fromarray(person.numpy() * 255)

        # out = remove(
        #     img,
        #     session=isnet,
        #     alpha_matting=True,
        #     alpha_matting_foreground_threshold=20,
        #     alpha_matting_background_threshold=20,
        #     alpha_matting_erode_size=11,
        # )

        # out = Image.composite(o1, o2, o1)
        # out.save(output, quality=100)
        print("Saved", output)
        draw_segmentation(
            output,
            model,
            panoptic_segmentation["segmentation"],
            panoptic_segmentation["segments_info"],
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="/data_storage/Lucky_Thesis_Data/dataset/20220629_sven/centered_resized/",
        required=False,
    )
    parser.add_argument(
        "--output", type=str, default="/home/lucky/Desktop/test/9/", required=False
    )
    parser.add_argument("--suffix", type=str, default=".JPG", required=False)

    arg = parser.parse_args()

    input_path = arg.input
    output_path = arg.output
    suffix = arg.suffix

    segments(input_path=input_path, output_path=output_path, suffix=suffix)
