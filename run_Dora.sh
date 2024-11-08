#!/bin/bash

# Define an array of prompts
prompts=("turn her into a rabbit" "Make her James Bond" "Make it look like Van Gogh painting" "Give her red hair and blue shirt" "turn her into Tolkien Elf" "turn her into a baby")


# Iterate over each prompt and execute the command
for prompt in "${prompts[@]}_1"; do
    # Start the Python script in the background
    /home/lucky/miniconda3/envs/igs2gs_custom/bin/python ../nerfstudio/nerfstudio/scripts/train.py \
        igs2gs \
        --vis wandb \
        --experiment-name "Dora_grn_$prompt" \
        --pipeline.model.cull_alpha_thresh 0.005 \
        --pipeline.model.output-depth-during-training True \
        --pipeline.model.use_scale_regularization True \
        --data /home/lucky/dataset/thesis_nerfstudio_data/coord_corrected/Dora_grn/ \
        --load-dir ./outputs/Dora_grn/splatfacto/2024-10-21_180506/nerfstudio_models/ \
        --pipeline.prompt "$prompt" \
        --pipeline.guidance-scale 7.5 \
        --pipeline.image-guidance-scale 2 \
        --pipeline.dataset-name Dora \
        nerfstudio-data \
        --eval_mode all \
        --downscale-factor 4 \
        --load-3D-points True \
        --center-method none \
        --auto-scale-poses False \
        --orientation-method none &


    # Wait for 60 minutes
    sleep 1800

    # Terminate the previous execution
    kill $PID
done