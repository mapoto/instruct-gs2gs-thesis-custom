#!/bin/bash

# Define an array of prompts
prompts=("Turn it into an anime" "Make it look like Fauvism painting" "Turn into 3d model" "make it look like Edward Munch Painting" "as if it were by modigliani" "give a scar on his left cheek" "make it look like a painting" "draw him in black and white" "make it look like a pencil sketch" "make it look like a watercolor" )

# Iterate over each prompt and execute the command
for prompt in "${prompts[@]}"; do
    # Start the Python script in the background
    /home/lucky/miniconda3/envs/igs2gs_custom/bin/python /home/lucky/miniconda3/envs/igs2gs_custom/lib/python3.8/site-packages/nerfstudio/scripts/train.py \
        igs2gs \
        --vis wandb \
        --pipeline.model.cull_alpha_thresh 0.005 \
        --pipeline.model.output-depth-during-training True \
        --pipeline.model.use_scale_regularization True \
        --data ./data/Simon_grn/ \
        --load-dir ./outputs/Dora_grn/splatfacto/2024-10-11_161635/nerfstudio_models/ \
        --pipeline.prompt "$prompt" \
        --pipeline.guidance-scale 5 \
        --pipeline.image-guidance-scale 2 \
        --pipeline.dataset-name Simon \
        nerfstudio-data \
        --eval_mode all \
        --downscale-factor 4 \
        --load-3D-points True &


    # Wait for 60 minutes
    sleep 7200

    # Terminate the previous execution
    kill $PID
done