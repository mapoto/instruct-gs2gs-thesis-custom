#!/bin/bash
#!/bin/bash

# Define an array of prompts
prompts=("turn him into a rabbit" "Make him look like a cyborg" "Make him James Bond")
# prompts=("Turn it into an anime" "Make it look like Fauvism painting" "Turn into 3d model" "Give him red hair and blue shirt" "as if it were by modigliani" "make him smile" "make it old" "turn him into Tolkien Elf" "make it look like a pencil sketch" "turn him into a baby" )

# Iterate over each prompt and execute the command
for prompt in "${prompts[@]}"; do
    # Start the Python script in the background
    /home/lucky/miniconda3/envs/igs2gs_custom/bin/python ../nerfstudio/nerfstudio/scripts/train.py \
        igs2gs \
        --vis wandb \
        --experiment-name Ephra_grn_"$prompt" \
        --pipeline.model.cull_alpha_thresh 0.005 \
        --pipeline.model.output-depth-during-training True \
        --pipeline.model.use_scale_regularization True \
        --data /home/lucky/dataset/thesis_nerfstudio_data/coord_corrected/Ephra_grn/ \
        --load-dir ./outputs/Ephra_grn/splatfacto/2024-10-21_182658/nerfstudio_models \
        --pipeline.prompt "$prompt" \
        --pipeline.guidance-scale 7.5 \
        --pipeline.image-guidance-scale 2 \
        --pipeline.dataset-name Ephra \
        nerfstudio-data \
        --eval_mode all \
        --downscale-factor 4 \
        --load-3D-points True \
        --center-method none \
        --auto-scale-poses False \
        --orientation-method none &

    # Store the PID of the last command executed
    PID=$!

    echo "Started processing: Prompt '$prompt'"
    echo "PID: $PID"


    # Wait for 60 minutes
    sleep 2400

    # Terminate the previous execution
    kill $PID

    echo "Terminated processing: Prompt '$prompt'"

done