#!/bin/bash

# Define an array of prompts
prompts=("Give this person red hair and blue shirt")
# prompts=("Make this person look like Tolkien Elf" "Turn this person into a stone statue" "Give this person red hair and blue shirt" "Give a cowboy hat" "As if a painting in Van Gogh style" "Turn into 3D model" "Have this person smile")

# Define an array of data directories
data_dirs=("./outputs/Dora_grn/splatfacto/2024-10-21_180506/nerfstudio_models/" "./outputs/Irene_grn/splatfacto/2024-10-21_181839/nerfstudio_models/" "./outputs/Ephra_grn/splatfacto/2024-10-21_182658/nerfstudio_models" "./outputs/Simon_grn/splatfacto/2024-10-21_192421/nerfstudio_models/")

# Total number of iterations
total_iterations=$(( ${#prompts[@]} * ${#data_dirs[@]} ))
current_iteration=1

# Iterate over each prompt and data directory
for prompt in "${prompts[@]}"; do
    for data_dir in "${data_dirs[@]}"; do
        # Determine the appropriate --data argument based on the data_dir
        if [[ "$data_dir" == *"Simon_grn"* ]]; then
            data_arg="/home/lucky/dataset/thesis_nerfstudio_data/coord_corrected/Simon_grn/"
            data_name="Simon"
        elif [[ "$data_dir" == *"Irene_grn"* ]]; then
            data_arg="/home/lucky/dataset/thesis_nerfstudio_data/coord_corrected/Irene_grn/"
            data_name="Irene"
        elif [[ "$data_dir" == *"Dora_grn"* ]]; then
            data_arg="/home/lucky/dataset/thesis_nerfstudio_data/coord_corrected/Dora_grn/"
            data_name="Dora"
        else
            data_arg="/home/lucky/dataset/thesis_nerfstudio_data/coord_corrected/Ephra_grn/"
            data_name="Ephra"
        fi

        echo "Running prompt: $prompt with data directory: $data_dir and data argument: $data_arg and data name: $data_name"

        # Start the Python script in the background
        /home/lucky/miniconda3/envs/igs2gs_custom/bin/python ../nerfstudio/nerfstudio/scripts/train.py \
            igs2gs \
            --vis wandb \
            --pipeline.model.cull_alpha_thresh 0.005 \
            --pipeline.model.output-depth-during-training True \
            --pipeline.model.use_scale_regularization True \
            --data "$data_arg" \
            --load-dir "$data_dir" \
            --pipeline.prompt "$prompt" \
            --pipeline.guidance-scale 7.5 \
            --pipeline.image-guidance-scale 2 \
            --pipeline.dataset-name "$data_name" \
            nerfstudio-data \
            --eval_mode all \
            --downscale-factor 4 \
            --load-3D-points True \
            --center-method none \
            --auto-scale-poses False \
            --orientation-method none &

        # Print progress message
        echo "Progress: $current_iteration/$total_iterations - Started processing: Prompt '$prompt' with Data Directory '$data_dir'"

        # Wait for 30 minutes
        sleep 900

        # Terminate the previous execution
        kill $PID

        # Print progress message
        echo "Progress: $current_iteration/$total_iterations - Completed processing: Prompt '$prompt' with Data Directory '$data_dir'"

        # Increment the current iteration counter
        ((current_iteration++))


    done
done