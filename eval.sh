#!/bin/bash

# Directory containing the folders
BASE_DIR="/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/igs2gs"

# Iterate over each folder in the base directory, sorted by creation date (oldest first)
for folder in $(ls -tr "$BASE_DIR"); do
    folder_path="$BASE_DIR/$folder"
    if [ -d "$folder_path" ]; then
        echo "Processing folder: $folder_path"
        python3 ./igs2gs/igs2gs_metrics/mv_metrics.py "$folder_path"
    fi
done