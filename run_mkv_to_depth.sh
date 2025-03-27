#!/bin/bash

# Define the base directory
BASE_DIR="../../data/raw_human_demos/simple_tasks/lifting"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate wilor

# Iterate over all directories inside the lifting folder
for dir in "$BASE_DIR"/*/; do
    # Extract the folder name
    folder_name=$(basename "$dir")
    
    # Construct the full data path
    DATA_ROOT_PATH="$BASE_DIR/$folder_name/"
    VIDEO_PATH="$DATA_ROOT_PATH/video.mkv"
    
    # Check if video file exists
    if [ -f "$VIDEO_PATH" ]; then
        echo "Processing depth data for: $folder_name"
        
        # Run the depth extraction script
        python mkv_to_depth.py --mkv_path "$VIDEO_PATH"
    else
        echo "Warning: No video.mkv found in $DATA_ROOT_PATH"
    fi
done 