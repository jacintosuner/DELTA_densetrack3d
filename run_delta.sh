#!/bin/bash\

# Important note: make sure to use run_mkv_to_depth.sh to generate depths.npy if using mkv files and don't have installed k4a-tools and pyk4a

# Define the base directory
BASE_DIR="../data/lifting"



source ~/miniconda3/etc/profile.d/conda.sh
conda activate densetrack3d

# Iterate over all directories inside the lifting folder
for dir in "$BASE_DIR"/*/; do
    # Extract the folder name
    folder_name=$(basename "$dir")
    
    # Construct the full data path
    DATA_ROOT_PATH="$BASE_DIR/$folder_name/"
    OUTPUT_PATH="$BASE_DIR/$folder_name/"

    # Run the command
    python delta_wrapper.py --ckpt checkpoints/densetrack3d.pth --data_root_path "$DATA_ROOT_PATH" --output_path "$OUTPUT_PATH"
done
