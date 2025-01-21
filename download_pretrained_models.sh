#!/bin/bash

# Create output directory
output_dir="output"
mkdir -p "$output_dir"

# Function to download a file from Google Drive
download_model() {
    local url="$1"
    local output_path="$2"
    
    echo "Downloading $(basename "$output_path")..."
    if gdown --fuzzy "$url" -O "$output_path"; then
        echo "Successfully downloaded model to $output_path"
    else
        echo "Error downloading model"
        exit 1
    fi
}

# Download models
echo -e "\n Starting Download for reallusion_v7.pth..."
download_model "https://drive.google.com/file/d/1zgES1LHt1izvGkVsSmw7d8urJtbI_pRI/view?usp=sharing" "$output_dir/reallusion_v7.pth"

echo -e "\n Starting Download for smpl_humanoid_v7.pth..."
download_model "https://drive.google.com/file/d/1lnCQ9cBtrGwwnA2QYNLgqVs2n-4_oZtn/view?usp=sharing" "$output_dir/smpl_humanoid_v7.pth"