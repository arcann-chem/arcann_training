#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Created: 2024/04/16
# Last modified: 2024/04/17
#----------------------------------------------

# Create a directory for storing downloaded files
mkdir -p ./downloads

# Define the output text file for listing downloaded files
output_file="arcann_environment_offline.txt"

# Initialize the output file with a header
echo "@EXPLICIT" > "$output_file"

# Loop through each link in the provided text file
while IFS= read -r url; do 
    # Extract filename from URL
    filename=$(basename "$url")
    
    # Full path for the file to be checked and/or downloaded
    filepath="./downloads/$filename"
    
    # Check if the file already exists
    if [ -f "$filepath" ]; then
        echo "$filepath already exists, skipping download."
        echo "$filepath" >> "$output_file"
    else
        # Download the file if it does not exist
        wget -P ./downloads "$url" --no-check-certificate

        # Check if the download was successful (file exists)
        if [ -f "$filepath" ]; then
            # If the file exists, append its path to the output file
            echo "$filepath" >> "$output_file"
        else
            # Optionally, log an error message if the file did not download successfully
            echo "Failed to download $url" >&2
        fi
    fi
done < arcann_environment.txt