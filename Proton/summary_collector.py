#!/usr/bin/env python3

import os
import glob

# Define the path to search
search_path = "/home/jarvis/photon_results"
# Define the output file
output_file = "summary_collection.txt"

# Find all summary.txt files
summary_files = glob.glob(os.path.join(search_path, "**", "summary.txt"), recursive=True)

# Open the output file for writing
with open(output_file, "w") as outfile:
    for file_path in summary_files:
        outfile.write(f"## Contents from: {file_path}\n")
        
        # Read and write the contents of each summary file
        try:
            with open(file_path, "r") as infile:
                outfile.write(infile.read())
        except UnicodeDecodeError:
            outfile.write("Error: Could not read file (possibly binary data)\n")
        
        outfile.write("\n\n")

print(f"Summary collection has been created at {output_file}")
