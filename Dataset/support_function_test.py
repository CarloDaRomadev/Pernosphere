'''
support_function_test.py

This file contains a simple utility function used to create a usable dataset
from a folder structure where each subfolder represents a different class.
'''

import os
import csv
import shutil
from pathlib import Path

def create_usable_dataset(
    root_dir, 
    output_dir='AI-Lab_project/Archive/all_leaf_test',
    csv_filename='AI-Lab_project/Dataset/dataset_leaf_labeled_test.csv'
):
    """

    Given a directory containing subfolders (one per class), this function:
    1. Copies all images into a single directory (all_leaf_test).
    2. Assigns a numeric label to each subfolder (B-r, ESCA, Healthy, L-b).
    3. Generates a CSV where each file is mapped with the number of its folder.

    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    label_map = {}       
    current_label = 0
    rows = []            

    for subfolder in sorted(root_dir.iterdir()):
        if subfolder.is_dir():
            label_map[subfolder.name] = current_label
            print(f"Processing '{subfolder.name}' with label {current_label}")

            for file in subfolder.iterdir():
                if file.is_file():
                    target_path = output_dir / file.name
                    i = 1

                    while target_path.exists():
                        target_path = output_dir / f"{i}_{file.name}"
                        i += 1

                    shutil.copy(file, target_path)
                    rows.append((target_path.name, current_label))

            current_label += 1

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])
        writer.writerows(rows)



# # # # #
create_usable_dataset('AI-Lab_project/Archive/trainTest/leaf/test')
