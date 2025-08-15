import os
import shutil
import random
import argparse

def generate_train_val_test_split(base_folder, output_folder, train_ratio=0.7, val_ratio=0.15):
    """
    Splits data into train, validation, and test sets, maintaining the class subfolder structure.
    The test ratio is inferred from the train and validation ratios (1 - train - val).

    Args:
        base_folder (str): The path to the base folder containing class subfolders (e.g., 'correct', 'incorrect').
        output_folder (str): The path to the output folder where the split data ('train', 'val', 'test') will be stored.
        train_ratio (float): The ratio of the data to use for training (default: 0.7).
        val_ratio (float): The ratio of the data to use for validation (default: 0.15).
    """
    # Ensure ratios are valid
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("The sum of train_ratio and val_ratio must be less than 1.0")

    # Clear the output folder if it exists, then recreate it and its main subdirectories
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    test_folder = os.path.join(output_folder, 'test')
    
    os.makedirs(train_folder)
    os.makedirs(val_folder)
    os.makedirs(test_folder)

    # Find class subdirectories (e.g., 'correct', 'incorrect')
    categories = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d)) and d != os.path.basename(output_folder)]

    for category in categories:
        category_path = os.path.join(base_folder, category)
        
        # Collect all file paths in the category
        files = [os.path.join(category_path, f) for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]

        # Shuffle and split the files
        random.shuffle(files)
        train_end_idx = int(len(files) * train_ratio)
        val_end_idx = train_end_idx + int(len(files) * val_ratio)

        train_files = files[:train_end_idx]
        val_files = files[train_end_idx:val_end_idx]
        test_files = files[val_end_idx:]

        # Create corresponding output folders for the category
        train_output_path = os.path.join(train_folder, category)
        val_output_path = os.path.join(val_folder, category)
        test_output_path = os.path.join(test_folder, category)
        
        os.makedirs(train_output_path, exist_ok=True)
        os.makedirs(val_output_path, exist_ok=True)
        os.makedirs(test_output_path, exist_ok=True)

        # Copy files to their new destination
        for file in train_files:
            shutil.copy(file, train_output_path)
        for file in val_files:
            shutil.copy(file, val_output_path)
        for file in test_files:
            shutil.copy(file, test_output_path)

        print(f"Processed '{category}': {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split image data into train, validation, and test sets and clean up original folders.")
    parser.add_argument('--label_type', required=True, help="The name of the parent folder containing the class subdirectories (e.g., 'correct', 'incorrect').")
    args = parser.parse_args()

    label_type = args.label_type
    base_folder = f"./{label_type}"
    output_folder = os.path.join(base_folder, "ds")

    if not os.path.isdir(base_folder):
        print(f"Error: Base folder '{base_folder}' not found. Please ensure it exists.")
    else:
        # 1. Generate the train/val/test split into the 'ds' subfolder
        print(f"Splitting data from '{base_folder}' into '{output_folder}'...")
        generate_train_val_test_split(base_folder, output_folder, train_ratio=0.7, val_ratio=0.15)
        print("\nSplit complete.")

        # 2. Clean up by deleting the original class folders
        print("Cleaning up original data folders...")
        try:
            shutil.rmtree(os.path.join(base_folder, 'correct'))
            print(f"  - Deleted: {os.path.join(base_folder, 'correct')}")
        except FileNotFoundError:
            print(f"  - Warning: Folder not found, skipping: {os.path.join(base_folder, 'correct')}")
            
        try:
            shutil.rmtree(os.path.join(base_folder, 'incorrect'))
            print(f"  - Deleted: {os.path.join(base_folder, 'incorrect')}")
        except FileNotFoundError:
            print(f"  - Warning: Folder not found, skipping: {os.path.join(base_folder, 'incorrect')}")

        print("\nCleanup finished. The final dataset is in:", output_folder)