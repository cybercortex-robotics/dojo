import os
import pandas as pd

def find_data_descriptor_files(folder_path):
    # List to store paths of found 'q.csv' files
    csv_files = []

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(folder_path):
        # Check if 'q.csv' is in the list of files in the current directory
        if 'data_descriptor.csv' in files:
            # Construct the full path to the file and add it to the list
            csv_files.append(os.path.join(root, 'data_descriptor.csv'))

    if csv_files:
        print(f"Found {len(csv_files)} 'data_descriptor.csv' files.")
    else:
        print("No 'data_descriptor.csv' files found.")

    return csv_files

def remove_first_column(found_files):
    for file_path in found_files:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Compare the header of first column with name 'timestamp_start'
        if df.columns[0] == 'timestamp_start':
            # Remove the first column
            df = df.iloc[:, 1:]
            print(f"Removed the first column from '{file_path}'.")

        # Save the modified DataFrame back to the CSV file
        df.to_csv(file_path, index=False)

if __name__ == '__main__':
    folder_path = r'C:\dev\src\CyberCortex.AI\dojo\data\fake_dataset'
    found_files = find_data_descriptor_files(folder_path)
    remove_first_column(found_files)
