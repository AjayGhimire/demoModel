import pandas as pd


def load_and_process_data(file_path):
    # Load the data from a CSV file
    data = pd.read_csv(file_path)

    # Perform data processing (e.g., normalization, encoding)
    processed_data = data  # Apply your processing steps here

    return processed_data
