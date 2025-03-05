import numpy as np
import pandas as pd

file_path = r'C:/data/UnitreeA1/robodog2_Annotation/datastream_1/data_descriptor.csv'

data_headers = pd.read_csv(file_path, nrows=0).columns.values
data = pd.read_csv(file_path)

# Sort the data based on the sort column
sorted_data = data.sort_values('timestamp_stop')

# print(sorted_data)

# Write the sorted data to the output CSV file
sorted_data.to_csv(file_path, index=False)
