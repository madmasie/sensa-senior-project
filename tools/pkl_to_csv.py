import pandas as pd

# Load the pickle file
df = pd.read_pickle('data/sen55_2026-05-22_11-39-43.pkl')

# Save as CSV
df.to_csv('output_file.csv', index=False)
