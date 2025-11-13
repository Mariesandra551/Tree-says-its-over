import pandas as pd
import glob
import os



"""
model.py
---------

This script prepares and consolidates raw economic data files into a single, standardized
dataset suitable for analysis and model training. It automatically detects header rows,
cleans inconsistent column names, converts numeric values to floats, and merges multiple
CSV files into one uniform dataset.

The goal is to create a reliable data foundation for the early warning system model
that predicts financial crisis risk in Greece.

Main steps:
    1. Read all CSV files from the 'data/' directory.
    2. Automatically detect the correct header row (first row containing 'Date').
    3. Standardize column names (convert to lowercase, replace spaces, symbols, etc.).
    4. Remove duplicate columns and add a 'country' column based on file names.
    5. Clean numeric values by removing commas and percent signs.
    6. Concatenate all cleaned datasets into one combined DataFrame.
    7. Save the final, cleaned dataset for machine learning.

Inputs:
    - data/*.csv : Raw economic data files (e.g., debt, deficit, yields, inflation)

Outputs:
    - data/merged_cleaned_dataset.csv : Combined and cleaned dataset ready for training

Intended Use:
    This file is part of the ECON 302 project on predicting financial crises using
    economic indicators and machine learning. It ensures that all input data are
    consistent, accurate, and formatted for reproducible modeling.
"""


files = glob.glob("data/*.csv")

dfs = []

for file in files:
    print("Processing:", file)

    raw = pd.read_csv(file, header=None)

    # detect header row (first row with the word "Date")
    header_row = raw[raw.apply(lambda row: row.astype(str).str.contains("Date", case=False)).any(axis=1)].index[0]

    temp = pd.read_csv(file, header=header_row)

    # Standardize column names
    temp.columns = (
        temp.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("%", "pct")
        .str.replace("/", "_")
    )

    # Remove duplicated columns
    temp = temp.loc[:, ~temp.columns.duplicated()]

    # Add country column based on filename
    temp["country"] = os.path.basename(file).replace(".csv", "")

    dfs.append(temp)

# Combine all cleaned data
df = pd.concat(dfs, ignore_index=True)

# Convert numeric columns (remove commas, convert to float)
for col in df.columns:
    try:
        df[col] = df[col].astype(str).str.replace(",", "").str.replace("%", "").astype(float)
    except:
        pass

print("\n Final dataset shape:", df.shape)
print(df.head())

df.to_csv("data/merged_cleaned_dataset.csv", index=False)
print("\n Saved cleaned dataset → data/merged_cleaned_dataset.csv")