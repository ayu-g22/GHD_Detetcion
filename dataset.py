import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv('train.csv')

# Ensure that the dataset has a column named 'boneage'
if 'boneage' not in df.columns:
    raise ValueError("The dataset must have a column named 'boneage'.")

# Generate random differences between 0 and 5 years
np.random.seed(42)  # For reproducibility
random_diffs = np.random.randint(0, 72, size=df.shape[0])  # Random differences from 0 to 5 years

# Assign the real age as bone age plus the random difference
df['real_age'] = df['boneage'] + random_diffs

# Save the modified DataFrame to a new CSV file
df.to_csv('boneage_with_real_age.csv', index=False)

print(df.head())  # Print the first few rows to check the result
