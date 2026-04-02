import pandas as pd
import numpy as np
import os

# Create data folder if not exists
os.makedirs("data", exist_ok=True)

# Load dataset
df = pd.read_csv("data/Crop_recommendation.csv")

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split indices into 3 parts
splits = np.array_split(df.index, 3)

# Create DataFrames for each client
client_1 = df.loc[splits[0]]
client_2 = df.loc[splits[1]]
client_3 = df.loc[splits[2]]

# Save client datasets
client_1.to_csv("data/client1_data.csv", index=False)
client_2.to_csv("data/client2_data.csv", index=False)
client_3.to_csv("data/client3_data.csv", index=False)

print("✅ Dataset successfully split into 3 federated clients")
print("Client 1 size:", client_1.shape)
print("Client 2 size:", client_2.shape)
print("Client 3 size:", client_3.shape)