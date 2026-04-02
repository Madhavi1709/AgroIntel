import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# Create processed data folder
os.makedirs("processed_data", exist_ok=True)

# Feature columns
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
LABEL = "label"

# Load all client datasets
clients = {
    "client1": pd.read_csv("data/client1_data.csv"),
    "client2": pd.read_csv("data/client2_data.csv"),
    "client3": pd.read_csv("data/client3_data.csv")
}

# Combine all data to fit scaler and encoder
combined_df = pd.concat(clients.values(), axis=0).reset_index(drop=True)

# Initialize scaler and encoder
scaler = StandardScaler()
label_encoder = LabelEncoder()

# Fit on combined data (important for federated consistency)
scaler.fit(combined_df[FEATURES])
label_encoder.fit(combined_df[LABEL])

# Save scaler and encoder
joblib.dump(scaler, "processed_data/scaler.pkl")
joblib.dump(label_encoder, "processed_data/label_encoder.pkl")

# Process each client separately
for client_name, df in clients.items():
    X = df[FEATURES]
    y = df[LABEL]

    X_scaled = scaler.transform(X)
    y_encoded = label_encoder.transform(y)

    processed_df = pd.DataFrame(X_scaled, columns=FEATURES)
    processed_df["label"] = y_encoded

    processed_df.to_csv(f"processed_data/{client_name}_processed.csv", index=False)
    print(f"✅ {client_name} data processed and saved")

# Save a global test set from combined data for evaluate_model.py
X_all = scaler.transform(combined_df[FEATURES])
y_all = label_encoder.transform(combined_df[LABEL])

full_df = pd.DataFrame(X_all, columns=FEATURES)
full_df["label"] = y_all

_, test_df = train_test_split(full_df, test_size=0.2, random_state=42)
test_df.to_csv("processed_data/test_data.csv", index=False)
print("✅ Global test_data.csv saved")

print("\nFeature scaling & label encoding completed successfully!")