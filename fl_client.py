import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import flwr as fl
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import sys

# -------------------------
# Load client-specific data
# -------------------------
client_id = sys.argv[1]  # client1 / client2 / client3

df = pd.read_csv(f"processed_data/{client_id}_processed.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

# Fixed num_classes = 22 (all crop classes in dataset)
NUM_CLASSES = 22
y = to_categorical(y, num_classes=NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"[{client_id}] Train: {len(X_train)} | Test: {len(X_test)}")

# -------------------------
# Define MLP Model
# -------------------------
def create_model():
    model = Sequential([
        Dense(64, activation="relu", input_shape=(X.shape[1],)),
        Dense(32, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = create_model()

# -------------------------
# Flower Client Definition
# -------------------------
class AgroClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        return loss, len(X_test), {"accuracy": accuracy}

# -------------------------
# Start Client
# Works with all flwr versions
# -------------------------
fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=AgroClient()
)