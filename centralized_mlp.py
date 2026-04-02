import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import joblib

# Load processed client datasets
df1 = pd.read_csv("processed_data/client1_processed.csv")
df2 = pd.read_csv("processed_data/client2_processed.csv")
df3 = pd.read_csv("processed_data/client3_processed.csv")

# Combine all data for centralized training
df = pd.concat([df1, df2, df3], axis=0).reset_index(drop=True)

# Split features and labels
X = df.drop("label", axis=1).values
y = df["label"].values

# Fixed num_classes
NUM_CLASSES = 22
y_cat = to_categorical(y, num_classes=NUM_CLASSES)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# Define MLP model
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

# Train model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Evaluate model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n✅ Centralized MLP Performance")
print("Accuracy:", accuracy_score(y_true, y_pred_classes))
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

# Save model
model.save("centralized_mlp_model.h5")
print("\n✅ Centralized MLP model saved")