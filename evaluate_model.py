import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import joblib

# -------------------------
# Load Test Data
# -------------------------
test_data = pd.read_csv("processed_data/test_data.csv")
X_test = test_data.drop("label", axis=1).values
y_test = test_data["label"].values

NUM_CLASSES = 22

# -------------------------
# Centralized Model Evaluation
# -------------------------
central_model = load_model("centralized_mlp_model.h5")
y_pred_central = np.argmax(central_model.predict(X_test), axis=1)

central_metrics = {
    "Accuracy":  accuracy_score(y_test, y_pred_central),
    "Precision": precision_score(y_test, y_pred_central, average="weighted", zero_division=0),
    "Recall":    recall_score(y_test, y_pred_central, average="weighted", zero_division=0),
    "F1-Score":  f1_score(y_test, y_pred_central, average="weighted", zero_division=0),
}

print("\n✅ Centralized MLP Metrics:")
for k, v in central_metrics.items():
    print(f"  {k}: {v:.4f}")

# -------------------------
# Federated Model Simulation
# (Simulates FedAvg by training 3 local models and averaging weights)
# -------------------------
def build_model():
    m = Sequential([
        Dense(64, activation="relu", input_shape=(X_test.shape[1],)),
        Dense(32, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

federated_accuracy = []
client_models = []

print("\n✅ Running Federated Learning Simulation (10 rounds)...")

for round_num in range(1, 11):
    round_weights = []

    for client_id in ["client1", "client2", "client3"]:
        df = pd.read_csv(f"processed_data/{client_id}_processed.csv")
        X_c = df.drop("label", axis=1).values
        y_c = to_categorical(df["label"].values, num_classes=NUM_CLASSES)
        X_tr, X_te, y_tr, y_te = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

        local_model = build_model()

        # Use global weights if available from previous round
        if round_num > 1:
            local_model.set_weights(global_weights)

        local_model.fit(X_tr, y_tr, epochs=5, batch_size=32, verbose=0)
        round_weights.append(local_model.get_weights())

    # FedAvg — average weights across all clients
    global_weights = [
        np.mean([client_w[layer] for client_w in round_weights], axis=0)
        for layer in range(len(round_weights[0]))
    ]

    # Evaluate global model on test set
    global_model = build_model()
    global_model.set_weights(global_weights)
    y_pred_fed_round = np.argmax(global_model.predict(X_test, verbose=0), axis=1)
    round_acc = accuracy_score(y_test, y_pred_fed_round)
    federated_accuracy.append(round_acc)
    print(f"  Round {round_num:02d} | Accuracy: {round_acc:.4f}")

# Final federated metrics
y_pred_fed = np.argmax(global_model.predict(X_test, verbose=0), axis=1)

federated_metrics = {
    "Accuracy":  accuracy_score(y_test, y_pred_fed),
    "Precision": precision_score(y_test, y_pred_fed, average="weighted", zero_division=0),
    "Recall":    recall_score(y_test, y_pred_fed, average="weighted", zero_division=0),
    "F1-Score":  f1_score(y_test, y_pred_fed, average="weighted", zero_division=0),
}

print("\n✅ Federated MLP Metrics:")
for k, v in federated_metrics.items():
    print(f"  {k}: {v:.4f}")

# Save final federated model
global_model.save("federated_mlp_model.h5")
print("\n✅ Federated model saved as federated_mlp_model.h5")

# -------------------------
# Comparison Table
# -------------------------
comparison_table = pd.DataFrame({
    "Metric":      ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Centralized": [central_metrics["Accuracy"], central_metrics["Precision"],
                    central_metrics["Recall"],   central_metrics["F1-Score"]],
    "Federated":   [federated_metrics["Accuracy"], federated_metrics["Precision"],
                    federated_metrics["Recall"],    federated_metrics["F1-Score"]],
})
print("\n📊 Centralized vs Federated Comparison:")
print(comparison_table.to_string(index=False))

# -------------------------
# Plot 1: Accuracy vs Rounds
# -------------------------
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), federated_accuracy, marker="o", label="Federated Accuracy", color="blue")
plt.axhline(y=central_metrics["Accuracy"], color="red", linestyle="--", label="Centralized Accuracy")
plt.xlabel("Communication Rounds")
plt.ylabel("Accuracy")
plt.title("Federated vs Centralized Accuracy over Rounds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_vs_rounds.png")
plt.show()
print("✅ Plot saved as accuracy_vs_rounds.png")

# -------------------------
# Plot 2: Confusion Matrix — Centralized
# -------------------------
cm_central = confusion_matrix(y_test, y_pred_central)
plt.figure(figsize=(14, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_central)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Centralized Model - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_centralized.png")
plt.show()
print("✅ Confusion matrix (centralized) saved")

# -------------------------
# Plot 3: Confusion Matrix — Federated
# -------------------------
cm_fed = confusion_matrix(y_test, y_pred_fed)
plt.figure(figsize=(14, 12))
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_fed)
disp2.plot(cmap="Greens", xticks_rotation=45)
plt.title("Federated Model - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_federated.png")
plt.show()
print("✅ Confusion matrix (federated) saved")

# -------------------------
# Plot 4: Metrics Bar Chart
# -------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
central_vals = [central_metrics[m] for m in metrics]
fed_vals = [federated_metrics[m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, central_vals, width, label="Centralized", color="steelblue")
plt.bar(x + width/2, fed_vals,     width, label="Federated",   color="seagreen")
plt.xticks(x, metrics)
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.title("Centralized vs Federated - Performance Metrics")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("metrics_comparison.png")
plt.show()
print("✅ Metrics bar chart saved as metrics_comparison.png")