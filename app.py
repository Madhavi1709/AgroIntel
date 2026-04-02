import os
# Force CPU and low memory usage for Render Free Tier
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import psycopg2
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, session, redirect, url_for
from keras.models import load_model
from crop_images import get_crop_image

# --- APP SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: By default, Flask looks for 'templates' in the root. 
app = Flask(__name__)
app.secret_key = "agrointel_secret_key_2025"

# --- DATABASE CONFIG ---
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_XzmcuGoLkw62@ep-summer-cell-anciiddp-pooler.c-6.us-east-1.aws.neon.tech/neondb?sslmode=require"
)

def get_connection():
    return psycopg2.connect(DATABASE_URL)

# --- LOAD MODELS (Paths match your screenshot) ---
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

fed_model_path = os.path.join(BASE_DIR, "federated_mlp_model.h5")
central_model_path = os.path.join(BASE_DIR, "centralized_mlp_model.h5")

if os.path.exists(fed_model_path):
    model = load_model(fed_model_path)
    print("✅ Federated model loaded")
else:
    model = load_model(central_model_path)
    print("⚠️ Centralized model loaded")

# Paths match your 'processed_data' folder
scaler  = joblib.load(os.path.join(BASE_DIR, "processed_data", "scaler.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "processed_data", "label_encoder.pkl"))

# --- ROUTES ---

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/index")
def index():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", username=session.get("username"), name=session.get("name"))

@app.route("/prediction")
def prediction_page():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("prediction.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        raw_vals = [
            float(request.form["N"]), float(request.form["P"]), float(request.form["K"]),
            float(request.form["temperature"]), float(request.form["humidity"]),
            float(request.form["ph"]), float(request.form["rainfall"])
        ]

        # Use DataFrame to avoid the Feature Name Warning
        input_df = pd.DataFrame([raw_vals], columns=FEATURES)
        scaled_data = scaler.transform(input_df)
        
        prediction = model.predict(scaled_data)
        crop = encoder.inverse_transform([prediction.argmax()])[0]

        img_data = get_crop_image(crop)

        # Basic Crop Info (Add more to this dict as needed)
        return render_template("result.html", prediction=crop, image_url=img_data["image_url"])
    except Exception as e:
        return render_template("prediction.html", error=str(e))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)