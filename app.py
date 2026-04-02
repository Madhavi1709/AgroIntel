import os
# Suppress TensorFlow logs and force CPU-only optimizations to save RAM
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import threading
import subprocess
import psycopg2
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, session, redirect, url_for
from keras.models import load_model
from crop_images import get_crop_image

# --- APP SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.secret_key = "agrointel_secret_key_2025"

# --- MEMORY OPTIMIZATIONS FOR RENDER ---
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# --- POSTGRESQL CONFIG ---
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_XzmcuGoLkw62@ep-summer-cell-anciiddp-pooler.c-6.us-east-1.aws.neon.tech/neondb?sslmode=require"
)

# --- DATABASE HELPERS ---
def get_connection():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id       SERIAL PRIMARY KEY,
                name     VARCHAR(100) NOT NULL,
                username VARCHAR(100) UNIQUE NOT NULL,
                mobile   VARCHAR(15)  NOT NULL,
                password VARCHAR(255) NOT NULL,
                city     VARCHAR(100) NOT NULL,
                state    VARCHAR(100) NOT NULL
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ PostgreSQL database ready")
    except Exception as e:
        print(f"❌ DB Init Error: {e}")

# Initialize DB on startup
init_db()

# --- LOAD ML MODELS (GLOBAL TO SAVE RAM) ---
# Define features exactly as used during training to avoid warnings
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

fed_model_path = os.path.join(BASE_DIR, "federated_mlp_model.h5")
central_model_path = os.path.join(BASE_DIR, "centralized_mlp_model.h5")

if os.path.exists(fed_model_path):
    model = load_model(fed_model_path)
    print("✅ Federated MLP model loaded")
else:
    model = load_model(central_model_path)
    print("⚠️ Using Centralized MLP model")

scaler  = joblib.load(os.path.join(BASE_DIR, "processed_data", "scaler.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "processed_data", "label_encoder.pkl"))

# --- CROP INFO DATA ---
CROP_INFO = {
    "rice": {"emoji": "🌾", "season": "Kharif", "water": "High", "duration": "120-150 days", "soil": "Clayey", "info": "Rice is a staple food crop grown in flooded paddies."},
    "maize": {"emoji": "🌽", "season": "Kharif", "water": "Medium", "duration": "80-110 days", "soil": "Loamy", "info": "Maize thrives in warm weather."},
    "chickpea": {"emoji": "🫘", "season": "Rabi", "water": "Low", "duration": "90-120 days", "soil": "Sandy Loam", "info": "Chickpea grows well in cool, dry climates."},
    # ... (Keep the rest of your CROP_INFO dictionary here)
}

# --- ROUTES ---

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/index")
def index():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", username=session["username"], name=session.get("name", ""))

@app.route("/register_user", methods=["POST"])
def register_user():
    # ... (Keep your existing registration logic)
    pass

@app.route("/login_user", methods=["POST"])
def login_user():
    # ... (Keep your existing login logic)
    pass

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Get data from form
        raw_vals = [
            float(request.form["N"]),
            float(request.form["P"]),
            float(request.form["K"]),
            float(request.form["temperature"]),
            float(request.form["humidity"]),
            float(request.form["ph"]),
            float(request.form["rainfall"])
        ]

        # 2. Convert to DataFrame with feature names to avoid Sklearn warnings
        input_df = pd.DataFrame([raw_vals], columns=FEATURES)
        
        # 3. Scale and Predict
        scaled_data = scaler.transform(input_df)
        prediction = model.predict(scaled_data)
        crop = encoder.inverse_transform([prediction.argmax()])[0]

        # 4. Get Crop Details
        crop_key = crop.lower().replace(" ", "")
        info = CROP_INFO.get(crop_key, {"emoji": "🌱", "season": "Varies", "water": "Moderate", "duration": "Varies", "soil": "Varies", "info": f"Recommended crop: {crop}"})
        img_data = get_crop_image(crop)

        return render_template("result.html", prediction=crop, emoji=info["emoji"], 
                               season=info["season"], water=info["water"], duration=info["duration"], 
                               soil=info["soil"], crop_info=info["info"], image_url=img_data["image_url"],
                               N=raw_vals[0], P=raw_vals[1], K=raw_vals[2], temperature=raw_vals[3], 
                               humidity=raw_vals[4], ph=raw_vals[5], rainfall=raw_vals[6])
    except Exception as e:
        return render_template("prediction.html", error=f"Prediction Error: {str(e)}")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    # Render sets the PORT environment variable automatically
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)