import os
import psycopg2
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, session, redirect, url_for
from keras.models import load_model
from crop_images import get_crop_image

# Force CPU and low memory usage for Render Free Tier
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- APP SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.secret_key = "agrointel_secret_key_2025"

# --- DATABASE CONFIG ---
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_XzmcuGoLkw62@ep-summer-cell-anciiddp-pooler.c-6.us-east-1.aws.neon.tech/neondb?sslmode=require"
)

def get_connection():
    return psycopg2.connect(DATABASE_URL)

# --- LOAD MODELS & ASSETS ---
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Load models from your root directory
fed_model_path = os.path.join(BASE_DIR, "federated_mlp_model.h5")
if os.path.exists(fed_model_path):
    model = load_model(fed_model_path)
else:
    model = load_model(os.path.join(BASE_DIR, "centralized_mlp_model.h5"))

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
        # Get data from prediction.html form
        n_val = float(request.form["N"])
        p_val = float(request.form["P"])
        k_val = float(request.form["K"])
        temp  = float(request.form["temperature"])
        hum   = float(request.form["humidity"])
        ph_val = float(request.form["ph"])
        rain  = float(request.form["rainfall"])

        # Convert to DataFrame to match scaler expectations
        input_df = pd.DataFrame([[n_val, p_val, k_val, temp, hum, ph_val, rain]], columns=FEATURES)
        scaled_data = scaler.transform(input_df)
        
        prediction = model.predict(scaled_data)
        crop = encoder.inverse_transform([prediction.argmax()])[0]
        
        # Fetch image via your crop_images utility
        img_data = get_crop_image(crop)
        
        # Important: Pass all variables required by result.html
        return render_template("result.html", 
                               prediction=crop, 
                               image_url=img_data["image_url"],
                               N=n_val, P=p_val, K=k_val,
                               temperature=temp, humidity=hum,
                               ph=ph_val, rainfall=rain,
                               emoji="🌱", season="Varies", water="Medium", 
                               duration="Varies", soil="Varies", crop_info="Recommended crop.")
    except Exception as e:
        return render_template("prediction.html", error=str(e))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)