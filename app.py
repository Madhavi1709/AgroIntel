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
# Explicitly setting template folder to ensure Flask finds your files
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
# Features must match the order in your 'About' page table
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

fed_model_path = os.path.join(BASE_DIR, "federated_mlp_model.h5")
if os.path.exists(fed_model_path):
    model = load_model(fed_model_path)
else:
    model = load_model(os.path.join(BASE_DIR, "centralized_mlp_model.h5"))

# Paths match your 'processed_data' folder
scaler  = joblib.load(os.path.join(BASE_DIR, "processed_data", "scaler.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "processed_data", "label_encoder.pkl"))

# --- CROP INFO DATA (Matches your result.html needs) ---
CROP_INFO = {
    "rice": {"emoji": "🌾", "season": "Kharif", "water": "High", "duration": "120-150 days", "soil": "Clayey", "info": "Rice is a staple food crop grown in flooded paddies."},
    # ... add other crops here following this structure
}

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

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/index")
def index():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", username=session.get("username"), name=session.get("name"))

# --- LOGIN LOGIC (Matches form action in login.html) ---
@app.route("/login_user", methods=["POST"])
def login_user():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    
    # Simple direct check or DB lookup
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name, password FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    cur.close()
    conn.close()

    if user and user[1] == password:
        session["username"] = username
        session["name"] = user[0]
        return redirect(url_for("index"))
    else:
        # Return to login page with error flag
        return render_template("login.html", login_error=True)

# --- PREDICTION LOGIC ---
@app.route("/prediction")
def prediction_page():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("prediction.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw_vals = [
            float(request.form["N"]), float(request.form["P"]), float(request.form["K"]),
            float(request.form["temperature"]), float(request.form["humidity"]),
            float(request.form["ph"]), float(request.form["rainfall"])
        ]
        
        input_df = pd.DataFrame([raw_vals], columns=FEATURES)
        scaled_data = scaler.transform(input_df)
        prediction = model.predict(scaled_data)
        crop = encoder.inverse_transform([prediction.argmax()])[0]
        
        crop_key = crop.lower().replace(" ", "")
        info = CROP_INFO.get(crop_key, {"emoji": "🌱", "season": "Varies", "water": "Medium", "duration": "Varies", "soil": "Varies", "info": "Recommended crop."})
        
        img_data = get_crop_image(crop)
        
        # Passing all variables required by result.html
        return render_template("result.html", 
                               prediction=crop, 
                               emoji=info["emoji"],
                               season=info["season"],
                               water=info["water"],
                               duration=info["duration"],
                               soil=info["soil"],
                               crop_info=info["info"],
                               image_url=img_data["image_url"],
                               N=raw_vals[0], P=raw_vals[1], K=raw_vals[2],
                               temperature=raw_vals[3], humidity=raw_vals[4],
                               ph=raw_vals[5], rainfall=raw_vals[6])
    except Exception as e:
        return render_template("prediction.html", error=str(e))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)