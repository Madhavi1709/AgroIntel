import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request, session, redirect, url_for
import numpy as np
import joblib
import webbrowser
import threading
import subprocess
import sys
import psycopg2  # Swapped from mysql.connector
from keras.models import load_model
from crop_images import get_crop_image

# ─── APP SETUP ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.secret_key = "agrointel_secret_key_2025"

# ─── POSTGRESQL CONFIG ────────────────────────────────────────
# This automatically uses Render's DB URL or your Neon URL
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "url"
)

# ─── DATABASE HELPERS ─────────────────────────────────────────
def get_connection():
    """Get PostgreSQL connection"""
    return psycopg2.connect(DATABASE_URL)

def init_db():
    """Create users table if not exists (PostgreSQL syntax)"""
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

def get_user(username):
    """Fetch user by username from PostgreSQL"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

def create_user(name, username, mobile, password, city, state):
    """Save new user to PostgreSQL"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (name, username, mobile, password, city, state) VALUES (%s, %s, %s, %s, %s, %s)",
        (name, username, mobile, password, city, state)
    )
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ New user saved to PostgreSQL: {username}")

# Initialize DB on every startup
init_db()

# ─── LOAD ML MODEL ────────────────────────────────────────────
fed_model_path = os.path.join(BASE_DIR, "federated_mlp_model.h5")
central_model_path = os.path.join(BASE_DIR, "centralized_mlp_model.h5")

if os.path.exists(fed_model_path):
    model = load_model(fed_model_path)
    print("✅ Federated MLP model loaded")
else:
    model = load_model(central_model_path)
    print("⚠️ Using Centralized MLP")

scaler  = joblib.load(os.path.join(BASE_DIR, "processed_data", "scaler.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "processed_data", "label_encoder.pkl"))

# ─── ROUTES ───────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/index")
def index():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", 
                           username=session["username"], 
                           name=session.get("name", ""))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/register_user", methods=["POST"])
def register_user():
    try:
        name     = request.form["name"].strip()
        username = request.form["username"].strip()
        mobile   = request.form["mobile"].strip()
        password = request.form["password"].strip()
        city     = request.form["city"].strip()
        state    = request.form["state"].strip()

        # Validations
        if not name or not name[0].isupper():
            return render_template("register.html", error="Name must start with a capital letter")
        if len(username) < 8:
            return render_template("register.html", error="Username must be at least 8 characters")
        if get_user(username):
            return render_template("register.html", error="Username already exists")
        
        create_user(name, username, mobile, password, city, state)
        return render_template("login.html", register_success=True)
    except Exception as e:
        return render_template("register.html", error=str(e))

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/login_user", methods=["POST"])
def login_user():
    username = request.form["username"].strip()
    password = request.form["password"].strip()
    user = get_user(username)

    # PostgreSQL tuple: (id, name, username, mobile, password, city, state)
    if user and user[4] == password:
        session["username"] = username
        session["name"]     = user[1]
        return redirect(url_for("index"))
    else:
        return render_template("login.html", login_error=True)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/prediction")
def prediction_page():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("prediction.html")

# ─── CROP INFO DATA ───────────────────────────────────────────
CROP_INFO = {
    "rice": {"emoji": "🌾", "season": "Kharif", "water": "High", "duration": "120-150 days", "soil": "Clayey", "info": "Rice thrives in warm, humid climates."},
    # ... (Keep all your CROP_INFO items here)
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw_data = [
            float(request.form["N"]), float(request.form["P"]), float(request.form["K"]),
            float(request.form["temperature"]), float(request.form["humidity"]),
            float(request.form["ph"]), float(request.form["rainfall"])
        ]
        
        data = scaler.transform(np.array([raw_data]))
        prediction = model.predict(data)
        crop = encoder.inverse_transform([prediction.argmax()])[0]
        
        crop_key = crop.lower().replace(" ", "")
        info = CROP_INFO.get(crop_key, {"emoji":"🌱", "season":"Varies", "water":"Medium", "duration":"Varies", "soil":"Varies", "info":"Recommended crop."})
        img_data = get_crop_image(crop)

        return render_template("result.html", prediction=crop, emoji=info["emoji"], 
                               season=info["season"], water=info["water"], 
                               duration=info["duration"], soil=info["soil"], 
                               crop_info=info["info"], image_url=img_data["image_url"],
                               N=raw_data[0], P=raw_data[1], K=raw_data[2], 
                               temperature=raw_data[3], humidity=raw_data[4], 
                               ph=raw_data[5], rainfall=raw_data[6])
    except Exception as e:
        return render_template("prediction.html", error=f"Error: {str(e)}")

@app.route("/train")
def train():
    # Federated training triggers subprocesses
    python = sys.executable
    try:
        subprocess.Popen([python, "fl_server.py"])
        for c in ["client1", "client2", "client3"]:
            subprocess.Popen([python, "fl_client.py", c])
        return render_template("index.html", username=session.get("username"), name=session.get("name"), train_success=True)
    except Exception as e:
        return render_template("index.html", username=session.get("username"), train_error=str(e))

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    # Local only: auto open browser
    if os.environ.get("RENDER") is None:
        threading.Timer(2, open_browser).start()
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)