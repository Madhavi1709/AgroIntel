import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request, session, redirect, url_for
import numpy as np
import joblib
import webbrowser
import threading
import subprocess
import sys
import psycopg2
from keras.models import load_model
from crop_images import get_crop_image

# ─── APP SETUP ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.secret_key = "agrointel_secret_key_2025"

# ─── POSTGRESQL CONFIG ────────────────────────────────────────
# On Render: DATABASE_URL is set automatically as environment variable
# Locally: update the fallback URL with your local PostgreSQL credentials
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_XzmcuGoLkw62@ep-summer-cell-anciiddp-pooler.c-6.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
)

# ─── DATABASE HELPERS ─────────────────────────────────────────
def get_connection():
    return psycopg2.connect(DATABASE_URL)

def init_db():
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
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

def create_user(name, username, mobile, password, city, state):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (name, username, mobile, password, city, state) VALUES (%s, %s, %s, %s, %s, %s)",
        (name, username, mobile, password, city, state)
    )
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ New user saved: {username}")

# Initialize DB on startup
init_db()

# ─── LOAD ML MODEL ────────────────────────────────────────────
fed_model_path     = os.path.join(BASE_DIR, "federated_mlp_model.h5")
central_model_path = os.path.join(BASE_DIR, "centralized_mlp_model.h5")

if os.path.exists(fed_model_path):
    model = load_model(fed_model_path)
    print("✅ Federated MLP model loaded")
else:
    model = load_model(central_model_path)
    print("⚠️  Using Centralized MLP model")

scaler  = joblib.load(os.path.join(BASE_DIR, "processed_data", "scaler.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "processed_data", "label_encoder.pkl"))
print("✅ Scaler and encoder loaded")

# ─── CROP INFO ────────────────────────────────────────────────
CROP_INFO = {
    "rice":        {"emoji": "🌾", "season": "Kharif",  "water": "High",   "duration": "120-150 days", "soil": "Clayey",       "info": "Rice is a staple food crop grown in flooded paddies. It thrives in warm, humid climates with abundant water supply."},
    "maize":       {"emoji": "🌽", "season": "Kharif",  "water": "Medium", "duration": "80-110 days",  "soil": "Loamy",        "info": "Maize is a versatile cereal crop used for food, feed and industrial purposes. It grows well in warm weather."},
    "chickpea":    {"emoji": "🫘", "season": "Rabi",    "water": "Low",    "duration": "90-120 days",  "soil": "Sandy Loam",   "info": "Chickpea is a protein-rich legume crop. It fixes nitrogen in soil and grows well in cool, dry climates."},
    "kidneybeans": {"emoji": "🫘", "season": "Kharif",  "water": "Medium", "duration": "90-120 days",  "soil": "Well-drained", "info": "Kidney beans are nutritious legumes that grow in warm climates and enrich soil with nitrogen."},
    "pigeonpeas":  {"emoji": "🌿", "season": "Kharif",  "water": "Low",    "duration": "120-180 days", "soil": "Sandy Loam",   "info": "Pigeon peas are drought-tolerant legumes widely grown in tropical and subtropical regions."},
    "mothbeans":   {"emoji": "🫘", "season": "Kharif",  "water": "Low",    "duration": "60-90 days",   "soil": "Sandy",        "info": "Moth beans are extremely drought-resistant legumes that grow in arid and semi-arid regions."},
    "mungbean":    {"emoji": "🫘", "season": "Kharif",  "water": "Low",    "duration": "60-90 days",   "soil": "Loamy",        "info": "Mung beans are fast-growing legumes rich in protein and minerals, used widely in Asian cuisine."},
    "blackgram":   {"emoji": "🫘", "season": "Kharif",  "water": "Low",    "duration": "70-90 days",   "soil": "Loamy",        "info": "Black gram is a nutritious pulse crop grown in tropical climates, widely used in Indian cuisine."},
    "lentil":      {"emoji": "🫘", "season": "Rabi",    "water": "Low",    "duration": "100-120 days", "soil": "Sandy Loam",   "info": "Lentils are cool-season legumes rich in protein and fiber, grown in drier climates."},
    "pomegranate": {"emoji": "🟣", "season": "Annual",  "water": "Low",    "duration": "180-240 days", "soil": "Well-drained", "info": "Pomegranate is a drought-tolerant fruit crop known for its antioxidant-rich seeds."},
    "banana":      {"emoji": "🍌", "season": "Annual",  "water": "High",   "duration": "270-365 days", "soil": "Loamy",        "info": "Banana is a tropical fruit crop that requires warm temperatures, high humidity and regular watering."},
    "mango":       {"emoji": "🥭", "season": "Summer",  "water": "Medium", "duration": "4-5 years",    "soil": "Alluvial",     "info": "Mango is the king of fruits, thriving in tropical climates with a distinct dry season for flowering."},
    "grapes":      {"emoji": "🍇", "season": "Annual",  "water": "Medium", "duration": "1-3 years",    "soil": "Well-drained", "info": "Grapes are climbing fruit crops grown in warm, sunny climates and widely used for juice and wine."},
    "watermelon":  {"emoji": "🍉", "season": "Summer",  "water": "Medium", "duration": "70-90 days",   "soil": "Sandy Loam",   "info": "Watermelon is a warm-season fruit crop that requires long warm summers and well-drained sandy soil."},
    "muskmelon":   {"emoji": "🍈", "season": "Summer",  "water": "Medium", "duration": "70-90 days",   "soil": "Sandy Loam",   "info": "Muskmelon is a sweet summer fruit that grows best in warm, dry climates with sandy loam soil."},
    "apple":       {"emoji": "🍎", "season": "Winter",  "water": "Medium", "duration": "4-5 years",    "soil": "Well-drained", "info": "Apples grow in temperate climates and require cold winters for dormancy and warm summers for ripening."},
    "orange":      {"emoji": "🍊", "season": "Winter",  "water": "Medium", "duration": "3-5 years",    "soil": "Loamy",        "info": "Oranges are citrus fruits that thrive in subtropical climates with moderate rainfall and warm temperatures."},
    "papaya":      {"emoji": "🍍", "season": "Annual",  "water": "Medium", "duration": "180-270 days", "soil": "Loamy",        "info": "Papaya is a fast-growing tropical fruit crop that produces fruit year-round in warm, frost-free climates."},
    "coconut":     {"emoji": "🥥", "season": "Annual",  "water": "High",   "duration": "6-7 years",    "soil": "Sandy Loam",   "info": "Coconut palms grow in coastal tropical regions and are known as the tree of life for their many uses."},
    "cotton":      {"emoji": "🪴", "season": "Kharif",  "water": "Medium", "duration": "150-180 days", "soil": "Black",        "info": "Cotton is a major cash crop grown in warm climates, primarily used for fiber in textile industry."},
    "jute":        {"emoji": "🌿", "season": "Kharif",  "water": "High",   "duration": "100-120 days", "soil": "Alluvial",     "info": "Jute is a natural fiber crop grown in warm, humid climates, used for making sacks, ropes and fabric."},
    "coffee":      {"emoji": "☕",  "season": "Annual",  "water": "Medium", "duration": "2-3 years",    "soil": "Well-drained", "info": "Coffee grows in tropical highland regions and requires consistent rainfall and moderate temperatures."},
}

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

# ── REGISTER ──────────────────────────────────────────────────
@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/register_user", methods=["POST"])
def register_user():
    name     = request.form["name"].strip()
    username = request.form["username"].strip()
    mobile   = request.form["mobile"].strip()
    password = request.form["password"].strip()
    city     = request.form["city"].strip()
    state    = request.form["state"].strip()

    if not name or not name[0].isupper():
        return render_template("register.html", error="Name must start with a capital letter")
    if len(username) < 8:
        return render_template("register.html", error="Username must be at least 8 characters")
    if get_user(username):
        return render_template("register.html", error="Username already exists. Please choose another.")
    if not (mobile.isdigit() and len(mobile) == 10 and mobile[0] in "6789"):
        return render_template("register.html", error="Enter a valid 10-digit mobile number")
    if len(password) < 8:
        return render_template("register.html", error="Password must be at least 8 characters")

    create_user(name, username, mobile, password, city, state)
    return render_template("login.html", register_success=True)

# ── LOGIN ─────────────────────────────────────────────────────
@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/login_user", methods=["POST"])
def login_user():
    username = request.form["username"].strip()
    password = request.form["password"].strip()
    user = get_user(username)
    # user tuple: (id, name, username, mobile, password, city, state)
    if user and user[4] == password:
        session["username"] = username
        session["name"]     = user[1]
        print(f"✅ Login: {username}")
        return redirect(url_for("index"))
    else:
        return render_template("login.html", login_error=True)

# ── LOGOUT ────────────────────────────────────────────────────
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# ── PREDICTION ────────────────────────────────────────────────
@app.route("/prediction")
def prediction_page():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("prediction.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        N           = float(request.form["N"])
        P           = float(request.form["P"])
        K           = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity    = float(request.form["humidity"])
        ph          = float(request.form["ph"])
        rainfall    = float(request.form["rainfall"])

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        data = scaler.transform(data)

        prediction = model.predict(data)
        crop = encoder.inverse_transform([prediction.argmax()])[0]
        print(f"✅ Prediction: {crop}")

        crop_key = crop.lower().replace(" ", "")
        info = CROP_INFO.get(crop_key, {
            "emoji": "🌱", "season": "Varies", "water": "Moderate",
            "duration": "Varies", "soil": "Varies",
            "info": f"{crop.capitalize()} is a recommended crop based on your soil and climate conditions."
        })

        img_data = get_crop_image(crop)

        return render_template("result.html",
                               prediction=crop,
                               emoji=info["emoji"],
                               season=info["season"],
                               water=info["water"],
                               duration=info["duration"],
                               soil=info["soil"],
                               crop_info=info["info"],
                               image_url=img_data["image_url"],
                               image_alt=img_data["alt"],
                               N=N, P=P, K=K,
                               temperature=temperature,
                               humidity=humidity,
                               ph=ph,
                               rainfall=rainfall)
    except Exception as e:
        return render_template("prediction.html", error=f"Error: {str(e)}")

# ── FEDERATED TRAINING ────────────────────────────────────────
@app.route("/train")
def train():
    python = sys.executable
    try:
        subprocess.Popen(
            [python, "fl_server.py"],
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
        )
        for c in ["client1", "client2", "client3"]:
            subprocess.Popen(
                [python, "fl_client.py", c],
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
            )
        return render_template("index.html",
                               username=session.get("username", ""),
                               name=session.get("name", ""),
                               train_success=True)
    except Exception as e:
        return render_template("index.html",
                               username=session.get("username", ""),
                               name=session.get("name", ""),
                               train_error=str(e))

# ─── AUTO OPEN BROWSER (local only) ──────────────────────────
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    threading.Timer(2, open_browser).start()
    app.run(debug=True, use_reloader=False)