from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

# -----------------------------------------
# Create Flask Application
# -----------------------------------------

app = Flask(__name__)

# -----------------------------------------
# Load AI Model
# -----------------------------------------

MODEL_PATH = "models/chatbot_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Please run train_model.py first."
    )

model = joblib.load(MODEL_PATH)

# -----------------------------------------
# Load Dataset
# -----------------------------------------

data = pd.read_csv("chatbot_data.csv")

# Create intent-response dictionary

intent_responses = (
    data
    .drop_duplicates("intent")
    .set_index("intent")["response"]
    .to_dict()
)

# -----------------------------------------
# Intent Display Names
# -----------------------------------------

intent_names = {
    "password_reset": "Password Reset",
    "wifi_issue": "Wi-Fi / Network Issue",
    "vpn_access": "VPN Access",
    "software_installation": "Software Installation",
    "slow_computer": "Slow Computer",
    "email_issue": "Email Problem",
    "hardware_issue": "Hardware Problem",
    "account_access": "Account Access",
    "printer_issue": "Printer Problem",
    "security_issue": "Security Issue"
}

# -----------------------------------------
# Home Page
# -----------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html",
        intent_names=intent_names
    )


# -----------------------------------------
# Chatbot API
# -----------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    data_received = request.get_json()

    user_message = data_received.get("message", "").strip()

    selected_intent = data_received.get("selected_intent", "")

    # Validate input

    if not user_message:

        return jsonify({
            "success": False,
            "message": "Please enter your IT problem."
        })

    # -----------------------------------------
    # AI Prediction
    # -----------------------------------------

    predicted_intent = model.predict([user_message])[0]

    # -----------------------------------------
    # Calculate Confidence
    # -----------------------------------------

    confidence = 0

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba([user_message])

        confidence = max(probabilities[0]) * 100

    # -----------------------------------------
    # If User Selected Category
    # -----------------------------------------

    if selected_intent and selected_intent != "auto":

        if selected_intent in intent_responses:

            predicted_intent = selected_intent

    # -----------------------------------------
    # Get Chatbot Response
    # -----------------------------------------

    chatbot_response = intent_responses.get(
        predicted_intent,
        "I could not understand your problem. Please contact the IT support team."
    )

    # -----------------------------------------
    # Return Result
    # -----------------------------------------

    return jsonify({

        "success": True,

        "intent": predicted_intent,

        "intent_name": intent_names.get(
            predicted_intent,
            predicted_intent
        ),

        "response": chatbot_response,

        "confidence": round(confidence, 2)

    })


# -----------------------------------------
# Run Application
# -----------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )