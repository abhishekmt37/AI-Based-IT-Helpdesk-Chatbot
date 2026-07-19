import pandas as pd
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# -----------------------------------------
# 1. Load Dataset
# -----------------------------------------

DATASET_PATH = "chatbot_data.csv"

data = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Total records:", len(data))

# -----------------------------------------
# 2. Select Input and Output
# -----------------------------------------

X = data["text"]
y = data["intent"]

# -----------------------------------------
# 3. Create Machine Learning Pipeline
# -----------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])

# -----------------------------------------
# 4. Train the Model
# -----------------------------------------

print("Training AI model...")

model.fit(X, y)

print("Model trained successfully!")

# -----------------------------------------
# 5. Create Models Folder
# -----------------------------------------

os.makedirs("models", exist_ok=True)

# -----------------------------------------
# 6. Save the Model
# -----------------------------------------

MODEL_PATH = "models/chatbot_model.pkl"

joblib.dump(model, MODEL_PATH)

print("Model saved successfully!")
print("Location:", MODEL_PATH)