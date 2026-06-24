from flask import Flask, render_template, request
import pandas as pd
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)


texts = []
labels = []

with open("dataset.txt", "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split("\t", 1)  # split label + text by tab
        if len(parts) == 2:
            labels.append(parts[0])
            texts.append(parts[1])

# 🔹 Convert to DataFrame
data = pd.DataFrame({
    "text": texts,
    "label": labels
})

# 🔹 Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["text"])
y = data["label"]

# 🔹 Train model with balanced class weights
model = MultinomialNB(class_prior=[0.5, 0.5])  # Equal priors to fix imbalance
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        text = request.form["speech_text"]

        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)[0]

        if prediction == "fraud":
            result = "⚠️ Spam / Fraud Call Detected"
        else:
            result = "✅ normal"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
