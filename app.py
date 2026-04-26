from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# 🔹 Load dataset
data = pd.read_csv("dataset/fraud_call.file", sep='\t', header=None, names=['label', 'text'], on_bad_lines='skip')

# 🔹 Features & Labels
X = data["text"]
y = data["label"]

# 🔹 Convert text → numbers
vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

# 🔹 Train model
model = MultinomialNB()
model.fit(X_vector, y)

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        text = request.form["speech_text"]

        # 🔹 Transform input text
        text_vector = vectorizer.transform([text])

        # 🔹 Prediction
        prediction = model.predict(text_vector)[0]

        if prediction == "spam":
            result = "⚠️ Spam / Fraud Call Detected"
        else:
            result = "✅ Safe Call"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
