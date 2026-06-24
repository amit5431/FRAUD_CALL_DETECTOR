from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

texts = []
labels = []
with open('dataset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t', 1)
        if len(parts) == 2:
            labels.append(parts[0])
            texts.append(parts[1])

data = pd.DataFrame({'text': texts, 'label': labels})
print(f"Total samples: {len(data)}")
print(f"Label distribution:\n{data['label'].value_counts()}")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with balanced class weights
model = MultinomialNB(class_prior=[0.5, 0.5])
model.fit(X_train, y_train)

# Calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n=== MODEL ACCURACY ===")
print(f"Accuracy: {accuracy * 100:.2f}%")

print(f"\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))