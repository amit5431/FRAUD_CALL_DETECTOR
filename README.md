# Fraud Call Detector

A simple Flask web app that predicts whether input text is a fraud/spam call or a normal call using a Naive Bayes model trained on the provided dataset.

## Project Structure

- `app.py` - Flask application and inference code
- `test_model.py` - Training and evaluation script for the Naive Bayes classifier
- `dataset.txt` - Labeled text dataset used for training
- `templates/index.html` - Web form and result display
- `static/style.css` - Page styling

## Requirements

- Python 3.8+
- Flask
- pandas
- scikit-learn

## Installation

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install flask pandas scikit-learn
```

## Usage

1. Run the app:

```powershell
python app.py
```

2. Open the web browser at `http://127.0.0.1:5000`
3. Enter a call transcript or message and submit to see whether it is detected as fraud/spam.

## Model Evaluation

To evaluate the model on the dataset, run:

```powershell
python test_model.py
```

This prints dataset statistics, accuracy, and a classification report.

## Notes

- The app uses TF-IDF vectorization and a Multinomial Naive Bayes classifier.
- The current model is trained on the entire dataset at startup, so large datasets may increase load time.
