"""Model training and prediction helpers for the SMS spam classifier."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB


DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "spam.csv"


def load_data(data_path: str | Path = DEFAULT_DATA_PATH) -> tuple[pd.Series, pd.Series]:
    """Load SMS labels and messages from the project's CSV dataset."""
    data = pd.read_csv(data_path, encoding="latin-1", usecols=[0, 1])
    data.columns = ["label", "message"]
    data = data.dropna(subset=["label", "message"])
    return data["message"], data["label"]


def train_model(data_path: str | Path = DEFAULT_DATA_PATH) -> Pipeline:
    """Train and return a TF-IDF plus Multinomial Naive Bayes pipeline."""
    messages, labels = load_data(data_path)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
            ("classifier", MultinomialNB()),
        ]
    )
    pipeline.fit(messages, labels)
    return pipeline


def predict_message(model: Pipeline, message: str) -> tuple[str, float]:
    """Predict whether one message is ham or spam and return confidence."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message must contain at least one non-whitespace character.")

    label = str(model.predict([message])[0])
    probabilities = model.predict_proba([message])[0]
    confidence = float(max(probabilities))
    return label, confidence
