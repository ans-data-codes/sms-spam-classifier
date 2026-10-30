from pathlib import Path

import pytest

from model import load_data, predict_message, train_model


DATA_PATH = Path(__file__).parents[1] / "data" / "spam.csv"


def test_load_data_reads_messages_and_labels():
    messages, labels = load_data(DATA_PATH)

    assert len(messages) == len(labels)
    assert len(messages) > 5000
    assert set(labels.unique()) == {"ham", "spam"}


def test_model_predicts_known_examples():
    model = train_model(DATA_PATH)

    spam_label, spam_confidence = predict_message(
        model, "Congratulations! You have won a free prize. Call now!"
    )
    ham_label, ham_confidence = predict_message(model, "Are we still meeting for lunch today?")

    assert spam_label == "spam"
    assert ham_label == "ham"
    assert 0 <= spam_confidence <= 1
    assert 0 <= ham_confidence <= 1


def test_predict_message_rejects_empty_input():
    model = train_model(DATA_PATH)

    with pytest.raises(ValueError):
        predict_message(model, "   ")
