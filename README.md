# SMS Spam Classifier

An SMS spam classification web application using TF-IDF and Multinomial Naive Bayes.

## Project structure

```text
sms-spam-classifier/
├── app.py              # Streamlit application
├── model.py            # Data loading, training, and predictions
├── requirements.txt    # Python dependencies
├── data/
│   └── spam.csv        # Local SMS dataset
├── assets/             # Application assets
└── tests/              # Automated tests
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app.py
```

The application trains the model from `data/spam.csv` when it starts. The dataset must contain the label in its first column and the message in its second column.

## Run tests

```bash
python -m pytest
```
