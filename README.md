
# AI-Powered Milk Adulteration Detection System

An AI Blend for Engineers project that uses machine learning to classify milk samples based on physicochemical parameters.

## Project Overview

The system uses a trained Random Forest model to classify milk samples into the categories supported by the model. It provides a Flask API for predictions and prediction history.

The results are preliminary ML classifications and are not a definitive food-safety determination.

## Team Members

- Nandana G
- Reeha Sachin
- Keerthana P R
- Amisha Devi U
- Devanarayanan P

## Technology Stack

- Python
- Flask
- scikit-learn
- Pandas and NumPy
- SQLite
- HTML, CSS and JavaScript

## Backend Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python app.py
```

The API runs locally on port 8000.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check application status |
| GET | `/features` | Retrieve model features and labels |
| POST | `/predict` | Predict a sample's class |
| GET | `/history` | Retrieve stored predictions when enabled |

## Model

The application uses the project's trained Random Forest model and its metadata files in the `models/` directory.

## Limitations

Predictions depend on the training data and model performance. The application is intended for preliminary classification, not certified milk-quality testing.