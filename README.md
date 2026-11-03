# ScholarIQ

Predicting a student's math score from the story behind their numbers.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render&logoColor=black)](https://render.com/)

**[Live Demo](https://scholariq-rzps.onrender.com/)** · **[Project Structure](#project-structure)** · **[Run Locally](#getting-started)**

---

## About the Project

ScholarIQ is an end-to-end machine learning application that predicts a student's **mathematics score** from academic and demographic information such as reading and writing scores, parental education, lunch type, and test preparation.

This project started as a question: what does it take to turn a Jupyter Notebook into something people can actually use? The answer turned out to be much more than training a model. It meant building a modular pipeline, preprocessing data in a way that stays consistent between training and prediction, comparing multiple algorithms, saving artifacts, wiring everything into a Flask app, and finally getting it live on the internet.

> **Note:** ScholarIQ predicts scores based on statistical patterns in the data. It does not show that any demographic factor *causes* a student to perform better or worse.

---

## Features

- Score prediction from seven student features
- Reproducible pipeline with a fixed random seed (`random_state=42`)
- Automated preprocessing using Scikit-learn `Pipeline` and `ColumnTransformer`
- Eight regression algorithms compared, with hyperparameters tuned via `GridSearchCV`
- Saved artifacts (model and preprocessor) so the app never has to retrain on startup
- Web interface built with Flask, HTML, and CSS
- Logging and custom exception handling for easier debugging
- Live deployment on Render, served with Gunicorn

---

## How It Works

```text
Student Dataset (stud.csv)
        │
        ▼
Data Ingestion ──► Train / Test Split (75 / 25)
        │
        ▼
Data Transformation
   ├── Numerical:   Median Imputer → Standard Scaler
   └── Categorical: Mode Imputer → One-Hot Encoder → Scaler
        │
        ▼
Model Training ──► 8 Regressors + GridSearchCV (4-fold CV)
        │
        ▼
Best Model ──► artifact/model.pkl
Preprocessor ──► artifact/preprocessor.pkl
        │
        ▼
Prediction Pipeline ──► Flask App ──► Predicted Math Score
```

The key idea is that **every new input must go through the exact same transformations the training data went through.** That's why the fitted preprocessor is saved and reused at prediction time.

---

## Dataset

The project uses a student performance dataset with **1,000 records** and **8 columns**.

| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | Student's gender |
| `race_ethnicity` | Categorical | Student's race/ethnicity group |
| `parental_level_of_education` | Categorical | Highest education level of parents |
| `lunch` | Categorical | Standard or free/reduced lunch |
| `test_preparation_course` | Categorical | Whether test preparation was completed |
| `reading_score` | Numerical | Reading score |
| `writing_score` | Numerical | Writing score |
| **`math_score`** | **Target** | Mathematics score (0–100) |

The data is split into **750 training** and **250 testing** records.

---

## Model Training & Evaluation

I compared eight regression algorithms:

| Algorithm | Core Idea |
|---|---|
| Linear Regression | Linear relationships between features and target |
| Decision Tree | Decision-based splits |
| Random Forest | Many trees combined together |
| Gradient Boosting | Sequentially correcting previous errors |
| AdaBoost | Adaptive boosting of weak learners |
| KNN | Predicts from the nearest data points |
| XGBoost | Optimized gradient boosting |
| CatBoost | Gradient boosting with strong categorical support |

Each model is tuned with `GridSearchCV` using 4-fold cross-validation and evaluated with **R²**. The model with the highest test R² is saved as the final model. A minimum threshold of **R² ≥ 0.6** is enforced, so training fails loudly if no model is good enough.

---

## Project Structure

```text
scholariq/
│
├── app.py                         # Flask application and routes
├── requirements.txt               # Project dependencies
├── setup.py                       # Package configuration
│
├── artifact/                      # Generated files (model, preprocessor, splits)
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── raw.csv
│   ├── train.csv
│   └── test.csv
│
├── src/
│   ├── stud.csv                   # Original dataset
│   ├── EDA.ipynb                  # Exploratory analysis and experiments
│   ├── exception.py               # Custom exception handling
│   ├── logger.py                  # Timestamp-based logging
│   ├── utils.py                   # Save/load objects, evaluation, GridSearchCV
│   │
│   ├── components/
│   │   ├── data_ingestion.py      # Reads and splits the dataset
│   │   ├── data_transformation.py # Preprocessing pipelines
│   │   └── model_trainer.py       # Model comparison and selection
│   │
│   └── pipeline/
│       ├── train_pipeline.py      # Runs the training workflow
│       └── predict_pipeline.py    # Loads artifacts and predicts
│
└── templates/
    ├── home.html
    └── index.html
```

---

## Getting Started

### Prerequisites

- Python 3.x
- pip

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/scholariq.git
cd scholariq
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the model (optional, artifacts are included)

```bash
python src/pipeline/train_pipeline.py
```

### 4. Run the app

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

---

## Deployment

ScholarIQ is deployed on **Render**:

1. The repository is connected to GitHub.
2. Render installs dependencies from `requirements.txt`.
3. The app is served with **Gunicorn**.
4. Pushing changes to the connected branch triggers a redeploy.

> **Lesson learned:** The app initially failed in production because the model artifacts were never committed to GitHub. A project working locally isn't the same as a project working in production. Everything the deployed app needs has to be in the repository.

---

## What I Learned

- **Project structure matters.** Splitting ingestion, transformation, training, and prediction into separate components makes the project far easier to maintain than a single notebook.
- **Consistency is everything.** Preprocessing fitted on training data must be reused exactly during inference.
- **Model comparison beats model loyalty.** Testing eight algorithms showed that the "fanciest" model isn't always the one to ship.
- **Deployment teaches humility.** Debugging a live app that behaved differently from my local setup taught me more than any tutorial.

---

## Limitations & Future Work

This project works, but there are clear areas to grow:

- [ ] **Cleaner evaluation:** Select models using cross-validation or a separate validation set, and touch the test set only once for the final evaluation.
- [ ] **Input validation:** Stronger checks on form inputs.
- [ ] **Unit and integration tests** for each pipeline component.
- [ ] **Automated retraining** when new data arrives.
- [ ] **Monitoring** for prediction drift over time.
- [ ] **Dimensionality reduction** experiments (e.g., PCA) for comparison.

---

## Contributing

Suggestions, issues, and pull requests are welcome. If you have an idea to make ScholarIQ better, feel free to open an issue.

---

## Author

**Dhanraj**
BCA student exploring machine learning, web development, and the engineering behind both.

[GitHub](https://github.com/<your-username>) · [Live Demo](https://scholariq-rzps.onrender.com/)

---

## License

This project is open source. Add your preferred license here (e.g., MIT).
