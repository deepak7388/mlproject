## End to End Machine Learning Project

# 📊 Student Performance Analysis

This project predicts a student's **math score** based on attributes like gender, ethnicity, parental education level, lunch type, test preparation, reading score, and writing score using Machine Learning. The application is served through a Flask web interface.

---

## ✅ 1. Problem Statement

The goal of this project is to understand how a student’s performance (test scores) is affected by different features:

- Gender
- Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course

The model aims to predict **math score** using the above features.

---

## 📁 2. Data Collection

- **Dataset Source:** [Kaggle - Student Performance Dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977)
- **Shape:** 1000 rows × 8 columns

**Columns include:**

- `gender`
- `race/ethnicity`
- `parental level of education`
- `lunch`
- `test preparation course`
- `math score` _(target)_
- `reading score`
- `writing score`

---

## ⚙️ 3. Tech Stack

| Purpose              | Tools & Libraries                        |
| -------------------- | ---------------------------------------- |
| Language             | Python                                   |
| ML Libraries         | scikit-learn, pandas, numpy              |
| Additional ML Models | XGBoost, CatBoost, AdaBoost              |
| Web Framework        | Flask                                    |
| Frontend             | HTML, CSS, Bootstrap 5                   |
| Deployment           | Localhost (can be extended to cloud)     |
| Serialization        | Pickle (`model.pkl`, `preprocessor.pkl`) |

---

## 🧱 4. Project Structure

student-performance-analysis/
│
├── app.py # Main Flask web app
├── requirements.txt # List of Python dependencies
├── README.md # Project documentation
│
├── templates/
│ └── home.html # HTML form for user input
│
├── artifacts/
│ ├── model.pkl # Trained ML model
│ └── preprocessor.pkl # Fitted preprocessor (encoders, scalers)
│
├── src/
│ ├── init.py
│ ├── logger.py # Logging functionality
│ ├── exception.py # Custom exception handler
│ ├── utils.py # Save/load object functions, model evaluation
│
│ └── pipeline/
│ ├── predict_pipeline.py # Code for prediction using trained model
│ └── train_pipeline.py # Code to train and save model and preprocessor

yaml
Copy
Edit

---

## 🚀 5. Setup Instructions

### Step 1: Clone the Repository

git clone https://github.com/your-username/student-performance-analysis.git
cd student-performance-analysis

### Step 2: Create Virtual Environment

bash
Copy
Edit
python -m venv venv

# For Windows:

venv\Scripts\activate

# For macOS/Linux:

source venv/bin/activate

### Step 3: Install Required Libraries

bash
Copy
Edit
pip install -r requirements.txt

## 🧠 6. Model Training

### Step 1: Download Dataset

Download the dataset from Kaggle and place it at an appropriate location like notebooks/data/StudentsPerformance.csv.

### Step 2: Train the Model

In your training pipeline (e.g., train_pipeline.py), make sure the following steps are covered:

Load dataset

Preprocess: handle categorical and numerical features

Train models: LinearRegression, RandomForest, XGBoost, CatBoost, etc.

Evaluate all models using R² Score

Save the best model and preprocessor using pickle

To start training:

bash
Copy
Edit
python src/pipeline/train_pipeline.py
After successful training, this generates:

artifacts/model.pkl

artifacts/preprocessor.pkl

These will be used for prediction.

## 🔮 7. Prediction Pipeline

📄 File: predict_pipeline.py
Handles:

Loading saved model.pkl and preprocessor.pkl

Accepting features from the user

Transforming and scaling input data

Predicting math score using the model

📄 File: app.py
Flask application entry point.

Defines two routes:

/ (GET): Renders the HTML form

/ (POST): Accepts form data, processes it using PredictPipeline, and returns the prediction.

bash
Copy
Edit
python app.py
Navigate to:

cpp
Copy
Edit
http://127.0.0.1:5000/

### 🖼️ 8. Frontend Form - home.html

A styled Bootstrap-based HTML form for user input

User fills out:

Gender

Ethnicity

Parental education level

Lunch type

Test preparation course

Reading and Writing scores

Submitting the form sends a POST request to / and shows the predicted math score on the same page

Example Form Inputs:

html
Copy
Edit

<form action="/" method="post">
    <input name="gender" ... />
    <input name="writing_score" ... />
    ...
</form>

## 🧪 9. Evaluation Metrics

R² Score is used to evaluate and select the best model.

If no model achieves a decent threshold (e.g., 0.6), training fails.

## 📷 10. Sample Output

![alt text](<Screenshot 2025-06-16 150219.png>)
![alt text](<Screenshot 2025-06-16 150243.png>)

## 📌 11. Features to Extend

Add Docker support

Deploy on Render or AWS EC2

Use MongoDB or SQLite to log predictions

Add CI/CD pipeline with GitHub Actions

Track model versions with MLflow

## 📃 12. requirements.txt (Sample)

txt
Copy
Edit
pandas
numpy
scikit-learn
xgboost
catboost
flask
Adjust as per your final environment

## 👨‍💻 13. Author

Chandra Prakash
This project was built as part of a self-initiated ML deployment practice.

## 📬 Contact

Email: deepak79232@gmail.com
GitHub: https://github.com/deepak7388

### 🏁 License

This project is open-source and available under the MIT License.
