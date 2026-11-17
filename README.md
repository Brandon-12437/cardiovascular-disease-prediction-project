## CARDIOVASCULAR DISEASE PREDICTION

## Problem Description

Cardiovascular diseases are among the major causes of illness and death worldwide. Early identification of potential cardiovascular disease risk is important because it can support timely medical assessment and intervention.

Identifying patterns in patient demographic information, clinical measurements, and diagnostic indicators can help explore the relationship between these factors and cardiovascular disease.
The Solution
The Cardiovascular Disease Prediction System addresses this problem by applying machine learning classification algorithms to estimate the likelihood of cardiovascular disease using patient-related data.

The system explores four classification algorithms: Logistic Regression, Decision Tree, Random Forest, and XGBoost.

The Random Forest Classifier was selected for the prediction application. The trained model and its associated DictVectorizer are serialized together using Python's Pickle module.

The model is exposed through a Flask REST API, allowing applications to submit patient information in JSON format and receive a prediction probability and classification result.
Exploratory Data Analysis
Exploratory Data Analysis (EDA) was performed to understand the structure of the cardiovascular disease dataset, examine the distribution of patient characteristics, and investigate relationships between clinical features and the target variable.

The dataset contains 919 records and 16 columns, consisting of 15 input features and the target variable num.
EDA Steps
The analysis included:
- Inspecting the shape and structure of the dataset.
- Checking for missing values.
- Examining duplicate records.
- Inspecting data types and feature distributions.
- Calculating descriptive statistics.
- Analyzing the distribution of the target variable.
- Examining relationships between patient characteristics and cardiovascular disease.
- Visualizing numerical feature distributions.
- Exploring correlations between clinical variables.
### Key Observations
The dataset contains demographic, clinical, and diagnostic information, including age, sex, resting blood pressure, cholesterol levels, maximum heart rate, exercise-induced angina, and other diagnostic indicators.

The original target variable, num, represents different levels of cardiovascular disease presence. It was transformed into a binary classification target to distinguish between patients with and without cardiovascular disease.

These findings guided the preprocessing and feature transformation stages before model development.
## DATA VISUALIZATION
<img width="700" alt="Heart Data Correlation Heatmap" src="screenshots/Screenshot%20from%202026-11-17%2010-41-09.png" />

<img width="700" alt="Cardiovascular Disease Target Distribution" src="screenshots/Screenshot%20from%202026-11-17%2010-41-23.png" />

### INSTALLATION AND SETUP

Requirements: Python 3.11, Conda, Ubuntu or Linux terminal
Clone the repository 
                                 
                                 https://github.com/Brandon-12437/cardiovascular-disease-prediction-project.git

CREATE A CONDA ENVIRONMENT

                                 conda create -n cardio-env python=3.11 -y
                                 conda activate cardio-env

INSTALL DEPENDENCIES
             
                                conda install flask pandas numpy scikit-learn=1.5.1 -y
                                 pip install xgboost gunicorn


### DATA PREPARATION

1. Dataset: cardiovascular_disease_cleaned(1).csv

2. Target variable: num (0 = no disease, 1 = disease)

3. Features include age, sex, chest pain type, blood pressure, cholesterol, etc.

Preprocessing steps:

a. Convert categorical variables to numeric

b. One-hot encode categorical features

c. Split data into train (60%), validation (20%), test (20%)


### MODEL TRAINING

~Logistic Regression, Decision Tree, Random Forest, and XGBoost were trained.

~Hyperparameters were tuned using validation set.

~Best performing model: Random Forest

~Model saved as Random_Forest_Model.bin using pickle.

### TRAINING THE MODEL

Train the model and save it as a binary file:

                                python train.py

### TESTING THE MODEL
                                 
                                Python predict-test.py 

<img width="1682" height="243" alt="Screenshot from 2025-11-17 21-40-50" src="https://github.com/user-attachments/assets/180e418c-00a3-4a5e-b04f-ae4e56353e6c" />


### RUNNING THE WEB SERVICE

Run Flask API locally:
<img width="1259" height="104" alt="Screenshot from 2025-11-17 22-55-17" src="https://github.com/user-attachments/assets/50ee563c-714a-4944-a573-c51409de5803" />
<img width="931" height="193" alt="Screenshot from 2025-11-17 22-58-35" src="https://github.com/user-attachments/assets/7854d516-ca57-442b-b0a4-fa148b14ead7" />

## CONTAINERIZATION

The Cardiovascular Disease Prediction System is containerized using Docker to provide a consistent and reproducible environment for running the Flask API.

The Docker image packages the application, trained Random Forest model, Python dependencies, and API configuration into a single deployable container.

### Docker Configuration

The application uses a Python 3.12-slim base image.

The Docker environment installs the required dependencies specified in `requirements.txt`, including Flask, NumPy, and Scikit-learn.

The application is served using Gunicorn as the production WSGI server.

### Building the Docker Image

The Docker image was built locally using the following command:

                        docker build -t cardiovascular-predictor .
                        
### Running the Docker Container

The container was started with port mapping to expose the Flask API on port 10000.

       docker run --name cardiovascular-test --rm \
       -p 10000:10000 \
       -e PORT=10000 \
       cardiovascular-predictor

Gunicorn starts the Flask application and listens on 0.0.0.0:10000.

<img width="800" alt="Docker Container Running" src="screenshots/docker-container-running.png" />Testing the API Locally
The containerized API was tested locally to verify that the application starts correctly and can process prediction requests.

API Health Check
A GET request was sent to the root endpoint:

                curl http://localhost:10000/

The API returned the following response:

          {
              "message": "Cardiovascular Disease Prediction API is running.",  "usage": "Send POST request to /predict with patient data."
          }


A POST request containing sample patient information was submitted to the /predict endpoint.

                   curl -X POST http://localhost:10000/predict \
                    -H "Content-Type: application/json" \
                   -d '{
                          "age": 55,
                          "sex": 1,
                          "cp": "typical angina",
                          "trestbps": 140,
                          "chol": 250,
                          "fbs": 0,
                          "restecg": "normal",
                          "thalch": 150,
                          "exang": 0,
                          "oldpeak": 1.0,
                          "slope": "flat",
                          "ca": 0,
                          "thal": "normal",
                          "dataset": "Cleveland",
                          "id": 1
                       }'

The API successfully processed the request and returned:

    {
        "Cardiovascular_Disease_Prediction_probability": 0.3242835286690758,
        "num": false
     }

This confirms that the serialized Random Forest model was successfully loaded and used to generate a prediction through the Flask API.
The local Docker testing confirmed that the container builds successfully, the API starts correctly, and the prediction endpoint processes sample requests.                  

### CLOUD DEPLOYMENT (Render)

This project is deployed on Render, allowing users to make real-time cardiovascular disease predictions through a publicly accessible API endpoint.

Render handles the infrastructure, builds the Docker image, and hosts the Flask API without requiring any manual server setup.
                                  
### Live API Endpoint

The model is live and accessible at: 

                 https://cardiovascular-disease-prediction-project.onrender.com

### Test the Deployed API

<img width="1469" height="349" alt="Screenshot from 2025-11-18 18-46-28" src="https://github.com/user-attachments/assets/432800c0-4a7c-4e37-89db-5659cc55d7bd" />

