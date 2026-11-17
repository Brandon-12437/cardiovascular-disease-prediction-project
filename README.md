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


#### Docker Containerization

The project is containerized using Docker, which ensures a consistent environment across development, testing, and deployment. Docker allows the application to run with all dependencies pre-installed and isolates it from your system’s Python environment.

Steps to Run the App with Docker

Build the Docker Image
Build the project image from the Dockerfile:
                                      
              sudo docker build -t cardio-predictor .
Run the Docker Container
Start the container and expose port 9698:
                                               
              sudo docker run -p 9698:9698 cardio-predictor

 ### Test the Application
Send a request to the service using:

               python predict-test.py

<img width="1101" height="158" alt="Screenshot from 2025-11-18 00-40-09" src="https://github.com/user-attachments/assets/b34ccefe-5ea9-45ea-aff5-252623cf4253" />
<img width="919" height="124" alt="Screenshot from 2025-11-18 00-41-25" src="https://github.com/user-attachments/assets/cbcea4af-6b1b-49fc-88b6-3cb8597f3c39" />


### CLOUD DEPLOYMENT (Render)

This project is deployed on Render, allowing users to make real-time cardiovascular disease predictions through a publicly accessible API endpoint.

Render handles the infrastructure, builds the Docker image, and hosts the Flask API without requiring any manual server setup.
                                  
### Live API Endpoint

The model is live and accessible at: 

                 https://cardiovascular-disease-prediction-project.onrender.com

### Test the Deployed API

<img width="1469" height="349" alt="Screenshot from 2025-11-18 18-46-28" src="https://github.com/user-attachments/assets/432800c0-4a7c-4e37-89db-5659cc55d7bd" />

