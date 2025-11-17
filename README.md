
                                                                      
🫀 CARDIOVASCULAR DISEASE PREDICTION SYSTEM


 🎯 EXECUTIVE SUMMARY
This project delivers a production-ready cardiovascular disease prediction system that achieves 96.5% AUC on test data. The end-to-end solution includes:

~Machine Learning Pipeline: 4 models trained and optimized
~REST API: Flask-based service with JSON I/O
~Containerization: Docker for reproducible deployments
`Cloud Hosting: Deployed on Railway with CI/CD
~Scalability: Gunicorn WSGI server with 4 workers

📊 DATASET & FEATURE ENGINEERING
Data Sources
Primary Dataset: UCI Machine Learning Repository - Heart Disease Dataset
Combined Sources:
~Cleveland
~Hungary
~Switzerland
~VA Long Beach

DATASET STATISTICS
Total Patients: 
Features: 16 (15 predictors + 1 target)
Missing Values: 0 (pre-cleaned)
Class Distribution:
  - No Disease (0):411 patients (55.277%)
  - Disease (1): 508 patients (44.722%)
  
Split Ratio:
  - Training: 551 samples (60%)
  - Validation:184samples (20%)
  - Test: 184 samples (20%)

    
                             FEATURE DICTIONARY
DEMOGRAPHIC FEATURES
| Feature                  | Type       | Description            | Clinical Significance                 |
|--------------------------|-----------|-----------------------|--------------------------------------|
| age                      | Continuous| Patient age (29-77 years) | Risk increases with age; >65 high risk |
| sex                      | Binary    | 0 = Female, 1 = Male  | Males 2x higher CVD risk             |

CLINICAL MEASUREMENTS
| Feature    | Type       | Description                  | Normal Range        | Risk Threshold            |
|-----------|------------|------------------------------|------------------|--------------------------|
| trestbps  | Continuous | Resting blood pressure (mm Hg)| 90-120           | >140 (hypertension)      |
| chol      | Continuous | Serum cholesterol (mg/dl)     | <200             | >240 (high risk)          |
| thalch    | Continuous | Max heart rate achieved       | Age-dependent    | <(220-age) × 0.85         |
| oldpeak   | Continuous | ST depression (mm)            | 0                | >2.0 (ischemia)           |

DIAGNOSTIC TESTS

| Feature   | Type       | Values          | Description                                                                 |
|-----------|------------|----------------|----------------------------------------------------------------------------|
| cp        | Categorical| 0-3            | Chest pain type: 0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic |
| fbs       | Binary     | 0/1            | Fasting blood sugar >120 mg/dl (1=true)                                    |
| restecg   | Categorical| 0-2            | Resting ECG: 0=normal, 1=ST-T abnormality, 2=LV hypertrophy                 |
| exang     | Binary     | 0/1            | Exercise-induced angina (1=yes)                                            |
| slope     | Categorical| 0-2            | Peak exercise ST slope: 0=upsloping, 1=flat, 2=downsloping                  |
| ca        | Categorical| 0-3            | Number of major vessels colored by fluoroscopy                              |
| thal      | Categorical| 3,6,7          | 3=normal, 6=fixed defect, 7=reversible defect                               |

TARGET VARIABLE
| Feature | Type   | Description              | Distribution       |
|---------|--------|--------------------------|------------------|
| num     | Binary | 0=No disease, 1=Disease | 55.277% / 44.722%    |

⚙️INSTALLATION AND SETUP

~Requirements: Python 3.11, Conda, Ubuntu or Linux terminal

~Clone the repository 
                                 
                                 https://github.com/Brandon-12437/cardiovascular-disease-prediction-project.git

CREATE A CONDA ENVIRONMENT

                                 conda create -n cardio-env python=3.11 -y
                                 conda activate cardio-env

INSTALL DEPENDENCIES
             
                                conda install flask pandas numpy scikit-learn=1.5.1 -y
                                 pip install xgboost gunicorn

VERIFY INSTALLATION

                              conda install flask pandas numpy scikit-learn=1.5.1 -y
                                                                              
                               pip install xgboost gunicorn



📊DATA PREPARATION

1. Dataset: cardiovascular_disease_cleaned(1).csv

2. Target variable: num (0 = no disease, 1 = disease)

3. Features include age, sex, chest pain type, blood pressure, cholesterol, etc.

Preprocessing steps:

a. Convert categorical variables to numeric

b. One-hot encode categorical features

c. Split data into train (60%), validation (20%), test (20%)


🏋️‍♂️MODEL TRAINING

~Logistic Regression, Decision Tree, Random Forest, and XGBoost were trained.

~Hyperparameters were tuned using validation set.

~Best performing model: Random Forest

~Model saved as Random_Forest_Model.bin using pickle.

i. 🏋️‍♂️ TRAINING THE MODEL

Train the model and save it as a binary file:

                                python train.py

TESTING THE MODEL
                                 
                                Python predict-test.py 

<img width="1682" height="243" alt="Screenshot from 2025-11-17 21-40-50" src="https://github.com/user-attachments/assets/180e418c-00a3-4a5e-b04f-ae4e56353e6c" />


🚀 RUNNING THE WEB SERVICE

Run Flask API locally:
<img width="1259" height="104" alt="Screenshot from 2025-11-17 22-55-17" src="https://github.com/user-attachments/assets/50ee563c-714a-4944-a573-c51409de5803" />
<img width="931" height="193" alt="Screenshot from 2025-11-17 22-58-35" src="https://github.com/user-attachments/assets/7854d516-ca57-442b-b0a4-fa148b14ead7" />


                                 
                                              
                                
 
