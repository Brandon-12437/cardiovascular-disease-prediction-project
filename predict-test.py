
import requests
url ="http://localhost:9698/predict"
patient = {'age': 67,
           'sex': 'male',
           'cp': 'asymptomatic',
           'trestbps': 145.0,
           'chol': 233.0,
           'fbs': 'low_fbs',
           'restecg': 'lv hypertrophy',
           'thalachh': 112.0,
           'exang': 'no',
           'oldpeak': 2.7,
           'slope': 'upsloping',
           'ca': 'no_vessel',
           'thal': 'reversal_defect'}


response = requests.post(url, json=patient).json()
print(response)
 
if response['num'] == True:
    print('Treatment plan should be considered for this patient')
else:
    print('patient is not at risk of cardiovascular disease, a treatment plan is not required.')
    