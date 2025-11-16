# %% [markdown]
#                                    CARDIOVASCULAR DISEASE PREDICTION
# PROBLEM DESCRIPTION AND HOW MACHINE LEARNING CAN HELP
# 
# Cardiovascular diseases (CVDs) are among the leading causes of illness and death worldwide. Many of these conditions develop gradually and can be prevented if the risk is identified early. However, traditional risk assessment methods can be time-consuming and may rely heavily on a doctor's experience and manual evaluation of multiple health indicators such as age, blood pressure, cholesterol level, and lifestyle habits.
# 
# 
# Machine Learning can analyze large amounts of patient health data and automatically learn patterns that indicate whether a person is at risk of developing cardiovascular disease. By training a model on past medical records, it can:
# 
# Predict risk levels accurately based on key health indicators.
# 
# Support doctors in making faster and more informed decisions.
# 
# Identify hidden patterns that may not be obvious in manual analysis.
# 
# Enable early intervention, which can improve patient outcomes and reduce healthcare costs.

# %%
import xgboost as xgb

# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
import seaborn as sns
import sklearn
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.ensemble  import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import sys


# %%
# System versions
print("Platform:", sys.platform)
print("Python version:", sys.version)
print("---" * 47)

# Libraries versions
print("matplotlib version:", matplotlib.__version__)
print("seaborn version:", sns.__version__)
print("xgboost version:", xgb.__version__)
print("sklearn version:", sklearn.__version__)
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)


# %% [markdown]
#                                     DATA PREPARATION AND EXPLORATORY DATA ANALYSIS

# %%
df = pd.read_csv('cardiovascular_disease_cleaned(1).csv')
df.head()

# %%
df.isnull().sum()


# %%
#CHECKING FOR MISSING VALUES IN THE DATASET
df.dataset.unique()

# %% [markdown]
# | Column       | Full Name / Meaning                                                                                       |
# | ------------ | --------------------------------------------------------------------------------------------------------- |
# | **id**       | Patient ID (Identification Number)                                                                        |
# | **age**      | Age of the patient (years)                                                                                |
# | **sex**      | Sex of the patient (1 = male, 0 = female)                                                                 |
# | **dataset**  | Dataset source identifier (used when multiple datasets are merged)                                        |
# | **cp**       | Chest Pain Type (4 categories: typical angina, atypical angina, non-anginal pain, asymptomatic)           |
# | **trestbps** | Resting Blood Pressure (mm Hg) measured upon hospital admission                                           |
# | **chol**     | Serum Cholesterol level (mg/dl)                                                                           |
# | **fbs**      | Fasting Blood Sugar (>120 mg/dl → 1 = true, 0 = false)                                                    |
# | **restecg**  | Resting Electrocardiographic Results (0, 1, 2 categories)                                                 |
# | **thalch**   | Maximum Heart Rate Achieved during exercise                                                               |
# | **exang**    | Exercise-Induced Angina (1 = yes, 0 = no)                                                                 |
# | **oldpeak**  | ST Depression induced by exercise relative to rest                                                        |
# | **slope**    | Slope of the Peak Exercise ST segment (0 = up, 1 = flat, 2 = down)                                        |
# | **ca**       | Number of Major Vessels Colored by Fluoroscopy (0–3)                                                      |
# | **thal**     | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)                                         |
# | **num**      | Target variable: Heart disease status (0–4 in raw data, usually converted to 0 = no disease, 1 = disease) |
# 

# %%
df.num = df.num.astype('object')
df.dtypes

# %%
df.describe().round(3)

# %%
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df.num = df.num.astype('object')
df.dtypes

# %%
df.num.max()


# %%
numerical = list(df.dtypes[(df.dtypes=='int64') | (df.dtypes=='float64')].index)
categorical = list(df.dtypes[df.dtypes=='object'].index)
numerical, categorical

# %%
df['sex'] = df['sex'].replace({'Male':1,'Female':0})
df['sex'].value_counts()

# %% [markdown]
#                   VISUALIZATION OF PATTERNS

# %%
sns.countplot(x='num', data=df)
plt.title('Distribution of Cardiovascular Disease')
plt.xlabel('Cardiovascular Disease')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize = (9,8))
plt.title("Heart Data Correlation Heatmap")
sns.heatmap(df[numerical].corr(), annot = True, linewidths = .1);

# %%
df[numerical].hist(figsize = (13, 13))

# %%
# VISUALIZE CONTINOUS VARIABLE VS TARGET VARIABLE
sns.boxplot(x='num', y='age', data=df)
plt.title('Age vs Cardiovascular Disease')
plt.xlabel('Cardiovascular Disease (0-4)')
plt.ylabel('Age')
plt.show()

# %%
sns.pairplot(df, hue='num', vars=['age', 'trestbps', 'chol', 'thalch', 'oldpeak'])
plt.suptitle('Pairplot of Numerical Features Colored by Cardiovascular Disease')
plt.show()

# %%
df.num = df.num.astype('int')

# %% [markdown]
#      SPLITTING THE DATA INTO TRAIN /VALIDATION/ TEST-SPLIT
# 
#      We will split it into 60% 20% 20% distribution

# %%
 #splitting the dataset into train,val and test sets
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

# %%
#check fo the length of the datasets
len(df_train), len(df_val), len(df_test)

# %%
df_train.reset_index(drop=True)
df_val.reset_index(drop=True)
df_test.reset_index(drop=True)    

# %%
y_train = df_train.num.values
y_val = df_val.num.values
y_test = df_test.num.values

# %%
# Drop `target` from our data sets
del df_train["num"]
del df_test["num"]
del df_val["num"]


# %% [markdown]
# PERFORMING ONE HOT ENCODING BEFORE WE TRAIN OUR DATA

# %%
train_dicts = df_train.to_dict(orient = 'records')
dv= DictVectorizer(sparse = False) 
X_train = dv.fit_transform(train_dicts)

# %%
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model

# %%
model.fit(X_train,y_train)

# %%
val_dicts = df_val.to_dict(orient = 'records')

X_val = dv.transform(val_dicts)

# %%
test_dicts = df_test.to_dict(orient = 'records')
X_test = dv.transform(test_dicts)

# %% [markdown]
#           TRAIN SEVERAL MODELS AND THEN DO FINE TUNING

# %% [markdown]
# TRAINING A LOGISTIC REGRESSION MODEL WITH SCIKIT-LEARN

# %%
model.coef_

# %%
model.coef_[0].round(3)

# %%
model.intercept_

# %%
model.intercept_[0]

# %%
model.fit(X_train,y_train)


# %%
model.predict_proba(X_test).round(2)

# %%
y_pred = model.predict(X_test)
y_pred


# %%
reg_params = [0.01, 0.1, 1, 2, 10, 100]
reg_params_scores = []
for param in reg_params:
    model = LogisticRegression(solver = 'liblinear', C = param,max_iter = 1000, random_state = 42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)    
    param_score = 100 * (y_pred == y_val).mean()

    reg_params_scores += [round(param_score, 3)]
param_scores = pd.Series(reg_params_scores, index = reg_params, name = "parameters_scores")
param_scores


# %%
#LogisticRegression MODEL HAS AN ACCURACY OF 87.5%

# %%
param_scores.plot(marker='o')
plt.xlabel("C (Regularization Strength)")
plt.ylabel("Validation Accuracy (%)")
plt.title("C parameter tuning for Logistic Regression")
plt.show()

# %%
X = df.drop('num', axis=1)  # features
y = df['num']               # target


# %%
print(X.shape)
print(y.value_counts())



# %% [markdown]
# TRAINING A DECISION TREEE CLASSIFFIER

# %%


depths = [1, 2, 3, 4, 5, 6, 10, 15, 20, None]
 
for depth in depths: 
    dt = DecisionTreeClassifier(max_depth=depth, random_state =42)
    dt.fit(X_train, y_train)
     
    # remember we need the column with negative scores
    y_pred = dt.predict(X_val)
    acc=100 *(y_pred ==y_val).mean()
     
    print('%4s -> %.3f' % (depth, acc))

# %%
from sklearn.metrics import roc_auc_score


scores = []
 
for d in [4, 5, 6]:
    for s in [1, 2, 5, 10, 15, 20, 100, 200, 500]:
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=s)
        dt.fit(X_train, y_train)
 
        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
         
        scores.append((d, s, auc))
 
columns = ['max_depth', 'min_samples_leaf', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)
df_scores.head()

# %%

# index - rows
df_scores_pivot = df_scores.pivot(index='min_samples_leaf', columns=['max_depth'], values=['auc'])
df_scores_pivot.round(3)

# %%
	
sns.heatmap(df_scores_pivot, annot=True, fmt=".3f")

# %%
# IT has min_sample_leaf  = 10-20  max_depth 6  and accuracy of 94.8%

# %% [markdown]
# RANDOM FOREST CLASSIFIER

# %%
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)
 
y_pred = rf.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)
 
rf.predict_proba(X_val[[0]])

# %%
scores = []
 
for n in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=n, random_state=1)
    rf.fit(X_train, y_train)
 
    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
     
    scores.append((n, auc))
 
df_scores = pd.DataFrame(scores, columns=['n_estimators', 'auc'])
df_scores

# %%
plt.plot(df_scores.n_estimators, df_scores.auc)

# %%
scores = []

for d in [5, 10, 15]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=d,
                                    random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((d, n, auc))

# %%
columns = ['max_depth', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)
df_scores.groupby("max_depth")["auc"].mean().round(4)

# %%
for d in [5, 10, 15]:
    df_subset = df_scores[df_scores.max_depth == d]
    
    plt.plot(df_subset.n_estimators, df_subset.auc,
             label='max_depth=%d' % d)

plt.legend()


# %%
max_depth = 15

# %%
scores = []

for s in [1, 3, 5, 10, 50]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=max_depth,
                                    min_samples_leaf=s,
                                    random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((s, n, auc))


# %%
columns = ['min_samples_leaf', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

# %% [markdown]
# 

# %%
colors = ['black', 'blue', 'orange', 'red', 'grey']
values = [1, 3, 5, 10, 50]

for s, col in zip(values, colors):
    df_subset = df_scores[df_scores.min_samples_leaf == s]
    
    plt.plot(df_subset.n_estimators, df_subset.auc,
             color=col,
             label='min_samples_leaf=%d' % s)

plt.legend()

# %%
min_samples_leaf = 1

# %%
f = RandomForestClassifier(n_estimators=200,
                            max_depth=max_depth,
                            min_samples_leaf= min_samples_leaf,
                            random_state=42)
rf.fit(X_train, y_train)


# %%
#    RANDOFORESTCLASSIFIER  HAS AN ACCURACY OF   96.5201%

# %% [markdown]
# XGBoost Classifier

# %%
import xgboost as xgb
print(xgb.__version__)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

watchlist = [(dtrain, "train"), (dval, "eval")]



# %%
xgb_params = {
    'eta' :0.3,
    'max_depth' :6,
    'min_child_weight' :1,
    'objective' : 'binary:logistic',
    'nthread' :8,

    'seed' :1,
    'verbosity' :1,
}

model = xgb.train(xgb_params,dtrain, num_boost_round=10)

# %%
y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)

# %%
#performance monitoring
watchlist = [(dtrain, 'train'), (dval, 'val')]

# %%
xgb_params = {
    'eta' :0.3,
    'max_depth' :6,
    'min_child_weight' :1,
    'objective' : 'binary:logistic',
    'nthread' :8,
    'seed' :1,
    'verbosity' :1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=200, evals=watchlist)

# %%
y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)

# %%
#XGBOOST HAS AN AACURACY OF 95.6516%

# %% [markdown]
# PICKING THE BEST MODEL
# 
# We will pick  RANDOM FOREST CLASSIFIER because it has an accuracy of  96.5201%

# %% [markdown]
# SAVING THE MODEL

# %%
import pickle


# %%
output_file = 'Random_Forest_Model.bin'
output_file

# %%
f_out = open(output_file, 'wb')
 
pickle.dump((dv, rf), f_out)
 
f_out.close()

# %%
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, rf), f_out)


# %%
input_file = 'Random_Forest_Model.bin'


with open(input_file,'rb') as f_in:
    
    dv, rf = pickle.load(f_in)

rf


# %%
# patient dictionary
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

# Convert to DataFrame
patient_df = pd.DataFrame([patient])

# Transform features using the loaded encoder
X_patient = dv.transform(patient_df.to_dict(orient='records'))

# Make prediction
pred_class = rf.predict(X_patient)[0]
pred_prob = rf.predict_proba(X_patient)[0, 1]  # probability of positive class

# Display input and results
print("Patient information:")
print(patient_df)

print(f"Predicted class: {pred_class}")
print(f"Predicted probability of disease: {pred_prob:.2f}")

# Define treatment if necessary
if pred_class == 1:
    print("Treatment plan should be considered for this patient.")



