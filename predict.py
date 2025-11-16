import pickle
from flask import Flask
from flask import request
from flask import jsonify
 
model_file = 'Random_Forest_Model.bin'
 
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)
 
app = Flask('Cardiovascular_Disease_Prediction')
 
@app.route('/predict', methods=['POST'])
def predict():
    # json = Python dictionary
    patient = request.get_json()
 
    X = dv.transform([patient])
    model.predict_proba(X)
    y_pred = model.predict_proba(X)[0,1] 
    num = y_pred >= 0.5
 
    result = {
        'Cardiovascular_Disease_Prediction_probability': float(y_pred),
        'num': bool(num)
    }
 
    return jsonify(result) 
 
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9698)