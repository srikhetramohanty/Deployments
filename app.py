from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np
import os
# Initalise the Flask app
app = Flask(__name__)
os.chdir('D:\App_')
# Loads pre-trained model
model = pickle.load(open('saved_model_test.pkl','rb'))

@app.route('/')
def home():
    return render_template("home.html")
    
    
@app.route('/predict',methods=['POST'])
def predict():
    #int_features = [x for x in request.form.values()]
    #final = np.array(int_features)
    #prediction = np.max(final)
    #data_unseen = pd.DataFrame([final], columns = cols)
    #prediction = predict_model(model, data=data_unseen, round = 0)
    #prediction = int(prediction.Label[0])
    csv_file = request.files['data_file']
    data = pd.read_csv(csv_file)
    
    #return data.shape
    return render_template('home.html',pred='Shape of dataset {}'.format(data.shape), tables=[data.to_html(classes='data')], titles=data.columns.values)
    
    
# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data = request.get_json(force=True)
#     data_unseen = pd.DataFrame([data])
#     prediction = predict_model(model, data=data_unseen)
#     output = prediction.Label[0]
#     return jsonify(output)
    
if __name__ == '__main__':
    app.run(debug=True)
