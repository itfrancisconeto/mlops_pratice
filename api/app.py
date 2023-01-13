# Load libraries
import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, -1)
    model_predictor = pickle.load(open('predictor.pkl', 'rb'))
    result = model_predictor.predict(to_predict)
    return result[0]

@app.route("/")
def index():
  return render_template('index.html')

@app.route('/result/', methods=['POST'])
def result():
  try:
    if request.method == 'POST':
      to_predict_list = request.form.to_dict()
      to_predict_list = list(to_predict_list.values())
      result = ValuePredictor(to_predict_list)       
      if int(result)== 1:
          prediction ='Diabetes True'
      else:
          prediction ='Diabetes False'
  except Exception as e:
    prediction = 'System error! Predict fail' + str(e)
  return render_template('index.html', predict=prediction)

if __name__ == "__main__":
  app.run(debug=True,host='0.0.0.0',port=3000)
