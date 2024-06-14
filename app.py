import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify,render_template
from sqlalchemy import create_engine
import sqlite3
from datetime import datetime
import joblib

app = Flask(__name__)

# Load the machine learning model
with open("model.pkl", "rb") as f:
    saved_model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
# def predict():
#     # Get input data from the form
#     input_data = request.form['input_data']

#     # Make predictions using the model
#     prediction = model.predict([input_data])

#     return render_template('result.html', prediction=prediction)
def predict():
    cylinders = request.form.get("cylinders", None)
    displacement = request.form.get("displacement", None)
    horsepower = request.form.get("horsepower", None)
    weight = request.form.get("weight", None)
    acceleration = request.form.get("acceleration", None)
    model_year = request.form.get("model_year", None)
    origin = request.form.get("origin", None)
    # input_data = request.form['input_data']
    # if any(value is None for value in [cylinders, displacement, horsepower, weight, acceleration, model_year, origin]):
    #     return jsonify({'error': -999})

    # Convert the values to float and use -999 if not provided
    # data = [
    #     float(cylinders) if cylinders is not None else -999,
    #     float(displacement) if displacement is not None else -999,
    #     float(horsepower) if horsepower is not None else -999,
    #     float(weight) if weight is not None else -999,
    #     float(acceleration) if acceleration is not None else -999,
    #     float(model_year) if model_year is not None else -999,
    #     float(origin) if origin is not None else -999]
    data = [cylinders, displacement, horsepower, weight,
            acceleration, model_year, origin]
    
    if None in data:
        return str(-999)
    else:
    #     # pred_df = pd.DataFrame(np.array(data).reshape(1,-1), 
    #     #                        columns=saved_model.feature_names_in_)
        
    #     # inputs = data
    #     # outputs = saved_model.predict(pred_df)[0]
    #     # date = str(datetime.now())[0:19]
    #     # log_df = pd.DataFrame({"inputs":[inputs], 
    #     #                        "outputs": [outputs], 
    #     #                        "date": [date]})
    #     # log_df.to_sql("logs", con=engine, if_exists="append", index=None)
        prediction = saved_model.predict([data])[0]

        return render_template("result.html",prediction=prediction)
        # return 

if __name__ == '__main__':
    app.run(debug=True)