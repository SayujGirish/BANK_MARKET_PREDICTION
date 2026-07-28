import pickle
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,predictPipline

application=Flask(__name__)
app=application

#Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
        age=request.form.get("age"),
        job=request.form.get("job"),
        marital=request.form.get("marital"),
        education=request.form.get("education"),
        default=request.form.get("default"),
        balance=request.form.get("balance"),
        housing=request.form.get("housing"),
        loan=request.form.get("loan"),
        contact=request.form.get("contact"),
        day=request.form.get("day"),
        month=request.form.get("month"),
        duration=request.form.get("duration"),
        campaign=request.form.get("campaign"),
        pdays=request.form.get("pdays"),
        previous=request.form.get("previous"),
        poutcome=request.form.get("poutcome")
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=predictPipline()
        results=predict_pipeline.predict(pred_df)
        prediction = "YES" if results[0] == 1 else "NO"
        return render_template('home.html',results=prediction)

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
    