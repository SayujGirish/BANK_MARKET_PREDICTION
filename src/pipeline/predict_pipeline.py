import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class predictPipline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(
        self,age,job,marital,education,default,balance,housing,loan,contact,day,
        month,duration,campaign,pdays,previous,poutcome):
        self.age = age
        self.job = job
        self.marital = marital
        self.education = education
        self.default = default
        self.balance = balance
        self.housing = housing
        self.loan = loan
        self.contact = contact
        self.day = day
        self.month = month
        self.duration = duration
        self.campaign = campaign
        self.pdays = pdays
        self.previous = previous
        self.poutcome = poutcome

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
            "age": [self.age],
            "job": [self.job],
            "marital": [self.marital],
            "education": [self.education],
            "default": [self.default],
            "balance": [self.balance],
            "housing": [self.housing],
            "loan": [self.loan],
            "contact": [self.contact],
            "day": [self.day],
            "month": [self.month],
            "duration": [self.duration],
            "campaign": [self.campaign],
            "pdays": [self.pdays],
            "previous": [self.previous],
            "poutcome": [self.poutcome]
        }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


    
