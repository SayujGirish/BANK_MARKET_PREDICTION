import sys
import os
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models

# Evaluation Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

#preprocessing
from sklearn.preprocessing import LabelEncoder

#Models
from sklearn.linear_model import LogisticRegression,LassoCV,RidgeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Support Vector Classifier": SVC(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Gaussian Naive Bayes": GaussianNB(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "XGBoost Classifier": XGBClassifier()
            }
            LabelEncoder_y = LabelEncoder()
            y_train = LabelEncoder_y.fit_transform(y_train)
            y_test = LabelEncoder_y.transform(y_test)
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            #to get best model score and name from dict
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found, Model Name: {best_model_name}, Accuracy Score: {best_model_score}")
            


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            confusion=confusion_matrix(y_test, predicted)
            recall=recall_score(y_test, predicted)
            precision=precision_score(y_test, predicted)
            f1=f1_score(y_test, predicted)
            roc_auc=roc_auc_score(y_test, predicted)
            return { "Model Name": best_model_name, "Accuracy Score": accuracy, "Confusion Matrix": confusion, "Recall": recall, "Precision": precision, "F1 Score": f1, "ROC AUC Score": roc_auc }
            
            

        except Exception as e:
            raise CustomException(e, sys)