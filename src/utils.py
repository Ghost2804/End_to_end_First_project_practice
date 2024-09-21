# # import os
# # import sys
# # import numpy as np 
# # import pandas as pd
# # import dill
# # import pickle
# # from sklearn.metrics import r2_score
# # from sklearn.model_selection import GridSearchCV

# # from src.exception import customException

# # def save_object(file_path, obj):
# #     try:
# #         dir_path = os.path.dirname(file_path)

# #         os.makedirs(dir_path, exist_ok=True)

# #         with open(file_path, "wb") as file_obj:
# #             pickle.dump(obj, file_obj)
            
# #     except Exception as e:
# #         raise customException(e, sys)


# # def evaluate_models(X_train, y_train,X_test,y_test,models,param):
# #     try:
# #         report = {}

# #         for i in range(len(list(models))):
# #             model = list(models.values())[i]
# #             para=param[list(models.keys())[i]]

# #             gs = GridSearchCV(model,para,cv=3)
# #             gs.fit(X_train,y_train)

# #             model.set_params(**gs.best_params_)
# #             model.fit(X_train,y_train)

# #             y_train_pred = model.predict(X_train)

# #             y_test_pred = model.predict(X_test)

# #             train_model_score = r2_score(y_train, y_train_pred)

# #             test_model_score = r2_score(y_test, y_test_pred)

# #             report[list(models.keys())[i]] = test_model_score

# #         return report

# #     except Exception as e:
# #         raise customException(e, sys)
    
# # def load_object(file_path):
# #     try:
# #         with open(file_path, 'rb') as file_obj:
# #             return dill.load(file_obj)
        
# #     except Exception as e:
# #         raise customException(e,sys)



# import sys
# import pandas as pd
# from src.exception import customException
# from src.utils import load_object
# import os
# import logging

# class PredictPipeline:
#     def __init__(self):
#         pass

#     def predict(self, features):
#         try:
#             model_path = os.path.join("artifacts", "model.pkl")
#             preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
#             logging.debug("Before Loading")
#             model = load_object(file_path=model_path)
#             preprocessor = load_object(file_path=preprocessor_path)
#             logging.debug("After Loading")
#             data_scaled = preprocessor.transform(features)
#             logging.debug(f"Scaled Data: {data_scaled}")
#             preds = model.predict(data_scaled)
#             logging.debug(f"Predictions: {preds}")
#             return preds
        
#         except Exception as e:
#             logging.error(f"Error in prediction: {e}")
#             raise customException(e, sys)

# class CustomData:
#     def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education, lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
#         self.gender = gender
#         self.race_ethnicity = race_ethnicity
#         self.parental_level_of_education = parental_level_of_education
#         self.lunch = lunch
#         self.test_preparation_course = test_preparation_course
#         self.reading_score = reading_score
#         self.writing_score = writing_score

#     def get_data_as_data_frame(self):
#         try:
#             custom_data_input_dict = {
#                 "gender": [self.gender],
#                 "race_ethnicity": [self.race_ethnicity],
#                 "parental_level_of_education": [self.parental_level_of_education],
#                 "lunch": [self.lunch],
#                 "test_preparation_course": [self.test_preparation_course],
#                 "reading_score": [self.reading_score],
#                 "writing_score": [self.writing_score],
#             }
#             df = pd.DataFrame(custom_data_input_dict)
#             logging.debug(f"Custom DataFrame: {df}")
#             return df

#         except Exception as e:
#             logging.error(f"Error in creating DataFrame: {e}")
#             raise customException(e, sys)


import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import customException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise customException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise customException(e, sys)