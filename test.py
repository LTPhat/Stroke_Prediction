import pickle
from sklearn.preprocessing import StandardScaler
import app
import Preprocessing
import pandas as pd
import numpy as np
# model = pickle.load(open('model.pkl', 'rb'))
# test_data = Preprocessing.X_test_copy
# age =40
gender = "Male"
BMI = '40'
work_type = 'Private'
mean_blood_pressure = 100
cholesterol = 'High'
smoke = 0
avg_glu_level= 180
alcohol = 1
active = 1
# new_point = app.create_data_input(age, gender, BMI, work_type, mean_blood_pressure, cholesterol, smoke, avg_glu_level,
#                               alcohol, active, data=test_data)
# data_final = app.add_to_test(new_point,test_data)
# # Categorical encode
# df_dummy = data_final[['gender', 'work_type', 'cholesterol']]
# df_dummy = pd.get_dummies(df_dummy)
# data_temp = data_final.drop(['gender', 'work_type', 'cholesterol'], axis=1)
# data_final = pd.concat([data_temp, df_dummy], axis=1)
# data_final = data_final.reset_index(drop=True)
#
# #Scaler
# scaler = StandardScaler()
# numeric_col = ['age', 'BMI', 'mean_blood_pressure', 'avg_glu_level']
# data_final[numeric_col] = scaler.fit_transform(data_final[numeric_col])
# def prediction(data):
#     data= np.array(data).reshape(1,-1)
#     predict = model.predict_proba(data)
#     return predict[0][1]

data = pd.read_excel('E:\StrokeProject\Main_Project\input_user.xlsx',index_col=0)
print(data)