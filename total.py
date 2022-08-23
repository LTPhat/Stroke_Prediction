import Main_Model
import Preprocessing
import pandas as pd
model = Main_Model.eclf
X_train,X_test,y_train,y_test = Preprocessing.X_train,Preprocessing.X_test,Preprocessing.y_train,Preprocessing.y_test
predict = model.predict(X_test)
predict_prob = model.predict_proba(X_test)
X_test_copy = Preprocessing.X_test_copy
df_compare = pd.DataFrame({'y_test':y_test})
df_compare['Proba_not_stroke'] = predict_prob[:,0]
df_compare['Proba_stroke'] = predict_prob[:,1]
df_compare['Predict'] = predict
# Lưu output dự đoán vào file csv
from pathlib import Path
filepath = Path('E:\StrokeProject\Main_Project\output_test.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
df_compare.to_csv(filepath)
