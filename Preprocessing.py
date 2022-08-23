import pandas as pd

# Load dataset after EDA

data = pd.read_csv('E:\StrokeProject\Main_Project\Dataset.csv')

# Delete unnamed columns
data.drop('Unnamed: 0',axis = 1, inplace = True)
#Train-test-split
from sklearn.model_selection import train_test_split
X = data.drop(['stroke'],axis=1) # Khai báo input
y = data['stroke'] # Khai báo output
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)
X_test_copy = X_test.copy()
# Train set size: (38108, 16)
# Test set size: (16333, 16)

# Categorical Encode for X_train, X_test
# Encode X_train
df_dummy = X_train[['gender','work_type','cholesterol']]
df_dummy = pd.get_dummies(df_dummy)
X_train_temp = X_train.drop(['gender','work_type','cholesterol'],axis = 1)
X_train_final = pd.concat([X_train_temp, df_dummy], axis=1)
X_train =X_train_final
# Encode X_test
df_dummy = X_test[['gender','work_type','cholesterol']]
df_dummy = pd.get_dummies(df_dummy)
X_test_temp = X_test.drop(['gender','work_type','cholesterol'],axis = 1)
X_test_final = pd.concat([X_test_temp, df_dummy], axis=1)
X_test =X_test_final

# scaler
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

# Khởi tạo list các biến numeric
numeric_col = ['age','BMI','mean_blood_pressure','avg_glu_level']
X_train[numeric_col]=scaler.fit_transform(X_train[numeric_col]) # Scale for train set
X_test[numeric_col]=scaler.transform(X_test[numeric_col]) # Scale for test set


