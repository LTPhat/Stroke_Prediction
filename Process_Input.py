import Preprocessing
import pickle
import app
model = pickle.load(open('model.pkl', 'rb'))
test_data = Preprocessing.X_test_copy
