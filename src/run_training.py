import joblib

# Get costumized functions from library
from packages.preprocess_data import *
from packages.train_model import run_model_training

# 0. Path to data
path_to_data = '../data/diabetes_data.csv'

# 1. Prepare the data
prepared_data = prepare_data(path_to_data)

# 2. Create train - test split
train_test_data = create_train_test_data(
    prepared_data['features'], prepared_data['label'], 0.33, 2021)

# 3. Run training
model = run_model_training(train_test_data['x_train'], train_test_data['x_test'],
                           train_test_data['y_train'], train_test_data['y_test'])

# 4. Save the trained model and vectorizer
joblib.dump(model, '../model/diabete_detector_model.pkl')
