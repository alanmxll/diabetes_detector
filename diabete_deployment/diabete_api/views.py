# Import necessary libraries
import json
import os
import sys

import joblib
import numpy as np
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Load the model from the static folder
path_to_model = os.path.join(settings.BASE_DIR, 'static/model/')
loaded_model = joblib.load(
    open(path_to_model+'diabete_detector_model.pkl', 'rb'))


# Create your view here
@api_view(['GET'])
def index(request):
    return_data = {
        "error_code": "0",
        "info": "success"
    }

    return Response(return_data)


@api_view(['POST'])
def predict_patient_status(request):
    try:
        # load the request data
        patient_json_info = request.data

        # Retrieve all the values from the json data
        patient_info = np.array(list(patient_json_info.values()))

        # Make prediction
        patient_status = loaded_model.predict([patient_info])

        # Model confidence score
        model_confidence_score = np.max(
            loaded_model.predict_proba([patient_info]))

        model_prediction = {
            'info': 'success',
            'patient_status': patient_status[0],
            'model_confidence_proba': float(f"{round(model_confidence_score*100, 2)}")
        }
    except ValueError as ve:
        model_prediction = {
            'error_code': '-1',
            'info': str(ve)
        }

    return Response(model_prediction)
