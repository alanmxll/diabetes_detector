from django.urls import path
from diabete_api import views


urlpatterns = [
    path('', views.index),
    path('predict', views.predict_patient_status)
]
