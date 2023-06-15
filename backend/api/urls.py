from django.urls import path
from . import views


urlpatterns = [
    path('predict_model/', views.predict_model),
    path('predict_ocr/', views.predict_ocr),
]
