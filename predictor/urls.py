from django.urls import path
from .views import PredictView, predict_view  # Only import what actually exists

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
    # If you want the web prediction page under /api/ too (optional)
    # path('web-predict/', predict_view, name='web-predict'),
]