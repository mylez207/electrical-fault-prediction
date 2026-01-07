from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PredictionSerializer
from .models import PredictionHistory
import joblib
import numpy as np
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    filename='predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load the model once at module level (shared across all views)
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), 
    'ml_models', 
    'best_fault_prediction_model.pkl'
)

try:
    MODEL = joblib.load(MODEL_PATH)
    print("Model loaded successfully at module import!")
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL = None


# Reusable prediction function (used by both web interface and API)
def make_prediction(features):
    if MODEL is None:
        raise Exception("Model not loaded")

    input_array = np.array(features).reshape(1, -1)
    prediction = MODEL.predict(input_array)[0]
    probabilities = MODEL.predict_proba(input_array)[0]
    confidence = float(max(probabilities))

    return {
        'prediction': int(prediction),
        'confidence': confidence,
        'probability_no_fault': float(probabilities[0]),
        'probability_fault': float(probabilities[1]),
    }


# Web Interface View - Now uses individual form fields (no more JSON input)
@login_required
def predict_view(request):
    result = None
    error = None

    if request.method == "POST":
        try:
            # Read each field individually from the form
            features = [
                float(request.POST['humidity']),
                float(request.POST['rainfall']),
                int(request.POST['lightning']),
                float(request.POST['temperature']),
                float(request.POST['wind_speed']),
                int(request.POST['weather_severity']),
                float(request.POST['voltage_unbalance']),
                float(request.POST['current_unbalance']),
                float(request.POST['power_factor']),
                float(request.POST['frequency']),
                float(request.POST['line_loading']),
                float(request.POST['active_power']),
                float(request.POST['reactive_power']),
                int(request.POST['equipment_age']),
                int(request.POST['thermal_stress']),
                float(request.POST['risk_score']),
            ]

            # Make real prediction using the shared function
            pred_result = make_prediction(features)

            # Save prediction to database
            PredictionHistory.objects.create(
                humidity=features[0],
                rainfall=features[1],
                lightning=features[2],
                temperature=features[3],
                wind_speed=features[4],
                weather_severity=features[5],
                voltage_unbalance=features[6],
                current_unbalance=features[7],
                power_factor=features[8],
                frequency=features[9],
                line_loading=features[10],
                active_power=features[11],
                reactive_power=features[12],
                equipment_age=features[13],
                thermal_stress=features[14],
                risk_score=features[15],
                prediction=pred_result['prediction'],
                confidence=pred_result['confidence']
            )

            # Log the prediction
            logger.info(f"Web prediction - User: {request.user.username} | Fault: {pred_result['prediction']} | Confidence: {pred_result['confidence']:.2f}")

            # Prepare simple result for the template
            result = {
                'prediction': pred_result['prediction'],  # 0 or 1
                'confidence': round(pred_result['confidence'] * 100, 1),  # percentage
                'class': 'fault' if pred_result['prediction'] == 1 else 'no-fault'
            }

        except ValueError as ve:
            error = "Please enter valid numbers in all fields."
        except KeyError as ke:
            error = f"Missing field: {str(ke)}"
        except Exception as e:
            error = f"Prediction failed: {str(e)}"
            logger.error(f"Web prediction error: {str(e)}")

    return render(request, 'predict.html', {
        'result': result,
        'error': error
    })


# API View remains unchanged - still accepts JSON for external use
class PredictView(APIView):
    def post(self, request):
        serializer = PredictionSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {'error': 'Invalid input', 'details': serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            features = [
                serializer.validated_data['humidity'],
                serializer.validated_data['rainfall'],
                serializer.validated_data['lightning'],
                serializer.validated_data['temperature'],
                serializer.validated_data['wind_speed'],
                serializer.validated_data['weather_severity'],
                serializer.validated_data['voltage_unbalance'],
                serializer.validated_data['current_unbalance'],
                serializer.validated_data['power_factor'],
                serializer.validated_data['frequency'],
                serializer.validated_data['line_loading'],
                serializer.validated_data['active_power'],
                serializer.validated_data['reactive_power'],
                serializer.validated_data['equipment_age'],
                serializer.validated_data['thermal_stress'],
                serializer.validated_data['risk_score'],
            ]

            pred_result = make_prediction(features)

            # Save to DB
            PredictionHistory.objects.create(
                humidity=features[0],
                rainfall=features[1],
                lightning=features[2],
                temperature=features[3],
                wind_speed=features[4],
                weather_severity=features[5],
                voltage_unbalance=features[6],
                current_unbalance=features[7],
                power_factor=features[8],
                frequency=features[9],
                line_loading=features[10],
                active_power=features[11],
                reactive_power=features[12],
                equipment_age=features[13],
                thermal_stress=features[14],
                risk_score=features[15],
                prediction=pred_result['prediction'],
                confidence=pred_result['confidence']
            )

            logger.info(f"API Prediction: {pred_result['prediction']}, Confidence: {pred_result['confidence']}")

            return Response({
                'success': True,
                'prediction': pred_result['prediction'],
                'confidence': pred_result['confidence'],
                'probability_no_fault': pred_result['probability_no_fault'],
                'probability_fault': pred_result['probability_fault'],
                'timestamp': datetime.now().isoformat(),
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"API Prediction error: {str(e)}")
            return Response({'success': False, 'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get(self, request):
        return Response({
            'message': 'Electrical Fault Prediction API',
            'description': 'Predicts power system faults using Random Forest',
            'method': 'POST',
            'required_features': [
                'humidity', 'rainfall', 'lightning', 'temperature', 'wind_speed',
                'weather_severity', 'voltage_unbalance', 'current_unbalance',
                'power_factor', 'frequency', 'line_loading', 'active_power',
                'reactive_power', 'equipment_age', 'thermal_stress', 'risk_score'
            ]
        })