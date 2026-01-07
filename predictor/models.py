from django.db import models

class PredictionHistory(models.Model):
    # Input features
    humidity = models.FloatField()
    rainfall = models.FloatField()
    lightning = models.FloatField()
    temperature = models.FloatField()
    wind_speed = models.FloatField()
    weather_severity = models.FloatField()
    voltage_unbalance = models.FloatField()
    current_unbalance = models.FloatField()
    power_factor = models.FloatField()
    frequency = models.FloatField()
    line_loading = models.FloatField()
    active_power = models.FloatField()
    reactive_power = models.FloatField()
    equipment_age = models.FloatField()
    thermal_stress = models.FloatField()
    risk_score = models.FloatField()
    
    # Output
    prediction = models.IntegerField()
    confidence = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name_plural = 'Prediction Histories'
    
    def __str__(self):
        return f"Prediction: {self.prediction} at {self.timestamp}"