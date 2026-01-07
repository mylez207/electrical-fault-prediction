from rest_framework import serializers

class PredictionSerializer(serializers.Serializer):
    # Environmental features
    humidity = serializers.FloatField(help_text="Humidity percentage (0-100)")
    rainfall = serializers.FloatField(help_text="Rainfall in mm")
    lightning = serializers.FloatField(help_text="Lightning activity indicator")
    temperature = serializers.FloatField(help_text="Temperature in degrees Celsius")
    wind_speed = serializers.FloatField(help_text="Wind speed in km/h")
    weather_severity = serializers.FloatField(help_text="Weather severity index")
    
    # Electrical features
    voltage_unbalance = serializers.FloatField(help_text="Voltage unbalance percentage")
    current_unbalance = serializers.FloatField(help_text="Current unbalance percentage")
    power_factor = serializers.FloatField(help_text="Power factor (0-1)")
    frequency = serializers.FloatField(help_text="System frequency in Hz")
    line_loading = serializers.FloatField(help_text="Line loading percentage")
    active_power = serializers.FloatField(help_text="Active power in kW")
    reactive_power = serializers.FloatField(help_text="Reactive power in kVAR")
    
    # Equipment features
    equipment_age = serializers.FloatField(help_text="Equipment age in years")
    thermal_stress = serializers.FloatField(help_text="Thermal stress indicator")
    risk_score = serializers.FloatField(help_text="Overall risk score (0-1)")