def get_high_risk_params(features_dict):
    """
    Returns list of strings describing parameters in HIGH range.
    Uses same thresholds as your frontend JS.
    """
    high = []

    # features_dict keys match your form field names
    if features_dict['humidity'] > 85:
        high.append(f"Humidity: High (>85%) – {features_dict['humidity']}%")
    if features_dict['rainfall'] > 30:
        high.append(f"Rainfall: High (>30 mm) – {features_dict['rainfall']} mm")
    if features_dict['lightning'] == 1:
        high.append("Lightning: High (present)")
    if features_dict['temperature'] > 42 or features_dict['temperature'] < 10:
        high.append(f"Temperature: High (extreme) – {features_dict['temperature']}°C")
    if features_dict['wind_speed'] > 80:
        high.append(f"Wind Speed: High (>80 km/h) – {features_dict['wind_speed']} km/h")
    if features_dict['weather_severity'] >= 7:
        high.append(f"Weather Severity: High (≥7) – {features_dict['weather_severity']}/10")
    if features_dict['voltage_unbalance'] > 2.0:
        high.append(f"Voltage Unbalance: High (>2.0%) – {features_dict['voltage_unbalance']}%")
    if features_dict['current_unbalance'] > 12:
        high.append(f"Current Unbalance: High (>12%) – {features_dict['current_unbalance']}%")
    if features_dict['power_factor'] < 0.85:
        high.append(f"Power Factor: High risk (<0.85) – {features_dict['power_factor']}")
    if abs(features_dict['frequency'] - 50) > 0.5:
        high.append(f"Frequency: High deviation (±>0.5 Hz) – {features_dict['frequency']} Hz")
    if features_dict['line_loading'] > 95:
        high.append(f"Line Loading: High (>95%) – {features_dict['line_loading']}%")
    # active_power & reactive_power → skip or customize based on your line rating
    if features_dict['equipment_age'] > 35:
        high.append(f"Equipment Age: High (>35 years) – {features_dict['equipment_age']} years")
    if features_dict['thermal_stress'] >= 7:
        high.append(f"Thermal Stress: High (≥7) – {features_dict['thermal_stress']}/10")
    if features_dict['risk_score'] > 60:
        high.append(f"Risk Score: High (>60) – {features_dict['risk_score']}/100")

    return high