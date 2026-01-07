# Fault Prediction API

## Endpoints

### 1. Single Prediction
**POST** `/api/predict/`

**Request:**
```json
{
    "humidity": 75.5,
    "rainfall": 10.2,
    ...
}
```

**Response:**
```json
{
    "success": true,
    "prediction": 1,
    "confidence": 0.85,
    ...
}
```

### 2. Batch Prediction
**POST** `/api/batch-predict/`

**Request:**
```json
[
    {"humidity": 75.5, "rainfall": 10.2, ...},
    {"humidity": 80.0, "rainfall": 5.0, ...}
]
```

### 3. Model Info
**GET** `/api/info/`