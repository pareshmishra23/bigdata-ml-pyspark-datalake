# Early Warning System (EWS) ML Pipeline

A comprehensive Machine Learning pipeline for IoT sensor anomaly detection with both web-based simulation and production-ready PySpark code.

## 🚀 Features

### Pipeline Components
- **Bronze Layer**: Raw IoT sensor data ingestion and storage
- **Silver Layer**: Data cleansing, validation, and quality scoring
- **Feature Engineering**: Time-series feature generation (lag, rolling averages, statistical features)
- **Anomaly Detection**: Isolation Forest-based anomaly detection
- **Real-time Inference**: Live anomaly scoring and alerts

### Web Dashboard
- Real-time sensor monitoring
- Pipeline status visualization
- Anomaly detection alerts
- Interactive sensor data exploration
- Model management interface

### Production Export
- Complete PySpark implementation
- Delta Lake integration
- Structured Streaming support
- Production configurations

## 🏗️ Architecture

```
IoT Sensors → Bronze (Raw) → Silver (Cleaned) → Features → ML Model → Anomalies
                ↓              ↓                ↓           ↓
            MongoDB        MongoDB         MongoDB    MongoDB
```

## 📊 Supported Sensor Types

- **Temperature**: Factory floor monitoring (°C)
- **Pressure**: Pipeline monitoring (PSI)  
- **Humidity**: Storage room monitoring (%)
- **Vibration**: Motor monitoring (mm/s)

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- MongoDB
- React development environment

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
yarn install
```

### Environment Configuration
Backend `.env`:
```
MONGO_URL=mongodb://localhost:27017
```

Frontend `.env`:
```
REACT_APP_BACKEND_URL=http://localhost:8001
```

## 🚦 Running the Application

### Development Mode
1. Start the backend:
```bash
cd backend
python server.py
```

2. Start the frontend:
```bash
cd frontend
yarn start
```

3. Access the dashboard at `http://localhost:3000`

### Production Mode
Use supervisor or similar process manager for backend services.

## 📈 ML Pipeline Details

### Feature Engineering
- **Statistical Features**: Mean, standard deviation, min, max, skewness, kurtosis
- **Lag Features**: Previous 1, 2, 3, 5 time steps
- **Rolling Window Features**: 5, 10, 20, 50-period rolling statistics
- **Trend Features**: Linear trend slope analysis
- **Change Features**: Rate of change and last change detection

### Anomaly Detection Model
- **Algorithm**: Isolation Forest
- **Contamination Rate**: 10% (configurable)
- **Estimators**: 100 trees
- **Feature Scaling**: StandardScaler normalization
- **Real-time Scoring**: Online anomaly detection with confidence scores

### Data Quality Framework
- **Validation Rules**: Sensor range checks, extreme value detection
- **Quality Scoring**: 0.0-1.0 quality score for each reading
- **Data Cleansing**: Outlier removal and value correction
- **Missing Data Handling**: Interpolation and forward filling

## 🔄 Pipeline Operations

### Data Flow
1. **Ingestion**: Simulated IoT sensor readings (can be replaced with Kafka/API)
2. **Bronze Storage**: Raw data storage with metadata
3. **Silver Processing**: Data quality checks and cleansing
4. **Feature Generation**: Time-series feature engineering
5. **Anomaly Detection**: ML-based anomaly scoring
6. **Alerting**: Real-time anomaly notifications

### API Endpoints
- `GET /api/health` - Pipeline health check
- `GET /api/dashboard/stats` - Overall statistics
- `GET /api/sensors` - List of active sensors
- `GET /api/sensors/{id}/recent` - Recent sensor data
- `GET /api/anomalies/recent` - Recent anomalies
- `POST /api/sensors/simulate` - Trigger data simulation
- `POST /api/model/retrain` - Retrain anomaly model
- `GET /api/export/pyspark` - Export PySpark code

## 📦 PySpark Export

The system generates production-ready PySpark code including:

### Components
- **Structured Streaming**: Real-time data ingestion
- **Delta Lake Integration**: ACID transactions and time travel
- **Feature Engineering**: Distributed time-series processing
- **ML Pipeline**: Scalable anomaly detection
- **Error Handling**: Robust error recovery

### Deployment
```python
# Example configuration
config = {
    "input_path": "/mnt/data/raw_sensors/",
    "bronze_path": "/mnt/data/bronze/sensor_data/",
    "silver_path": "/mnt/data/silver/sensor_data/",
    "features_path": "/mnt/data/features/sensor_features/",
    "anomalies_path": "/mnt/data/gold/anomalies/"
}
```

## 🎯 Use Cases

### Industrial IoT
- Factory equipment monitoring
- Predictive maintenance
- Quality control

### Infrastructure
- Pipeline monitoring
- Environmental sensing  
- Safety systems

### Smart Buildings
- HVAC optimization
- Energy management
- Occupancy detection

## 🔧 Configuration

### Sensor Configuration
Modify `IoTSensorSimulator` class to add new sensor types:
```python
sensors = {
    'NEW_SENSOR_001': {
        'type': 'custom_type',
        'location': 'Custom_Location',
        'normal_range': (min_val, max_val),
        'unit': 'unit'
    }
}
```

### Model Parameters
Adjust anomaly detection parameters:
```python
anomaly_model = IsolationForest(
    contamination=0.1,      # Expected anomaly rate
    n_estimators=100,       # Number of trees
    random_state=42
)
```

### Feature Engineering
Customize feature windows:
```python
window_sizes = [5, 10, 20, 50]  # Rolling window sizes
lag_features = [1, 2, 3, 5]     # Lag periods
```

## 📊 Sample Data

The system includes realistic IoT sensor data simulation:
- **Temperature**: 18-28°C with seasonal variations
- **Pressure**: 95-108 PSI with process variations
- **Humidity**: 45-65% with environmental changes
- **Vibration**: 0.1-0.8 mm/s with mechanical variations

Anomalies are injected at 10% rate with:
- Temperature spikes/drops (±10-20°C)
- Pressure surges (±15-25 PSI)
- Humidity extremes (±20-30%)
- Vibration peaks (±5-10x normal)

## 🚨 Alerting & Monitoring

### Anomaly Detection
- Real-time scoring with confidence intervals
- Configurable threshold settings
- Multi-sensor correlation analysis
- Historical anomaly tracking

### Dashboard Metrics
- Data ingestion rates
- Processing latency
- Model accuracy metrics
- System health indicators

## 🔒 Security & Compliance

### Data Privacy
- No sensitive data stored in logs
- Configurable data retention
- Audit trail support

### Model Security
- Model versioning
- A/B testing capability
- Rollback mechanisms

## 🎨 Customization

### UI Themes
The dashboard supports custom styling through Tailwind CSS classes and can be extended with additional sensor types and visualization components.

### Model Algorithms
Replace Isolation Forest with other algorithms:
- One-Class SVM
- Local Outlier Factor
- Deep learning approaches (Autoencoders)

## 📝 License

This project is available for educational and research purposes.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional sensor types
- Advanced ML algorithms
- Enhanced visualizations
- Performance optimizations

## 📞 Support

For questions and support regarding the EWS pipeline implementation, please refer to the documentation or create an issue in the repository.