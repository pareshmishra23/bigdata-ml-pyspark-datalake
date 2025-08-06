from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pymongo import MongoClient
import uuid
import json
import asyncio
import random
import time
from threading import Thread
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for real-time simulation
sensor_data_stream = []
anomaly_model = None
scaler = None
feature_store = []
is_streaming = False

# MongoDB connection
MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = MongoClient(MONGO_URL)
db = client.ews_pipeline

# Collections for different layers
bronze_collection = db.bronze_sensor_data
silver_collection = db.silver_sensor_data
feature_collection = db.feature_store
anomaly_collection = db.anomaly_predictions

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting EWS Pipeline...")
    await initialize_pipeline()
    yield
    # Shutdown
    logger.info("Shutting down EWS Pipeline...")

app = FastAPI(title="Early Warning System ML Pipeline", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class SensorReading(BaseModel):
    sensor_id: str
    sensor_type: str  # temperature, pressure, humidity, vibration
    value: float
    unit: str
    timestamp: datetime
    location: str
    metadata: Dict[str, Any] = {}

class BronzeData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sensor_readings: List[SensorReading]
    ingestion_timestamp: datetime
    source: str
    batch_id: str

class SilverData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sensor_id: str
    sensor_type: str
    cleaned_value: float
    original_value: float
    is_valid: bool
    quality_score: float
    enriched_data: Dict[str, Any]
    processed_timestamp: datetime

class FeatureData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sensor_id: str
    timestamp: datetime
    features: Dict[str, float]
    feature_version: str = "v1.0"

class AnomalyPrediction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sensor_id: str
    timestamp: datetime
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    feature_values: Dict[str, float]
    model_version: str = "v1.0"

# IoT Sensor Data Generator
class IoTSensorSimulator:
    def __init__(self):
        self.sensors = {
            'TEMP_001': {'type': 'temperature', 'location': 'Factory_Floor_A', 'normal_range': (18, 25), 'unit': '°C'},
            'TEMP_002': {'type': 'temperature', 'location': 'Factory_Floor_B', 'normal_range': (20, 28), 'unit': '°C'},
            'PRESS_001': {'type': 'pressure', 'location': 'Pipeline_1', 'normal_range': (95, 105), 'unit': 'PSI'},
            'PRESS_002': {'type': 'pressure', 'location': 'Pipeline_2', 'normal_range': (98, 108), 'unit': 'PSI'},
            'HUMID_001': {'type': 'humidity', 'location': 'Storage_Room_1', 'normal_range': (45, 65), 'unit': '%'},
            'VIB_001': {'type': 'vibration', 'location': 'Motor_1', 'normal_range': (0.1, 0.8), 'unit': 'mm/s'},
        }
        
    def generate_reading(self, sensor_id: str, timestamp: datetime = None, inject_anomaly: bool = False) -> SensorReading:
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        sensor_config = self.sensors[sensor_id]
        min_val, max_val = sensor_config['normal_range']
        
        if inject_anomaly and random.random() < 0.1:  # 10% chance of anomaly
            # Generate anomalous reading
            if random.random() < 0.5:
                value = min_val - random.uniform(5, 15)  # Below normal
            else:
                value = max_val + random.uniform(5, 20)  # Above normal
        else:
            # Generate normal reading with some noise
            center = (min_val + max_val) / 2
            noise = random.gauss(0, (max_val - min_val) * 0.1)
            value = center + noise
            value = max(min_val * 0.8, min(max_val * 1.2, value))  # Clamp to reasonable bounds
        
        return SensorReading(
            sensor_id=sensor_id,
            sensor_type=sensor_config['type'],
            value=round(value, 3),
            unit=sensor_config['unit'],
            timestamp=timestamp,
            location=sensor_config['location'],
            metadata={
                'quality': random.choice(['good', 'fair', 'excellent']),
                'source': 'simulator'
            }
        )
    
    def generate_batch(self, batch_size: int = 50, inject_anomalies: bool = True) -> List[SensorReading]:
        readings = []
        base_time = datetime.utcnow()
        
        for i in range(batch_size):
            timestamp = base_time - timedelta(minutes=i)
            sensor_id = random.choice(list(self.sensors.keys()))
            reading = self.generate_reading(sensor_id, timestamp, inject_anomalies)
            readings.append(reading)
        
        return readings

# Feature Engineering
class TimeSeriesFeatureEngine:
    def __init__(self):
        self.window_sizes = [5, 10, 20, 50]
        
    def extract_features(self, df: pd.DataFrame, sensor_id: str) -> Dict[str, float]:
        """Extract time-series features for a specific sensor"""
        sensor_data = df[df['sensor_id'] == sensor_id].sort_values('timestamp')
        
        if len(sensor_data) < 5:
            return {}
        
        values = sensor_data['cleaned_value'].values
        features = {}
        
        # Basic statistical features
        features['mean'] = np.mean(values)
        features['std'] = np.std(values)
        features['min'] = np.min(values)
        features['max'] = np.max(values)
        features['range'] = features['max'] - features['min']
        features['skewness'] = float(pd.Series(values).skew())
        features['kurtosis'] = float(pd.Series(values).kurtosis())
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            if len(values) > lag:
                features[f'lag_{lag}'] = values[-1-lag]
        
        # Rolling window features
        for window in self.window_sizes:
            if len(values) >= window:
                window_values = values[-window:]
                features[f'rolling_mean_{window}'] = np.mean(window_values)
                features[f'rolling_std_{window}'] = np.std(window_values)
                features[f'rolling_max_{window}'] = np.max(window_values)
                features[f'rolling_min_{window}'] = np.min(window_values)
        
        # Trend features
        if len(values) >= 10:
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            features['trend_slope'] = slope
            
        # Change features
        if len(values) > 1:
            features['last_change'] = values[-1] - values[-2]
            features['change_rate'] = (values[-1] - values[0]) / len(values) if len(values) > 1 else 0
        
        return features

# Initialize components
simulator = IoTSensorSimulator()
feature_engine = TimeSeriesFeatureEngine()

async def initialize_pipeline():
    """Initialize the ML pipeline components"""
    global anomaly_model, scaler
    
    logger.info("Initializing ML Pipeline...")
    
    # Train initial model with synthetic data
    await train_anomaly_model()
    
    # Start background data generation
    start_background_data_generation()
    
    logger.info("Pipeline initialized successfully!")

def start_background_data_generation():
    """Start background thread for continuous data generation"""
    global is_streaming
    
    def generate_streaming_data():
        global is_streaming
        is_streaming = True
        
        while is_streaming:
            try:
                # Generate batch of sensor readings
                readings = simulator.generate_batch(batch_size=6, inject_anomalies=True)
                
                # Process through pipeline
                asyncio.create_task(process_sensor_batch(readings))
                
                time.sleep(30)  # Generate new batch every 30 seconds
            except Exception as e:
                logger.error(f"Error in background data generation: {e}")
                time.sleep(5)
    
    thread = Thread(target=generate_streaming_data, daemon=True)
    thread.start()

async def process_sensor_batch(readings: List[SensorReading]):
    """Process a batch of sensor readings through the entire pipeline"""
    try:
        # Stage 1: Bronze Layer - Raw data ingestion
        bronze_data = BronzeData(
            sensor_readings=readings,
            ingestion_timestamp=datetime.utcnow(),
            source="iot_sensors",
            batch_id=str(uuid.uuid4())
        )
        
        bronze_collection.insert_one(bronze_data.dict())
        
        # Stage 2: Silver Layer - Data cleansing and validation
        silver_records = []
        for reading in readings:
            silver_data = await cleanse_and_validate(reading)
            silver_records.append(silver_data)
            silver_collection.insert_one(silver_data.dict())
        
        # Stage 3: Feature Engineering
        await generate_features(silver_records)
        
        # Stage 4: Anomaly Detection
        await detect_anomalies(silver_records)
        
        logger.info(f"Processed batch of {len(readings)} sensor readings")
        
    except Exception as e:
        logger.error(f"Error processing sensor batch: {e}")

async def cleanse_and_validate(reading: SensorReading) -> SilverData:
    """Cleanse and validate sensor reading"""
    
    # Data quality checks
    is_valid = True
    quality_score = 1.0
    cleaned_value = reading.value
    
    # Check for reasonable sensor ranges
    sensor_config = simulator.sensors.get(reading.sensor_id)
    if sensor_config:
        min_val, max_val = sensor_config['normal_range']
        # Extended range check (allowing some variance)
        if cleaned_value < min_val * 0.5 or cleaned_value > max_val * 1.5:
            quality_score *= 0.5
            
        # Check for extreme outliers
        if cleaned_value < min_val * 0.2 or cleaned_value > max_val * 2.0:
            is_valid = False
            quality_score *= 0.1
    
    # Simple data cleaning - remove obvious outliers
    if abs(cleaned_value) > 1000:  # Unreasonably high value
        cleaned_value = np.median([reading.value, cleaned_value])
        quality_score *= 0.7
    
    return SilverData(
        sensor_id=reading.sensor_id,
        sensor_type=reading.sensor_type,
        cleaned_value=cleaned_value,
        original_value=reading.value,
        is_valid=is_valid,
        quality_score=quality_score,
        enriched_data={
            'location': reading.location,
            'unit': reading.unit,
            'metadata': reading.metadata
        },
        processed_timestamp=datetime.utcnow()
    )

async def generate_features(silver_records: List[SilverData]):
    """Generate time-series features for each sensor"""
    
    # Get recent historical data for feature engineering
    for sensor_id in set(record.sensor_id for record in silver_records):
        # Get last 100 records for this sensor
        historical_data = list(silver_collection.find(
            {"sensor_id": sensor_id, "is_valid": True}
        ).sort("processed_timestamp", -1).limit(100))
        
        if len(historical_data) >= 5:
            # Convert to DataFrame for feature engineering
            df = pd.DataFrame([{
                'sensor_id': record['sensor_id'],
                'cleaned_value': record['cleaned_value'],
                'timestamp': record['processed_timestamp']
            } for record in historical_data])
            
            # Extract features
            features = feature_engine.extract_features(df, sensor_id)
            
            if features:
                feature_data = FeatureData(
                    sensor_id=sensor_id,
                    timestamp=datetime.utcnow(),
                    features=features
                )
                
                feature_collection.insert_one(feature_data.dict())

async def detect_anomalies(silver_records: List[SilverData]):
    """Detect anomalies using trained model"""
    global anomaly_model, scaler
    
    if not anomaly_model or not scaler:
        return
    
    for record in silver_records:
        # Get latest features for this sensor
        latest_features = feature_collection.find_one(
            {"sensor_id": record.sensor_id},
            sort=[("timestamp", -1)]
        )
        
        if latest_features and latest_features['features']:
            try:
                # Prepare feature vector
                feature_names = sorted(latest_features['features'].keys())
                feature_vector = np.array([[latest_features['features'][name] for name in feature_names]])
                
                # Scale features
                feature_vector_scaled = scaler.transform(feature_vector)
                
                # Predict anomaly
                anomaly_score = anomaly_model.decision_function(feature_vector_scaled)[0]
                is_anomaly = anomaly_model.predict(feature_vector_scaled)[0] == -1
                
                # Calculate confidence (normalized anomaly score)
                confidence = min(1.0, max(0.0, (abs(anomaly_score) + 0.5) / 1.5))
                
                anomaly_prediction = AnomalyPrediction(
                    sensor_id=record.sensor_id,
                    timestamp=datetime.utcnow(),
                    is_anomaly=is_anomaly,
                    anomaly_score=float(anomaly_score),
                    confidence=confidence,
                    feature_values=latest_features['features']
                )
                
                anomaly_collection.insert_one(anomaly_prediction.dict())
                
                if is_anomaly:
                    logger.warning(f"ANOMALY DETECTED - Sensor: {record.sensor_id}, Score: {anomaly_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error in anomaly detection for sensor {record.sensor_id}: {e}")

async def train_anomaly_model():
    """Train the anomaly detection model"""
    global anomaly_model, scaler
    
    logger.info("Training anomaly detection model...")
    
    # Generate training data
    training_readings = []
    for _ in range(500):  # Generate 500 historical readings
        readings = simulator.generate_batch(batch_size=10, inject_anomalies=True)
        training_readings.extend(readings)
    
    # Process training data through silver layer
    training_silver = []
    for reading in training_readings:
        silver_data = await cleanse_and_validate(reading)
        training_silver.append(silver_data)
    
    # Group by sensor and extract features
    training_features = []
    sensor_groups = {}
    
    for record in training_silver:
        if record.sensor_id not in sensor_groups:
            sensor_groups[record.sensor_id] = []
        sensor_groups[record.sensor_id].append(record)
    
    for sensor_id, records in sensor_groups.items():
        if len(records) >= 10:
            # Convert to DataFrame
            df = pd.DataFrame([{
                'sensor_id': record.sensor_id,
                'cleaned_value': record.cleaned_value,
                'timestamp': record.processed_timestamp
            } for record in records])
            
            # Extract features for multiple time windows
            for i in range(5, len(records)):
                subset_df = df.iloc[:i+1]
                features = feature_engine.extract_features(subset_df, sensor_id)
                if features:
                    training_features.append(features)
    
    if len(training_features) > 50:
        # Convert to feature matrix
        feature_names = sorted(set().union(*[f.keys() for f in training_features]))
        feature_matrix = []
        
        for features in training_features:
            feature_vector = [features.get(name, 0.0) for name in feature_names]
            feature_matrix.append(feature_vector)
        
        feature_matrix = np.array(feature_matrix)
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix)
        
        # Scale features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Train Isolation Forest
        anomaly_model = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_estimators=100
        )
        anomaly_model.fit(feature_matrix_scaled)
        
        # Save model artifacts
        joblib.dump(anomaly_model, '/tmp/anomaly_model.pkl')
        joblib.dump(scaler, '/tmp/scaler.pkl')
        
        logger.info(f"Model trained successfully with {len(training_features)} feature vectors")
    else:
        logger.warning("Insufficient training data for model training")

# API Endpoints

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "pipeline_initialized": anomaly_model is not None,
        "streaming": is_streaming,
        "timestamp": datetime.utcnow()
    }

@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """Get overall pipeline statistics"""
    
    bronze_count = bronze_collection.count_documents({})
    silver_count = silver_collection.count_documents({})
    feature_count = feature_collection.count_documents({})
    anomaly_count = anomaly_collection.count_documents({"is_anomaly": True})
    
    return {
        "total_ingested": bronze_count,
        "total_processed": silver_count,
        "total_features": feature_count,
        "total_anomalies": anomaly_count,
        "active_sensors": len(simulator.sensors),
        "pipeline_health": "healthy" if anomaly_model else "initializing"
    }

@app.get("/api/sensors")
async def get_sensors():
    """Get list of active sensors"""
    return {
        "sensors": [
            {
                "sensor_id": sensor_id,
                "type": config["type"],
                "location": config["location"],
                "unit": config["unit"],
                "normal_range": config["normal_range"]
            }
            for sensor_id, config in simulator.sensors.items()
        ]
    }

@app.get("/api/sensors/{sensor_id}/recent")
async def get_recent_sensor_data(sensor_id: str, limit: int = 50):
    """Get recent data for a specific sensor"""
    
    recent_data = list(silver_collection.find(
        {"sensor_id": sensor_id}
    ).sort("processed_timestamp", -1).limit(limit))
    
    return {
        "sensor_id": sensor_id,
        "data": recent_data
    }

@app.get("/api/anomalies/recent")
async def get_recent_anomalies(limit: int = 20):
    """Get recent anomalies across all sensors"""
    
    recent_anomalies = list(anomaly_collection.find(
        {"is_anomaly": True}
    ).sort("timestamp", -1).limit(limit))
    
    return {
        "anomalies": recent_anomalies
    }

@app.get("/api/features/{sensor_id}")
async def get_sensor_features(sensor_id: str):
    """Get latest features for a sensor"""
    
    latest_features = feature_collection.find_one(
        {"sensor_id": sensor_id},
        sort=[("timestamp", -1)]
    )
    
    return {
        "sensor_id": sensor_id,
        "features": latest_features
    }

@app.post("/api/sensors/simulate")
async def simulate_sensor_batch(batch_size: int = 10, inject_anomalies: bool = True):
    """Manually trigger sensor data simulation"""
    
    readings = simulator.generate_batch(batch_size, inject_anomalies)
    await process_sensor_batch(readings)
    
    return {
        "message": f"Generated and processed {len(readings)} sensor readings",
        "batch_size": len(readings)
    }

@app.post("/api/model/retrain")
async def retrain_model():
    """Retrain the anomaly detection model"""
    
    await train_anomaly_model()
    
    return {
        "message": "Model retrained successfully",
        "model_ready": anomaly_model is not None
    }

@app.get("/api/export/pyspark")
async def export_pyspark_code():
    """Export equivalent PySpark code for the pipeline"""
    
    pyspark_code = generate_pyspark_code()
    
    return {
        "pyspark_code": pyspark_code,
        "description": "Production-ready PySpark implementation of the EWS pipeline"
    }

def generate_pyspark_code() -> str:
    """Generate equivalent PySpark code"""
    
    return '''
# PySpark Implementation of EWS Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from delta.tables import *
import pyspark.sql.functions as F

# Initialize Spark Session
spark = SparkSession.builder \\
    .appName("EWS_Pipeline") \\
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \\
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \\
    .getOrCreate()

# Bronze Layer - Raw Data Ingestion
def ingest_to_bronze(input_path, bronze_path):
    """Ingest raw IoT sensor data to Bronze layer"""
    
    schema = StructType([
        StructField("sensor_id", StringType(), True),
        StructField("sensor_type", StringType(), True),
        StructField("value", DoubleType(), True),
        StructField("unit", StringType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("location", StringType(), True)
    ])
    
    # Read streaming data
    bronze_df = spark \\
        .readStream \\
        .format("json") \\
        .schema(schema) \\
        .option("path", input_path) \\
        .load()
    
    # Add metadata columns
    bronze_df = bronze_df \\
        .withColumn("ingestion_timestamp", current_timestamp()) \\
        .withColumn("source", lit("iot_sensors"))
    
    # Write to Delta Bronze table
    query = bronze_df.writeStream \\
        .format("delta") \\
        .outputMode("append") \\
        .option("checkpointLocation", f"{bronze_path}/_checkpoints") \\
        .start(bronze_path)
    
    return query

# Silver Layer - Data Cleansing and Validation
def process_to_silver(bronze_path, silver_path):
    """Process Bronze data to Silver layer with cleansing"""
    
    bronze_df = spark \\
        .readStream \\
        .format("delta") \\
        .load(bronze_path)
    
    # Data quality and cleansing
    silver_df = bronze_df \\
        .withColumn("is_valid", 
                   when((col("value").isNotNull()) & 
                        (col("value") > -999) & 
                        (col("value") < 999), True).otherwise(False)) \\
        .withColumn("cleaned_value", 
                   when(col("is_valid"), col("value")).otherwise(0.0)) \\
        .withColumn("quality_score", 
                   when(col("is_valid"), 1.0).otherwise(0.0)) \\
        .withColumn("processed_timestamp", current_timestamp())
    
    # Write to Delta Silver table
    query = silver_df.writeStream \\
        .format("delta") \\
        .outputMode("append") \\
        .option("checkpointLocation", f"{silver_path}/_checkpoints") \\
        .start(silver_path)
    
    return query

# Feature Engineering
def generate_features(silver_path, features_path):
    """Generate time-series features"""
    
    silver_df = spark \\
        .readStream \\
        .format("delta") \\
        .load(silver_path) \\
        .where(col("is_valid") == True)
    
    # Window-based aggregations
    windowed_df = silver_df \\
        .withWatermark("processed_timestamp", "10 minutes") \\
        .groupBy(
            col("sensor_id"),
            window(col("processed_timestamp"), "5 minutes", "1 minute")
        ) \\
        .agg(
            mean("cleaned_value").alias("mean_5min"),
            stddev("cleaned_value").alias("std_5min"),
            min("cleaned_value").alias("min_5min"),
            max("cleaned_value").alias("max_5min"),
            count("cleaned_value").alias("count_5min")
        )
    
    # Calculate additional features
    features_df = windowed_df \\
        .withColumn("range_5min", col("max_5min") - col("min_5min")) \\
        .withColumn("timestamp", col("window.start")) \\
        .drop("window")
    
    # Write features to Delta table
    query = features_df.writeStream \\
        .format("delta") \\
        .outputMode("append") \\
        .option("checkpointLocation", f"{features_path}/_checkpoints") \\
        .start(features_path)
    
    return query

# Anomaly Detection (using K-means as example)
def detect_anomalies(features_path, anomalies_path):
    """Detect anomalies using unsupervised learning"""
    
    # Load features for batch training
    features_df = spark.read.format("delta").load(features_path)
    
    # Prepare feature vector
    feature_cols = ["mean_5min", "std_5min", "range_5min", "count_5min"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    # Scale features
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    
    # K-means clustering for anomaly detection
    kmeans = KMeans(featuresCol="scaled_features", k=3, predictionCol="cluster")
    
    # Create pipeline
    pipeline = Pipeline(stages=[assembler, scaler, kmeans])
    
    # Train model
    model = pipeline.fit(features_df)
    
    # Apply to streaming data
    streaming_features = spark \\
        .readStream \\
        .format("delta") \\
        .load(features_path)
    
    # Predict clusters
    predictions = model.transform(streaming_features)
    
    # Define anomaly logic (cluster 0 as normal, others as anomalies)
    anomalies_df = predictions \\
        .withColumn("is_anomaly", when(col("cluster") != 0, True).otherwise(False)) \\
        .withColumn("anomaly_score", rand()) \\  # Replace with actual distance calculation
        .withColumn("detection_timestamp", current_timestamp())
    
    # Write anomalies to Delta table
    query = anomalies_df.writeStream \\
        .format("delta") \\
        .outputMode("append") \\
        .option("checkpointLocation", f"{anomalies_path}/_checkpoints") \\
        .start(anomalies_path)
    
    return query

# Main Pipeline Execution
def run_ews_pipeline(config):
    """Run the complete EWS pipeline"""
    
    # Paths
    bronze_path = config["bronze_path"]
    silver_path = config["silver_path"] 
    features_path = config["features_path"]
    anomalies_path = config["anomalies_path"]
    
    # Start pipeline stages
    bronze_query = ingest_to_bronze(config["input_path"], bronze_path)
    silver_query = process_to_silver(bronze_path, silver_path)
    features_query = generate_features(silver_path, features_path)
    anomaly_query = detect_anomalies(features_path, anomalies_path)
    
    # Wait for termination
    spark.streams.awaitAnyTermination()

# Configuration
config = {
    "input_path": "/mnt/data/raw_sensors/",
    "bronze_path": "/mnt/data/bronze/sensor_data/",
    "silver_path": "/mnt/data/silver/sensor_data/",
    "features_path": "/mnt/data/features/sensor_features/",
    "anomalies_path": "/mnt/data/gold/anomalies/"
}

# Run pipeline
if __name__ == "__main__":
    run_ews_pipeline(config)
'''

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)