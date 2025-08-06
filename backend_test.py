#!/usr/bin/env python3
"""
Comprehensive Backend API Testing for Early Warning System ML Pipeline
Tests all API endpoints and ML pipeline functionality
"""

import requests
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, List

class EWSPipelineAPITester:
    def __init__(self, base_url: str = "https://9a9ff788-fc70-480d-8dda-1d7d62c601cd.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name: str, success: bool, message: str = "", response_data: Any = None):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name}: PASSED - {message}")
        else:
            print(f"❌ {name}: FAILED - {message}")
        
        self.test_results.append({
            "test": name,
            "success": success,
            "message": message,
            "response_data": response_data
        })

    def make_request(self, method: str, endpoint: str, data: Dict = None, timeout: int = 30) -> tuple:
        """Make HTTP request and return success status and response"""
        url = f"{self.base_url}{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            else:
                return False, {"error": f"Unsupported method: {method}"}
            
            # Try to parse JSON response
            try:
                response_data = response.json()
            except:
                response_data = {"raw_response": response.text}
            
            return response.status_code == 200, response_data, response.status_code
            
        except requests.exceptions.Timeout:
            return False, {"error": "Request timeout"}, 408
        except requests.exceptions.ConnectionError:
            return False, {"error": "Connection error"}, 503
        except Exception as e:
            return False, {"error": str(e)}, 500

    def test_health_endpoint(self):
        """Test /api/health endpoint"""
        success, response_data, status_code = self.make_request('GET', '/api/health')
        
        if success:
            required_fields = ['status', 'pipeline_initialized', 'streaming', 'timestamp']
            missing_fields = [field for field in required_fields if field not in response_data]
            
            if missing_fields:
                self.log_test("Health Endpoint", False, f"Missing fields: {missing_fields}", response_data)
            else:
                pipeline_status = "initialized" if response_data.get('pipeline_initialized') else "not initialized"
                streaming_status = "active" if response_data.get('streaming') else "inactive"
                self.log_test("Health Endpoint", True, 
                            f"Status: {response_data.get('status')}, Pipeline: {pipeline_status}, Streaming: {streaming_status}")
        else:
            self.log_test("Health Endpoint", False, f"HTTP {status_code}: {response_data.get('error', 'Unknown error')}")

    def test_dashboard_stats(self):
        """Test /api/dashboard/stats endpoint"""
        success, response_data, status_code = self.make_request('GET', '/api/dashboard/stats')
        
        if success:
            required_fields = ['total_ingested', 'total_processed', 'total_features', 'total_anomalies', 'active_sensors', 'pipeline_health']
            missing_fields = [field for field in required_fields if field not in response_data]
            
            if missing_fields:
                self.log_test("Dashboard Stats", False, f"Missing fields: {missing_fields}", response_data)
            else:
                stats_summary = f"Ingested: {response_data.get('total_ingested', 0)}, " \
                              f"Processed: {response_data.get('total_processed', 0)}, " \
                              f"Features: {response_data.get('total_features', 0)}, " \
                              f"Anomalies: {response_data.get('total_anomalies', 0)}, " \
                              f"Sensors: {response_data.get('active_sensors', 0)}"
                self.log_test("Dashboard Stats", True, stats_summary, response_data)
        else:
            self.log_test("Dashboard Stats", False, f"HTTP {status_code}: {response_data.get('error', 'Unknown error')}")

    def test_sensors_list(self):
        """Test /api/sensors endpoint"""
        success, response_data, status_code = self.make_request('GET', '/api/sensors')
        
        if success:
            if 'sensors' not in response_data:
                self.log_test("Sensors List", False, "Missing 'sensors' field in response", response_data)
                return []
            
            sensors = response_data['sensors']
            expected_sensor_count = 6
            
            if len(sensors) != expected_sensor_count:
                self.log_test("Sensors List", False, f"Expected {expected_sensor_count} sensors, got {len(sensors)}")
                return sensors
            
            # Validate sensor structure
            required_sensor_fields = ['sensor_id', 'type', 'location', 'unit', 'normal_range']
            sensor_types = set()
            
            for sensor in sensors:
                missing_fields = [field for field in required_sensor_fields if field not in sensor]
                if missing_fields:
                    self.log_test("Sensors List", False, f"Sensor {sensor.get('sensor_id', 'unknown')} missing fields: {missing_fields}")
                    return sensors
                sensor_types.add(sensor['type'])
            
            expected_types = {'temperature', 'pressure', 'humidity', 'vibration'}
            if not expected_types.issubset(sensor_types):
                missing_types = expected_types - sensor_types
                self.log_test("Sensors List", False, f"Missing sensor types: {missing_types}")
            else:
                sensor_summary = f"{len(sensors)} sensors with types: {', '.join(sorted(sensor_types))}"
                self.log_test("Sensors List", True, sensor_summary)
            
            return sensors
        else:
            self.log_test("Sensors List", False, f"HTTP {status_code}: {response_data.get('error', 'Unknown error')}")
            return []

    def test_simulate_data(self):
        """Test /api/sensors/simulate endpoint"""
        payload = {
            "batch_size": 10,
            "inject_anomalies": True
        }
        
        success, response_data, status_code = self.make_request('POST', '/api/sensors/simulate', payload)
        
        if success:
            if 'message' in response_data and 'batch_size' in response_data:
                batch_size = response_data.get('batch_size', 0)
                self.log_test("Data Simulation", True, f"Generated {batch_size} sensor readings")
            else:
                self.log_test("Data Simulation", False, "Invalid response format", response_data)
        else:
            self.log_test("Data Simulation", False, f"HTTP {status_code}: {response_data.get('error', 'Unknown error')}")

    def test_model_retrain(self):
        """Test /api/model/retrain endpoint"""
        success, response_data, status_code = self.make_request('POST', '/api/model/retrain', timeout=60)
        
        if success:
            if 'message' in response_data and 'model_ready' in response_data:
                model_status = "ready" if response_data.get('model_ready') else "not ready"
                self.log_test("Model Retrain", True, f"Model retrained successfully, Status: {model_status}")
            else:
                self.log_test("Model Retrain", False, "Invalid response format", response_data)
        else:
            self.log_test("Model Retrain", False, f"HTTP {status_code}: {response_data.get('error', 'Unknown error')}")

    def test_recent_anomalies(self):
        """Test /api/anomalies/recent endpoint"""
        success, response_data, status_code = self.make_request('GET', '/api/anomalies/recent')
        
        if success:
            if 'anomalies' not in response_data:
                self.log_test("Recent Anomalies", False, "Missing 'anomalies' field in response", response_data)
                return
            
            anomalies = response_data['anomalies']
            anomaly_count = len(anomalies)
            
            if anomaly_count > 0:
                # Validate anomaly structure
                required_fields = ['sensor_id', 'timestamp', 'is_anomaly', 'anomaly_score', 'confidence']
                sample_anomaly = anomalies[0]
                missing_fields = [field for field in required_fields if field not in sample_anomaly]
                
                if missing_fields:
                    self.log_test("Recent Anomalies", False, f"Anomaly missing fields: {missing_fields}")
                else:
                    self.log_test("Recent Anomalies", True, f"Found {anomaly_count} recent anomalies")
            else:
                self.log_test("Recent Anomalies", True, "No recent anomalies (system healthy)")
        else:
            self.log_test("Recent Anomalies", False, f"HTTP {status_code}: {response_data.get('error', 'Unknown error')}")

    def test_sensor_data(self, sensors: List[Dict]):
        """Test /api/sensors/{sensor_id}/recent endpoint for each sensor"""
        if not sensors:
            self.log_test("Sensor Data", False, "No sensors available for testing")
            return
        
        # Test first sensor
        test_sensor = sensors[0]
        sensor_id = test_sensor['sensor_id']
        
        success, response_data, status_code = self.make_request('GET', f'/api/sensors/{sensor_id}/recent')
        
        if success:
            if 'sensor_id' not in response_data or 'data' not in response_data:
                self.log_test("Sensor Data", False, f"Invalid response format for sensor {sensor_id}", response_data)
                return
            
            sensor_data = response_data['data']
            data_count = len(sensor_data)
            
            if data_count > 0:
                # Validate data structure
                required_fields = ['sensor_id', 'cleaned_value', 'original_value', 'is_valid', 'quality_score']
                sample_data = sensor_data[0]
                missing_fields = [field for field in required_fields if field not in sample_data]
                
                if missing_fields:
                    self.log_test("Sensor Data", False, f"Sensor data missing fields: {missing_fields}")
                else:
                    self.log_test("Sensor Data", True, f"Retrieved {data_count} data points for sensor {sensor_id}")
            else:
                self.log_test("Sensor Data", True, f"No data available for sensor {sensor_id} (may be new)")
        else:
            self.log_test("Sensor Data", False, f"HTTP {status_code}: {response_data.get('error', 'Unknown error')}")

    def test_sensor_features(self, sensors: List[Dict]):
        """Test /api/features/{sensor_id} endpoint"""
        if not sensors:
            self.log_test("Sensor Features", False, "No sensors available for testing")
            return
        
        test_sensor = sensors[0]
        sensor_id = test_sensor['sensor_id']
        
        success, response_data, status_code = self.make_request('GET', f'/api/features/{sensor_id}')
        
        if success:
            if 'sensor_id' not in response_data:
                self.log_test("Sensor Features", False, f"Invalid response format for sensor {sensor_id}", response_data)
                return
            
            features = response_data.get('features')
            if features and 'features' in features:
                feature_count = len(features['features'])
                self.log_test("Sensor Features", True, f"Retrieved {feature_count} features for sensor {sensor_id}")
            else:
                self.log_test("Sensor Features", True, f"No features available for sensor {sensor_id} (may be new)")
        else:
            self.log_test("Sensor Features", False, f"HTTP {status_code}: {response_data.get('error', 'Unknown error')}")

    def test_export_pyspark(self):
        """Test /api/export/pyspark endpoint"""
        success, response_data, status_code = self.make_request('GET', '/api/export/pyspark')
        
        if success:
            if 'pyspark_code' not in response_data or 'description' not in response_data:
                self.log_test("PySpark Export", False, "Missing required fields in response", response_data)
                return
            
            pyspark_code = response_data['pyspark_code']
            code_length = len(pyspark_code)
            
            # Basic validation of PySpark code
            required_keywords = ['SparkSession', 'Delta', 'Pipeline', 'bronze', 'silver']
            missing_keywords = [kw for kw in required_keywords if kw not in pyspark_code]
            
            if missing_keywords:
                self.log_test("PySpark Export", False, f"PySpark code missing keywords: {missing_keywords}")
            else:
                self.log_test("PySpark Export", True, f"Generated PySpark code ({code_length} characters)")
        else:
            self.log_test("PySpark Export", False, f"HTTP {status_code}: {response_data.get('error', 'Unknown error')}")

    def run_comprehensive_test(self):
        """Run all API tests"""
        print("🚀 Starting Comprehensive EWS Pipeline API Testing")
        print("=" * 60)
        
        # Test basic endpoints
        print("\n📋 Testing Basic Endpoints...")
        self.test_health_endpoint()
        self.test_dashboard_stats()
        
        # Test sensors
        print("\n🔧 Testing Sensor Endpoints...")
        sensors = self.test_sensors_list()
        self.test_sensor_data(sensors)
        self.test_sensor_features(sensors)
        
        # Test ML pipeline operations
        print("\n🤖 Testing ML Pipeline Operations...")
        self.test_simulate_data()
        
        # Wait a bit for data processing
        print("⏳ Waiting for data processing...")
        time.sleep(5)
        
        # Test anomaly detection
        self.test_recent_anomalies()
        
        # Test model operations (this might take longer)
        print("\n🧠 Testing Model Operations...")
        self.test_model_retrain()
        
        # Test export functionality
        print("\n📤 Testing Export Functionality...")
        self.test_export_pyspark()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed / self.tests_run * 100):.1f}%")
        
        # Print failed tests
        failed_tests = [result for result in self.test_results if not result['success']]
        if failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"  • {test['test']}: {test['message']}")
        
        return self.tests_passed == self.tests_run

def main():
    """Main test execution"""
    print("Early Warning System ML Pipeline - Backend API Testing")
    print(f"Test started at: {datetime.now()}")
    
    tester = EWSPipelineAPITester()
    success = tester.run_comprehensive_test()
    
    print(f"\nTest completed at: {datetime.now()}")
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())