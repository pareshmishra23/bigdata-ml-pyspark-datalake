import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Badge } from './components/ui/badge';
import { Button } from './components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Progress } from './components/ui/progress';
import { Alert, AlertDescription } from './components/ui/alert';
import { AlertTriangle, Activity, Database, TrendingUp, Zap, RefreshCw, Play, Pause } from 'lucide-react';
import './App.css';

function App() {
  const [stats, setStats] = useState({});
  const [sensors, setSensors] = useState([]);
  const [anomalies, setAnomalies] = useState([]);
  const [selectedSensor, setSelectedSensor] = useState(null);
  const [sensorData, setSensorData] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

  const API_BASE = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

  // Fetch dashboard statistics
  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/dashboard/stats`);
      const data = await response.json();
      setStats(data);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  // Fetch sensors list
  const fetchSensors = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/sensors`);
      const data = await response.json();
      setSensors(data.sensors || []);
    } catch (error) {
      console.error('Error fetching sensors:', error);
    }
  };

  // Fetch recent anomalies
  const fetchAnomalies = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/anomalies/recent`);
      const data = await response.json();
      setAnomalies(data.anomalies || []);
    } catch (error) {
      console.error('Error fetching anomalies:', error);
    }
  };

  // Fetch sensor data
  const fetchSensorData = async (sensorId) => {
    try {
      const response = await fetch(`${API_BASE}/api/sensors/${sensorId}/recent`);
      const data = await response.json();
      setSensorData(data.data || []);
    } catch (error) {
      console.error('Error fetching sensor data:', error);
    }
  };

  // Simulate new data batch
  const simulateData = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/sensors/simulate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          batch_size: 12,
          inject_anomalies: true
        })
      });
      const data = await response.json();
      console.log('Simulation result:', data);
      
      // Refresh data after simulation
      setTimeout(() => {
        fetchStats();
        fetchAnomalies();
        if (selectedSensor) {
          fetchSensorData(selectedSensor);
        }
      }, 2000);
    } catch (error) {
      console.error('Error simulating data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Retrain model
  const retrainModel = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/model/retrain`, {
        method: 'POST'
      });
      const data = await response.json();
      console.log('Retrain result:', data);
    } catch (error) {
      console.error('Error retraining model:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-refresh data
  useEffect(() => {
    fetchStats();
    fetchSensors();
    fetchAnomalies();

    const interval = setInterval(() => {
      fetchStats();
      fetchAnomalies();
    }, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, []);

  // Fetch sensor data when sensor is selected
  useEffect(() => {
    if (selectedSensor) {
      fetchSensorData(selectedSensor);
    }
  }, [selectedSensor]);

  const getSensorTypeColor = (type) => {
    const colors = {
      temperature: 'bg-red-100 text-red-800',
      pressure: 'bg-blue-100 text-blue-800',
      humidity: 'bg-green-100 text-green-800',
      vibration: 'bg-purple-100 text-purple-800'
    };
    return colors[type] || 'bg-gray-100 text-gray-800';
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatValue = (value, unit) => {
    return `${value?.toFixed(3)} ${unit}`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-r from-blue-500 to-purple-600 p-2 rounded-lg">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900">Early Warning System</h1>
                <p className="text-sm text-slate-600">ML Pipeline Dashboard</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <Button
                variant="outline"
                size="sm"
                onClick={simulateData}
                disabled={isLoading}
                className="bg-white hover:bg-slate-50"
              >
                {isLoading ? (
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Play className="w-4 h-4 mr-2" />
                )}
                Simulate Data
              </Button>
              
              <Button
                variant="outline"
                size="sm"
                onClick={retrainModel}
                disabled={isLoading}
                className="bg-white hover:bg-slate-50"
              >
                <TrendingUp className="w-4 h-4 mr-2" />
                Retrain Model
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card className="border-0 shadow-lg bg-gradient-to-br from-blue-50 to-blue-100">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-blue-600">Total Ingested</p>
                  <p className="text-3xl font-bold text-blue-900">{stats.total_ingested || 0}</p>
                </div>
                <Database className="w-8 h-8 text-blue-500" />
              </div>
            </CardContent>
          </Card>

          <Card className="border-0 shadow-lg bg-gradient-to-br from-green-50 to-green-100">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-green-600">Processed Records</p>
                  <p className="text-3xl font-bold text-green-900">{stats.total_processed || 0}</p>
                </div>
                <Activity className="w-8 h-8 text-green-500" />
              </div>
            </CardContent>
          </Card>

          <Card className="border-0 shadow-lg bg-gradient-to-br from-purple-50 to-purple-100">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-purple-600">Features Generated</p>
                  <p className="text-3xl font-bold text-purple-900">{stats.total_features || 0}</p>
                </div>
                <TrendingUp className="w-8 h-8 text-purple-500" />
              </div>
            </CardContent>
          </Card>

          <Card className="border-0 shadow-lg bg-gradient-to-br from-red-50 to-red-100">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-red-600">Anomalies Detected</p>
                  <p className="text-3xl font-bold text-red-900">{stats.total_anomalies || 0}</p>
                </div>
                <AlertTriangle className="w-8 h-8 text-red-500" />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Pipeline Status */}
        <Card className="mb-8 border-0 shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Zap className="w-5 h-5 text-yellow-500" />
              <span>Pipeline Status</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-4">
                <Badge variant={stats.pipeline_health === 'healthy' ? 'default' : 'secondary'}>
                  {stats.pipeline_health || 'Initializing'}
                </Badge>
                <Badge variant="outline">{stats.active_sensors || 0} Active Sensors</Badge>
              </div>
              {lastUpdated && (
                <p className="text-sm text-slate-500">
                  Last updated: {lastUpdated.toLocaleTimeString()}
                </p>
              )}
            </div>
            
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Data Ingestion</span>
                <span className="text-sm text-green-600">Active</span>
              </div>
              <Progress value={100} className="h-2" />
              
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Feature Engineering</span>
                <span className="text-sm text-green-600">Running</span>
              </div>
              <Progress value={85} className="h-2" />
              
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Anomaly Detection</span>
                <span className="text-sm text-green-600">Online</span>
              </div>
              <Progress value={100} className="h-2" />
            </div>
          </CardContent>
        </Card>

        {/* Main Content Tabs */}
        <Tabs defaultValue="sensors" className="space-y-6">
          <TabsList className="bg-white shadow-sm border">
            <TabsTrigger value="sensors">Sensors</TabsTrigger>
            <TabsTrigger value="anomalies">Anomalies</TabsTrigger>
            <TabsTrigger value="pipeline">Pipeline</TabsTrigger>
            <TabsTrigger value="export">Export</TabsTrigger>
          </TabsList>

          {/* Sensors Tab */}
          <TabsContent value="sensors" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Sensors List */}
              <div className="lg:col-span-1">
                <Card className="border-0 shadow-lg">
                  <CardHeader>
                    <CardTitle>Active Sensors</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {sensors.map((sensor) => (
                        <div
                          key={sensor.sensor_id}
                          className={`p-3 rounded-lg cursor-pointer transition-all ${
                            selectedSensor === sensor.sensor_id
                              ? 'bg-blue-50 border-2 border-blue-200'
                              : 'bg-slate-50 hover:bg-slate-100 border border-slate-200'
                          }`}
                          onClick={() => setSelectedSensor(sensor.sensor_id)}
                        >
                          <div className="flex items-center justify-between">
                            <div>
                              <p className="font-medium text-sm">{sensor.sensor_id}</p>
                              <p className="text-xs text-slate-600">{sensor.location}</p>
                            </div>
                            <Badge className={getSensorTypeColor(sensor.type)}>
                              {sensor.type}
                            </Badge>
                          </div>
                          <div className="mt-2 text-xs text-slate-500">
                            Range: {sensor.normal_range[0]} - {sensor.normal_range[1]} {sensor.unit}
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Sensor Details */}
              <div className="lg:col-span-2">
                <Card className="border-0 shadow-lg">
                  <CardHeader>
                    <CardTitle>
                      {selectedSensor ? `Sensor Data: ${selectedSensor}` : 'Select a Sensor'}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {selectedSensor ? (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="bg-slate-50 p-3 rounded-lg">
                            <p className="text-xs font-medium text-slate-600">Latest Value</p>
                            <p className="text-lg font-bold">
                              {sensorData[0] ? formatValue(sensorData[0].cleaned_value, sensorData[0].enriched_data?.unit) : 'N/A'}
                            </p>
                          </div>
                          <div className="bg-slate-50 p-3 rounded-lg">
                            <p className="text-xs font-medium text-slate-600">Quality Score</p>
                            <p className="text-lg font-bold">
                              {sensorData[0] ? (sensorData[0].quality_score * 100).toFixed(1) + '%' : 'N/A'}
                            </p>
                          </div>
                          <div className="bg-slate-50 p-3 rounded-lg">
                            <p className="text-xs font-medium text-slate-600">Status</p>
                            <Badge variant={sensorData[0]?.is_valid ? 'default' : 'destructive'}>
                              {sensorData[0]?.is_valid ? 'Valid' : 'Invalid'}
                            </Badge>
                          </div>
                          <div className="bg-slate-50 p-3 rounded-lg">
                            <p className="text-xs font-medium text-slate-600">Records</p>
                            <p className="text-lg font-bold">{sensorData.length}</p>
                          </div>
                        </div>

                        {/* Recent Readings */}
                        <div>
                          <h4 className="font-medium mb-3">Recent Readings</h4>
                          <div className="space-y-2 max-h-96 overflow-y-auto">
                            {sensorData.slice(0, 10).map((reading, index) => (
                              <div key={index} className="flex items-center justify-between p-2 bg-slate-50 rounded">
                                <div className="flex-1">
                                  <span className="font-medium">
                                    {formatValue(reading.cleaned_value, reading.enriched_data?.unit)}
                                  </span>
                                  {reading.cleaned_value !== reading.original_value && (
                                    <span className="text-xs text-slate-500 ml-2">
                                      (orig: {reading.original_value})
                                    </span>
                                  )}
                                </div>
                                <div className="text-xs text-slate-500">
                                  {formatTimestamp(reading.processed_timestamp)}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-12 text-slate-500">
                        <Activity className="w-12 h-12 mx-auto mb-4 opacity-50" />
                        <p>Select a sensor to view detailed data</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Anomalies Tab */}
          <TabsContent value="anomalies">
            <Card className="border-0 shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <AlertTriangle className="w-5 h-5 text-red-500" />
                  <span>Recent Anomalies</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {anomalies.length > 0 ? (
                  <div className="space-y-4">
                    {anomalies.map((anomaly, index) => (
                      <Alert key={index} className="border-red-200 bg-red-50">
                        <AlertTriangle className="w-4 h-4 text-red-500" />
                        <AlertDescription>
                          <div className="flex items-center justify-between">
                            <div>
                              <p className="font-medium text-red-800">
                                Anomaly detected in sensor <span className="font-mono">{anomaly.sensor_id}</span>
                              </p>
                              <p className="text-sm text-red-600 mt-1">
                                Score: {anomaly.anomaly_score?.toFixed(3)} | 
                                Confidence: {(anomaly.confidence * 100)?.toFixed(1)}% |
                                Time: {formatTimestamp(anomaly.timestamp)}
                              </p>
                            </div>
                            <Badge variant="destructive">Alert</Badge>
                          </div>
                        </AlertDescription>
                      </Alert>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12 text-slate-500">
                    <AlertTriangle className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>No recent anomalies detected</p>
                    <p className="text-sm mt-2">Your systems are operating normally</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Pipeline Tab */}
          <TabsContent value="pipeline">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card className="border-0 shadow-lg">
                <CardHeader>
                  <CardTitle>Pipeline Architecture</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    <div className="flex items-center space-x-4">
                      <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                        <Database className="w-6 h-6 text-blue-600" />
                      </div>
                      <div>
                        <h4 className="font-medium">Bronze Layer</h4>
                        <p className="text-sm text-slate-600">Raw sensor data ingestion</p>
                        <Badge variant="outline" className="mt-1">Active</Badge>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-4">
                      <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                        <Activity className="w-6 h-6 text-green-600" />
                      </div>
                      <div>
                        <h4 className="font-medium">Silver Layer</h4>
                        <p className="text-sm text-slate-600">Data cleansing & validation</p>
                        <Badge variant="outline" className="mt-1">Processing</Badge>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-4">
                      <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                        <TrendingUp className="w-6 h-6 text-purple-600" />
                      </div>
                      <div>
                        <h4 className="font-medium">Feature Engineering</h4>
                        <p className="text-sm text-slate-600">Time-series feature generation</p>
                        <Badge variant="outline" className="mt-1">Running</Badge>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-4">
                      <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center">
                        <AlertTriangle className="w-6 h-6 text-red-600" />
                      </div>
                      <div>
                        <h4 className="font-medium">Anomaly Detection</h4>
                        <p className="text-sm text-slate-600">ML-based anomaly detection</p>
                        <Badge variant="outline" className="mt-1">Online</Badge>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-0 shadow-lg">
                <CardHeader>
                  <CardTitle>Model Information</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="bg-slate-50 p-4 rounded-lg">
                      <h4 className="font-medium mb-2">Isolation Forest</h4>
                      <div className="text-sm text-slate-600 space-y-1">
                        <p>• Algorithm: Isolation Forest</p>
                        <p>• Contamination Rate: 10%</p>
                        <p>• Estimators: 100</p>
                        <p>• Feature Engineering: Time-series</p>
                      </div>
                    </div>
                    
                    <div className="bg-slate-50 p-4 rounded-lg">
                      <h4 className="font-medium mb-2">Features Used</h4>
                      <div className="text-sm text-slate-600 space-y-1">
                        <p>• Rolling averages (5, 10, 20, 50 windows)</p>
                        <p>• Statistical moments (mean, std, skew)</p>
                        <p>• Lag features (1, 2, 3, 5)</p>
                        <p>• Trend analysis</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Export Tab */}
          <TabsContent value="export">
            <Card className="border-0 shadow-lg">
              <CardHeader>
                <CardTitle>Export PySpark Code</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <p className="text-slate-600">
                    Export production-ready PySpark code equivalent to this pipeline for deployment 
                    on your Spark infrastructure with Delta Lake support.
                  </p>
                  
                  <div className="bg-slate-100 p-4 rounded-lg">
                    <h4 className="font-medium mb-2">Included Components:</h4>
                    <ul className="text-sm text-slate-600 space-y-1">
                      <li>• Bronze layer with Delta Lake ingestion</li>
                      <li>• Silver layer data quality processing</li>
                      <li>• Structured streaming feature engineering</li>
                      <li>• ML-based anomaly detection</li>
                      <li>• Complete pipeline orchestration</li>
                    </ul>
                  </div>
                  
                  <Button
                    onClick={() => {
                      fetch(`${API_BASE}/api/export/pyspark`)
                        .then(response => response.json())
                        .then(data => {
                          const blob = new Blob([data.pyspark_code], { type: 'text/plain' });
                          const url = URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url;
                          a.download = 'ews_pyspark_pipeline.py';
                          document.body.appendChild(a);
                          a.click();
                          document.body.removeChild(a);
                          URL.revokeObjectURL(url);
                        });
                    }}
                    className="bg-gradient-to-r from-blue-500 to-purple-600 text-white hover:from-blue-600 hover:to-purple-700"
                  >
                    <Database className="w-4 h-4 mr-2" />
                    Download PySpark Code
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}

export default App;