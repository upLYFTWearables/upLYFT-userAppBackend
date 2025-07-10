import React, { useState, useEffect, useRef, useCallback } from 'react';
import Plot from 'react-plotly.js';
import './MedicalRehabDashboard.css';

// Configuration constants
const WINDOW_SIZE = 1000;
const PLOT_WINDOW = 5.0;
const FS_DEFAULT = 100;
const FILTER_CUTOFF = 5.0;
const HEIGHT_THRESHOLD = 1.5;
const DISTANCE_THRESHOLD = 50;
const POWER_WINDOW_SIZE = 500;
const FREQ_WINDOW_SIZE = 60;
const MAX_POINTS = 300;
const ANNOTATION_LIMIT = 5;
const DATA_DECIMATION = 3;

// IMU Mapping
const IMU_MAPPING = {
  "IMU1": "Left_Below_Knee",
  "IMU2": "Right_Below_Knee",
  "IMU3": "Left_Above_Knee",
  "IMU4": "Right_Above_Knee",
  "IMU7": "Pelvis"
};

// Stickman kinematic model dimensions
const PELVIS_LENGTH = 1.0;
const PELVIS_WIDTH = 2.0;
const UPPER_LEG_LENGTH = 4.0;
const LOWER_LEG_LENGTH = 4.0;
const FOOT_LENGTH = 1.0;

// Utility functions
const butter_filter = (data) => {
  if (data.length < 10) return data;
  // Simple low-pass filter implementation
  const alpha = 0.1;
  const filtered = [data[0]];
  for (let i = 1; i < data.length; i++) {
    filtered[i] = alpha * data[i] + (1 - alpha) * filtered[i - 1];
  }
  return filtered;
};

const find_peaks = (signal, height, distance) => {
  const peaks = [];
  for (let i = 1; i < signal.length - 1; i++) {
    if (signal[i] > signal[i - 1] && 
        signal[i] > signal[i + 1] && 
        signal[i] > height) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] > distance) {
        peaks.push(i);
      }
    }
  }
  return peaks;
};

const quaternionToMatrix = (q) => {
  const [w, x, y, z] = q;
  const norm = Math.sqrt(w*w + x*x + y*y + z*z);
  if (norm < 1e-12) return [[1,0,0],[0,1,0],[0,0,1]];
  const [nw, nx, ny, nz] = [w/norm, x/norm, y/norm, z/norm];
  
  return [
    [1 - 2*(ny*ny + nz*nz), 2*(nx*ny - nw*nz), 2*(nx*nz + nw*ny)],
    [2*(nx*ny + nw*nz), 1 - 2*(nx*nx + nz*nz), 2*(ny*nz - nw*nx)],
    [2*(nx*nz - nw*ny), 2*(ny*nz + nw*nx), 1 - 2*(nx*nx + ny*ny)]
  ];
};

const matrixToEulerZYX = (R) => {
  const beta_y = -Math.asin(R[2][0]);
  const cos_beta = Math.cos(beta_y);
  let alpha_z, gamma_x;
  
  if (Math.abs(cos_beta) < 1e-6) {
    alpha_z = Math.atan2(-R[0][1], R[1][1]);
    gamma_x = 0.0;
  } else {
    alpha_z = Math.atan2(R[1][0], R[0][0]);
    gamma_x = Math.atan2(R[2][1], R[2][2]);
  }
  
  return [alpha_z, beta_y, gamma_x];
};

// Data generators
const useStepDataGenerator = () => {
  const [stepData, setStepData] = useState({});
  
  useEffect(() => {
    const interval = setInterval(() => {
      const t = Date.now() / 1000;
      const newData = {};
      
      Object.keys(IMU_MAPPING).forEach(imuId => {
        const baseAccel = 9.8;
        const walkingFreq = 1.5;
        const walkingAmplitude = 8.0;
        const accel = baseAccel + walkingAmplitude * Math.sin(2 * Math.PI * walkingFreq * t) + 
                     (Math.random() - 0.5) * 1.0;
        
        if (!newData[imuId]) {
          newData[imuId] = {
            timestamp: [],
            accel_magnitude: [],
            step_count: 0,
            cadence: 0
          };
        }
        
        // Keep only last WINDOW_SIZE points
        const existing = stepData[imuId] || { timestamp: [], accel_magnitude: [] };
        const timestamps = [...existing.timestamp, t].slice(-WINDOW_SIZE);
        const accels = [...existing.accel_magnitude, accel].slice(-WINDOW_SIZE);
        
        newData[imuId] = {
          timestamp: timestamps,
          accel_magnitude: accels,
          step_count: Math.floor(walkingFreq * (t % 60)),
          cadence: walkingFreq * 60
        };
      });
      
      setStepData(newData);
    }, 10); // 100 Hz simulation
    
    return () => clearInterval(interval);
  }, [stepData]);
  
  return stepData;
};

const usePowerDataGenerator = () => {
  const [powerData, setPowerData] = useState({
    timestamp: [],
    accel_x: [], accel_y: [], accel_z: [],
    gyro_x: [], gyro_y: [], gyro_z: []
  });
  
  useEffect(() => {
    const interval = setInterval(() => {
      const t = Date.now() / 1000;
      const baseAccel = 9.8;
      const walkingFreq = 1.5;
      const walkingAmplitude = 8.0;
      
      const accel_x = baseAccel * 0.1 + walkingAmplitude * Math.sin(2 * Math.PI * walkingFreq * t) + (Math.random() - 0.5) * 1.0;
      const accel_y = baseAccel * 0.1 + walkingAmplitude * Math.cos(2 * Math.PI * walkingFreq * t * 1.1) + (Math.random() - 0.5) * 1.0;
      const accel_z = baseAccel + walkingAmplitude * 0.3 * Math.sin(2 * Math.PI * walkingFreq * t * 0.8) + (Math.random() - 0.5) * 0.6;
      
      const gyro_x = 10 * Math.sin(2 * Math.PI * walkingFreq * t * 0.7) + (Math.random() - 0.5) * 2.0;
      const gyro_y = 10 * Math.cos(2 * Math.PI * walkingFreq * t * 0.9) + (Math.random() - 0.5) * 2.0;
      const gyro_z = 5 * Math.sin(2 * Math.PI * walkingFreq * t * 1.2) + (Math.random() - 0.5) * 1.0;
      
      setPowerData(prev => ({
        timestamp: [...prev.timestamp, t * 1e6].slice(-POWER_WINDOW_SIZE),
        accel_x: [...prev.accel_x, accel_x].slice(-POWER_WINDOW_SIZE),
        accel_y: [...prev.accel_y, accel_y].slice(-POWER_WINDOW_SIZE),
        accel_z: [...prev.accel_z, accel_z].slice(-POWER_WINDOW_SIZE),
        gyro_x: [...prev.gyro_x, gyro_x].slice(-POWER_WINDOW_SIZE),
        gyro_y: [...prev.gyro_y, gyro_y].slice(-POWER_WINDOW_SIZE),
        gyro_z: [...prev.gyro_z, gyro_z].slice(-POWER_WINDOW_SIZE)
      }));
    }, 10);
    
    return () => clearInterval(interval);
  }, []);
  
  return powerData;
};

const useFrequencyDataGenerator = () => {
  const [freqData, setFreqData] = useState({});
  const [freqStats, setFreqStats] = useState({});
  const startTime = useRef(Date.now() / 1000);
  
  useEffect(() => {
    const interval = setInterval(() => {
      const currentTime = Date.now() / 1000;
      const relativeTime = currentTime - startTime.current;
      
      const imuList = ["IMU1", "IMU2", "IMU3", "IMU4", "IMU5", "IMU6", "IMU7"];
      const newFreqData = { ...freqData };
      const newFreqStats = { ...freqStats };
      
      imuList.forEach(imuId => {
        const baseFreq = 200;
        const variation = 20 * Math.sin(2 * Math.PI * 0.1 * currentTime) + (Math.random() - 0.5) * 10;
        const frequency = Math.max(80, Math.min(250, baseFreq + variation));
        
        if (!newFreqData[imuId]) {
          newFreqData[imuId] = { frequencies: [], times: [] };
        }
        
        // Add new data point every 0.8 seconds
        const lastTime = newFreqData[imuId].times[newFreqData[imuId].times.length - 1] || 0;
        if (relativeTime - lastTime >= 0.8) {
          newFreqData[imuId].frequencies.push(frequency);
          newFreqData[imuId].times.push(relativeTime);
          
          // Keep only last MAX_POINTS
          if (newFreqData[imuId].frequencies.length > MAX_POINTS) {
            newFreqData[imuId].frequencies = newFreqData[imuId].frequencies.slice(-MAX_POINTS);
            newFreqData[imuId].times = newFreqData[imuId].times.slice(-MAX_POINTS);
          }
          
          // Update stats
          newFreqStats[imuId] = {
            current: frequency,
            avg: newFreqData[imuId].frequencies.reduce((a, b) => a + b, 0) / newFreqData[imuId].frequencies.length,
            count: newFreqData[imuId].frequencies.length
          };
        }
      });
      
      setFreqData(newFreqData);
      setFreqStats(newFreqStats);
    }, 10);
    
    return () => clearInterval(interval);
  }, [freqData, freqStats]);
  
  return { freqData, freqStats };
};

const useStickmanDataGenerator = () => {
  const [stickmanData, setStickmanData] = useState({
    IMU1: { quat: [1.0, 0.0, 0.0, 0.0], timestamp: 0 },
    IMU2: { quat: [1.0, 0.0, 0.0, 0.0], timestamp: 0 },
    IMU3: { quat: [1.0, 0.0, 0.0, 0.0], timestamp: 0 },
    IMU4: { quat: [1.0, 0.0, 0.0, 0.0], timestamp: 0 },
    IMU7: { quat: [1.0, 0.0, 0.0, 0.0], timestamp: 0 }
  });
  
  useEffect(() => {
    const interval = setInterval(() => {
      const currentTime = Date.now() / 1000;
      const walkingFreq = 0.5;
      const t = currentTime * walkingFreq;
      
      const newData = {};
      
      Object.keys(stickmanData).forEach(imuId => {
        let angle = 0;
        
        if (imuId === 'IMU7') { // Pelvis
          angle = 0.1 * Math.sin(2 * Math.PI * t) + (Math.random() - 0.5) * 0.04;
        } else if (imuId === 'IMU1') { // Left upper leg
          angle = 0.3 * Math.sin(2 * Math.PI * t) + (Math.random() - 0.5) * 0.1;
        } else if (imuId === 'IMU3') { // Left lower leg
          angle = 0.2 * Math.sin(2 * Math.PI * t + Math.PI/4) + (Math.random() - 0.5) * 0.1;
        } else if (imuId === 'IMU2') { // Right upper leg
          angle = 0.3 * Math.sin(2 * Math.PI * t + Math.PI) + (Math.random() - 0.5) * 0.1;
        } else if (imuId === 'IMU4') { // Right lower leg
          angle = 0.2 * Math.sin(2 * Math.PI * t + Math.PI + Math.PI/4) + (Math.random() - 0.5) * 0.1;
        }
        
        const quat = [Math.cos(angle/2), Math.sin(angle/2), 0, 0];
        const norm = Math.sqrt(quat.reduce((sum, q) => sum + q*q, 0));
        
        newData[imuId] = {
          quat: norm > 0 ? quat.map(q => q/norm) : [1, 0, 0, 0],
          timestamp: currentTime
        };
      });
      
      setStickmanData(newData);
    }, 100); // 10 Hz simulation
    
    return () => clearInterval(interval);
  }, []);
  
  return stickmanData;
};

const usePelvicDataGenerator = () => {
  const [pelvicData, setPelvicData] = useState({
    timestamp: [],
    tilt: [],
    obliquity: [],
    rotation: []
  });
  const [isCalibrated, setIsCalibrated] = useState(false);
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const startTime = useRef(Date.now() / 1000);
  const calibrationQuats = useRef([]);
  const R0 = useRef([[1,0,0],[0,1,0],[0,0,1]]);
  
  useEffect(() => {
    const interval = setInterval(() => {
      const currentTime = Date.now() / 1000;
      const walkingFreq = 0.3;
      const t = (currentTime - startTime.current) * walkingFreq;
      
      const baseAngle = 0.1 * Math.sin(2 * Math.PI * t) + (Math.random() - 0.5) * 0.04;
      let quat = [
        Math.cos(baseAngle/2),
        Math.sin(baseAngle/2) * 0.5,
        Math.sin(baseAngle/2) * 0.3,
        Math.sin(baseAngle/2) * 0.2
      ];
      
      const norm = Math.sqrt(quat.reduce((sum, q) => sum + q*q, 0));
      if (norm > 0) quat = quat.map(q => q/norm);
      
      // Calibration phase
      if (!isCalibrated) {
        calibrationQuats.current.push(quat);
        setCalibrationProgress(calibrationQuats.current.length);
        
        if (calibrationQuats.current.length >= 100) {
          // Compute calibration matrix
          let R_sum = [[0,0,0],[0,0,0],[0,0,0]];
          calibrationQuats.current.forEach(q => {
            const R = quaternionToMatrix(q);
            for (let i = 0; i < 3; i++) {
              for (let j = 0; j < 3; j++) {
                R_sum[i][j] += R[i][j];
              }
            }
          });
          
          // Average and orthogonalize
          for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
              R_sum[i][j] /= calibrationQuats.current.length;
            }
          }
          
          R0.current = R_sum;
          setIsCalibrated(true);
        }
        return;
      }
      
      // Process quaternion to angles
      const R_current = quaternionToMatrix(quat);
      
      // Apply calibration (simplified)
      const [alpha_z, beta_y, gamma_x] = matrixToEulerZYX(R_current);
      
      let tilt = -beta_y * 180 / Math.PI;
      let obliquity = gamma_x * 180 / Math.PI;
      let rotation = alpha_z * 180 / Math.PI;
      
      // Add realistic variation
      tilt += 5 * Math.sin(2 * Math.PI * t * 1.2) + (Math.random() - 0.5) * 2;
      obliquity += 3 * Math.sin(2 * Math.PI * t * 0.8) + (Math.random() - 0.5) * 1;
      rotation += 2 * Math.sin(2 * Math.PI * t * 1.5) + (Math.random() - 0.5) * 0.6;
      
      setPelvicData(prev => ({
        timestamp: [...prev.timestamp, currentTime],
        tilt: [...prev.tilt, tilt],
        obliquity: [...prev.obliquity, obliquity],
        rotation: [...prev.rotation, rotation]
      }));
      
    }, 50); // 20 Hz simulation
    
    return () => clearInterval(interval);
  }, [isCalibrated]);
  
  return { pelvicData, isCalibrated, calibrationProgress, startTime: startTime.current };
};

// React Components for each visualization
const StepLengthVisualization = () => {
  const stepData = useStepDataGenerator();
  
  const getStepPlots = () => {
    const plots = [];
    const currentTime = Date.now() / 1000;
    
    Object.entries(IMU_MAPPING).forEach(([imuId, imuName], index) => {
      const buffer = stepData[imuId];
      if (!buffer || buffer.timestamp.length === 0) return;
      
      const timeWindow = currentTime - PLOT_WINDOW;
      const validIndices = buffer.timestamp.map((t, i) => t >= timeWindow ? i : -1).filter(i => i !== -1);
      
      if (validIndices.length === 0) return;
      
      const plotTimes = validIndices.map(i => buffer.timestamp[i] - currentTime);
      const plotAccel = validIndices.map(i => buffer.accel_magnitude[i]);
      const filteredAccel = butter_filter(plotAccel);
      const peaks = find_peaks(filteredAccel, HEIGHT_THRESHOLD, DISTANCE_THRESHOLD);
      
      plots.push({
        data: [
          {
            x: plotTimes,
            y: plotAccel,
            type: 'scatter',
            mode: 'lines',
            name: 'Raw Accel',
            line: { color: '#4285f4', width: 1 },
            opacity: 0.5
          },
          {
            x: plotTimes,
            y: filteredAccel,
            type: 'scatter',
            mode: 'lines',
            name: 'Filtered',
            line: { color: '#34a853', width: 2 }
          },
          ...(peaks.length > 0 ? [{
            x: peaks.map(p => plotTimes[p]),
            y: peaks.map(p => filteredAccel[p]),
            type: 'scatter',
            mode: 'markers',
            name: 'Steps',
            marker: { color: 'red', size: 8, symbol: 'x' }
          }] : [])
        ],
        layout: {
          title: { text: imuName, font: { color: '#2c3e50', size: 12 } },
          xaxis: { 
            title: 'Time (s)', 
            range: [-PLOT_WINDOW, 0],
            color: '#2c3e50',
            gridcolor: '#e0e0e0'
          },
          yaxis: { 
            title: 'Accel (m/s²)', 
            range: [8, 18],
            color: '#2c3e50',
            gridcolor: '#e0e0e0'
          },
          plot_bgcolor: '#ffffff',
          paper_bgcolor: '#ffffff',
          font: { color: '#2c3e50' },
          showlegend: true,
          legend: { orientation: 'h', y: -0.2 },
          margin: { t: 40, b: 60, l: 60, r: 20 }
        },
        config: { displayModeBar: false },
        style: { height: '250px', marginBottom: '10px' }
      });
    });
    
    return plots;
  };
  
  const plots = getStepPlots();
  
  return (
    <div className="visualization-container">
      {plots.map((plot, index) => (
        <Plot
          key={index}
          data={plot.data}
          layout={plot.layout}
          config={plot.config}
          style={plot.style}
        />
      ))}
    </div>
  );
};

export default StepLengthVisualization; 