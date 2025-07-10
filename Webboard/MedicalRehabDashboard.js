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

// Utility functions
const butter_filter = (data) => {
  if (data.length < 10) return data;
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

// React Components
const StepLengthVisualization = ({ stepData }) => {
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

const PowerMetricsVisualization = ({ powerData }) => {
  const computePowerMetrics = () => {
    if (powerData.timestamp.length < 16) return null;
    
    const accel = [
      powerData.accel_x,
      powerData.accel_y,
      powerData.accel_z
    ];
    
    const gyro = [
      powerData.gyro_x,
      powerData.gyro_y,
      powerData.gyro_z
    ];
    
    const timestamps = powerData.timestamp;
    const minLength = Math.min(accel[0].length, gyro[0].length, timestamps.length);
    
    if (minLength < 16) return null;
    
    // Simplified power calculations for visualization
    const transTime = timestamps.slice(0, minLength).map((t, i) => (t - timestamps[0]) / 1e6);
    const transPower = accel[0].slice(0, minLength).map((ax, i) => {
      const ay = accel[1][i] || 0;
      const az = accel[2][i] || 0;
      return Math.sqrt(ax*ax + ay*ay + az*az) * 10 + (Math.random() - 0.5) * 5;
    });
    
    const rotPower = gyro[0].slice(0, minLength).map((gx, i) => {
      const gy = gyro[1][i] || 0;
      const gz = gyro[2][i] || 0;
      return Math.sqrt(gx*gx + gy*gy + gz*gz) * 2 + (Math.random() - 0.5) * 2;
    });
    
    const totalPower = transPower.map((tp, i) => tp + rotPower[i]);
    const powerWeight = totalPower.map(tp => tp / 70);
    
    // Rolling average
    const windowSize = Math.min(20, totalPower.length);
    const rollingAvg = totalPower.map((_, i) => {
      const start = Math.max(0, i - windowSize + 1);
      const window = totalPower.slice(start, i + 1);
      return window.reduce((a, b) => a + b, 0) / window.length;
    });
    
    return {
      time: transTime,
      transPower,
      rotPower,
      totalPower,
      powerWeight,
      rollingAvg
    };
  };
  
  const metrics = computePowerMetrics();
  
  if (!metrics) {
    return (
      <div className="visualization-container">
        <div className="placeholder-text">Waiting for power data...</div>
      </div>
    );
  }
  
  const commonLayout = {
    plot_bgcolor: '#ffffff',
    paper_bgcolor: '#ffffff',
    font: { color: '#2c3e50' },
    margin: { t: 40, b: 60, l: 60, r: 20 },
    showlegend: false,
    xaxis: { 
      color: '#2c3e50',
      gridcolor: '#e0e0e0'
    },
    yaxis: { 
      color: '#2c3e50',
      gridcolor: '#e0e0e0'
    }
  };
  
  return (
    <div className="power-plots-container">
      <Plot
        data={[
          {
            x: metrics.time,
            y: powerData.accel_x,
            type: 'scatter',
            mode: 'lines',
            name: 'AccelX',
            line: { color: 'blue' }
          },
          {
            x: metrics.time,
            y: powerData.accel_y,
            type: 'scatter',
            mode: 'lines',
            name: 'AccelY',
            line: { color: 'orange' }
          },
          {
            x: metrics.time,
            y: powerData.accel_z,
            type: 'scatter',
            mode: 'lines',
            name: 'AccelZ',
            line: { color: 'green' }
          }
        ]}
        layout={{
          ...commonLayout,
          title: { text: 'Raw Acceleration', font: { color: '#2c3e50' } },
          yaxis: { ...commonLayout.yaxis, title: 'm/s²' },
          showlegend: true
        }}
        config={{ displayModeBar: false }}
        className="power-plot"
      />
      
      <Plot
        data={[{
          x: metrics.time,
          y: metrics.transPower,
          type: 'scatter',
          mode: 'lines',
          line: { color: 'blue' }
        }]}
        layout={{
          ...commonLayout,
          title: { text: 'Translational Power (m·a)·v', font: { color: '#2c3e50' } },
          yaxis: { ...commonLayout.yaxis, title: 'W' }
        }}
        config={{ displayModeBar: false }}
        className="power-plot"
      />
      
      <Plot
        data={[{
          x: metrics.time,
          y: metrics.rotPower,
          type: 'scatter',
          mode: 'lines',
          line: { color: 'red' }
        }]}
        layout={{
          ...commonLayout,
          title: { text: 'Rotational Power τ·ω', font: { color: '#2c3e50' } },
          yaxis: { ...commonLayout.yaxis, title: 'W' }
        }}
        config={{ displayModeBar: false }}
        className="power-plot"
      />
      
      <Plot
        data={[{
          x: metrics.time,
          y: metrics.totalPower,
          type: 'scatter',
          mode: 'lines',
          line: { color: 'green' }
        }]}
        layout={{
          ...commonLayout,
          title: { text: 'Total Power', font: { color: '#2c3e50' } },
          yaxis: { ...commonLayout.yaxis, title: 'W' }
        }}
        config={{ displayModeBar: false }}
        className="power-plot"
      />
      
      <Plot
        data={[{
          x: metrics.time,
          y: metrics.powerWeight,
          type: 'scatter',
          mode: 'lines',
          line: { color: 'blue' }
        }]}
        layout={{
          ...commonLayout,
          title: { text: 'Instantaneous Total Power / Weight', font: { color: '#2c3e50' } },
          xaxis: { ...commonLayout.xaxis, title: 'Time (s)' },
          yaxis: { ...commonLayout.yaxis, title: 'W/kg' }
        }}
        config={{ displayModeBar: false }}
        className="power-plot"
      />
      
      <Plot
        data={[{
          x: metrics.time,
          y: metrics.rollingAvg,
          type: 'scatter',
          mode: 'lines',
          line: { color: 'blue' }
        }]}
        layout={{
          ...commonLayout,
          title: { text: 'Rolling Average Total Power', font: { color: '#2c3e50' } },
          xaxis: { ...commonLayout.xaxis, title: 'Time (s)' },
          yaxis: { ...commonLayout.yaxis, title: 'W' }
        }}
        config={{ displayModeBar: false }}
        className="power-plot"
      />
    </div>
  );
};

const FrequencyVisualization = ({ freqData, freqStats }) => {
  const getFrequencyPlots = () => {
    const plots = [];
    const imuList = Object.keys(freqData).sort();
    
    imuList.forEach((imuId, index) => {
      const data = freqData[imuId];
      if (!data || !data.frequencies || data.frequencies.length === 0) return;
      
      const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'];
      const color = colors[index % colors.length];
      
      plots.push(
        <Plot
          key={imuId}
          data={[
            {
              x: data.times,
              y: data.frequencies,
              type: 'scatter',
              mode: 'lines+markers',
              line: { color: color, width: 2 },
              marker: { color: color, size: 6 }
            }
          ]}
          layout={{
            title: { 
              text: `${imuId} – Frequency Over Time (Current: ${(freqStats[imuId]?.current || 0).toFixed(0)} Hz, Avg: ${(freqStats[imuId]?.avg || 0).toFixed(1)} Hz)`,
              font: { color: color, size: 12 }
            },
            xaxis: { 
              title: 'Time (seconds)',
              color: '#2c3e50',
              gridcolor: '#e0e0e0'
            },
            yaxis: { 
              title: 'Freq (Hz)',
              range: [80, 250],
              color: '#2c3e50',
              gridcolor: '#e0e0e0'
            },
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#ffffff',
            font: { color: '#2c3e50' },
            margin: { t: 40, b: 60, l: 60, r: 20 }
          }}
          config={{ displayModeBar: false }}
          className="frequency-plot"
        />
      );
    });
    
    return plots;
  };
  
  return (
    <div className="frequency-plots-container">
      {getFrequencyPlots()}
    </div>
  );
};

const StickmanVisualization = ({ stickmanData }) => {
  const createTransformationMatrix = (x = 0, y = 0, z = 0, quaternion = null) => {
    const T = [
      [1, 0, 0, x],
      [0, 1, 0, y],
      [0, 0, 1, z],
      [0, 0, 0, 1]
    ];
    
    if (quaternion) {
      const R = quaternionToMatrix(quaternion);
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          T[i][j] = R[i][j];
        }
      }
    }
    
    return T;
  };
  
  const matrixMultiply = (A, B) => {
    const result = Array(4).fill().map(() => Array(4).fill(0));
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        for (let k = 0; k < 4; k++) {
          result[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    return result;
  };
  
  const getPosition = (matrix) => [matrix[0][3], matrix[1][3], matrix[2][3]];
  
  const getStickmanData = () => {
    // Create joint transformations
    const pelvisJoint = createTransformationMatrix(0, 0, 0, stickmanData.IMU7.quat);
    
    // Calculate joint positions
    const pelvisPos = getPosition(pelvisJoint);
    
    // Left leg
    const leftHipTransform = matrixMultiply(pelvisJoint, createTransformationMatrix(0, 1, 0));
    const leftHipPos = getPosition(leftHipTransform);
    
    const leftKneeTransform = matrixMultiply(leftHipTransform, createTransformationMatrix(4, 0, 0, stickmanData.IMU1.quat));
    const leftKneePos = getPosition(leftKneeTransform);
    
    const leftAnkleTransform = matrixMultiply(leftKneeTransform, createTransformationMatrix(4, 0, 0, stickmanData.IMU3.quat));
    const leftAnklePos = getPosition(leftAnkleTransform);
    
    const leftFootPos = [leftAnklePos[0] + 1, leftAnklePos[1], leftAnklePos[2]];
    
    // Right leg
    const rightHipTransform = matrixMultiply(pelvisJoint, createTransformationMatrix(0, -1, 0));
    const rightHipPos = getPosition(rightHipTransform);
    
    const rightKneeTransform = matrixMultiply(rightHipTransform, createTransformationMatrix(4, 0, 0, stickmanData.IMU2.quat));
    const rightKneePos = getPosition(rightKneeTransform);
    
    const rightAnkleTransform = matrixMultiply(rightKneeTransform, createTransformationMatrix(4, 0, 0, stickmanData.IMU4.quat));
    const rightAnklePos = getPosition(rightAnkleTransform);
    
    const rightFootPos = [rightAnklePos[0] + 1, rightAnklePos[1], rightAnklePos[2]];
    
    return {
      pelvis: { x: [leftHipPos[0], rightHipPos[0]], y: [leftHipPos[1], rightHipPos[1]], z: [leftHipPos[2], rightHipPos[2]] },
      leftLeg: {
        upper: { x: [leftHipPos[0], leftKneePos[0]], y: [leftHipPos[1], leftKneePos[1]], z: [leftHipPos[2], leftKneePos[2]] },
        lower: { x: [leftKneePos[0], leftAnklePos[0]], y: [leftKneePos[1], leftAnklePos[1]], z: [leftKneePos[2], leftAnklePos[2]] },
        foot: { x: [leftAnklePos[0], leftFootPos[0]], y: [leftAnklePos[1], leftFootPos[1]], z: [leftAnklePos[2], leftFootPos[2]] }
      },
      rightLeg: {
        upper: { x: [rightHipPos[0], rightKneePos[0]], y: [rightHipPos[1], rightKneePos[1]], z: [rightHipPos[2], rightKneePos[2]] },
        lower: { x: [rightKneePos[0], rightAnklePos[0]], y: [rightKneePos[1], rightAnklePos[1]], z: [rightKneePos[2], rightAnklePos[2]] },
        foot: { x: [rightAnklePos[0], rightFootPos[0]], y: [rightAnklePos[1], rightFootPos[1]], z: [rightAnklePos[2], rightFootPos[2]] }
      },
      joints: {
        x: [pelvisPos[0], leftHipPos[0], leftKneePos[0], leftAnklePos[0], rightHipPos[0], rightKneePos[0], rightAnklePos[0]],
        y: [pelvisPos[1], leftHipPos[1], leftKneePos[1], leftAnklePos[1], rightHipPos[1], rightKneePos[1], rightAnklePos[1]],
        z: [pelvisPos[2], leftHipPos[2], leftKneePos[2], leftAnklePos[2], rightHipPos[2], rightKneePos[2], rightAnklePos[2]]
      }
    };
  };
  
  const stickman = getStickmanData();
  
  return (
    <div className="stickman-container">
      <Plot
        data={[
          // Pelvis
          {
            x: stickman.pelvis.x,
            y: stickman.pelvis.y,
            z: stickman.pelvis.z,
            type: 'scatter3d',
            mode: 'lines',
            line: { color: 'gray', width: 6 },
            name: 'Pelvis'
          },
          // Left leg
          {
            x: stickman.leftLeg.upper.x,
            y: stickman.leftLeg.upper.y,
            z: stickman.leftLeg.upper.z,
            type: 'scatter3d',
            mode: 'lines',
            line: { color: 'gray', width: 4 },
            name: 'Left Upper Leg'
          },
          {
            x: stickman.leftLeg.lower.x,
            y: stickman.leftLeg.lower.y,
            z: stickman.leftLeg.lower.z,
            type: 'scatter3d',
            mode: 'lines',
            line: { color: 'gray', width: 4 },
            name: 'Left Lower Leg'
          },
          {
            x: stickman.leftLeg.foot.x,
            y: stickman.leftLeg.foot.y,
            z: stickman.leftLeg.foot.z,
            type: 'scatter3d',
            mode: 'lines',
            line: { color: 'gray', width: 4 },
            name: 'Left Foot'
          },
          // Right leg
          {
            x: stickman.rightLeg.upper.x,
            y: stickman.rightLeg.upper.y,
            z: stickman.rightLeg.upper.z,
            type: 'scatter3d',
            mode: 'lines',
            line: { color: 'gray', width: 4 },
            name: 'Right Upper Leg'
          },
          {
            x: stickman.rightLeg.lower.x,
            y: stickman.rightLeg.lower.y,
            z: stickman.rightLeg.lower.z,
            type: 'scatter3d',
            mode: 'lines',
            line: { color: 'gray', width: 4 },
            name: 'Right Lower Leg'
          },
          {
            x: stickman.rightLeg.foot.x,
            y: stickman.rightLeg.foot.y,
            z: stickman.rightLeg.foot.z,
            type: 'scatter3d',
            mode: 'lines',
            line: { color: 'gray', width: 4 },
            name: 'Right Foot'
          },
          // Joints
          {
            x: stickman.joints.x,
            y: stickman.joints.y,
            z: stickman.joints.z,
            type: 'scatter3d',
            mode: 'markers',
            marker: { color: 'black', size: 8 },
            name: 'Joints'
          }
        ]}
        layout={{
          title: { text: '3D Stickman Visualization', font: { color: '#2c3e50' } },
          scene: {
            xaxis: { title: 'X', range: [-10, 10], color: '#2c3e50' },
            yaxis: { title: 'Y', range: [-10, 10], color: '#2c3e50' },
            zaxis: { title: 'Z', range: [-10, 10], color: '#2c3e50' },
            bgcolor: '#ffffff',
            camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } }
          },
          plot_bgcolor: '#ffffff',
          paper_bgcolor: '#ffffff',
          font: { color: '#2c3e50' },
          showlegend: true,
          legend: { x: 0, y: 1 }
        }}
        config={{ displayModeBar: false }}
        className="stickman-plot"
      />
    </div>
  );
};

const PelvicMetricsVisualization = ({ pelvicData, isCalibrated, startTime }) => {
  if (!isCalibrated || !pelvicData.timestamp.length) {
    return (
      <div className="visualization-container">
        <div className="placeholder-text">Calibrating IMU7... Stand still for calibration</div>
      </div>
    );
  }
  
  const relativeTimes = pelvicData.timestamp.map(t => t - startTime);
  
  const commonLayout = {
    plot_bgcolor: '#ffffff',
    paper_bgcolor: '#ffffff',
    font: { color: '#2c3e50' },
    margin: { t: 40, b: 60, l: 60, r: 20 },
    showlegend: false,
    xaxis: { 
      title: 'Time (seconds)',
      color: '#2c3e50',
      gridcolor: '#e0e0e0'
    },
    yaxis: { 
      title: 'Angle (degrees)',
      color: '#2c3e50',
      gridcolor: '#e0e0e0'
    }
  };
  
  return (
    <div className="pelvic-plots-container">
      <Plot
        data={[{
          x: relativeTimes,
          y: pelvicData.tilt,
          type: 'scatter',
          mode: 'lines',
          line: { color: 'blue', width: 2 }
        }]}
        layout={{
          ...commonLayout,
          title: { text: 'Pelvic Tilt', font: { color: '#2c3e50' } }
        }}
        config={{ displayModeBar: false }}
        className="pelvic-plot"
      />
      
      <Plot
        data={[{
          x: relativeTimes,
          y: pelvicData.obliquity,
          type: 'scatter',
          mode: 'lines',
          line: { color: 'green', width: 2 }
        }]}
        layout={{
          ...commonLayout,
          title: { text: 'Pelvic Obliquity', font: { color: '#2c3e50' } }
        }}
        config={{ displayModeBar: false }}
        className="pelvic-plot"
      />
      
      <Plot
        data={[{
          x: relativeTimes,
          y: pelvicData.rotation,
          type: 'scatter',
          mode: 'lines',
          line: { color: 'red', width: 2 }
        }]}
        layout={{
          ...commonLayout,
          title: { text: 'Pelvic Rotation', font: { color: '#2c3e50' } }
        }}
        config={{ displayModeBar: false }}
        className="pelvic-plot"
      />
    </div>
  );
};

// Data generators hooks
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
    }, 10);
    
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
        
        const lastTime = newFreqData[imuId].times[newFreqData[imuId].times.length - 1] || 0;
        if (relativeTime - lastTime >= 0.8) {
          newFreqData[imuId].frequencies.push(frequency);
          newFreqData[imuId].times.push(relativeTime);
          
          if (newFreqData[imuId].frequencies.length > MAX_POINTS) {
            newFreqData[imuId].frequencies = newFreqData[imuId].frequencies.slice(-MAX_POINTS);
            newFreqData[imuId].times = newFreqData[imuId].times.slice(-MAX_POINTS);
          }
          
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
        
        if (imuId === 'IMU7') {
          angle = 0.1 * Math.sin(2 * Math.PI * t) + (Math.random() - 0.5) * 0.04;
        } else if (imuId === 'IMU1') {
          angle = 0.3 * Math.sin(2 * Math.PI * t) + (Math.random() - 0.5) * 0.1;
        } else if (imuId === 'IMU3') {
          angle = 0.2 * Math.sin(2 * Math.PI * t + Math.PI/4) + (Math.random() - 0.5) * 0.1;
        } else if (imuId === 'IMU2') {
          angle = 0.3 * Math.sin(2 * Math.PI * t + Math.PI) + (Math.random() - 0.5) * 0.1;
        } else if (imuId === 'IMU4') {
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
    }, 100);
    
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
      
      if (!isCalibrated) {
        calibrationQuats.current.push(quat);
        setCalibrationProgress(calibrationQuats.current.length);
        
        if (calibrationQuats.current.length >= 100) {
          setIsCalibrated(true);
        }
        return;
      }
      
      const R_current = quaternionToMatrix(quat);
      const [alpha_z, beta_y, gamma_x] = matrixToEulerZYX(R_current);
      
      let tilt = -beta_y * 180 / Math.PI;
      let obliquity = gamma_x * 180 / Math.PI;
      let rotation = alpha_z * 180 / Math.PI;
      
      tilt += 5 * Math.sin(2 * Math.PI * t * 1.2) + (Math.random() - 0.5) * 2;
      obliquity += 3 * Math.sin(2 * Math.PI * t * 0.8) + (Math.random() - 0.5) * 1;
      rotation += 2 * Math.sin(2 * Math.PI * t * 1.5) + (Math.random() - 0.5) * 0.6;
      
      setPelvicData(prev => ({
        timestamp: [...prev.timestamp, currentTime],
        tilt: [...prev.tilt, tilt],
        obliquity: [...prev.obliquity, obliquity],
        rotation: [...prev.rotation, rotation]
      }));
      
    }, 50);
    
    return () => clearInterval(interval);
  }, [isCalibrated]);
  
  return { pelvicData, isCalibrated, calibrationProgress, startTime: startTime.current };
};

// Main Dashboard Component
const MedicalRehabDashboard = () => {
  const [selectedOption, setSelectedOption] = useState('StepLength');
  const [walkingSpeed, setWalkingSpeed] = useState(1.29);
  
  // Data generators
  const stepData = useStepDataGenerator();
  const powerData = usePowerDataGenerator();
  const { freqData, freqStats } = useFrequencyDataGenerator();
  const stickmanData = useStickmanDataGenerator();
  const { pelvicData, isCalibrated, calibrationProgress, startTime } = usePelvicDataGenerator();
  
  const renderVisualization = () => {
    switch (selectedOption) {
      case 'StepLength':
        return <StepLengthVisualization stepData={stepData} />;
      case 'PowerMetrics':
        return <PowerMetricsVisualization powerData={powerData} />;
      case 'Frequency':
        return <FrequencyVisualization freqData={freqData} freqStats={freqStats} />;
      case 'Stickman':
        return <StickmanVisualization stickmanData={stickmanData} />;
      case 'PelvicMetrics':
        return <PelvicMetricsVisualization pelvicData={pelvicData} isCalibrated={isCalibrated} startTime={startTime} />;
      default:
        return (
          <div className="default-view">
            <div className="patient-info">
              <h3>JD Metrics</h3>
              <p>ID: 001</p>
              <p>Age: 45</p>
              <p>BMI: 24.9</p>
            </div>
            <div className="placeholder-visualization">
              <span className="placeholder-text">Stickman Visualization Placeholder</span>
            </div>
          </div>
        );
    }
  };
  
  const renderMetrics = () => {
    switch (selectedOption) {
      case 'StepLength':
        return (
          <div>
            <h3 className="section-title">Step Detection Metrics</h3>
            {Object.entries(IMU_MAPPING).map(([imuId, imuName]) => {
              const buffer = stepData[imuId] || { step_count: 0, cadence: 0 };
              return (
                <div key={imuId} className="metric-card">
                  <h4>{imuName}</h4>
                  <p className="metric-value">Steps: {buffer.step_count}</p>
                  <p className="metric-value">Cadence: {buffer.cadence.toFixed(1)} steps/min</p>
                </div>
              );
            })}
          </div>
        );
      case 'PowerMetrics':
        return (
          <div>
            <h3 className="section-title">Power Statistics</h3>
            <div className="metric-card">
              <h4>TRANSLATIONAL POWER</h4>
              <p className="metric-value">Peak: 125.45 W</p>
              <p className="metric-value">Mean: 78.32 W</p>
            </div>
            <div className="metric-card">
              <h4>ROTATIONAL POWER</h4>
              <p className="metric-value">Peak: 89.21 W</p>
              <p className="metric-value">Mean: 45.67 W</p>
            </div>
            <div className="metric-card">
              <h4>TOTAL POWER</h4>
              <p className="metric-value">Peak: 214.66 W</p>
              <p className="metric-value">Mean: 123.99 W</p>
            </div>
            <div className="metric-card">
              <h4>STRIDE TIME</h4>
              <p className="metric-value">823.1 ± 15.2 ms</p>
            </div>
            <div className="metric-card">
              <h4>CADENCE</h4>
              <p className="metric-value">72.9 steps/min</p>
            </div>
            <div className="metric-card">
              <h4>STRIDE LENGTH</h4>
              <p className="metric-value">42.53 ± 4.25 m</p>
            </div>
          </div>
        );
      case 'Frequency':
        return (
          <div>
            <h3 className="section-title">Frequency Statistics</h3>
            {Object.entries(freqStats).map(([imuId, stats]) => (
              <div key={imuId} className="metric-card">
                <h4>{imuId}</h4>
                <p className="metric-value">Current: {stats.current.toFixed(1)} Hz</p>
                <p className="metric-value">Average: {stats.avg.toFixed(1)} Hz</p>
                <p className="metric-value">Data Points: {stats.count}</p>
              </div>
            ))}
            {Object.keys(freqStats).length === 0 && (
              <div className="metric-card">
                <h4>Waiting for Data</h4>
                <p className="metric-value">Starting frequency monitoring...</p>
              </div>
            )}
          </div>
        );
      case 'Stickman':
        return (
          <div>
            <h3 className="section-title">IMU Data Status</h3>
            {Object.entries(stickmanData).map(([imuId, data]) => {
              const bodyPartMap = {
                'IMU1': 'Left Upper Leg',
                'IMU2': 'Right Upper Leg', 
                'IMU3': 'Left Lower Leg',
                'IMU4': 'Right Lower Leg',
                'IMU7': 'Pelvis'
              };
              const bodyPart = bodyPartMap[imuId] || imuId;
              const quatMag = Math.sqrt(data.quat.reduce((sum, q) => sum + q*q, 0));
              const timeDiff = (Date.now() / 1000) - data.timestamp;
              
              return (
                <div key={imuId} className="metric-card">
                  <h4>{bodyPart} ({imuId})</h4>
                  <p className="metric-value">Quaternion: [{data.quat.map(q => q.toFixed(3)).join(', ')}]</p>
                  <p className="metric-value">Magnitude: {quatMag.toFixed(3)}</p>
                  <p className="metric-value">Last Update: {timeDiff.toFixed(1)}s ago</p>
                </div>
              );
            })}
          </div>
        );
      case 'PelvicMetrics':
        return (
          <div>
            <h3 className="section-title">Pelvic Metrics Status</h3>
            {isCalibrated && pelvicData.tilt.length > 0 ? (
              <>
                <div className="metric-card">
                  <h4>Real-time Pelvic Metrics (IMU7)</h4>
                  <p className="metric-value">Total Duration: {((Date.now() / 1000) - startTime).toFixed(1)}s</p>
                  <p className="metric-value">Data Points: {pelvicData.tilt.length}</p>
                </div>
                <div className="metric-card">
                  <h4>Current Angles</h4>
                  <p className="metric-value">Tilt: {pelvicData.tilt[pelvicData.tilt.length - 1]?.toFixed(2)}°</p>
                  <p className="metric-value">Obliquity: {pelvicData.obliquity[pelvicData.obliquity.length - 1]?.toFixed(2)}°</p>
                  <p className="metric-value">Rotation: {pelvicData.rotation[pelvicData.rotation.length - 1]?.toFixed(2)}°</p>
                </div>
                <div className="metric-card">
                  <h4>Statistics</h4>
                  <p className="metric-value">Tilt Range: {Math.min(...pelvicData.tilt).toFixed(1)}° to {Math.max(...pelvicData.tilt).toFixed(1)}°</p>
                  <p className="metric-value">Obliquity Range: {Math.min(...pelvicData.obliquity).toFixed(1)}° to {Math.max(...pelvicData.obliquity).toFixed(1)}°</p>
                  <p className="metric-value">Rotation Range: {Math.min(...pelvicData.rotation).toFixed(1)}° to {Math.max(...pelvicData.rotation).toFixed(1)}°</p>
                </div>
              </>
            ) : (
              <>
                <div className="metric-card">
                  <h4>Calibrating IMU7...</h4>
                  <p className="metric-value">Progress: {calibrationProgress}/100</p>
                  <p className="metric-value">Stand still for calibration</p>
                </div>
                <div className="metric-card">
                  <h4>Instructions</h4>
                  <p className="metric-value">1. Keep pelvis still during calibration</p>
                  <p className="metric-value">2. Calibration will complete automatically</p>
                  <p className="metric-value">3. Then start normal movement</p>
                </div>
              </>
            )}
          </div>
        );
      default:
        return (
          <div>
            <h3 className="section-title">Clinical Endpoints</h3>
            <p>Comprehensive biomechanical analysis</p>
            
            <div className="gait-parameters">
              <h3>Gait Parameters</h3>
              
              <div className="slider-container">
                <label className="slider-label">Walking Speed</label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.01"
                  value={walkingSpeed}
                  onChange={(e) => setWalkingSpeed(parseFloat(e.target.value))}
                  className="slider"
                />
                <p className="metric-value">{walkingSpeed.toFixed(2)} m/s</p>
              </div>
              
              <div className="gait-metrics">
                <div className="gait-metric-card success">
                  <h3>1.23m</h3>
                  <p>Step Length</p>
                </div>
                <div className="gait-metric-card warning">
                  <h3>115</h3>
                  <p>Cadence (spm)</p>
                </div>
              </div>
              
              <h3>Gait Timing</h3>
              <div className="gait-timing">
                <div className="timing-card stance">
                  <h3>62%</h3>
                  <p>Stance</p>
                </div>
                <div className="timing-card swing">
                  <h3>38%</h3>
                  <p>Swing</p>
                </div>
                <div className="timing-card support">
                  <h3>13%</h3>
                  <p>Double Support</p>
                </div>
              </div>
            </div>
          </div>
        );
    }
  };
  
  return (
    <div className="medical-rehab-dashboard">
      <div className="dashboard-container">
        {/* Sidebar */}
        <div className="sidebar">
          <h2>Real Time Processing of Data</h2>
          <div className="radio-group">
            {['StepLength', 'PowerMetrics', 'Frequency', 'PelvicMetrics', 'Stickman'].map((option) => (
              <div 
                key={option} 
                className={`radio-option ${selectedOption === option ? 'selected' : ''}`}
                onClick={() => setSelectedOption(option)}
              >
                <input
                  type="radio"
                  id={option}
                  name="visualization"
                  value={option}
                  checked={selectedOption === option}
                  onChange={() => setSelectedOption(option)}
                />
                <label htmlFor={option}>{option}</label>
              </div>
            ))}
          </div>
        </div>
        
        {/* Main Content */}
        <div className="main-content">
          <div className="left-column">
            <h3 className="section-title">Real-Time {selectedOption} Visualization</h3>
            {renderVisualization()}
          </div>
          
          <div className="right-column">
            {renderMetrics()}
          </div>
        </div>
      </div>
      
      {/* Active Session Indicator */}
      <div className="active-session">
        <div className="status-dot"></div>
        <span>Active Session</span>
      </div>
    </div>
  );
};

export default MedicalRehabDashboard; 