import React from 'react';
import ReactDOM from 'react-dom/client';
import MedicalRehabDashboard from './MedicalRehabDashboard';
import './MedicalRehabDashboard.css';

// Create root and render the application
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <MedicalRehabDashboard />
  </React.StrictMode>
);

// Optional: Performance measuring
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';

function sendToAnalytics(metric) {
  console.log(metric);
}

getCLS(sendToAnalytics);
getFID(sendToAnalytics);
getFCP(sendToAnalytics);
getLCP(sendToAnalytics);
getTTFB(sendToAnalytics); 