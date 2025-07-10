# Medical Rehabilitation Dashboard - React Version

A React-based medical rehabilitation dashboard that provides real-time visualization of IMU data, gait analysis, and biomechanical metrics. This implementation addresses the refresh/rerun issues present in the Streamlit version while maintaining identical functionality and user interface.

## Features

### 🔄 Real-Time Visualizations
- **Step Length Detection**: 5-panel step detection with Butterworth filtering and peak detection
- **Power Metrics**: 6-panel power analysis (acceleration, translational/rotational/total power, power/weight ratio, rolling averages)
- **Frequency Monitoring**: Multi-IMU frequency tracking with dynamic graphs and statistics
- **3D Stickman Visualization**: Real-time kinematic model with quaternion-based joint movements
- **Pelvic Metrics**: 3-panel pelvic angle analysis with automatic calibration

### 📊 Key Capabilities
- **No Refresh Issues**: Smooth, persistent visualizations without page reloads
- **Real-Time Updates**: Live data updates every 0.5 seconds
- **Simulated Test Data**: Built-in data generators for demonstration purposes
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Professional UI**: Matches the original Streamlit design with medical-grade styling

## Installation & Setup

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn package manager

### Quick Start

1. **Navigate to the React directory**:
   ```bash
   cd Webboard
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm start
   ```

4. **Open your browser**:
   - The application will automatically open at `http://localhost:3000`
   - If it doesn't open automatically, navigate to `http://localhost:3000`

### Production Build

To create a production build:

```bash
npm run build
```

The build files will be generated in the `build/` directory and can be served by any static file server.

## Project Structure

```
Webboard/
├── package.json              # Dependencies and scripts
├── public/
│   └── index.html            # Main HTML template
├── MedicalRehabDashboard.js  # Main React application
├── MedicalRehabDashboard.css # Styling and layout
├── index.js                  # React app entry point
└── README_React.md           # This file
```

## Technical Implementation

### Architecture
- **React 18**: Modern React with hooks and functional components
- **Plotly.js**: High-performance, interactive plotting library
- **CSS Grid & Flexbox**: Responsive layout system
- **Real-Time Data**: Custom hooks for data generation and management

### Key Components

#### 1. StepLengthVisualization
- Multi-IMU step detection with filtering
- Real-time peak detection algorithms
- Step counting and cadence calculation

#### 2. PowerMetricsVisualization
- 6-panel power analysis dashboard
- Translational and rotational power calculations
- Rolling averages and statistical metrics

#### 3. FrequencyVisualization
- Multi-IMU frequency monitoring
- Dynamic data decimation for performance
- Color-coded IMU identification

#### 4. StickmanVisualization
- 3D kinematic model with joint transformations
- Quaternion-to-rotation matrix conversion
- Real-time walking motion simulation

#### 5. PelvicMetricsVisualization
- 3-panel pelvic angle tracking
- Automatic calibration system
- Euler angle extraction from quaternions

### Data Generation
Each visualization includes built-in test data generators that simulate realistic:
- Walking patterns with proper frequency and amplitude
- IMU sensor noise and variations
- Coordinated multi-sensor data streams
- Calibration phases for pelvic metrics

## Usage

### Navigation
Use the sidebar radio buttons to switch between different visualizations:
- **StepLength**: Real-time step detection and gait parameters
- **PowerMetrics**: Comprehensive power analysis
- **Frequency**: IMU data frequency monitoring
- **PelvicMetrics**: Pelvic angle tracking with calibration
- **Stickman**: 3D kinematic visualization

### Features Per View

#### StepLength View
- **Left Panel**: 5 real-time plots showing acceleration data and detected steps
- **Right Panel**: Step counts and cadence metrics for each IMU

#### PowerMetrics View
- **Left Panel**: 6 power analysis plots in a 3x2 grid layout
- **Right Panel**: Statistical summaries and peak/mean values

#### Frequency View
- **Left Panel**: Dynamic frequency plots for all active IMUs
- **Right Panel**: Current frequency, averages, and data point counts

#### Stickman View
- **Left Panel**: 3D interactive stickman with joint movements
- **Right Panel**: Real-time quaternion data and IMU status

#### PelvicMetrics View
- **Left Panel**: 3 pelvic angle plots (tilt, obliquity, rotation)
- **Right Panel**: Calibration status and angle statistics

## Performance Optimizations

### Real-Time Performance
- **Efficient Updates**: Container-based rendering without full page refreshes
- **Data Management**: Circular buffers with configurable window sizes
- **Throttled Rendering**: 0.5-second update intervals for smooth performance
- **Memory Management**: Automatic cleanup of old data points

### Visualization Optimizations
- **Data Decimation**: Reduced point density for frequency plots
- **Selective Rendering**: Only update changed components
- **Efficient Layouts**: CSS Grid for optimal rendering performance

## Customization

### Configuration Options
Modify constants in `MedicalRehabDashboard.js`:

```javascript
const WINDOW_SIZE = 1000;          // Data buffer size
const PLOT_WINDOW = 5.0;           // Time window for plots (seconds)
const POWER_WINDOW_SIZE = 500;     // Power metrics buffer size
const MAX_POINTS = 300;            // Maximum points per frequency plot
```

### Styling Customization
Edit `MedicalRehabDashboard.css` to modify:
- Color schemes and themes
- Layout dimensions and spacing
- Font styles and sizes
- Responsive breakpoints

### Adding New Visualizations
1. Create a new component in `MedicalRehabDashboard.js`
2. Add data generator hook if needed
3. Update the sidebar options array
4. Add case in `renderVisualization()` switch statement
5. Add corresponding metrics in `renderMetrics()`

## Comparison with Streamlit Version

### ✅ Advantages of React Version
- **No Refresh Issues**: Persistent visualizations without page reloads
- **Better Performance**: More efficient rendering and data management
- **Smoother Updates**: Consistent real-time updates without interruptions
- **Professional Feel**: Native web application experience
- **Mobile Responsive**: Better mobile and tablet support
- **Customizable**: Easier to modify and extend

### 🔄 Maintained Features
- **Identical UI**: Same visual design and layout
- **All Visualizations**: Complete feature parity
- **Real-Time Data**: Same update frequencies and data patterns
- **Color Schemes**: Consistent medical-grade styling
- **Metrics Display**: Identical statistical information

## Troubleshooting

### Common Issues

1. **Plots not loading**:
   - Check browser console for errors
   - Ensure Plotly.js loaded correctly
   - Verify network connectivity

2. **Performance issues**:
   - Reduce `MAX_POINTS` constant
   - Increase update interval
   - Close other browser tabs

3. **Layout problems**:
   - Check CSS Grid support in browser
   - Verify viewport meta tag
   - Test responsive design tools

### Development Mode
For development and debugging:

```bash
npm start
```

- Hot reloading enabled
- Detailed error messages
- React DevTools integration

## Future Enhancements

### Planned Features
- WebSocket integration for real IMU data
- Data export functionality
- Historical data analysis
- User authentication
- Advanced filtering options
- Custom color themes

### Integration Options
- REST API connections
- WebSocket real-time data streams
- Database integration
- Cloud deployment options

## Support

For issues specific to the React implementation:
1. Check browser console for error messages
2. Verify all dependencies are installed
3. Ensure Node.js version compatibility
4. Review network requests in browser DevTools

## License

This React implementation maintains the same license as the original Streamlit version.

---

**Note**: This React version provides a complete replacement for the Streamlit dashboard with improved performance and no refresh/rerun issues while maintaining 100% feature compatibility and identical user interface design. 