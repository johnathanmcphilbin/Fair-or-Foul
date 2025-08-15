# Fair-or-Foul

Python-based system for analysing youth sports referee decisions. Includes computer vision for call detection, data processing, and statistical analysis to identify patterns of potential bias. Supports basketball, soccer, and martial arts with modular code for video tracking, tagging, and visualisation.

## Features

- **Computer Vision Analysis**: Advanced video analysis for automatic call detection
- **Statistical Analysis**: Identify patterns and biases in referee decisions
- **Multi-Sport Support**: Basketball, soccer, and martial arts
- **Web Interface**: Modern, responsive web application for easy data upload and analysis
- **Data Processing**: CSV data analysis with team call rates and county alignment bias detection

## Web Application

The project now includes a modern web frontend that provides:

- **File Upload**: Drag-and-drop interface for CSV and video files
- **Data Analysis**: Interactive analysis tools for referee decision patterns
- **Results Visualization**: Clean, tabular display of analysis results
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### Running the Web Application

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Web Server**:
   ```bash
   python run_web.py
   ```
   Or directly:
   ```bash
   python app.py
   ```

3. **Open Your Browser**:
   Navigate to `http://localhost:5000`

### Web Interface Features

- **Home**: Overview and introduction to the system
- **Upload**: File upload with drag-and-drop support
- **Analysis**: Run statistical analysis on uploaded data
- **About**: Information about supported sports and features

## Command Line Interface

The original command-line interface is still available:

```bash
# Analyze team call rates
python scripts/ff.py rates data/raw/calls.csv

# Analyze county alignment bias
python scripts/ff.py alignment data/raw/calls.csv

# Tag video files
python scripts/ff.py tag video.mp4 basketball match123 ref456 teamA teamB
```

## Project Structure

```
Fair-or-Foul/
├── app.py                 # Flask web application
├── run_web.py            # Web app startup script
├── requirements.txt       # Python dependencies
├── templates/            # HTML templates
│   └── index.html       # Main web interface
├── static/               # Static assets
│   ├── css/
│   │   └── style.css    # Custom styling
│   └── js/
│       └── main.js      # Interactive functionality
├── scripts/              # Command-line scripts
│   └── ff.py            # Main CLI tool
├── src/                  # Core modules
│   └── fairorfoul/      # Main package
├── data/                 # Data storage
│   ├── raw/             # Raw input data
│   └── processed/       # Analysis results
└── uploads/              # Web upload directory
```

## Supported Sports

### Basketball
- Foul, travel, double dribble, charge, block, shooting foul, technical, out of bounds

### Soccer
- Foul, yellow card, red card, offside, penalty, handball, out of play

### Martial Arts
- Warning, point deduction, disqualification, stalling, illegal move

## Development

The web application is built with:
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Styling**: Bootstrap 5, custom CSS
- **Icons**: Font Awesome
- **Charts**: Chart.js (for future visualizations)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is designed for educational and research purposes in youth sports fairness analysis.
