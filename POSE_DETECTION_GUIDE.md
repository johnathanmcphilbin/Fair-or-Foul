# Live Pose Detection & Limb Crossing Analysis

## Overview

The Fair-or-Foul web application now includes a **Live Pose Detection** module that:
- Captures live video feed from your camera
- Detects human pose using TensorFlow.js PoseNet model
- Draws ellipses around each limb of the detected person
- Counts every time a limb (ellipse) crosses with another limb
- Displays real-time crossing counter in the top-right corner
- Records detailed session data with frame-by-frame information
- Automatically saves data to CSV format for analysis

## Features

### 1. Live Camera Feed
- Streams live video from your device's camera
- Displays video in real-time on an HTML5 canvas
- Shows pose skeleton with joint positions
- 720p resolution for accurate pose detection

### 2. Pose Detection
- Uses TensorFlow.js PoseNet model
- Detects 17 key joints on the human body:
  - Head joints: nose, eyes, ears
  - Arm joints: shoulders, elbows, wrists
  - Torso joints: hips
  - Leg joints: knees, ankles

### 3. Limb Visualization
- Draws ellipses around each detected limb
- Color-coded ellipses for different limbs
- Semi-transparent rendering for clarity
- Shows joint positions as green dots
- Skeleton connections in cyan

### 4. Crossing Detection
- Detects when limb ellipses intersect with each other
- Counts each crossing event
- Displays total count in the top-right corner
- Prevents double-counting from the same limb pair

### 5. Session Data Recording
- Records comprehensive frame-by-frame data including:
  - Timestamp (in seconds)
  - Frame number
  - Current crossing count
  - Confidence score for pose detection
  - Number of detected joints
  - X/Y coordinates for each joint
  - Confidence score for each joint

### 6. Automatic CSV Saving
- Session data automatically saved to CSV format
- Filename format: `pose_session_YYYYMMDD_HHMMSS.csv`
- Saved in the `data/` directory
- Downloadable from the web interface

## How to Use

### Starting a Session

1. Navigate to the **"Live Detection"** section in the navigation menu
2. Click the **"Start Camera"** button
3. Allow browser access to your camera when prompted
4. Wait for the status message to show "Camera started. Detecting poses..."

### During Recording

- **Red Counter (Top Right)**: Shows real-time crossing count
- **Live Video Feed**: Shows pose skeleton and ellipses around limbs
- **Status Bar**: Updates with current operation status
- Session automatically tracks all data in the background

### Stopping a Session

1. Click the **"Stop & Save"** button to end the session
2. Session data is automatically saved to CSV
3. Session statistics update on the page
4. Your saved session appears in the "Saved Sessions" section

### Resetting

- Click **"Reset"** to clear all counters and prepare for a new session
- Does not delete previously saved sessions

## Session Statistics

After each session, the following statistics are displayed:

| Statistic | Description |
|-----------|-------------|
| **Total Crossings** | Total number of limb crossings detected |
| **Duration** | Session length in seconds |
| **Frames Processed** | Total number of video frames analyzed |
| **Avg Confidence** | Average pose detection confidence (0-100%) |

## CSV Data Format

Each saved session creates a CSV file with the following columns:

```
timestamp, frame_number, crossings_so_far, confidence, joints_detected,
nose_x, nose_y, nose_confidence,
left_eye_x, left_eye_y, left_eye_confidence,
... (for all 17 joints)
```

**Example row:**
```
0.03, 1, 0, 0.985, 16, 
640.2, 240.5, 0.998, 620.1, 215.3, 0.997, ...
```

## Accessing Saved Sessions

1. Scroll down to the **"Saved Sessions"** section
2. Each saved session shows:
   - Filename with timestamp
   - Creation date and time
   - File size in KB
   - Download button for CSV file

3. Click **"Download CSV"** to export the data

## Technical Details

### Backend (Flask)

**New Routes:**
- `POST /save_pose_session` - Saves session data to CSV
- `GET /get_saved_sessions` - Retrieves list of saved sessions
- `GET /download_session/<filename>` - Downloads a specific session CSV

**Data Directory:**
- Session data stored in: `/data/pose_session_*.csv`
- Auto-created on first session save

### Frontend (JavaScript)

**New Module:** `static/js/pose-detection.js`

**Main Class:** `PoseDetector`
- Manages camera access and streaming
- Handles pose estimation
- Draws visualization
- Detects limb crossings
- Records session data
- Communicates with backend

### Libraries Used

- **TensorFlow.js 4.x** - ML computation
- **PoseNet Model** - Pre-trained pose detection
- **HTML5 Canvas** - Real-time visualization
- **MediaDevices API** - Camera access
- **Fetch API** - Server communication

## Pose Model Details

**Model:** PoseNet (MobileNet backbone)

**Accuracy Factors:**
- Lighting conditions (well-lit environments work best)
- Camera resolution (720p recommended)
- Distance from camera (1-3 meters optimal)
- Body position (full body visible works best)

**Confidence Threshold:**
- Joints with confidence < 0.5 are not displayed
- Helps reduce false positives

## Troubleshooting

### Camera Not Accessible
- Check browser permissions for camera access
- Ensure camera is not in use by another application
- Try a different browser (Chrome/Edge recommended)

### Poor Pose Detection
- Improve lighting conditions
- Move further from camera (1-2 meters)
- Ensure full body is visible
- Clean camera lens

### Data Not Saving
- Check browser console for errors (F12)
- Ensure `data/` directory has write permissions
- Try refreshing the page and starting again

### Low Confidence Scores
- Improve lighting
- Wear contrasting colored clothing
- Move slower for better tracking
- Ensure camera is at appropriate distance

## Browser Compatibility

- **Chrome/Edge** (Recommended) - Full support
- **Firefox** - Full support
- **Safari** - Full support with some performance limits
- **Mobile Browsers** - Limited support for camera access

## Data Privacy

- All processing happens locally in the browser
- Data only sent to server for saving
- No data sent to external servers
- Session data stored locally on server

## Future Enhancements

Possible improvements:
- Video recording alongside data
- Real-time analytics graphs
- Pose comparison features
- Multi-person detection
- Limb-specific crossing counters
- Configurable detection parameters
