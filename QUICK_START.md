# Quick Start Guide: Live Pose Detection

## Installation & Setup

### Prerequisites
- Web browser with camera access (Chrome/Edge recommended)
- Microphone/speaker optional (for audio feedback if added later)

### Running the Application

```bash
# Navigate to project directory
cd /workspaces/Fair-or-Foul

# Start the Flask web server
python run_web.py
```

You should see:
```
Starting Fair-or-Foul Web Application...
Open your browser and go to: http://localhost:5000
Press Ctrl+C to stop the server
```

### Accessing the Feature

1. Open browser to: **http://localhost:5000**
2. Scroll down or click **"Live Detection"** in the navbar
3. You're now in the Pose Detection section

## Quick Start Steps

### Session 1: First-Time Setup

1. **Allow Camera Access**
   - Click "Start Camera"
   - Browser will prompt for camera permission
   - Click "Allow" when prompted

2. **View Live Pose Detection**
   - Green dots appear on your joints
   - Cyan lines connect the skeleton
   - Colored ellipses show around limbs

3. **Generate Some Crossings**
   - Move your limbs around
   - Cross arms over body
   - Cross legs
   - Watch counter in top-right increase

4. **Stop Recording**
   - Click "Stop & Save" button
   - Wait for "Session saved" message
   - Session data automatically saved to CSV

5. **Download Results**
   - Scroll to "Saved Sessions" section
   - Find your session timestamp
   - Click "Download CSV" button
   - Open the CSV in Excel/Google Sheets

### Example Session Data

When you download the CSV, you'll see:

```
timestamp,frame_number,crossings_so_far,confidence,joints_detected,...
0.03,1,0,0.985,16,640.2,240.5,0.998,620.1,215.3,0.997,...
0.06,2,0,0.987,16,641.1,241.2,0.999,619.8,216.1,0.998,...
0.09,3,1,0.989,17,642.3,239.8,0.997,621.2,214.5,0.996,...
...
```

## Understanding the Interface

### Top Section: Canvas with Video
```
┌─────────────────────────────────────────────────┐
│                                       [Crossings: 5] │
│  Live pose detection video feed                │
│  • Green dots = Joint positions                 │
│  • Cyan lines = Skeleton connections            │
│  • Colored ellipses = Limbs                     │
│                                                  │
│  (Person on camera with pose overlay)           │
└─────────────────────────────────────────────────┘
```

### Control Buttons
```
[🟢 Start Camera] [🔴 Stop & Save] [🟡 Reset]
```

### Status Messages
- "Ready to start" - Waiting for action
- "Camera started. Detecting poses..." - Running
- "Session saved to pose_session_..." - Complete
- "Error..." - Something went wrong

### Statistics Display (After Session)
```
┌─────────────────────────────────────────────────┐
│  Total Crossings: 42  │  Duration: 45s          │
│  Frames: 1350         │  Avg Confidence: 94.2%  │
└─────────────────────────────────────────────────┘
```

## Common Scenarios

### Scenario 1: Testing Limb Detection
**Goal:** Verify pose detection is working

**Steps:**
1. Click "Start Camera"
2. Stand in front of camera (1-2 meters away)
3. Raise arms slowly
4. Verify green dots appear on your joints
5. Watch cyan skeleton lines connect

**Expected Result:** All 17 body joints visible

### Scenario 2: Counting Limb Crossings
**Goal:** Record a specific movement pattern

**Steps:**
1. Click "Start Camera"
2. Cross arms over chest 5 times slowly
3. Cross legs 5 times slowly
4. Watch counter increase in top-right
5. Click "Stop & Save"

**Expected Result:** Counter should show ~10 crossings

### Scenario 3: Analyzing Session Data
**Goal:** Export and analyze movement data

**Steps:**
1. Perform complete session (30+ seconds)
2. Click "Stop & Save"
3. Find session in "Saved Sessions"
4. Click "Download CSV"
5. Open in Excel/Google Sheets
6. Analyze columns:
   - `timestamp` - When each frame occurred
   - `crossings_so_far` - Running total
   - `joints_detected` - How many limbs visible
   - `*_confidence` - Accuracy per joint

**Expected Result:** CSV file with detailed frame-by-frame data

## Troubleshooting

### Issue: Camera doesn't start
**Solutions:**
1. Check browser camera permissions (Settings → Privacy)
2. Try Chrome instead of Firefox
3. Ensure camera isn't in use by another app
4. Refresh page and try again

### Issue: Model takes too long to load
**Solutions:**
1. Wait 10-15 seconds on first use
2. Check internet connection
3. Try different browser
4. Check browser console (F12) for errors

### Issue: Poor pose detection (few joints visible)
**Solutions:**
1. Improve lighting (natural light preferred)
2. Move 1-2 meters from camera
3. Wear contrasting clothing
4. Show full body to camera
5. Clear any obstructions

### Issue: Crossings not being detected
**Solutions:**
1. Move limbs more dramatically
2. Cross limbs completely (not just close)
3. Move slower for better tracking
4. Ensure limbs are visible to camera

### Issue: Session didn't save
**Solutions:**
1. Check browser console for errors (F12)
2. Verify `data/` folder exists
3. Check browser download folder
4. Try again after refresh

## Tips for Best Results

### Camera Setup
- ✅ Good lighting (natural daylight best)
- ✅ 1-2 meters from camera
- ✅ Full body visible
- ✅ Plain background helpful
- ❌ Avoid backlighting
- ❌ Don't wear loose/baggy clothes

### Movement
- ✅ Slow, deliberate movements
- ✅ Keep full body visible
- ✅ Use wide range of motion
- ✅ Ensure limbs cross completely
- ❌ Don't move too fast
- ❌ Avoid sudden jerky movements

### Data Collection
- ✅ Sessions 30+ seconds work best
- ✅ Repeat specific patterns
- ✅ Name your files clearly
- ✅ Label what movement was performed
- ❌ Don't remove model from view
- ❌ Don't cover camera

## Sample Data Use Cases

### Use Case 1: Sports Analysis
**Record:** Athlete's movements during training
**Analyze:** Crossing patterns and limb coordination
**Output:** Movement efficiency reports

### Use Case 2: Rehabilitation
**Record:** Patient performing therapy exercises
**Analyze:** Range of motion and joint movements
**Output:** Progress tracking data

### Use Case 3: Fitness Tracking
**Record:** Workout movements (jumping jacks, etc)
**Analyze:** Movement accuracy and speed
**Output:** Performance metrics

### Use Case 4: Research
**Record:** Human movement patterns
**Analyze:** Statistical analysis of joint positions
**Output:** Movement databases

## CSV Column Reference

| Column | Description | Example |
|--------|-------------|---------|
| `timestamp` | Seconds since session start | 0.03 |
| `frame_number` | Frame count | 1 |
| `crossings_so_far` | Total crossings to this point | 0 |
| `confidence` | Overall pose confidence (0-1) | 0.985 |
| `joints_detected` | Number of visible joints | 16 |
| `nose_x`, `nose_y` | Nose position | 640.2, 240.5 |
| `nose_confidence` | Nose detection confidence | 0.998 |
| ... | (same for all 17 joints) | ... |

## Advanced: Manual Data Analysis

### In Excel/Google Sheets

```
1. Open downloaded CSV
2. Create pivot table on "crossings_so_far"
3. Graph crossing count vs time
4. Analyze joint positions:
   - Find max/min X,Y values
   - Calculate movement range
   - Track accuracy (confidence)
```

### In Python

```python
import pandas as pd

# Load session data
df = pd.read_csv('data/pose_session_20250101_120000.csv')

# Analyze crossings
print(f"Total Crossings: {df['crossings_so_far'].max()}")
print(f"Average Confidence: {df['confidence'].mean()}")

# Plot crossing pattern
df.plot(x='timestamp', y='crossings_so_far')
plt.show()

# Export statistics
df.describe().to_csv('analysis.csv')
```

## Next Steps

1. **Run your first session** - Try basic movements
2. **Experiment with different motions** - See what gets detected
3. **Download and analyze data** - Open CSV in spreadsheet
4. **Plan your analysis** - What patterns do you want to track?
5. **Automate analysis** - Use Python scripts for batch processing

## Support

For issues or questions:
1. Check the POSE_DETECTION_GUIDE.md for detailed documentation
2. Check browser console (F12) for error messages
3. Review IMPLEMENTATION_SUMMARY.md for technical details
4. Try troubleshooting section above

---

**Ready to start?** Click "Live Detection" in the navbar and "Start Camera"!
