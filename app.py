from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    redirect,
    url_for,
    flash,
)
import os
from pathlib import Path
import sys
from werkzeug.utils import secure_filename
import csv
from datetime import datetime

# Add the src directory to the path to import fairorfoul modules
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from fairorfoul.io import load_calls_csv, save_processed
from fairorfoul.analysis import team_call_rates, county_alignment_bias
from fairorfoul.config import CALL_TYPES

app = Flask(__name__)
app.secret_key = "fair-or-foul-secret-key-2024"

# Configuration
UPLOAD_FOLDER = "uploads"
DATA_FOLDER = "data"
ALLOWED_EXTENSIONS = {"csv", "mp4", "avi", "mov", "mkv"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max file size

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html", sports=CALL_TYPES.keys())


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        flash("No file selected")
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        flash(f"File {filename} uploaded successfully")
        return redirect(url_for("index"))

    flash("Invalid file type")
    return redirect(url_for("index"))


@app.route("/analyze", methods=["POST"])
def analyze_data():
    try:
        data = request.get_json()
        csv_path = data.get("csv_path")
        analysis_type = data.get("analysis_type")

        if not csv_path or not os.path.exists(csv_path):
            return jsonify({"error": "CSV file not found"}), 400

        df = load_calls_csv(csv_path)

        if analysis_type == "team_rates":
            result = team_call_rates(df)
            output_path = "data/processed/team_rates.csv"
        elif analysis_type == "county_alignment":
            result = county_alignment_bias(df)
            output_path = "data/processed/county_alignment.csv"
        else:
            return jsonify({"error": "Invalid analysis type"}), 400

        save_processed(result, output_path)

        # Convert result to JSON-serializable format
        result_dict = result.to_dict("records")

        return jsonify(
            {"success": True, "data": result_dict, "output_path": output_path}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_uploaded_files")
def get_uploaded_files():
    files = []
    for filename in os.listdir(app.config["UPLOAD_FOLDER"]):
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if os.path.isfile(filepath):
            files.append(
                {
                    "name": filename,
                    "path": filepath,
                    "size": os.path.getsize(filepath),
                    "type": (
                        filename.rsplit(".", 1)[1].lower()
                        if "." in filename
                        else "unknown"
                    ),
                }
            )
    return jsonify(files)


@app.route("/download/<path:filename>")
def download_file(filename):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return "File not found", 404


@app.route("/get_sport_call_types/<sport>")
def get_sport_call_types(sport):
    if sport.lower() in CALL_TYPES:
        return jsonify(CALL_TYPES[sport.lower()])
    return jsonify([])


@app.route("/save_pose_session", methods=["POST"])
def save_pose_session():
    """Save pose detection session data to CSV"""
    try:
        data = request.get_json()
        session_data = data.get("session_data", [])
        
        if not session_data:
            return jsonify({"error": "No session data provided"}), 400
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pose_session_{timestamp}.csv"
        filepath = os.path.join(DATA_FOLDER, filename)
        
        # Write CSV file
        if session_data:
            keys = session_data[0].keys()
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                writer.writeheader()
                writer.writerows(session_data)
        
        return jsonify({
            "success": True,
            "filename": filename,
            "filepath": filepath,
            "message": f"Session saved to {filename}"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_saved_sessions")
def get_saved_sessions():
    """Get list of saved pose detection sessions"""
    try:
        sessions = []
        if os.path.exists(DATA_FOLDER):
            for filename in os.listdir(DATA_FOLDER):
                if filename.startswith("pose_session_") and filename.endswith(".csv"):
                    filepath = os.path.join(DATA_FOLDER, filename)
                    file_size = os.path.getsize(filepath)
                    file_time = os.path.getmtime(filepath)
                    
                    sessions.append({
                        "filename": filename,
                        "filepath": filepath,
                        "size": file_size,
                        "created": datetime.fromtimestamp(file_time).isoformat()
                    })
        
        # Sort by creation time (newest first)
        sessions.sort(key=lambda x: x["created"], reverse=True)
        return jsonify(sessions)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download_session/<filename>")
def download_session(filename):
    """Download a saved pose detection session"""
    try:
        filepath = os.path.join(DATA_FOLDER, filename)
        
        # Security check
        if not filename.startswith("pose_session_") or not os.path.exists(filepath):
            return "File not found", 404
        
        return send_file(filepath, as_attachment=True, download_name=filename)
    
    except Exception as e:
        return str(e), 500


        return jsonify(CALL_TYPES[sport.lower()])
    return jsonify([])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
