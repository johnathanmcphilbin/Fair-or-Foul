#!/usr/bin/env python3
"""
Simple script to run the Fair-or-Foul web application
"""

from app import app

if __name__ == "__main__":
    print("Starting Fair-or-Foul Web Application...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")

    try:
        app.run(debug=True, host="0.0.0.0", port=5000)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {e}")
