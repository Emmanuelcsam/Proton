#!/usr/bin/env python3

import os
import re
import subprocess
import webbrowser
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Change this path to the directory where you want to look for summary.txt files
BASE_DIRECTORY = "/home/jarvis/photon_results"

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Regular expression pattern to match URLs
URL_PATTERN = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'

def gather_summary_data(base_path):
    """
    Recursively walk through 'base_path' and gather information about all 'summary.txt' files.
    Returns a list of dictionaries with id, folder_path and file_path.
    """
    all_summaries = []
    id_counter = 1

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower() == "summary.txt":
                filepath = os.path.join(root, file)
                try:
                    relative_folder = os.path.relpath(root, base_path)
                    all_summaries.append({
                        "id": id_counter,
                        "folder": relative_folder,
                        "filepath": filepath
                    })
                    id_counter += 1
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    return all_summaries

def get_summary_content(filepath):
    """Read and return the content of a summary file"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return f"Error reading file: {str(e)}"

def open_file_location(filepath):
    """Open the file's directory in the system file explorer"""
    directory = os.path.dirname(filepath)
    
    # Platform-specific commands to open the file explorer
    try:
        if os.name == 'nt':  # Windows
            os.startfile(directory)
        elif os.name == 'posix':  # Linux, macOS
            if os.system(f'xdg-open "{directory}"') != 0:  # Try Linux first
                os.system(f'open "{directory}"')  # Try macOS
        return True
    except Exception as e:
        print(f"Error opening location: {e}")
        return False

# API Routes
@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/api/summaries')
def get_summaries():
    """API endpoint to get all summaries"""
    try:
        summaries = gather_summary_data(BASE_DIRECTORY)
        return jsonify(summaries)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/summary/<int:summary_id>')
def get_summary(summary_id):
    """API endpoint to get a specific summary's content"""
    try:
        summaries = gather_summary_data(BASE_DIRECTORY)
        
        # Find the summary with the matching ID
        summary = next((s for s in summaries if s["id"] == summary_id), None)
        
        if not summary:
            return jsonify({"error": "Summary not found"}), 404
        
        content = get_summary_content(summary["filepath"])
        
        return jsonify({
            "summary": summary,
            "content": content
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/open-location', methods=['POST'])
def api_open_location():
    """API endpoint to open a file location"""
    try:
        data = request.json
        if not data or 'path' not in data:
            return jsonify({"error": "No path provided"}), 400
        
        filepath = data['path']
        success = open_file_location(filepath)
        
        if success:
            return jsonify({"status": "success"})
        else:
            return jsonify({"error": "Failed to open location"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/open-url', methods=['POST'])
def api_open_url():
    """API endpoint to open a URL in the browser"""
    try:
        data = request.json
        if not data or 'url' not in data:
            return jsonify({"error": "No URL provided"}), 400
        
        url = data['url']
        # Validate URL
        if not re.match(URL_PATTERN, url):
            return jsonify({"error": "Invalid URL"}), 400
            
        webbrowser.open(url)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Create the static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Copy the HTML file to the static directory (in a real app, you'd have a build process)
    # For now, assuming the HTML file is in the same directory as this script
    try:
        with open('frontend.html', 'r') as src, open('static/index.html', 'w') as dst:
            dst.write(src.read())
    except FileNotFoundError:
        print("Warning: frontend.html not found. Please place the HTML file in the static directory.")
    
    # Start the Flask server
    print(f"Starting server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
