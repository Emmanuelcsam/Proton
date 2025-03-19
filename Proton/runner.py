#!/usr/bin/env python3
"""
OSINT Suite Runner

This script automates running the entire OSINT workflow:
1. Running proton.py to search and crawl websites
2. Running protonizer.py to analyze and summarize the results
3. Starting the app.py web server
4. Opening the frontend in a web browser

Usage:
    python3 osint_runner.py "search keywords" "example.com" 50
"""

import argparse
import subprocess
import time
import webbrowser
import os
import sys
import signal
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the complete OSINT workflow")
    parser.add_argument("keyword", help="Search keyword for proton.py")
    parser.add_argument("site", nargs='?', default=None, help="Site to search/crawl (optional)")
    parser.add_argument("depth", type=int, help="Depth value (1-300)")
    parser.add_argument("--skip-proton", action="store_true", 
                        help="Skip running proton.py (use existing results)")
    parser.add_argument("--skip-protonizer", action="store_true", 
                        help="Skip running protonizer.py")
    parser.add_argument("--port", type=int, default=5000, 
                        help="Port for the web server (default: 5000)")
    
    # Handle the case where user provides only keyword and depth
    args = parser.parse_args()
    
    # If user provided just two arguments, the second is actually the depth
    if len(sys.argv) == 3 and args.site and args.site.isdigit() and args.depth is None:
        args.depth = int(args.site)
        args.site = None
    
    return args

def run_command(cmd, description):
    """Run a command and handle any errors."""
    print(f"\n[+] {description}...")
    try:
        print(f"[*] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"[+] Command completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] Error: {e}")
        return False
    except Exception as e:
        print(f"[!] Unexpected error: {e}")
        return False

def start_web_server(port=5000):
    """Start the Flask web server in the background."""
    print(f"\n[+] Starting web server on port {port}...")
    try:
        # Define the correct static directory path
        static_dir = Path("/home/jarvis/Proton/static")
        
        # Make sure the static directory exists
        static_dir.mkdir(exist_ok=True, parents=True)
        print(f"[*] Ensuring static directory exists: {static_dir}")
        
        # Copy frontend.html to the static directory if needed
        frontend_path = Path("frontend.html")
        static_index_path = static_dir / "index.html"
        
        if frontend_path.exists():
            print(f"[*] Copying frontend.html to {static_index_path}")
            with open(frontend_path, "r") as src, open(static_index_path, "w") as dst:
                dst.write(src.read())
        else:
            print(f"[!] Warning: frontend.html not found in current directory")
            # Check if it's already in the destination
            if not static_index_path.exists():
                print(f"[!] Warning: {static_index_path} doesn't exist either!")
                print(f"[*] The web interface may not function correctly")
        
        # Start the server as a background process
        cmd = ["python3", "app.py"]
        server_process = subprocess.Popen(cmd, 
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
        
        # Give the server a moment to start up
        time.sleep(2)
        
        # Check if the process is still running
        if server_process.poll() is None:
            print(f"[+] Web server started successfully (PID: {server_process.pid})")
            return server_process
        else:
            stdout, stderr = server_process.communicate()
            print(f"[!] Web server failed to start: {stderr.decode()}")
            return None
    except Exception as e:
        print(f"[!] Error starting web server: {e}")
        return None

def open_web_browser(port=5000):
    """Open the web browser to view the interface."""
    url = f"http://localhost:{port}"
    print(f"\n[+] Opening web browser to {url}...")
    try:
        webbrowser.open(url)
        print(f"[+] Browser opened successfully")
        return True
    except Exception as e:
        print(f"[!] Failed to open browser: {e}")
        print(f"[+] Please manually navigate to {url}")
        return False

def main():
    args = parse_arguments()
    
    # Run proton.py if not skipped
    if not args.skip_proton:
        # Build the command based on whether a site was provided
        if args.site:
            cmd = ["python3", "proton.py", args.keyword, args.site, str(args.depth)]
            description = f"Running proton.py for '{args.keyword}' on site '{args.site}' with depth {args.depth}"
        else:
            cmd = ["python3", "proton.py", args.keyword, str(args.depth)]
            description = f"Running proton.py for '{args.keyword}' with depth {args.depth}"
        
        success = run_command(cmd, description)
        if not success:
            choice = input("[!] proton.py encountered errors. Continue anyway? (y/n): ").lower()
            if choice != 'y':
                return
    else:
        print("\n[*] Skipping proton.py as requested")
    
    # Run protonizer.py if not skipped
    if not args.skip_protonizer:
        success = run_command(
            ["python3", "protonizer.py"],
            "Running protonizer.py to analyze and summarize the results"
        )
        if not success:
            choice = input("[!] protonizer.py encountered errors. Continue anyway? (y/n): ").lower()
            if choice != 'y':
                return
    else:
        print("\n[*] Skipping protonizer.py as requested")
    
    # Start the web server
    server_process = start_web_server(args.port)
    if not server_process:
        print("[!] Failed to start web server. Exiting.")
        return
    
    # Open the web browser
    open_web_browser(args.port)
    
    print("\n[+] OSINT suite is now running")
    print(f"[+] The web interface is available at http://localhost:{args.port}")
    print("[+] Press Ctrl+C to stop the server and exit")
    
    try:
        # Keep the script running until the user interrupts
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[+] Stopping web server...")
        os.kill(server_process.pid, signal.SIGTERM)
        time.sleep(1)
        print("[+] Done. Goodbye!")

if __name__ == "__main__":
    main()
