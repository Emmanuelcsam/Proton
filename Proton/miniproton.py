#!/usr/bin/env python3
import argparse
import subprocess
import sys
import logging
import shutil
import traceback
import json
import os

VENV_PATH = os.path.expanduser("~/newvenv/bin/activate")  # Adjust virtualenv path as needed

def check_command(command):
    """Verify if a command is available in the system PATH."""
    if shutil.which(command) is None:
        logging.error(f"'{command}' is not installed. Please install it to proceed.")
        sys.exit(1)
    logging.debug(f"'{command}' is available.")

def run_command(command, label, debug=False, use_venv=False):
    """Execute a command, optionally using a virtual environment."""
    try:
        check_command(command[0])
        logging.debug(f"Executing command: {' '.join(command)}")

        # If running Photon, ensure it is within the virtual environment
        if use_venv:
            command = f"source {VENV_PATH} && " + " ".join(command)
            result = subprocess.run(command, shell=True, capture_output=True, text=True, executable="/bin/bash")
        else:
            result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"Error running {label}: {result.stderr.strip()}")
            if debug:
                logging.debug("Full stderr: " + result.stderr)
            return None

        logging.debug(f"{label} output: {result.stdout.strip()}")
        return result.stdout.strip()

    except Exception as exc:
        logging.error(f"Exception running {label}: {exc}")
        if debug:
            traceback.print_exc()
        return None

def run_ddgr(query, debug=False):
    """Run ddgr with the given search query and output formatted results."""
    print(f"\n[+] DuckDuckGo (ddgr) search results for: {query}\n")
    output = run_command(['ddgr', '--json', query], "ddgr", debug)
    
    if output:
        try:
            results = json.loads(output)
            for result in results[:10]:  # Limit output to 10 results
                print(f"- {result.get('title', 'No Title')}\n  {result.get('url', 'No URL')}\n")
        except json.JSONDecodeError:
            logging.error("Failed to parse ddgr JSON output.")

def run_photon(target, debug=False):
    """Run Photon OSINT tool on the given target URL."""
    print(f"\n[+] Photon OSINT results for: {target}\n")
    output = run_command(['python3', 'photon.py', '-u', target, '-o', 'json'], "Photon OSINT", debug, use_venv=True)
    
    if output:
        print(output)

def main():
    parser = argparse.ArgumentParser(description="OSINT tool using Photon OSINT and ddgr.")
    parser.add_argument('--query', type=str, help="Search query for DuckDuckGo (ddgr)")
    parser.add_argument('--target', type=str, help="Target URL for Photon OSINT scraping")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode for detailed output")
    args = parser.parse_args()

    # Configure logging based on the debug flag.
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging_level, format='[%(levelname)s] %(message)s')

    if args.debug:
        logging.debug("Debug mode enabled.")

    if not args.query and not args.target:
        parser.error("You must provide at least --query or --target (or both).")

    if args.query:
        run_ddgr(args.query, args.debug)

    if args.target:
        run_photon(args.target, args.debug)

if __name__ == '__main__':
    main()
