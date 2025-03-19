#!/usr/bin/env python3
"""
Deep Research Assistant UI

A simple GUI wrapper for the Deep Research Assistant that provides a user-friendly
interface for configuring and running research queries without having to remember
command-line arguments.
"""

import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import threading
import time
import re
import signal  # Add signal module for sending SIGINT

class DeepResearchUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Deep Research Assistant UI")
        self.root.geometry("800x700")
        self.root.minsize(800, 700)
        
        # Process tracking
        self.current_process = None
        self.is_running = False
        
        # Configuration save/load
        self.config_dir = "config"
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            
        self.config_file = os.path.join(self.config_dir, "last_config.json")
        
        # Create the UI
        self.create_ui()
        
        # Load last configuration if available
        self.load_config(self.config_file)
    
    def create_ui(self):
        """Create the user interface."""
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Progress frame at the top
        self.progress_frame = ttk.Frame(main_frame)
        
        # Progress bar
        ttk.Label(self.progress_frame, text="Progress:").pack(side=tk.LEFT, padx=5)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, 
                                           length=400, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Time estimate
        self.time_var = tk.StringVar(value="Estimated time: --:--")
        ttk.Label(self.progress_frame, textvariable=self.time_var).pack(side=tk.RIGHT, padx=5)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Basic settings tab
        basic_frame = ttk.Frame(notebook, padding="10")
        notebook.add(basic_frame, text="Basic Settings")
        
        # Advanced settings tab
        advanced_frame = ttk.Frame(notebook, padding="10")
        notebook.add(advanced_frame, text="Advanced Settings")
        
        # Output tab
        output_frame = ttk.Frame(notebook, padding="10")
        notebook.add(output_frame, text="Output")
        
        # === Basic Settings Tab ===
        # Query input
        ttk.Label(basic_frame, text="Research Query:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.query_var = tk.StringVar()
        ttk.Entry(basic_frame, textvariable=self.query_var, width=60).grid(row=0, column=1, sticky=tk.W+tk.E, pady=5)
        
        # Research depth
        ttk.Label(basic_frame, text="Research Depth (1-5):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.depth_var = tk.IntVar(value=3)
        depth_frame = ttk.Frame(basic_frame)
        depth_frame.grid(row=1, column=1, sticky=tk.W, pady=5)
        for i in range(1, 6):
            text = {1: "Quick", 2: "Basic", 3: "Standard", 4: "Deep", 5: "Exhaustive"}
            ttk.Radiobutton(depth_frame, text=f"{i} - {text[i]}", variable=self.depth_var, value=i).pack(side=tk.LEFT, padx=10)
        
        # Summarization method
        ttk.Label(basic_frame, text="Summarization Method:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.method_var = tk.StringVar(value="textrank")
        method_frame = ttk.Frame(basic_frame)
        method_frame.grid(row=2, column=1, sticky=tk.W, pady=5)
        methods = [
            ("TextRank (no API required)", "textrank"),
            ("Ollama (local AI)", "ollama"),
            ("OpenAI (API key required)", "openai")
        ]
        for text, value in methods:
            ttk.Radiobutton(method_frame, text=text, variable=self.method_var, value=value, 
                          command=self.toggle_api_fields).pack(anchor=tk.W, pady=2)
        
        # Model selection (for Ollama/OpenAI)
        ttk.Label(basic_frame, text="AI Model:").grid(row=3, column=0, sticky=tk.W, pady=5)
        model_frame = ttk.Frame(basic_frame)
        model_frame.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        self.model_var = tk.StringVar(value="gemma3:12b")
        self.model_combobox = ttk.Combobox(model_frame, textvariable=self.model_var, width=30, state="readonly")
        self.model_combobox['values'] = ("gemma3:12b", "llama3:latest", "deepseek-r1:8b", "gpt-3.5-turbo", "gpt-4")
        self.model_combobox.current(0)  # Set initial selection explicitly
        self.model_combobox.pack(side=tk.LEFT, padx=5)
        
        # Summary depth
        ttk.Label(basic_frame, text="Summary Depth:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.summary_depth_var = tk.StringVar(value="medium")
        depth_opts_frame = ttk.Frame(basic_frame)
        depth_opts_frame.grid(row=4, column=1, sticky=tk.W, pady=5)
        for text, value in [("Short", "short"), ("Medium", "medium"), ("Detailed", "detailed")]:
            ttk.Radiobutton(depth_opts_frame, text=text, variable=self.summary_depth_var, value=value).pack(side=tk.LEFT, padx=10)
        
        # Output format
        ttk.Label(basic_frame, text="Output Format:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.format_var = tk.StringVar(value="markdown")
        format_frame = ttk.Frame(basic_frame)
        format_frame.grid(row=5, column=1, sticky=tk.W, pady=5)
        for text, value in [("Markdown", "markdown"), ("Plain Text", "text"), ("JSON", "json")]:
            ttk.Radiobutton(format_frame, text=text, variable=self.format_var, value=value).pack(side=tk.LEFT, padx=10)
        
        # === Advanced Settings Tab ===
        # API Key for OpenAI
        ttk.Label(advanced_frame, text="OpenAI API Key:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(advanced_frame, textvariable=self.api_key_var, width=50, show="*")
        self.api_key_entry.grid(row=0, column=1, sticky=tk.W+tk.E, pady=5)
        
        # Ollama URL
        ttk.Label(advanced_frame, text="Ollama API URL:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.ollama_url_var = tk.StringVar(value="http://localhost:11434/api/generate")
        self.ollama_url_entry = ttk.Entry(advanced_frame, textvariable=self.ollama_url_var, width=50)
        self.ollama_url_entry.grid(row=1, column=1, sticky=tk.W+tk.E, pady=5)
        
        # Output Directory
        ttk.Label(advanced_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, pady=5)
        dir_frame = ttk.Frame(advanced_frame)
        dir_frame.grid(row=2, column=1, sticky=tk.W+tk.E, pady=5)
        self.output_dir_var = tk.StringVar(value="research_results")
        ttk.Entry(dir_frame, textvariable=self.output_dir_var, width=40).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(dir_frame, text="Browse...", command=self.browse_output_dir).pack(side=tk.RIGHT, padx=5)
        
        # Max Sources
        ttk.Label(advanced_frame, text="Maximum Sources:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.max_sources_var = tk.StringVar(value="0")
        ttk.Entry(advanced_frame, textvariable=self.max_sources_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=5)
        ttk.Label(advanced_frame, text="(0 = auto-scale based on depth)").grid(row=3, column=1, sticky=tk.W, padx=120, pady=5)
        
        # Excluded Domains
        ttk.Label(advanced_frame, text="Excluded Domains:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.excluded_domains_var = tk.StringVar()
        ttk.Entry(advanced_frame, textvariable=self.excluded_domains_var, width=50).grid(row=4, column=1, sticky=tk.W+tk.E, pady=5)
        ttk.Label(advanced_frame, text="(space-separated list)").grid(row=4, column=1, sticky=tk.E, pady=5)
        
        # Photon Path
        ttk.Label(advanced_frame, text="Photon Script Path:").grid(row=5, column=0, sticky=tk.W, pady=5)
        photon_frame = ttk.Frame(advanced_frame)
        photon_frame.grid(row=5, column=1, sticky=tk.W+tk.E, pady=5)
        self.photon_path_var = tk.StringVar(value="./photon.py")
        ttk.Entry(photon_frame, textvariable=self.photon_path_var, width=40).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(photon_frame, text="Browse...", command=self.browse_photon_path).pack(side=tk.RIGHT, padx=5)
        
        # Verbose and Debug options
        options_frame = ttk.Frame(advanced_frame)
        options_frame.grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=10)
        self.verbose_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Verbose Output", variable=self.verbose_var).pack(side=tk.LEFT, padx=10)
        self.debug_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Debug Mode", variable=self.debug_var).pack(side=tk.LEFT, padx=10)
        
        # === Output Tab ===
        # Command preview
        ttk.Label(output_frame, text="Command Preview:").pack(anchor=tk.W, pady=5)
        self.command_text = scrolledtext.ScrolledText(output_frame, height=4, wrap=tk.WORD)
        self.command_text.pack(fill=tk.X, expand=False, pady=5)
        
        # Output display
        ttk.Label(output_frame, text="Output:").pack(anchor=tk.W, pady=5)
        self.output_text = scrolledtext.ScrolledText(output_frame, height=15, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Button Frame (at the bottom of the main window)
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Save configuration button
        ttk.Button(button_frame, text="Save Configuration", command=self.save_config_dialog).pack(side=tk.LEFT, padx=5)
        
        # Load configuration button
        ttk.Button(button_frame, text="Load Configuration", command=self.load_config_dialog).pack(side=tk.LEFT, padx=5)
        
        # Update command button
        ttk.Button(button_frame, text="Update Command", command=self.update_command).pack(side=tk.LEFT, padx=5)
        
        # Run button
        self.run_button = ttk.Button(button_frame, text="Run Research", command=self.run_research)
        self.run_button.pack(side=tk.RIGHT, padx=5)
        
        # Stop button (initially disabled)
        self.stop_button = ttk.Button(button_frame, text="Stop & Extract Results", 
                                   command=self.stop_research, state=tk.DISABLED)
        self.stop_button.pack(side=tk.RIGHT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        # Update command initially
        self.update_command()
        
        # Initial API field state
        self.toggle_api_fields()
    
    def toggle_api_fields(self):
        """Enable/disable API fields based on the selected method."""
        method = self.method_var.get()
        
        if method == "openai":
            self.api_key_entry.config(state="normal")
            self.ollama_url_entry.config(state="disabled")
            self.model_combobox.config(state="readonly")  # Changed to readonly
            self.model_combobox['values'] = ("gpt-3.5-turbo", "gpt-4", "gpt-4-turbo")
            if not self.model_var.get() in ("gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"):
                self.model_var.set("gpt-3.5-turbo")
        elif method == "ollama":
            self.api_key_entry.config(state="disabled")
            self.ollama_url_entry.config(state="normal")
            self.model_combobox.config(state="readonly")  # Changed to readonly
            self.model_combobox['values'] = ("gemma3:12b", "llama3:latest", "deepseek-r1:8b", "mistral:latest")
            if not self.model_var.get() in ("gemma3:12b", "llama3:latest", "deepseek-r1:8b", "mistral:latest"):
                self.model_var.set("gemma3:12b")
            # Force update the display
            self.model_combobox.selection_clear()
            self.model_combobox.update()
        else:  # textrank
            self.api_key_entry.config(state="disabled")
            self.ollama_url_entry.config(state="disabled")
            self.model_combobox.config(state="disabled")
    
    def browse_output_dir(self):
        """Open directory browser to select output directory."""
        directory = filedialog.askdirectory(initialdir=self.output_dir_var.get())
        if directory:
            self.output_dir_var.set(directory)
    
    def browse_photon_path(self):
        """Open file browser to select photon.py path."""
        filepath = filedialog.askopenfilename(
            initialdir=os.path.dirname(self.photon_path_var.get()),
            title="Select Photon Script",
            filetypes=(("Python files", "*.py"), ("All files", "*.*"))
        )
        if filepath:
            self.photon_path_var.set(filepath)
    
    def update_command(self):
        """Update the command preview based on current settings."""
        command = ["python", "research-report.py"]
        
        # Add arguments
        if self.depth_var.get() != 3:  # Only add if not default
            command.extend(["--depth", str(self.depth_var.get())])
        
        if self.method_var.get() != "textrank":
            command.extend(["--method", self.method_var.get()])
        
        if self.model_var.get() and self.method_var.get() != "textrank":
            command.extend(["--model", self.model_var.get()])
        
        if self.summary_depth_var.get() != "medium":
            command.extend(["--summary-depth", self.summary_depth_var.get()])
        
        if self.format_var.get() != "markdown":
            command.extend(["--format", self.format_var.get()])
        
        if self.output_dir_var.get() != "research_results":
            command.extend(["--output-dir", self.output_dir_var.get()])
        
        if self.max_sources_var.get() and self.max_sources_var.get() != "0":
            command.extend(["--max-sources", self.max_sources_var.get()])
        
        if self.excluded_domains_var.get():
            excluded = self.excluded_domains_var.get().split()
            command.extend(["--exclude-domains"] + excluded)
        
        if self.method_var.get() == "ollama" and self.ollama_url_var.get() != "http://localhost:11434/api/generate":
            command.extend(["--ollama-url", self.ollama_url_var.get()])
        
        if self.method_var.get() == "openai" and self.api_key_var.get():
            command.extend(["--openai-key", self.api_key_var.get()])
        
        if self.photon_path_var.get() != "./photon.py":
            command.extend(["--photon-path", self.photon_path_var.get()])
        
        if self.verbose_var.get():
            command.append("--verbose")
        
        if self.debug_var.get():
            command.append("--debug")
        
        # Add query at the end
        if self.query_var.get():
            command.append(f'"{self.query_var.get()}"')
        
        # Display the command (with API key masked)
        display_command = command.copy()
        if "--openai-key" in display_command:
            key_index = display_command.index("--openai-key") + 1
            if key_index < len(display_command):
                display_command[key_index] = "****"
        
        self.command_text.delete(1.0, tk.END)
        self.command_text.insert(tk.END, " ".join(display_command))
        
        # Store the actual command
        self.current_command = command
    
    def run_research(self):
        """Run the research process in a separate thread."""
        # Validate query
        if not self.query_var.get().strip():
            messagebox.showerror("Error", "Please enter a research query")
            return
        
        # Update command
        self.update_command()
        
        # Disable run button, enable stop button
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.is_running = True
        self.status_var.set("Running research...")
        
        # Clear output
        self.output_text.delete(1.0, tk.END)
        
        # Reset and show progress bar
        self.progress_var.set(0)
        
        # Show progress frame - fixed position
        main_frame = self.root.children["!frame"]
        self.progress_frame.pack(in_=main_frame, fill=tk.X, pady=5, before=main_frame.children["!notebook"])
        
        # Calculate estimated time based on depth and max sources
        self._update_time_estimate()
        
        # Save current configuration
        self.save_config(self.config_file)
        
        # Run in a separate thread
        threading.Thread(target=self._run_process, daemon=True).start()
    
    def stop_research(self):
        """Stop the currently running research process and try to extract results."""
        if not self.is_running or not self.current_process:
            return
            
        try:
            # Indicate stopping
            self.status_var.set("Stopping research and extracting results...")
            self.output_text.insert(tk.END, "\n\n----- STOPPING RESEARCH EARLY (EXTRACTING PARTIAL RESULTS) -----\n\n")
            
            # Send SIGINT to the process (equivalent to Ctrl+C) which should trigger the research
            # script to save partial results if it has any
            if sys.platform == 'win32':
                # On Windows we need to use taskkill
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.current_process.pid)], 
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                # On Unix systems we can send SIGINT for a more graceful shutdown
                import signal
                self.current_process.send_signal(signal.SIGINT)
                
            # Give the process a moment to clean up
            self.root.after(500, self._check_process_stopped)
            
        except Exception as e:
            self.output_text.insert(tk.END, f"\nError stopping research: {str(e)}\n")
            self.reset_ui_after_run()
    
    def _check_process_stopped(self):
        """Check if the process has stopped, force kill if necessary."""
        try:
            if self.current_process and self.current_process.poll() is None:
                # Still running, wait a bit longer (up to 5 seconds total)
                if getattr(self, '_stop_wait_count', 0) < 10:
                    self._stop_wait_count = getattr(self, '_stop_wait_count', 0) + 1
                    self.root.after(500, self._check_process_stopped)
                else:
                    # Force terminate if it's taking too long
                    self.current_process.terminate()
                    self.output_text.insert(tk.END, "\nForced termination of research process.\n")
                    self.reset_ui_after_run()
            else:
                # Process has stopped
                self.output_text.insert(tk.END, "\nResearch stopped. Check output directory for partial results.\n")
                self.reset_ui_after_run()
                
                # Look for output file path in the output text
                output_text = self.output_text.get(1.0, tk.END)
                file_match = re.search(r'Research results saved to: (.+\.(?:md|txt|json))', output_text)
                if file_match:
                    filepath = file_match.group(1)
                    if os.path.exists(filepath):
                        self.output_text.insert(tk.END, f"\nPartial results were saved to: {filepath}\n")
                        if messagebox.askyesno("Open Results", "Would you like to open the partial results file?"):
                            self._open_file(filepath)
                
        except Exception as e:
            self.output_text.insert(tk.END, f"\nError checking process status: {str(e)}\n")
            self.reset_ui_after_run()
    
    def _open_file(self, filepath):
        """Open a file with the default system application."""
        try:
            if sys.platform == 'win32':
                os.startfile(filepath)
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', filepath])
            else:  # Linux
                subprocess.run(['xdg-open', filepath])
        except Exception as e:
            self.output_text.insert(tk.END, f"\nError opening file: {str(e)}\n")
    
    def reset_ui_after_run(self):
        """Reset UI elements after a run completes or is stopped."""
        self.is_running = False
        self.current_process = None
        self._stop_wait_count = 0
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        # Hide progress frame after delay
        self.root.after(2000, self.progress_frame.pack_forget)
    
    def _update_time_estimate(self):
        """Update the estimated time based on depth and max sources."""
        depth = self.depth_var.get()
        max_sources = int(self.max_sources_var.get() or "0")
        
        # Base time in minutes for each depth level
        base_times = {
            1: 2,    # Quick: ~2 minutes
            2: 5,    # Basic: ~5 minutes
            3: 10,   # Standard: ~10 minutes
            4: 20,   # Deep: ~20 minutes
            5: 40    # Exhaustive: ~40 minutes
        }
        
        # Calculate estimate
        if max_sources > 0:
            # If max_sources is specified, scale time accordingly
            estimate_minutes = max(base_times[depth] * (max_sources / 20), base_times[depth])
        else:
            # Otherwise use the base time for the depth level
            estimate_minutes = base_times[depth]
        
        # Format as MM:SS
        minutes = int(estimate_minutes)
        seconds = int((estimate_minutes - minutes) * 60)
        self.time_var.set(f"Estimated time: {minutes:02d}:{seconds:02d}")
        
        # Store for progress calculations
        self.estimate_seconds = minutes * 60 + seconds
    
    def _run_process(self):
        """Run the subprocess in a thread."""
        try:
            # Build command
            command = self.current_command.copy()
            
            # Replace query with proper formatting
            if command[-1].startswith('"') and command[-1].endswith('"'):
                command[-1] = command[-1][1:-1]
            
            # Run the command
            self.current_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Initialize progress tracking
            start_time = time.time()
            steps_completed = 0
            total_steps = 5  # The process has 5 main steps
            progress_pattern = re.compile(r'\[(\d+)/5\]')
            extraction_pattern = re.compile(r'Extracting content from (\d+) sources')
            source_pattern = re.compile(r'Successfully extracted content from (\d+) sources')
            
            # Process and display output in real-time
            for line in iter(self.current_process.stdout.readline, ''):
                # Check if we should exit early
                if not self.is_running:
                    break
                    
                self.output_text.insert(tk.END, line)
                self.output_text.see(tk.END)
                self.output_text.update_idletasks()
                
                # Parse output to update progress
                step_match = progress_pattern.search(line)
                if step_match:
                    current_step = int(step_match.group(1))
                    if current_step > steps_completed:
                        steps_completed = current_step
                        self.progress_var.set((steps_completed / total_steps) * 100)
                        
                        # Update time estimate
                        elapsed = time.time() - start_time
                        if steps_completed > 0:
                            # Estimate total time based on completed steps
                            estimated_total = elapsed * (total_steps / steps_completed)
                            remaining = max(0, estimated_total - elapsed)
                            mins, secs = divmod(int(remaining), 60)
                            self.time_var.set(f"Remaining: {mins:02d}:{secs:02d}")
                
                # Extract more detailed progress from source extraction step
                if steps_completed == 3:  # During extraction step
                    extract_match = extraction_pattern.search(line)
                    if extract_match:
                        total_sources = int(extract_match.group(1))
                        
                    source_match = source_pattern.search(line)
                    if source_match:
                        extracted_sources = int(source_match.group(1))
                        if total_sources > 0:
                            # Calculate progress within this step
                            step_progress = (extracted_sources / total_sources) * 20  # 20% for this step
                            total_progress = 40 + step_progress  # Step 3 starts at 40%
                            self.progress_var.set(total_progress)
            
            # Wait for process to complete
            returncode = self.current_process.wait()
            
            # Set progress to 100% on success
            if returncode == 0:
                self.progress_var.set(100)
                self.status_var.set("Research completed successfully")
                
                # Check if results file was generated and offer to open it
                output_text = self.output_text.get(1.0, tk.END)
                file_match = re.search(r'Research results saved to: (.+\.(?:md|txt|json))', output_text)
                if file_match:
                    filepath = file_match.group(1)
                    if os.path.exists(filepath):
                        self.output_text.insert(tk.END, f"\nResults ready at: {filepath}\n")
                        if messagebox.askyesno("Open Results", "Would you like to open the results file?"):
                            self._open_file(filepath)
            else:
                self.status_var.set(f"Research failed with return code {returncode}")
            
            # Reset UI
            self.reset_ui_after_run()
        
        except Exception as e:
            self.output_text.insert(tk.END, f"\nError: {str(e)}\n")
            self.status_var.set("Error occurred during execution")
            self.reset_ui_after_run()
    
    def save_config_dialog(self):
        """Open a dialog to save the current configuration."""
        filepath = filedialog.asksaveasfilename(
            initialdir=self.config_dir,
            title="Save Configuration",
            defaultextension=".json",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        if filepath:
            self.save_config(filepath)
            self.status_var.set(f"Configuration saved to {filepath}")
    
    def save_config(self, filepath):
        """Save the current configuration to a file."""
        config = {
            "query": self.query_var.get(),
            "depth": self.depth_var.get(),
            "method": self.method_var.get(),
            "model": self.model_var.get(),
            "summary_depth": self.summary_depth_var.get(),
            "format": self.format_var.get(),
            "output_dir": self.output_dir_var.get(),
            "max_sources": self.max_sources_var.get(),
            "excluded_domains": self.excluded_domains_var.get(),
            "ollama_url": self.ollama_url_var.get(),
            "api_key": self.api_key_var.get(),
            "photon_path": self.photon_path_var.get(),
            "verbose": self.verbose_var.get(),
            "debug": self.debug_var.get()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
    
    def load_config_dialog(self):
        """Open a dialog to load a configuration file."""
        filepath = filedialog.askopenfilename(
            initialdir=self.config_dir,
            title="Load Configuration",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        if filepath:
            self.load_config(filepath)
            self.status_var.set(f"Configuration loaded from {filepath}")
    
    def load_config(self, filepath):
        """Load configuration from a file."""
        if not os.path.exists(filepath):
            return
        
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            # Apply configuration to UI elements
            if "query" in config:
                self.query_var.set(config["query"])
            if "depth" in config:
                self.depth_var.set(config["depth"])
            if "method" in config:
                self.method_var.set(config["method"])
            if "model" in config:
                self.model_var.set(config["model"])
            if "summary_depth" in config:
                self.summary_depth_var.set(config["summary_depth"])
            if "format" in config:
                self.format_var.set(config["format"])
            if "output_dir" in config:
                self.output_dir_var.set(config["output_dir"])
            if "max_sources" in config:
                self.max_sources_var.set(config["max_sources"])
            if "excluded_domains" in config:
                self.excluded_domains_var.set(config["excluded_domains"])
            if "ollama_url" in config:
                self.ollama_url_var.set(config["ollama_url"])
            if "api_key" in config:
                self.api_key_var.set(config["api_key"])
            if "photon_path" in config:
                self.photon_path_var.set(config["photon_path"])
            if "verbose" in config:
                self.verbose_var.set(config["verbose"])
            if "debug" in config:
                self.debug_var.set(config["debug"])
            
            # Update dependencies
            self.toggle_api_fields()
            self.update_command()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = DeepResearchUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
