#!/usr/bin/env python3

import os
import re
import tkinter as tk
from tkinter import ttk, messagebox
import webbrowser
import threading
from functools import partial

# Change this path to the directory where you want to look for summary.txt files
BASE_DIRECTORY = "/home/jarvis/photon_results"

# Regular expression pattern to match URLs
URL_PATTERN = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'

def gather_summary_data(base_path):
    """
    Recursively walk through 'base_path' and gather text from all 'summary.txt' files.
    Returns a list of tuples (folder_path, file_contents, file_path).
    """
    all_summaries = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower() == "summary.txt":
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                    relative_folder = os.path.relpath(root, base_path)
                    all_summaries.append((relative_folder, content, filepath))
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

    return all_summaries

def open_file_location(filepath):
    """Open the file's directory in the system file explorer"""
    directory = os.path.dirname(filepath)
    
    # Platform-specific commands to open the file explorer
    if os.name == 'nt':  # Windows
        os.startfile(directory)
    elif os.name == 'posix':  # Linux, macOS
        try:
            # Try using xdg-open for Linux
            os.system(f'xdg-open "{directory}"')
        except:
            try:
                # Try using open for macOS
                os.system(f'open "{directory}"')
            except:
                messagebox.showinfo("Information", f"Path: {directory}")

class SummaryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hacker Information Board")
        self.root.geometry("1100x750")
        
        # Define color scheme - hacker theme
        self.colors = {
            'bg_dark': "#0D0D0D",
            'bg_light': "#1A1A1A",
            'accent': "#00FF41",      # Matrix green
            'text_light': "#00FF41",  # Matrix green
            'text_dark': "#0D0D0D",
            'highlight': "#FFD700",   # Gold for highlights
            'error': "#FF3131"        # Red for errors
        }
        
        # Configure styles
        self.configure_styles()
        
        # Create UI elements
        self.create_ui()
        
        # Initialize variables
        self.summaries_data = {}
        self.search_results = []
        self.current_match_index = tk.IntVar(value=-1)
        
        # Load data
        self.load_summaries()
    
    def configure_styles(self):
        """Configure the ttk styles for the application"""
        style = ttk.Style()
        style.theme_use("clam")
        
        # Main frames
        style.configure(
            "Custom.TFrame", 
            background=self.colors['bg_dark']
        )
        style.configure(
            "Search.TFrame", 
            background=self.colors['bg_light']
        )
        
        # Labels
        style.configure(
            "Custom.TLabel", 
            background=self.colors['bg_dark'], 
            foreground=self.colors['text_light'], 
            font=("Courier New", 14, "bold")
        )
        style.configure(
            "Search.TLabel", 
            background=self.colors['bg_light'], 
            foreground=self.colors['text_light'], 
            font=("Courier New", 10)
        )
        
        # Buttons
        style.configure(
            "Custom.TButton",
            background=self.colors['accent'],
            foreground=self.colors['text_dark'],
            padding=6,
            font=("Courier New", 10, "bold")
        )
        style.map(
            "Custom.TButton",
            background=[('active', self.colors['highlight'])]
        )
        
        # Scrollbars
        style.configure(
            "Custom.Vertical.TScrollbar",
            troughcolor=self.colors['bg_light'],
            gripcount=0,
            background=self.colors['accent'],
            bordercolor=self.colors['bg_dark'],
            arrowcolor=self.colors['text_dark']
        )
        style.configure(
            "Custom.Horizontal.TScrollbar",
            troughcolor=self.colors['bg_light'],
            gripcount=0,
            background=self.colors['accent'],
            bordercolor=self.colors['bg_dark'],
            arrowcolor=self.colors['text_dark']
        )
        
        # Entry fields
        style.configure(
            "Custom.TEntry",
            fieldbackground=self.colors['bg_light'],
            foreground=self.colors['text_light'],
            bordercolor=self.colors['accent'],
            lightcolor=self.colors['accent'],
            darkcolor=self.colors['accent'],
        )
        
        # Checkbutton
        style.configure(
            "Search.TCheckbutton", 
            background=self.colors['bg_light'], 
            foreground=self.colors['text_light']
        )
        
        # Treeview
        style.configure(
            "Treeview", 
            background=self.colors['bg_light'], 
            foreground=self.colors['text_light'], 
            fieldbackground=self.colors['bg_light'],
            bordercolor=self.colors['bg_dark']
        )
        style.configure(
            "Treeview.Heading", 
            background=self.colors['bg_dark'], 
            foreground=self.colors['text_light'], 
            bordercolor=self.colors['bg_dark'],
            font=("Courier New", 10, "bold")
        )
        style.map(
            "Treeview", 
            background=[('selected', self.colors['accent'])],
            foreground=[('selected', self.colors['text_dark'])]
        )
    
    def create_ui(self):
        """Create the user interface"""
        # Main frame
        self.main_frame = ttk.Frame(self.root, style="Custom.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        self.create_header()
        
        # Search bar
        self.create_search_bar()
        
        # Split pane for list and content
        self.create_content_panes()
        
        # Status bar
        self.create_status_bar()
        
        # Bind keyboard shortcuts
        self.bind_shortcuts()
    
    def create_header(self):
        """Create the header section"""
        header_frame = ttk.Frame(self.main_frame, style="Custom.TFrame")
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        title_label = ttk.Label(
            header_frame, 
            text="[ ACCESSING COMPROMISED DATA ARCHIVE ]", 
            style="Custom.TLabel"
        )
        title_label.pack(side=tk.LEFT, padx=5)
    
    def create_search_bar(self):
        """Create the search bar"""
        search_frame = ttk.Frame(self.main_frame, style="Search.TFrame")
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Search label
        search_label = ttk.Label(search_frame, text="[ SEARCH ]:", style="Search.TLabel")
        search_label.pack(side=tk.LEFT, padx=5)
        
        # Search entry
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=40, style="Custom.TEntry")
        self.search_entry.pack(side=tk.LEFT, padx=5)
        
        # Case-sensitive search option
        self.case_sensitive_var = tk.BooleanVar(value=False)
        case_check = ttk.Checkbutton(
            search_frame, 
            text="Case Sensitive", 
            variable=self.case_sensitive_var,
            style="Search.TCheckbutton"
        )
        case_check.pack(side=tk.LEFT, padx=10)
        
        # Search count display
        self.search_count_var = tk.StringVar(value="")
        search_count_label = ttk.Label(
            search_frame, 
            textvariable=self.search_count_var, 
            style="Search.TLabel"
        )
        search_count_label.pack(side=tk.RIGHT, padx=10)
        
        # Navigation buttons
        self.search_prev_button = ttk.Button(
            search_frame, 
            text="◄ Prev", 
            style="Custom.TButton",
            command=self.prev_match
        )
        self.search_prev_button.pack(side=tk.RIGHT, padx=2)
        
        self.search_next_button = ttk.Button(
            search_frame, 
            text="Next ►", 
            style="Custom.TButton",
            command=self.next_match
        )
        self.search_next_button.pack(side=tk.RIGHT, padx=2)
        
        self.search_button = ttk.Button(
            search_frame, 
            text="Scan", 
            style="Custom.TButton",
            command=self.perform_search
        )
        self.search_button.pack(side=tk.RIGHT, padx=5)
    
    def create_content_panes(self):
        """Create the split pane for list and content"""
        # Create a paned window
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for summary list
        self.list_frame = ttk.Frame(self.paned_window, style="Custom.TFrame")
        
        # Create a treeview for the list of summaries
        list_columns = ('index', 'folder')
        self.summary_tree = ttk.Treeview(self.list_frame, columns=list_columns, show='headings')
        self.summary_tree.heading('index', text='#')
        self.summary_tree.column('index', width=40, anchor=tk.CENTER)
        self.summary_tree.heading('folder', text='Source Location')
        
        # Add scrollbar for the treeview
        list_scrollbar = ttk.Scrollbar(
            self.list_frame, 
            orient=tk.VERTICAL, 
            command=self.summary_tree.yview,
            style="Custom.Vertical.TScrollbar"
        )
        self.summary_tree.configure(yscrollcommand=list_scrollbar.set)
        
        # Pack the treeview and scrollbar
        self.summary_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right panel for summary content
        self.content_frame = ttk.Frame(self.paned_window, style="Custom.TFrame")
        
        # Add the panels to the paned window
        self.paned_window.add(self.list_frame, weight=1)
        self.paned_window.add(self.content_frame, weight=3)
        
        # Create a text widget with scrollbars for the content
        text_container = ttk.Frame(self.content_frame, style="Custom.TFrame")
        text_container.pack(fill=tk.BOTH, expand=True)
        
        # Vertical scrollbar
        v_scrollbar = ttk.Scrollbar(
            text_container, 
            orient=tk.VERTICAL, 
            style="Custom.Vertical.TScrollbar"
        )
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(
            text_container, 
            orient=tk.HORIZONTAL, 
            style="Custom.Horizontal.TScrollbar"
        )
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Text widget with custom cursor
        self.text_widget = tk.Text(
            text_container,
            wrap="none",  # Allow horizontal scrolling
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set,
            bg=self.colors['bg_light'],
            fg=self.colors['text_light'],
            font=("Courier New", 12),
            padx=5,
            pady=5,
            insertbackground=self.colors['accent'],  # Cursor color
            selectbackground=self.colors['accent'],
            selectforeground=self.colors['text_dark'],
            cursor="arrow"
        )
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbars
        v_scrollbar.config(command=self.text_widget.yview)
        h_scrollbar.config(command=self.text_widget.xview)
        
        # Create text tags for formatting
        self.text_widget.tag_configure("title", font=("Courier New", 14, "bold"), foreground=self.colors['accent'])
        self.text_widget.tag_configure("location", font=("Courier New", 12, "italic"), foreground=self.colors['text_light'])
        self.text_widget.tag_configure("hyperlink", foreground=self.colors['accent'], underline=1)
        self.text_widget.tag_configure("search_match", background=self.colors['highlight'], foreground=self.colors['text_dark'])
        self.text_widget.tag_configure("current_match", background=self.colors['accent'], foreground=self.colors['text_dark'])
    
    def create_status_bar(self):
        """Create the status bar"""
        status_frame = ttk.Frame(self.main_frame, style="Search.TFrame")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar(value="[ SYSTEM READY ]")
        status_label = ttk.Label(
            status_frame, 
            textvariable=self.status_var, 
            style="Search.TLabel"
        )
        status_label.pack(side=tk.LEFT, padx=10, pady=3)

        # Add font size controls
        font_frame = ttk.Frame(status_frame, style="Search.TFrame")
        font_frame.pack(side=tk.RIGHT, padx=10)
        
        font_label = ttk.Label(font_frame, text="Font Size:", style="Search.TLabel")
        font_label.pack(side=tk.LEFT, padx=(0, 5))
        
        font_decrease = ttk.Button(font_frame, text="-", width=2, style="Custom.TButton", command=self.decrease_font)
        font_decrease.pack(side=tk.LEFT, padx=2)
        
        font_increase = ttk.Button(font_frame, text="+", width=2, style="Custom.TButton", command=self.increase_font)
        font_increase.pack(side=tk.LEFT, padx=2)
        
        # Add word wrap toggle
        self.wrap_var = tk.BooleanVar(value=False)
        wrap_check = ttk.Checkbutton(
            status_frame, 
            text="Word Wrap", 
            variable=self.wrap_var,
            style="Search.TCheckbutton",
            command=self.toggle_word_wrap
        )
        wrap_check.pack(side=tk.RIGHT, padx=10)
    
    def bind_shortcuts(self):
        """Bind keyboard shortcuts"""
        self.search_entry.bind("<Key>", self.search_key_handler)
        self.summary_tree.bind("<<TreeviewSelect>>", self.display_summary)
        self.summary_tree.bind("<Button-3>", self.context_menu_for_tree)
        self.root.bind("<Control-f>", lambda e: self.search_entry.focus())
        self.root.bind("<F3>", lambda e: self.next_match())
        self.root.bind("<Shift-F3>", lambda e: self.prev_match())
    
    def load_summaries(self):
        """Load summary data in a background thread"""
        self.status_var.set("[ SCANNING FILE SYSTEM ]")
        
        def background_load():
            summaries = gather_summary_data(BASE_DIRECTORY)
            
            # Update the UI in the main thread
            self.root.after(0, lambda: self.populate_summary_list(summaries))
        
        # Start background thread
        thread = threading.Thread(target=background_load)
        thread.daemon = True
        thread.start()
    
    def populate_summary_list(self, summaries):
        """Populate the summary list with data"""
        if not summaries:
            self.text_widget.config(state=tk.NORMAL)
            self.text_widget.insert(tk.END, "[ NO DATA FOUND ]\n")
            self.text_widget.config(state=tk.DISABLED)
            self.status_var.set("[ NO DATA FOUND IN TARGET DIRECTORY ]")
            return
        
        # Clear existing items
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)
        
        # Add new items
        for index, (subfolder, content, filepath) in enumerate(summaries, start=1):
            item_id = self.summary_tree.insert(
                '', 'end', 
                values=(index, subfolder)
            )
            # Store the summary data for reference
            self.summaries_data[item_id] = (subfolder, content, filepath)
        
        # Select first item
        if self.summary_tree.get_children():
            first_item = self.summary_tree.get_children()[0]
            self.summary_tree.selection_set(first_item)
            self.summary_tree.focus(first_item)
            self.display_summary()
        
        self.status_var.set(f"[ {len(summaries)} DATA ARCHIVES IDENTIFIED ]")
    
    def find_urls_in_text(self, text):
        """Find all URLs in the given text and return a list of (start, end, url) tuples"""
        urls = []
        for match in re.finditer(URL_PATTERN, text):
            start, end = match.span()
            url = match.group(0)
            urls.append((start, end, url))
        return urls
    
    def open_url(self, url):
        """Open a URL in the default web browser"""
        try:
            webbrowser.open(url)
            self.status_var.set(f"[ ACCESSING EXTERNAL RESOURCE: {url} ]")
        except Exception as e:
            self.status_var.set(f"[ CONNECTION ERROR: {str(e)} ]")
    
    def display_summary(self, event=None):
        """Display the selected summary in the text widget"""
        selected_items = self.summary_tree.selection()
        if not selected_items:
            return
        
        item_id = selected_items[0]
        item_data = self.summaries_data.get(item_id)
        if not item_data:
            return
        
        folder_path, content, full_path = item_data
        
        # Clear the text widget
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete(1.0, tk.END)
        
        # Insert title
        self.text_widget.insert(tk.END, f"[ DATA FROM: {os.path.basename(folder_path)} ]\n", "title")
        
        # Insert location with a button to open it
        location_text = f"Location: {folder_path}"
        self.text_widget.insert(tk.END, location_text + "\n\n", "location")
        
        # Make the location clickable
        location_start = "1.0 + 10c"
        location_end = f"1.0 + {10 + len(folder_path)}c"
        location_tag = "location_link"
        self.text_widget.tag_configure(location_tag, underline=1)
        self.text_widget.tag_bind(location_tag, "<Button-1>", lambda e: open_file_location(full_path))
        self.text_widget.tag_add(location_tag, location_start, location_end)
        
        # Insert content and process URLs
        start_pos = self.text_widget.index(tk.END)
        self.text_widget.insert(tk.END, content)
        
        # Find and mark URLs
        all_content = self.text_widget.get(start_pos, tk.END)
        url_positions = self.find_urls_in_text(all_content)
        
        for start, end, url in url_positions:
            # Calculate positions in the text widget
            start_index = f"{start_pos} + {start} chars"
            end_index = f"{start_pos} + {end} chars"
            
            # Apply hyperlink tag
            self.text_widget.tag_add("hyperlink", start_index, end_index)
            
            # Bind click event
            tag_name = f"url_{start}_{end}"
            self.text_widget.tag_configure(tag_name)
            self.text_widget.tag_bind(tag_name, "<Button-1>", lambda e, u=url: self.open_url(u))
            self.text_widget.tag_add(tag_name, start_index, end_index)
        
        self.text_widget.config(state=tk.DISABLED)
        
        # Update status
        self.status_var.set(f"[ DISPLAYING: {folder_path} ]")
        
        # Clear search highlights when changing summaries
        self.search_results.clear()
        self.current_match_index.set(-1)
        self.search_count_var.set("")
    
    def perform_search(self):
        """Search for text in the current summary"""
        search_text = self.search_var.get()
        if not search_text:
            self.status_var.set("[ ENTER SEARCH TERM ]")
            return
        
        # Clear previous search highlighting
        self.text_widget.tag_remove("search_match", "1.0", tk.END)
        self.text_widget.tag_remove("current_match", "1.0", tk.END)
        
        # Find all matches
        self.search_results.clear()
        self.current_match_index.set(-1)
        
        start_idx = "1.0"
        count_var = tk.StringVar()
        
        while True:
            # Find the next match
            match_pos = self.text_widget.search(
                search_text, start_idx, tk.END, 
                count=count_var,
                nocase=not self.case_sensitive_var.get()
            )
            
            if not match_pos:
                break
                
            # Calculate end index
            match_length = int(count_var.get())
            match_end = f"{match_pos}+{match_length}c"
            
            # Add to results
            self.search_results.append((match_pos, match_end))
            
            # Highlight match
            self.text_widget.tag_add("search_match", match_pos, match_end)
            
            # Move to next starting position
            start_idx = match_end
        
        # Update search count
        num_results = len(self.search_results)
        if num_results > 0:
            self.search_count_var.set(f"[ {num_results} MATCHES ]")
            self.go_to_match(0)  # Go to first match
        else:
            self.search_count_var.set("[ NO MATCHES ]")
            self.status_var.set(f"[ NO RESULTS FOR: '{search_text}' ]")
    
    def go_to_match(self, index):
        """Go to a specific search match by index"""
        if not self.search_results:
            return
        
        # Make sure index is in bounds
        num_results = len(self.search_results)
        if index < 0:
            index = num_results - 1
        elif index >= num_results:
            index = 0
        
        # Remove current match highlight
        self.text_widget.tag_remove("current_match", "1.0", tk.END)
        
        # Highlight current match
        match_pos, match_end = self.search_results[index]
        self.text_widget.tag_add("current_match", match_pos, match_end)
        
        # Ensure visibility
        self.text_widget.see(match_pos)
        
        # Update current match index
        self.current_match_index.set(index)
        
        # Update status
        self.status_var.set(f"[ MATCH {index + 1} OF {num_results} ]")
        self.search_count_var.set(f"[ {index + 1} / {num_results} ]")
    
    def next_match(self):
        """Go to the next search match"""
        current = self.current_match_index.get()
        if current >= 0:
            self.go_to_match(current + 1)
    
    def prev_match(self):
        """Go to the previous search match"""
        current = self.current_match_index.get()
        if current >= 0:
            self.go_to_match(current - 1)
    
    def search_key_handler(self, event):
        """Handle key events in the search entry"""
        if event.keysym == "Return":
            self.perform_search()
            return "break"
        elif event.keysym == "Escape":
            # Clear search
            self.search_var.set("")
            self.text_widget.tag_remove("search_match", "1.0", tk.END)
            self.text_widget.tag_remove("current_match", "1.0", tk.END)
            self.search_results.clear()
            self.current_match_index.set(-1)
            self.search_count_var.set("")
            return "break"
    
    def context_menu_for_tree(self, event):
        """Show context menu for treeview items"""
        # Get the item under the mouse
        item = self.summary_tree.identify('item', event.x, event.y)
        if item:
            # Select the item
            self.summary_tree.selection_set(item)
            
            # Create a context menu
            context_menu = tk.Menu(self.root, tearoff=0, bg=self.colors['bg_light'], fg=self.colors['text_light'])
            
            # Get the item data
            item_data = self.summaries_data.get(item)
            if item_data:
                subfolder, content, filepath = item_data
                
                # Add menu items
                context_menu.add_command(
                    label="Open File Location", 
                    command=lambda: open_file_location(filepath)
                )
                
                # Show the menu
                context_menu.post(event.x_root, event.y_root)
    
    def increase_font(self):
        """Increase the font size in the text widget"""
        current_font = self.text_widget['font']
        if isinstance(current_font, str):
            # Parse font string
            parts = current_font.split()
            family = parts[0]
            size = int(parts[1])
        else:
            # Font is a tuple
            family, size = current_font
        
        new_size = min(size + 2, 36)  # Maximum size 36
        self.text_widget.configure(font=(family, new_size))
    
    def decrease_font(self):
        """Decrease the font size in the text widget"""
        current_font = self.text_widget['font']
        if isinstance(current_font, str):
            # Parse font string
            parts = current_font.split()
            family = parts[0]
            size = int(parts[1])
        else:
            # Font is a tuple
            family, size = current_font
        
        new_size = max(size - 2, 8)  # Minimum size 8
        self.text_widget.configure(font=(family, new_size))
    
    def toggle_word_wrap(self):
        """Toggle word wrap in the text widget"""
        if self.wrap_var.get():
            self.text_widget.configure(wrap="word")
        else:
            self.text_widget.configure(wrap="none")


def create_ui():
    """Create the main application UI"""
    root = tk.Tk()
    app = SummaryApp(root)
    root.mainloop()


if __name__ == "__main__":
    create_ui()
