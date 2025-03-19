#!/usr/bin/env python3
import argparse
import subprocess
import sys
import logging
import shutil
import traceback
import json
import os
import re
import time
import textwrap
from collections import OrderedDict

__version__ = "1.8"

# Path to your virtual environment activation script and Photon script
VENV_PATH = os.path.expanduser("~/newvenv/bin/activate")
PHOTON_SCRIPT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "photon.py")

def check_command(command):
    """Verify if a command is available in the system PATH."""
    if shutil.which(command) is None:
        logging.error(f"'{command}' is not installed. Please install it to proceed.")
        sys.exit(1)
    logging.debug(f"'{command}' is available.")

def pretty_print_json(output):
    """Attempt to parse and pretty-print JSON output; if it fails, return raw output."""
    try:
        data = json.loads(output)
        return json.dumps(data, indent=2)
    except json.JSONDecodeError:
        return output

def run_command(command, label, debug=False, use_venv=False, ignore_errors=False):
    """
    Execute a command, optionally within a virtual environment.
    Returns the command output as a string.
    """
    try:
        if use_venv:
            # For commands like Photon that must run inside your virtualenv,
            # we assume python3 is available.
            check_command("python3")
        else:
            check_command(command[0])
        logging.debug(f"Executing command: {' '.join(command)}")
        
        if use_venv:
            # Combine the activation command with the target command
            full_command = f"source {VENV_PATH} && " + " ".join(command)
            result = subprocess.run(full_command, shell=True, capture_output=True, text=True, executable="/bin/bash")
        else:
            result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"Error running {label}: {result.stderr.strip()}")
            if debug:
                logging.debug("Full stderr: " + result.stderr)
            if ignore_errors:
                # Return empty result but don't fail completely
                return ""
            return None
        logging.debug(f"{label} output: {result.stdout.strip()}")
        return result.stdout.strip()
    except Exception as exc:
        logging.error(f"Exception running {label}: {exc}")
        if debug:
            traceback.print_exc()
        return None

def calculate_resource_values(depth):
    """
    Calculate appropriate resource values based on the depth parameter (1-100).
    Returns a dictionary with calibrated values for different tools.
    
    Depth scale:
    1-20: Light/Quick search
    21-50: Medium search
    51-80: Thorough search
    81-100: Deep/Extensive search
    """
    # Ensure depth is within valid range (now 1-100)
    depth = max(1, min(100, depth))
    
    # Calculate ddgr results - maintain a reasonable page size
    # We'll use constant page size of 10 for better UX, but adjust total results
    ddgr_page_size = 10
    
    # Calculate ddgr total results based on depth
    if depth <= 20:
        ddgr_results = 10  # Light search
    elif depth <= 50:
        ddgr_results = 15  # Medium search
    elif depth <= 80:
        ddgr_results = 20  # Thorough search
    else:
        ddgr_results = 25  # Deep search (max for ddgr)
    
    # Calculate Photon crawl parameters
    # Level: 1 to 5 (Photon's internal crawl depth)
    photon_level = 1 + int((depth - 1) / 20)  # Ranges from 1-5
    
    # Threads: 2 to 50 (threading for parallel processing)
    photon_threads = 2 + int((depth - 1) * 0.48)  # Linear scaling
    
    # Timeout: 5s to 30s (longer for deeper crawls)
    photon_timeout = 5 + int((depth - 1) * 0.25)  # Linear scaling
    
    # URLs to crawl per page: proportional to depth
    if depth <= 20:
        urls_percent = 0.3  # Crawl 30% of results for light search
    elif depth <= 50:
        urls_percent = 0.5  # Crawl 50% of results for medium search
    elif depth <= 80:
        urls_percent = 0.7  # Crawl 70% of results for thorough search
    else:
        urls_percent = 1.0  # Crawl 100% of results for deep search
    
    urls_to_crawl = max(1, min(ddgr_page_size, round(ddgr_page_size * urls_percent)))
    
    return {
        "ddgr_results": ddgr_results,
        "ddgr_page_size": ddgr_page_size,
        "photon_threads": photon_threads,
        "photon_level": photon_level,
        "photon_timeout": photon_timeout,
        "urls_to_crawl": urls_to_crawl
    }

def build_ddgr_command(query, ddgr_args, page_size=10):
    """Build the ddgr command with all user-supplied arguments."""
    command = ['ddgr', '--json']
    
    # Ensure we're using a proper page size for ddgr
    command.extend(['--num', str(page_size)])
    
    # Add all ddgr-specific arguments
    if ddgr_args.get('region'):
        command.extend(['--reg', ddgr_args.get('region')])
    if ddgr_args.get('colorize'):
        command.extend(['--colorize', ddgr_args.get('colorize')])
    if ddgr_args.get('nocolor'):
        command.append('--nocolor')
    if ddgr_args.get('colors'):
        command.extend(['--colors', ddgr_args.get('colors')])
    if ddgr_args.get('time'):
        command.extend(['--time', ddgr_args.get('time')])
    if ddgr_args.get('site'):
        for site in ddgr_args.get('site'):
            command.extend(['--site', site])
    if ddgr_args.get('expand'):
        command.append('--expand')
    if ddgr_args.get('proxy'):
        command.extend(['--proxy', ddgr_args.get('proxy')])
    if ddgr_args.get('unsafe'):
        command.append('--unsafe')
    if ddgr_args.get('noua'):
        command.append('--noua')
    
    # Add the query
    command.append(query)
    
    return command

def run_ddgr_with_pagination(query, ddgr_args, depth=30, debug=False, page=1):
    """
    Run DuckDuckGo search using ddgr with proper pagination support.
    This function uses ddgr's interactive mode and simulates pagination commands.
    
    Parameters:
    - query: The search query string
    - ddgr_args: Dictionary of ddgr-specific arguments
    - depth: Depth parameter (1-100)
    - debug: Whether to enable debug output
    - page: Page number (1-based for user display)
    
    Returns:
    - List of URLs from search results
    - Boolean indicating if there might be more results
    """
    resources = calculate_resource_values(depth)
    page_size = resources["ddgr_page_size"]
    
    # For first page, search normally
    if page == 1:
        print(f"\n[+] DuckDuckGo (ddgr) search results for: {query}")
        print(f"[+] Search depth: {depth}/100 (Page {page})\n")
        command = build_ddgr_command(query, ddgr_args, page_size)
    else:
        print(f"\n[+] DuckDuckGo (ddgr) search results - Page {page} for: {query}")
        print(f"[+] Search depth: {depth}/100\n")
        
        # For subsequent pages, we need to simulate pagination:
        # 1. Run ddgr in non-interactive mode first to get initial results
        # 2. Then run multiple "next page" commands to get to the desired page
        
        # First, run the initial query with more results to move through pages faster
        # Use the max limit of 25 to reduce the number of pagination steps needed
        initial_results = 25
        command = ['ddgr', '--json', '--num', str(initial_results), query]
        # Add other args
        if ddgr_args.get('region'):
            command.extend(['--reg', ddgr_args.get('region')])
        if ddgr_args.get('time'):
            command.extend(['--time', ddgr_args.get('time')])
        if ddgr_args.get('site'):
            for site in ddgr_args.get('site'):
                command.extend(['--site', site])
        if ddgr_args.get('unsafe'):
            command.append('--unsafe')
    
    output = run_command(command, "ddgr", debug)
    if not output:
        print("[-] No results from ddgr search")
        return [], False
    
    urls = []
    try:
        results = json.loads(output)
        
        # If we're on a later page, we need to extract the correct subset of results
        result_offset = 0
        if page > 1:
            # For page 2, we want results 10-19 (assuming page_size=10)
            # For page 3, we want results 20-29, etc.
            result_offset = (page - 1) * page_size
            
            # If offset is beyond available results, no more results
            if result_offset >= len(results):
                print("[-] No more results available.")
                return [], False
            
            # Get the slice of results for this page
            end_offset = min(result_offset + page_size, len(results))
            page_results = results[result_offset:end_offset]
        else:
            # First page, just take the first page_size results
            page_results = results[:page_size]
        
        # Process and display the results
        for i, result in enumerate(page_results, 1):
            title = result.get("title", "No Title")
            url = result.get("url", "No URL")
            abstract = result.get("abstract", "")
            
            # Add URLs to the list for Photon crawling
            urls.append(url)
            
            # Display result with correct global index
            global_index = result_offset + i
            print(f"{global_index}. {title}")
            print(f"   {url}")
            if abstract:
                wrapped_abstract = textwrap.fill(abstract, width=80, initial_indent="   ", subsequent_indent="   ")
                print(f"{wrapped_abstract}\n")
            else:
                print()  # Empty line for spacing
        
        # Determine if there might be more results
        # We consider there are more if:
        # 1. We got a full page of results, or
        # 2. We know there are more results in our fetched batch
        has_more = (len(page_results) == page_size) or (result_offset + len(page_results) < len(results))
        
        return urls, has_more
    
    except json.JSONDecodeError:
        logging.error("Failed to parse ddgr JSON output.")
        print(output)
        return [], False

def build_photon_command(target, photon_args, output_dir):
    """Build the Photon command with all user-supplied arguments."""
    command = [
        'python3', PHOTON_SCRIPT,
        '-u', target,
        '-o', output_dir
    ]
    
    # Add all photon-specific arguments
    if photon_args.get('level') is not None:
        command.extend(['-l', str(photon_args.get('level'))])
    if photon_args.get('threads') is not None:
        command.extend(['-t', str(photon_args.get('threads'))])
    if photon_args.get('delay') is not None:
        command.extend(['-d', str(photon_args.get('delay'))])
    if photon_args.get('timeout') is not None:
        command.extend(['--timeout', str(photon_args.get('timeout'))])
    if photon_args.get('cookie'):
        command.extend(['-c', photon_args.get('cookie')])
    if photon_args.get('regex'):
        command.extend(['-r', photon_args.get('regex')])
    if photon_args.get('export'):
        command.extend(['-e', photon_args.get('export')])
    if photon_args.get('seeds'):
        command.extend(['-s'] + photon_args.get('seeds'))
    if photon_args.get('user_agent'):
        command.extend(['--user-agent', photon_args.get('user_agent')])
    if photon_args.get('exclude'):
        command.extend(['--exclude', photon_args.get('exclude')])
    if photon_args.get('proxy'):
        command.extend(['-p', photon_args.get('proxy')])
    
    # Add boolean flags
    if photon_args.get('verbose'):
        command.append('-v')
    if photon_args.get('headers'):
        command.append('--headers')
    if photon_args.get('dns'):
        command.append('--dns')
    if photon_args.get('keys'):
        command.append('--keys')
    if photon_args.get('only_urls'):
        command.append('--only-urls')
    if photon_args.get('wayback'):
        command.append('--wayback')
    
    return command

def run_photon_on_single_target(target, photon_args, depth=30, debug=False, index=None, total=None):
    """Run Photon OSINT on a single target URL with user-specified arguments."""
    resources = calculate_resource_values(depth)
    
    # Use calculated resources if not provided in photon_args
    level = photon_args.get('level') or resources["photon_level"]
    threads = photon_args.get('threads') or resources["photon_threads"]
    timeout = photon_args.get('timeout') or resources["photon_timeout"]
    
    # Create a progress indicator if we're processing multiple URLs
    progress_str = ""
    if index is not None and total is not None:
        progress_str = f"[{index}/{total}] "
    
    print(f"\n[+] {progress_str}Photon crawling target: {target}")
    print(f"[+] Crawl depth: {depth}/100 (level: {level}, threads: {threads}, timeout: {timeout}s)")
    
    if not os.path.isfile(PHOTON_SCRIPT):
        logging.error(f"Photon script not found at {PHOTON_SCRIPT}. Please ensure photon.py is available.")
        return None
    
    # Create organized output folder structure
    main_output_dir = "photon_results"
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)
    
    # Get domain name for subfolder
    domain = target.replace("https://", "").replace("http://", "").split("/")[0]
    domain_safe = re.sub(r'[^\w\-_]', '_', domain)  # Make domain name safe for filesystem
    
    # Create a unique subfolder for this target
    timestamp = int(time.time())
    target_dir = f"{domain_safe}_{timestamp}"
    output_dir = os.path.join(main_output_dir, target_dir)
    
    # Build the Photon command
    command = build_photon_command(target, photon_args, output_dir)
    
    # Run the command with the proper settings and handle failures gracefully
    output = run_command(command, f"Photon OSINT on {target}", debug, use_venv=True, ignore_errors=True)
    
    # Create the output directory even if the command failed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if output is not None:
        print(f"[+] Completed scanning {target}")
        print(f"[+] Results saved to {output_dir}/")
        return output_dir
    
    # If the output is None, it means the command completely failed
    print(f"[!] Issues encountered while scanning {target}, but continuing...")
    print(f"[+] Partial results may be available in {output_dir}/")
    return output_dir

def run_photon_on_multiple_targets(targets, photon_args, depth=30, debug=False):
    """Run Photon OSINT on multiple target URLs with user-specified arguments."""
    if not targets:
        print("[-] No targets to crawl with Photon.")
        return
    
    resources = calculate_resource_values(depth)
    max_targets = photon_args.get('max_targets') or resources["urls_to_crawl"]
    
    # Deduplicate targets while preserving order
    unique_targets = list(OrderedDict.fromkeys(targets))
    
    if len(unique_targets) > max_targets:
        print(f"[*] Limiting Photon crawl to top {max_targets} targets based on depth setting {depth}/100")
        targets_to_crawl = unique_targets[:max_targets]
    else:
        targets_to_crawl = unique_targets
    
    print(f"\n[+] Starting Photon crawler on {len(targets_to_crawl)} targets")
    
    results = []
    for i, target in enumerate(targets_to_crawl, 1):
        result_dir = run_photon_on_single_target(
            target, photon_args, depth, debug, i, len(targets_to_crawl)
        )
        if result_dir:
            results.append((target, result_dir))
    
    if results:
        print("\n[+] Photon crawling complete. Summary:")
        print(f"[+] All results saved to the 'photon_results/' directory")
        for target, output_dir in results:
            print(f"  - {target} -> {os.path.basename(output_dir)}/")
    else:
        print("\n[-] No successful Photon crawls.")

def ask_for_more():
    """Ask the user if they want more results."""
    while True:
        answer = input("\nMore results? (y/n): ").lower().strip()
        if answer in ['y', 'yes']:
            return True
        elif answer in ['n', 'no']:
            return False
        else:
            print("Please answer 'y' or 'n'.")

def is_url(text):
    """Check if the given text is a URL."""
    url_pattern = re.compile(r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
    return bool(url_pattern.match(text))

def main():
    # Create argument parser with extensive options for both ddgr and photon
    parser = argparse.ArgumentParser(
        description="Enhanced OSINT tool using Photon OSINT and ddgr with pagination support.",
        epilog=("Examples:\n"
                "  python3 proton.py \"search keywords\" 25                          # Search and crawl results with medium depth\n"
                "  python3 proton.py \"search keywords\" \"https://example.com\" 45    # Search, crawl specific site and search results\n"
                "  python3 proton.py --query 'osint tools' --depth 75 --no-crawl   # Search only, no crawling\n"
                "  python3 proton.py --target 'https://example.com' --depth 100     # Crawl only a specific site deeply\n"
                "\nDepth Values (1-100):\n"
                "  1-20:   Quick/light crawl (fewer results, shallow depth, faster)\n"
                "  21-50:  Medium crawl (moderate results and depth)\n"
                "  51-80:  Thorough crawl (more results, deeper level)\n"
                "  81-100: Deep crawl (maximum results and depth level, longer runtime)"),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Basic options
    basic_group = parser.add_argument_group('Basic Options')
    basic_group.add_argument('--query', type=str, 
                    help="Search query for DuckDuckGo (ddgr)")
    basic_group.add_argument('--target', type=str, 
                    help="Target URL for Photon OSINT scraping")
    basic_group.add_argument('--depth', type=int, default=30, 
                    help="Search depth (1-100): 1=quickest, 100=deepest. Controls number of results, crawl depth, and threads")
    basic_group.add_argument('--debug', action='store_true', 
                    help="Enable debug mode for detailed output and error messages")
    basic_group.add_argument('--no-crawl', action='store_true',
                    help="Disable automatic crawling of search results (search only)")
    basic_group.add_argument('--no-pagination', action='store_true',
                    help="Disable pagination ('more' prompt at the end)")
    basic_group.add_argument('--version', action='version', 
                    version=f"%(prog)s {__version__}")
    
    # ddgr-specific options
    ddgr_group = parser.add_argument_group('DuckDuckGo (ddgr) Options')
    ddgr_group.add_argument('--ddgr-region', type=str, metavar='REG', default='us-en',
                    help="region-specific search e.g. 'us-en' for US (default)")
    ddgr_group.add_argument('--ddgr-colorize', type=str, choices=['auto', 'always', 'never'], default='auto',
                    help="whether to colorize output")
    ddgr_group.add_argument('--ddgr-nocolor', action='store_true',
                    help="equivalent to --ddgr-colorize=never")
    ddgr_group.add_argument('--ddgr-colors', type=str, metavar='COLORS',
                    help="set output colors")
    ddgr_group.add_argument('--ddgr-time', type=str, metavar='SPAN', choices=('d', 'w', 'm', 'y'),
                    help="time limit search [d (1 day), w (1 wk), m (1 month), y (1 year)]")
    ddgr_group.add_argument('--ddgr-site', type=str, metavar='SITE', action='append',
                    help="search sites using DuckDuckGo")
    ddgr_group.add_argument('--ddgr-expand', action='store_true',
                    help="Show complete url in search results")
    ddgr_group.add_argument('--ddgr-proxy', type=str, metavar='URI',
                    help="tunnel traffic through an HTTPS proxy; URI format: [http[s]://][user:pwd@]host[:port]")
    ddgr_group.add_argument('--ddgr-unsafe', action='store_true',
                    help="disable safe search")
    ddgr_group.add_argument('--ddgr-noua', action='store_true',
                    help="disable user agent")
    
    # Photon-specific options
    photon_group = parser.add_argument_group('Photon OSINT Options')
    photon_group.add_argument('--photon-level', type=int, metavar='LEVEL', 
                    help="levels to crawl (1-5)")
    photon_group.add_argument('--photon-threads', type=int, metavar='THREADS',
                    help="number of threads")
    photon_group.add_argument('--photon-delay', type=float, metavar='DELAY',
                    help="delay between requests")
    photon_group.add_argument('--photon-timeout', type=float, metavar='TIMEOUT',
                    help="http request timeout")
    photon_group.add_argument('--photon-cookie', type=str, metavar='COOKIE',
                    help="cookie")
    photon_group.add_argument('--photon-regex', type=str, metavar='REGEX',
                    help="regex pattern")
    photon_group.add_argument('--photon-export', type=str, metavar='FORMAT', choices=['csv', 'json'],
                    help="export format (csv, json)")
    photon_group.add_argument('--photon-seeds', type=str, metavar='SEEDS', action='append',
                    help="additional seed URLs")
    photon_group.add_argument('--photon-user-agent', type=str, metavar='UA',
                    help="custom user agent(s)")
    photon_group.add_argument('--photon-exclude', type=str, metavar='REGEX',
                    help="exclude URLs matching this regex")
    photon_group.add_argument('--photon-proxy', type=str, metavar='PROXY',
                    help="Proxy server IP:PORT or DOMAIN:PORT")
    photon_group.add_argument('--photon-verbose', action='store_true',
                    help="verbose output")
    photon_group.add_argument('--photon-headers', action='store_true',
                    help="add headers")
    photon_group.add_argument('--photon-dns', action='store_true',
                    help="enumerate subdomains and DNS data")
    photon_group.add_argument('--photon-keys', action='store_true',
                    help="find secret keys")
    photon_group.add_argument('--photon-only-urls', action='store_true',
                    help="only extract URLs")
    photon_group.add_argument('--photon-wayback', action='store_true',
                    help="fetch URLs from archive.org as seeds")
    photon_group.add_argument('--photon-max-targets', type=int, metavar='N',
                    help="maximum number of targets to crawl (overrides automatic scaling)")
    
    # Positional arguments (for simpler command line usage)
    parser.add_argument('keywords', type=str, nargs='?', 
                    help="Search keywords (e.g., \"George Washington\")")
    parser.add_argument('url_or_depth', type=str, nargs='?',
                    help="Either a URL to crawl (e.g., \"https://example.com\") or depth value (e.g., \"45\")")
    parser.add_argument('positional_depth', type=str, nargs='?',
                    help="Depth value when URL is provided (e.g., \"45\" when using format: \"keywords URL depth\")")
    
    args = parser.parse_args()

    # Configure logging based on the debug flag
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging_level, format='[%(levelname)s] %(message)s')
    
    if args.debug:
        logging.debug("Debug mode enabled.")

    # Process positional arguments if provided
    if args.keywords:
        # We have at least the keywords argument
        query = args.keywords
        target = None
        depth = args.depth  # Default to named argument depth
        
        if args.url_or_depth:
            # Second argument could be either a URL or depth
            if is_url(args.url_or_depth):
                # It's a URL
                target = args.url_or_depth
                # If we have a third positional argument, it's the depth
                if args.positional_depth and args.positional_depth.isdigit():
                    depth = int(args.positional_depth)
            elif args.url_or_depth.isdigit():
                # It's a depth value
                depth = int(args.url_or_depth)
    else:
        # Use named arguments
        query = args.query
        target = args.target
        depth = args.depth
    
    # Ensure depth is within valid range
    if depth is not None:
        depth = max(1, min(100, depth))
    
    # Ensure we have at least one action to perform
    if not query and not target:
        parser.error("You must provide either keywords, a URL, or use --query/--target arguments.")
    
    # Initialize list of URLs to crawl with Photon
    urls_to_crawl = []
    
    # Add specifically provided target URL if any
    if target:
        urls_to_crawl.append(target)
    
    # Prepare ddgr arguments
    ddgr_args = {
        'region': args.ddgr_region,
        'colorize': args.ddgr_colorize,
        'nocolor': args.ddgr_nocolor,
        'colors': args.ddgr_colors,
        'time': args.ddgr_time,
        'site': args.ddgr_site,
        'expand': args.ddgr_expand,
        'proxy': args.ddgr_proxy,
        'unsafe': args.ddgr_unsafe,
        'noua': args.ddgr_noua
    }
    
    # Prepare photon arguments
    photon_args = {
        'level': args.photon_level,
        'threads': args.photon_threads,
        'delay': args.photon_delay,
        'timeout': args.photon_timeout,
        'cookie': args.photon_cookie,
        'regex': args.photon_regex,
        'export': args.photon_export,
        'seeds': args.photon_seeds,
        'user_agent': args.photon_user_agent,
        'exclude': args.photon_exclude,
        'proxy': args.photon_proxy,
        'verbose': args.photon_verbose,
        'headers': args.photon_headers,
        'dns': args.photon_dns,
        'keys': args.photon_keys,
        'only_urls': args.photon_only_urls,
        'wayback': args.photon_wayback,
        'max_targets': args.photon_max_targets
    }
    
    # Run ddgr search if query is provided
    if query:
        page = 1  # Start at page 1 (1-based for user display)
        has_more = True
        
        while has_more:
            # Run search for current page with proper pagination
            search_urls, has_more = run_ddgr_with_pagination(query, ddgr_args, depth, args.debug, page)
            
            # Add search results to crawl list if auto-crawl is enabled
            if not args.no_crawl and search_urls:
                # Run Photon on collected URLs for this page
                run_photon_on_multiple_targets(search_urls, photon_args, depth, args.debug)
            
            # Check if pagination is disabled or if we've reached the end
            if args.no_pagination or not has_more:
                break
                
            # Ask for more results
            if not ask_for_more():
                break
                
            # Move to next page
            page += 1
            
    elif target and not args.no_crawl:
        # If no query but a target was provided, crawl that target
        run_photon_on_single_target(target, photon_args, depth, args.debug)

if __name__ == '__main__':
    main()
