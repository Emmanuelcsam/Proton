#!/usr/bin/env python3
"""
Deep Research Assistant

A comprehensive research tool that combines:
- Advanced search capabilities (DuckDuckGo)
- Deep web crawling (Photon)
- Intelligent content extraction
- Multi-model summarization (TextRank, Ollama, OpenAI)
- Source hierarchy tracking
- Detailed research reports with hierarchical organization

Usage:
  python deep_research_assistant.py "your research topic"
  python deep_research_assistant.py --depth 4 --model gemma3:12b "quantum computing"
"""

import os
import sys
import re
import json
import time
import math
import random
import argparse
import logging
import subprocess
import requests
import hashlib
import urllib.parse
import urllib.request
from datetime import datetime
from urllib.parse import urlparse, urljoin
from collections import Counter, defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import string
import traceback
import shutil
import textwrap

__version__ = "1.0.0"

# Try to import dependencies, provide helpful error message if not available
try:
    from bs4 import BeautifulSoup, Comment
except ImportError:
    print("Error: BeautifulSoup4 is required. Install it with: pip install beautifulsoup4")
    sys.exit(1)

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    
    # Check for NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
    
    NLTK_AVAILABLE = True
except ImportError:
    print("Warning: NLTK not available. Some advanced NLP features will be disabled.")
    print("To enable all features: pip install nltk")
    NLTK_AVAILABLE = False

# Configuration Constants
DEFAULT_CONFIG = {
    # Research parameters
    "depth": 3,               # 1-5 scale: higher means deeper research
    "max_sources": 25,        # Maximum sources to analyze
    "exclude_domains": [],    # Domains to exclude from search/crawl
    "batch_size": 5,          # Number of URLs to process at once
    
    # Search settings
    "search_engine": "duckduckgo",
    "max_search_results": 30,
    "search_time_limit": "year",
    
    # Crawler settings
    "crawl_depth": 2,         # Photon crawl level
    "max_links_per_source": 3, # How many links to follow from each source
    "respect_robots_txt": True,
    
    # Content extraction settings
    "min_content_length": 300,
    "max_content_length": 20000,
    
    # Summarization settings
    "summary_method": "textrank",  # 'textrank', 'ollama', 'openai'
    "summary_model": "gemma3:12b",  # Model for Ollama or OpenAI
    "summary_depth": "medium",     # 'short', 'medium', 'detailed'
    "ollama_api_url": "http://localhost:11434/api/generate",
    "openai_api_key": "",
    "openai_api_url": "https://api.openai.com/v1/chat/completions",
    
    # Output settings
    "output_format": "markdown",    # 'text', 'markdown', 'json'
    "output_dir": "research_results",
    "include_source_hierarchy": True,
    
    # Performance settings
    "timeout": 30,            # Request timeout in seconds  
    "max_retries": 3,         # Max retries for failed requests
    "max_threads": 8,         # Maximum threads for parallel processing
    
    # Paths for external tools
    "photon_path": "./photon.py",
    
    # Logging and debugging
    "verbose": False,
    "debug": False,
    
    # Scaling parameters based on depth
    "source_count_by_depth": {
        1: 10,   # Quick overview
        2: 20,   # Basic research
        3: 30,   # Comprehensive
        4: 50,   # In-depth
        5: 80    # Exhaustive
    },
    "crawl_depth_by_depth": {
        1: 1,    # Minimal crawling
        2: 1,    # Basic crawling
        3: 2,    # Moderate crawling
        4: 3,    # Deep crawling
        5: 4     # Very deep crawling
    }
}

# Default HTTP headers for web requests
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive"
}

# URL pattern for extraction
URL_PATTERN = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[^)\s]*)?')

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeepResearch")

#=======================================================#
# Utility Functions                                     #
#=======================================================#

def clean_text(text):
    """Clean text by removing extra whitespace, HTML tags, etc."""
    if not text:
        return ""
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'&\w+;', ' ', text)    # Remove HTML entities
    text = re.sub(r'\s+', ' ', text)      # Replace multiple spaces with single space
    return text.strip()

def extract_keywords(text, limit=20):
    """Extract keywords from text with stop word filtering."""
    if not text or not NLTK_AVAILABLE:
        return []
    
    try:
        # Clean and lowercase text
        text = clean_text(text).lower()
        
        # Tokenize
        words = word_tokenize(text)
        
        # Filter out stopwords, short words, punctuation, and numbers
        stop_words = set(stopwords.words('english'))
        keywords = [word for word in words 
                   if word not in stop_words 
                   and len(word) > 3 
                   and word not in string.punctuation 
                   and not word.isdigit()]
        
        # Count frequencies
        word_freq = Counter(keywords)
        
        # Return most common words
        return [word for word, count in word_freq.most_common(limit)]
    except Exception as e:
        logger.debug(f"Error extracting keywords: {str(e)}")
        return []

def is_valid_url(url):
    """Check if a URL is valid."""
    if not url:
        return False
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)
    except:
        return False

def get_domain(url):
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return ""

def fingerprint_text(text, length=32):
    """Create a fingerprint hash of text for deduplication."""
    if not text:
        return ""
    text = clean_text(text)
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:length]

def print_header(text):
    """Print a formatted header."""
    try:
        width = min(os.get_terminal_size().columns, 80)
    except OSError:
        width = 80
    print(f"\n\033[1;36m{'=' * width}\n{text.center(width)}\n{'=' * width}\033[0m\n")

def print_progress(current, total, prefix='', suffix='', length=50, fill='█'):
    """Print a progress bar."""
    percent = int(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if current == total:
        print()

def create_spinner():
    """Create a simple spinner to indicate progress."""
    import itertools
    import threading
    
    spinner_active = [True]
    spinner_thread = None
    
    def spin():
        for c in itertools.cycle('|/-\\'):
            if not spinner_active[0]:
                break
            sys.stdout.write(f"\r{c} ")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\r')
        sys.stdout.flush()
    
    spinner_thread = threading.Thread(target=spin)
    spinner_thread.daemon = True
    spinner_thread.start()
    
    def stop_spinner():
        spinner_active[0] = False
        if spinner_thread:
            spinner_thread.join(0.5)
    
    return stop_spinner

def generate_output_filename(query, format="markdown", output_dir="research_results"):
    """Generate a filename for the research output based on the query."""
    # Create output dir if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Clean the query to create a valid filename
    clean_query = re.sub(r'[^\w\s-]', '', query).strip().lower()
    clean_query = re.sub(r'[-\s]+', '-', clean_query)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    extensions = {
        "markdown": "md",
        "text": "txt",
        "json": "json"
    }
    ext = extensions.get(format, "txt")
    
    return os.path.join(output_dir, f"research_{clean_query[:40]}_{timestamp}.{ext}")

def estimate_reading_time(text, wpm=200):
    """Estimate reading time in minutes based on text length."""
    if not text:
        return 1
    word_count = len(text.split())
    return max(1, round(word_count / wpm))

#=======================================================#
# Search Class                                          #
#=======================================================#

class WebSearch:
    """Search for information on the web using various search engines."""
    
    def __init__(self, config):
        """Initialize search module with configuration."""
        self.config = config
        self.timeout = config.get("timeout", 30)
        self.max_results = config.get("max_search_results", 30)
        self.search_time = config.get("search_time_limit", "year")
        
        # Check if ddgr command-line tool is available
        try:
            subprocess.run(['ddgr', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            self.ddgr_available = True
            logger.info("ddgr command-line tool found, will use it for search")
        except (subprocess.SubprocessError, FileNotFoundError):
            self.ddgr_available = False
            logger.info("ddgr not found, will use direct API calls instead")
    
    def search(self, query, depth=3):
        """
        Execute a search query and return results.
        
        Args:
            query: The search query string
            depth: Research depth (1-5)
            
        Returns:
            List of search results with URLs, titles, and snippets
        """
        logger.info(f"Searching for: {query}")
        
        # Scale max results based on depth
        max_results = min(100, depth * 10 + 10)  # 20 for depth 1, 60 for depth 5
        
        # Execute search with appropriate method
        if self.ddgr_available:
            results = self._search_with_ddgr(query, max_results)
        else:
            results = self._search_with_api(query, max_results)
            
        # If no results, try fallback method
        if not results:
            logger.warning("No search results found via primary method")
            results = self._search_with_fallback(query, max_results)
            
        if not results:
            logger.error("All search methods failed")
            return []
            
        # Filter and enhance results
        filtered_results = self._filter_results(results, self.config.get("exclude_domains", []))
        logger.info(f"Found {len(filtered_results)} relevant search results")
        
        return filtered_results
    
    def _search_with_ddgr(self, query, max_results):
        """Use ddgr command-line tool to perform DuckDuckGo search."""
        results = []
        
        try:
            # Build ddgr command
            cmd = ['ddgr', '--json', '--num', str(min(max_results, 25))]
            
            # Add time limit if specified
            if self.search_time:
                time_map = {'day': 'd', 'week': 'w', 'month': 'm', 'year': 'y'}
                if self.search_time in time_map:
                    cmd.extend(['--time', time_map[self.search_time]])
            
            # Add region
            cmd.extend(['--reg', 'us-en'])
            
            # Add query
            cmd.append(query)
            
            logger.debug(f"Executing ddgr command: {' '.join(cmd)}")
            
            # Run the command
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if process.returncode != 0:
                logger.error(f"ddgr search failed: {process.stderr}")
                return []
                
            # Parse JSON output
            try:
                search_results = json.loads(process.stdout)
                
                # Process results
                for result in search_results:
                    results.append({
                        'title': result.get('title', 'No Title'),
                        'url': result.get('url', ''),
                        'snippet': result.get('abstract', ''),
                        'source': 'ddgr'
                    })
            except json.JSONDecodeError:
                logger.error("Failed to parse ddgr JSON output")
                return []
                
        except Exception as e:
            logger.error(f"Error using ddgr: {str(e)}")
            return []
            
        return results
    
    def _search_with_api(self, query, max_results):
        """Use direct API calls to DuckDuckGo for search results."""
        results = []
        
        try:
            # Prepare API request to DuckDuckGo
            escaped_query = urllib.parse.quote_plus(query)
            api_url = f"https://html.duckduckgo.com/html/?q={escaped_query}"
            
            headers = DEFAULT_HEADERS.copy()
            request = urllib.request.Request(api_url, headers=headers)
            
            logger.debug(f"Fetching search results from: {api_url}")
            
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                html_content = response.read().decode('utf-8')
                
                # Use regex to extract search results
                result_pattern = re.compile(
                    r'<h2 class="result__title">.*?<a href="(.*?)".*?>(.*?)</a>.*?'
                    r'<a class="result__snippet".*?>(.*?)</a>',
                    re.DOTALL
                )
                
                for match in result_pattern.finditer(html_content):
                    url, title, snippet = match.groups()
                    
                    # Clean up HTML entities
                    url = self._cleanup_html_entities(url)
                    title = self._cleanup_html_entities(title)
                    snippet = self._cleanup_html_entities(snippet)
                    
                    # Extract actual URL from DuckDuckGo's redirect
                    if url.startswith('/'):
                        continue  # Skip internal links
                        
                    if 'duckduckgo.com/l/?' in url:
                        try:
                            # Extract the actual URL from the redirect
                            url_param = re.search(r'uddg=(.*?)&', url)
                            if url_param:
                                url = urllib.parse.unquote_plus(url_param.group(1))
                        except Exception:
                            # If extraction fails, keep the original URL
                            pass
                    
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet,
                        'source': 'duckduckgo_api'
                    })
                    
                    if len(results) >= max_results:
                        break
                        
        except Exception as e:
            logger.error(f"Error searching with DuckDuckGo API: {str(e)}")
            return []
            
        return results
    
    def _search_with_fallback(self, query, max_results):
        """Fallback search method using a different endpoint."""
        results = []
        
        try:
            # Try using the DuckDuckGo lite API
            escaped_query = urllib.parse.quote_plus(query)
            api_url = f"https://lite.duckduckgo.com/lite/?q={escaped_query}"
            
            headers = DEFAULT_HEADERS.copy()
            request = urllib.request.Request(api_url, headers=headers)
            
            logger.debug(f"Using fallback search API: {api_url}")
            
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                html_content = response.read().decode('utf-8')
                
                # Extract results from the lite version
                result_pattern = re.compile(
                    r'<a class="result-link" href="(.*?)".*?>(.*?)</a>.*?'
                    r'<td class="result-snippet">(.*?)</td>',
                    re.DOTALL
                )
                
                for match in result_pattern.finditer(html_content):
                    url, title, snippet = match.groups()
                    
                    # Clean up HTML entities
                    url = self._cleanup_html_entities(url)
                    title = self._cleanup_html_entities(title)
                    snippet = self._cleanup_html_entities(snippet)
                    
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet,
                        'source': 'duckduckgo_lite'
                    })
                    
                    if len(results) >= max_results:
                        break
                        
        except Exception as e:
            logger.error(f"Backup search method also failed: {str(e)}")
            
        return results
    
    def _cleanup_html_entities(self, text):
        """Clean up HTML entities and tags from text."""
        if not text:
            return ""
            
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Replace common HTML entities
        entities = {
            '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
            '&#39;': "'", '&nbsp;': ' ',
        }
        
        for entity, replacement in entities.items():
            text = text.replace(entity, replacement)
            
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _filter_results(self, results, exclude_domains):
        """Filter search results based on various criteria."""
        filtered_results = []
        seen_domains = set()
        seen_urls = set()
        
        for result in results:
            url = result.get('url', '')
            
            # Skip empty URLs
            if not url:
                continue
                
            # Skip already seen URLs
            url_normalized = url.lower()
            if url_normalized in seen_urls:
                continue
                
            # Skip excluded domains
            domain = get_domain(url)
            if not domain:
                continue
                
            if any(excluded in domain for excluded in exclude_domains):
                logger.debug(f"Excluding URL from excluded domain: {url}")
                continue
                
            # Avoid too many results from the same domain for diversity
            if domain in seen_domains and len([d for d in seen_domains if d == domain]) >= 3:
                continue
                
            # Skip URLs that are likely to be unhelpful
            if self._should_skip_url(url, result.get('title', ''), result.get('snippet', '')):
                continue
                
            # Add to filtered results
            seen_urls.add(url_normalized)
            seen_domains.add(domain)
            filtered_results.append(result)
            
        return filtered_results
    
    def _should_skip_url(self, url, title, snippet):
        """Determine if a URL should be skipped based on various criteria."""
        # Skip social media profiles
        social_media_patterns = [
            r'facebook\.com/[^/]+$', r'twitter\.com/[^/]+$', r'instagram\.com/[^/]+$',
            r'linkedin\.com/in/', r'pinterest\.com/[^/]+$', r'tiktok\.com/@[^/]+$'
        ]
        
        for pattern in social_media_patterns:
            if re.search(pattern, url):
                return True
                
        # Skip shopping and product pages
        shopping_patterns = [
            r'amazon\.com/[^/]+/dp/', r'ebay\.com/itm/', r'etsy\.com/listing/'
        ]
        
        for pattern in shopping_patterns:
            if re.search(pattern, url):
                return True
                
        # Skip login and signup pages
        if re.search(r'(login|signin|signup|register)\.(php|html|aspx)$', url):
            return True
            
        # Skip PDF files
        if url.endswith('.pdf'):
            return True
            
        return False

#=======================================================#
# Web Crawler Class                                     #
#=======================================================#

class WebCrawler:
    """Web crawler that uses Photon for discovering new content."""
    
    def __init__(self, config):
        """Initialize the WebCrawler with configuration."""
        self.config = config
        self.photon_path = config.get("photon_path", "./photon.py")
        self.crawl_depth = config.get("crawl_depth", 2)
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        self.max_threads = config.get("max_threads", 8)
        self.output_dir = "photon_results"
        
        # Verify Photon availability
        if not os.path.isfile(self.photon_path):
            logger.warning(f"Photon not found at {self.photon_path}. Photon-based crawling will be disabled.")
    
    def crawl_url(self, url, depth=1, parent_url=None):
        """
        Crawl a URL with Photon to discover linked content.
        
        Args:
            url: The URL to crawl
            depth: Crawl depth (1-5)
            parent_url: The parent URL that led to this URL (for hierarchy tracking)
            
        Returns:
            Dictionary with crawl results including discovered URLs
        """
        # Check if Photon is available
        if not os.path.isfile(self.photon_path):
            logger.error("Photon script not found. Cannot crawl URL.")
            return {"url": url, "found_urls": [], "error": True}
        
        # Make sure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Create domain-specific subfolder
        domain = get_domain(url)
        timestamp = int(time.time())
        domain_safe = re.sub(r'[^\w\-_]', '_', domain)
        target_dir = f"{domain_safe}_{timestamp}"
        output_path = os.path.join(self.output_dir, target_dir)
        
        # Build Photon command
        command = [
            'python3', self.photon_path,
            '-u', url,
            '-l', str(min(depth, 3)),  # Photon level 1-3
            '-t', str(min(self.max_threads, 10)),  # Threads
            '-o', output_path
        ]
        
        # Run Photon
        logger.info(f"Crawling {url} with Photon (depth {depth})")
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout * 2  # Give Photon more time
            )
            
            # Check for errors
            if result.returncode != 0:
                logger.error(f"Photon crawl failed: {result.stderr}")
                return {"url": url, "found_urls": [], "error": True, "parent_url": parent_url}
            
            # Extract results from Photon output directory
            crawl_results = self._extract_photon_results(output_path)
            crawl_results["parent_url"] = parent_url
            
            return crawl_results
            
        except subprocess.TimeoutExpired:
            logger.error(f"Photon crawl timed out for {url}")
            return {"url": url, "found_urls": [], "error": True, "parent_url": parent_url}
            
        except Exception as e:
            logger.error(f"Error during Photon crawl: {str(e)}")
            return {"url": url, "found_urls": [], "error": True, "parent_url": parent_url}
    
    def _extract_photon_results(self, output_path):
        """Extract URLs and other data from Photon results directory."""
        results = {
            "found_urls": [],
            "files": [],
            "error": False
        }
        
        try:
            # Extract internal URLs
            internal_file = os.path.join(output_path, "internal.txt")
            if os.path.isfile(internal_file):
                with open(internal_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        url = line.strip()
                        if url and is_valid_url(url):
                            results["found_urls"].append(url)
            
            # Extract external URLs if needed
            external_file = os.path.join(output_path, "external.txt")
            if os.path.isfile(external_file) and len(results["found_urls"]) < 20:
                with open(external_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        url = line.strip()
                        if url and is_valid_url(url):
                            results["found_urls"].append(url)
            
            # Extract discovered files
            files_file = os.path.join(output_path, "files.txt")
            if os.path.isfile(files_file):
                with open(files_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        file_url = line.strip()
                        if file_url:
                            results["files"].append(file_url)
            
            # Deduplicate URLs
            results["found_urls"] = list(OrderedDict.fromkeys(results["found_urls"]))
            results["files"] = list(OrderedDict.fromkeys(results["files"]))
            
            return results
            
        except Exception as e:
            logger.error(f"Error extracting Photon results: {str(e)}")
            results["error"] = True
            return results
    
    def crawl_urls_with_hierarchy(self, urls, max_depth=2, max_per_source=3):
        """
        Crawl multiple URLs while maintaining source hierarchy.
        
        Args:
            urls: List of seed URLs to crawl
            max_depth: Maximum crawl depth
            max_per_source: Maximum URLs to follow from each source
            
        Returns:
            Dictionary with hierarchical crawl results
        """
        if not urls:
            return {"crawled_urls": [], "source_hierarchy": {}}
        
        hierarchy = {}  # URL -> list of child URLs
        all_crawled = set()
        
        # Crawl each seed URL
        seed_results = []
        with ThreadPoolExecutor(max_workers=min(len(urls), 5)) as executor:
            future_to_url = {executor.submit(self.crawl_url, url, max_depth): url for url in urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if not result["error"] and result["found_urls"]:
                        # Limit the number of URLs per source
                        child_urls = result["found_urls"][:max_per_source]
                        hierarchy[url] = child_urls
                        all_crawled.add(url)
                        seed_results.append(result)
                except Exception as e:
                    logger.error(f"Error crawling {url}: {str(e)}")
        
        # Create full hierarchy with nested levels
        source_hierarchy = self._build_source_hierarchy(hierarchy)
        
        return {
            "crawled_urls": list(all_crawled),
            "source_hierarchy": source_hierarchy
        }
    
    def _build_source_hierarchy(self, url_children):
        """Build a nested hierarchy of sources."""
        # Create a dictionary for the hierarchy
        hierarchy = {}
        
        # Build the hierarchy
        for url, children in url_children.items():
            hierarchy[url] = {child: {} for child in children}
        
        return hierarchy

#=======================================================#
# Content Extractor Class                               #
#=======================================================#

class ContentExtractor:
    """Extract and process content from web pages."""
    
    def __init__(self, config):
        """Initialize the ContentExtractor with configuration."""
        self.config = config
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        self.min_content_length = config.get("min_content_length", 300)
        self.max_content_length = config.get("max_content_length", 20000)
    
    def extract_content(self, url, source_info=None):
        """
        Extract content from a webpage with metadata.
        
        Args:
            url: The URL to extract content from
            source_info: Information about the source (for hierarchy tracking)
            
        Returns:
            Dictionary with extracted content and metadata
        """
        # Initialize result structure
        result = {
            "url": url,
            "domain": get_domain(url),
            "title": "",
            "text": "",
            "html": "",
            "metadata": {},
            "links": [],
            "timestamp": datetime.now().isoformat(),
            "source_info": source_info,
            "error": None
        }
        
        # Validate URL
        if not is_valid_url(url):
            result["error"] = "Invalid URL format"
            return result
        
        # Attempt extraction with retries
        for attempt in range(self.max_retries):
            try:
                # Fetch the webpage
                response = self._fetch_url(url)
                
                if not response or response.get("error"):
                    result["error"] = response.get("error") if response else "Failed to fetch URL"
                    if attempt < self.max_retries - 1:
                        time.sleep(1)  # Wait before retry
                        continue
                    return result
                
                # Update result with response info
                result.update(response)
                
                # Extract content based on type
                if 'html' in result.get("content_type", "").lower():
                    extracted = self._extract_html_content(result["html"], url)
                    result.update(extracted)
                elif 'json' in result.get("content_type", "").lower():
                    extracted = self._extract_json_content(result["html"])
                    result.update(extracted)
                elif 'text/plain' in result.get("content_type", "").lower():
                    result["text"] = result["html"]  # For plain text, html field contains the text
                    result["title"] = url.split('/')[-1] or "Text Document"
                else:
                    # Try HTML extraction as fallback
                    extracted = self._extract_html_content(result["html"], url)
                    result.update(extracted)
                
                # Extract keywords if content is sufficient
                if len(result["text"]) >= self.min_content_length:
                    result["keywords"] = extract_keywords(result["text"])
                    
                    # Generate content fingerprint for deduplication
                    result["content_hash"] = fingerprint_text(result["text"])
                    
                    # Estimate reading time
                    result["reading_time_min"] = estimate_reading_time(result["text"])
                    
                    return result
                else:
                    result["error"] = "Content too short"
                    if attempt < self.max_retries - 1:
                        time.sleep(1)  # Wait before retry
                        continue
                    
            except Exception as e:
                logger.error(f"Error extracting content from {url}: {str(e)}")
                result["error"] = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    continue
        
        return result
    
    def _fetch_url(self, url):
        """Fetch content from a URL with proper error handling."""
        try:
            # Set up request headers
            headers = DEFAULT_HEADERS.copy()
            
            # Create request
            req = urllib.request.Request(url, headers=headers)
            
            # Open connection with timeout
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                # Get response info
                status_code = response.getcode()
                info = dict(response.info())
                content_type = info.get('Content-Type', 'text/html').split(';')[0]
                
                # Read content
                content = response.read()
                
                # Try to decode based on content type or headers
                charset = None
                
                # Check for charset in Content-Type header
                content_type_header = info.get('Content-Type', '')
                charset_match = re.search(r'charset=([^\s;]+)', content_type_header)
                if charset_match:
                    charset = charset_match.group(1)
                
                # Decode with proper charset, or fallback to UTF-8
                try:
                    if charset:
                        decoded_content = content.decode(charset)
                    else:
                        decoded_content = content.decode('utf-8')
                except UnicodeDecodeError:
                    # Fallback with error handling
                    decoded_content = content.decode('utf-8', errors='replace')
                
                # Return response info and content
                return {
                    "status_code": status_code,
                    "content_type": content_type,
                    "html": decoded_content,
                    "headers": info
                }
                
        except urllib.error.HTTPError as e:
            logger.warning(f"HTTP error while fetching {url}: {e.code}")
            return {"error": f"HTTP error: {e.code}"}
            
        except urllib.error.URLError as e:
            logger.warning(f"URL error while fetching {url}: {str(e)}")
            return {"error": f"URL error: {str(e)}"}
            
        except Exception as e:
            logger.warning(f"Exception while fetching {url}: {str(e)}")
            return {"error": f"Error: {str(e)}"}
    
    def _extract_html_content(self, html, url):
        """Extract useful content from HTML."""
        if not html:
            return {"title": "", "text": "", "links": []}
        
        try:
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)
            
            # Extract meta description
            description = ""
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                description = meta_desc.get('content', '')
            
            # Remove unwanted elements that typically contain boilerplate/ads
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'form', 'iframe']):
                tag.decompose()
            
            # Remove comments
            for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # Try to find main content
            main_content = self._find_main_content(soup)
            
            # Extract text from main content or fallback to body
            text = ""
            if main_content:
                paragraphs = main_content.find_all('p')
                if paragraphs:
                    text = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
                
                # If no substantial paragraphs, get all text from main content
                if len(text) < self.min_content_length:
                    text = main_content.get_text(separator='\n\n', strip=True)
            
            # Fallback to all paragraphs
            if len(text) < self.min_content_length:
                paragraphs = soup.find_all('p')
                if paragraphs:
                    text = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
            
            # Last resort: extract from body
            if len(text) < self.min_content_length:
                body = soup.find('body')
                if body:
                    text = body.get_text(separator='\n\n', strip=True)
            
            # Prepend description if found and not already in text
            if description and description not in text:
                text = description + "\n\n" + text
                
            # Clean up text
            text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excess newlines
            
            # Truncate if too long
            if len(text) > self.max_content_length:
                text = text[:self.max_content_length] + "... [truncated]"
            
            # Extract links if needed
            links = []
            if self.config.get("extract_links", True):
                for a in soup.find_all('a', href=True):
                    href = a.get('href', '').strip()
                    if href and not href.startswith('#') and not href.startswith('javascript:'):
                        # Resolve relative URLs
                        try:
                            abs_url = urljoin(url, href)
                            link_text = a.get_text(strip=True)
                            if is_valid_url(abs_url):
                                links.append({
                                    "url": abs_url,
                                    "text": link_text
                                })
                        except Exception:
                            continue
            
            return {
                "title": title,
                "text": text,
                "links": links,
                "metadata": {
                    "description": description
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing HTML: {str(e)}")
            # Fallback to regex extraction
            return self._extract_with_regex(html)
    
    def _find_main_content(self, soup):
        """Find the main content container in HTML."""
        # Try common content containers by ID
        for id_pattern in ['content', 'main', 'article', 'post', 'entry']:
            main = soup.find(id=re.compile(f'.*{id_pattern}.*', re.I))
            if main:
                return main
        
        # Try common content containers by class
        for class_pattern in ['content', 'main', 'article', 'post', 'entry']:
            main = soup.find(class_=re.compile(f'.*{class_pattern}.*', re.I))
            if main:
                return main
        
        # Try semantic HTML5 elements
        for tag in ['article', 'main', 'section']:
            main = soup.find(tag)
            if main:
                return main
        
        # Find the div with the most paragraphs
        divs = soup.find_all('div')
        if divs:
            # Compute the number of paragraphs for each div
            div_counts = [(div, len(div.find_all('p'))) for div in divs]
            # Sort by count in descending order
            div_counts.sort(key=lambda x: x[1], reverse=True)
            # Return the div with the most paragraphs if it has at least 2
            if div_counts and div_counts[0][1] >= 2:
                return div_counts[0][0]
        
        # Fallback to body
        return soup.find('body')
    
    def _extract_with_regex(self, html):
        """Fallback extraction method using regex."""
        title = ""
        text = ""
        
        # Extract title
        title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = self._cleanup_html_text(title_match.group(1))
        
        # Extract paragraphs
        paragraphs = re.findall(r'<p>(.*?)</p>', html, re.IGNORECASE | re.DOTALL)
        if paragraphs:
            text = '\n\n'.join([self._cleanup_html_text(p) for p in paragraphs if len(self._cleanup_html_text(p)) > 20])
        
        # Fallback to stripping all HTML
        if not text:
            text = self._cleanup_html_text(html)
        
        return {
            "title": title,
            "text": text,
            "links": []
        }
    
    def _cleanup_html_text(self, text):
        """Clean up text extracted from HTML."""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]*>', ' ', text)
        
        # Replace HTML entities
        text = re.sub(r'&[a-z]+;', ' ', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_json_content(self, json_text):
        """Extract content from JSON."""
        try:
            data = json.loads(json_text)
            
            # Try to find title and text in common JSON structures
            title = ""
            text = ""
            
            # Look for title-like fields
            for field in ['title', 'name', 'heading', 'subject']:
                if field in data and isinstance(data[field], str):
                    title = data[field].strip()
                    break
            
            # Look for content-like fields
            for field in ['content', 'text', 'body', 'description', 'article']:
                if field in data:
                    if isinstance(data[field], str):
                        text = data[field].strip()
                        break
                    elif isinstance(data[field], dict):
                        # Handle nested content
                        for subfield in ['text', 'value', 'content']:
                            if subfield in data[field] and isinstance(data[field][subfield], str):
                                text = data[field][subfield].strip()
                                break
            
            # If nothing found, use the whole JSON as text
            if not text:
                text = json.dumps(data, indent=2)
                
            return {
                "title": title or "JSON Document",
                "text": text,
                "links": []
            }
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON content")
            return {
                "title": "Invalid JSON",
                "text": json_text,
                "links": []
            }

    def extract_batch(self, urls, source_hierarchy=None):
        """
        Extract content from multiple URLs in parallel.
        
        Args:
            urls: List of URLs to extract content from
            source_hierarchy: Dictionary mapping URLs to their source information
            
        Returns:
            List of extracted content items
        """
        if not urls:
            return []
        
        results = []
        
        # Process URLs in parallel
        with ThreadPoolExecutor(max_workers=min(len(urls), self.config.get("max_threads", 8))) as executor:
            # Create tasks for each URL
            future_to_url = {}
            for url in urls:
                # Get source info if available
                source_info = None
                if source_hierarchy and url in source_hierarchy:
                    source_info = {"parent": source_hierarchy.get(url)}
                
                future = executor.submit(self.extract_content, url, source_info)
                future_to_url[future] = url
            
            # Process completed tasks
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    content = future.result()
                    results.append(content)
                except Exception as e:
                    logger.error(f"Error extracting content from {url}: {str(e)}")
                    results.append({
                        "url": url,
                        "error": str(e),
                        "title": "Extraction Error",
                        "text": ""
                    })
        
        return results

#=======================================================#
# Source Analyzer Class                                 #
#=======================================================#

class SourceAnalyzer:
    """Analyze and process extracted content for research."""
    
    def __init__(self, config):
        """Initialize the SourceAnalyzer with configuration."""
        self.config = config
    
    def analyze_sources(self, sources, query):
        """
        Analyze extracted sources for relevance and quality.
        
        Args:
            sources: List of extracted content items
            query: Original research query
            
        Returns:
            List of processed and prioritized sources
        """
        if not sources:
            return []
        
        # Filter out failed/error sources
        valid_sources = [s for s in sources if s.get("text") and not s.get("error")]
        
        if not valid_sources:
            logger.warning("No valid sources to analyze")
            return []
        
        # Analyze each source
        for source in valid_sources:
            # Calculate relevance to query
            source["relevance"] = self._calculate_relevance(source, query)
            
            # Calculate credibility score
            source["credibility"] = self._calculate_credibility(source)
            
            # Detect content type
            source["content_type"] = self._detect_content_type(source)
            
            # Calculate quality score
            source["quality"] = self._calculate_quality(source)
            
            # Calculate overall score
            source["score"] = self._calculate_overall_score(source)
        
        # Sort by overall score
        valid_sources.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Apply domain diversity (avoid too many sources from same domain)
        diversified = self._apply_domain_diversity(valid_sources)
        
        return diversified
    
    def _calculate_relevance(self, source, query):
        """Calculate relevance of source to the query."""
        text = source.get("text", "")
        title = source.get("title", "")
        
        # Extract query keywords
        query_keywords = set(extract_keywords(query))
        if not query_keywords:
            return 50  # Default value if no keywords
        
        # Count keyword matches in title (weighted higher)
        title_matches = sum(1 for kw in query_keywords if kw in title.lower())
        title_score = min(100, title_matches * 20)  # 20 points per match, max 100
        
        # Count keyword matches in text
        text_keywords = set(extract_keywords(text[:5000]))  # Limit to first 5000 chars
        text_matches = len(query_keywords.intersection(text_keywords))
        text_score = min(100, (text_matches / len(query_keywords)) * 100)
        
        # Calculate combined score (70% title, 30% text)
        relevance_score = (title_score * 0.7) + (text_score * 0.3)
        
        return relevance_score
    
    def _calculate_credibility(self, source):
        """Calculate credibility score for a source."""
        url = source.get("url", "")
        domain = source.get("domain", "")
        
        # Base credibility score
        score = 50
        
        # Boost for academic and educational domains
        if domain.endswith(".edu") or domain.endswith(".gov") or domain.endswith(".org"):
            score += 20
        elif domain.endswith(".ac.uk") or domain.endswith(".gov.uk"):
            score += 20
        elif any(domain.endswith(tld) for tld in [".gov", ".mil"]):
            score += 20
        
        # Boost for reputable news sources
        reputable_news = [
            "nytimes.com", "washingtonpost.com", "wsj.com", "economist.com",
            "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com", "bloomberg.com"
        ]
        if any(domain == site for site in reputable_news):
            score += 15
        
        # Boost for academic publications
        academic_sites = [
            "springer.com", "nature.com", "science.org", "wiley.com",
            "sciencedirect.com", "tandfonline.com", "jstor.org",
            "arxiv.org", "ncbi.nlm.nih.gov", "pubmed.gov"
        ]
        if any(site in domain for site in academic_sites):
            score += 25
        
        # Boost for tech documentation
        tech_docs = [
            "developer.mozilla.org", "docs.python.org", "developer.apple.com",
            "learn.microsoft.com", "docs.microsoft.com", "developer.android.com"
        ]
        if any(site == domain for site in tech_docs):
            score += 15
        
        # Wikipedia gets its own category
        if domain == "wikipedia.org" or domain.endswith(".wikipedia.org"):
            score += 10
        
        # Cap at 100
        return min(100, score)
    
    def _detect_content_type(self, source):
        """Detect the type of content."""
        url = source.get("url", "")
        text = source.get("text", "")
        title = source.get("title", "")
        domain = source.get("domain", "")
        
        # Academic paper patterns
        academic_patterns = [
            r'\b(abstract|introduction|methodology|results|conclusion|references)\b',
            r'\b(published in|journal of|proceedings of|vol\.|volume|doi:)\b'
        ]
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in academic_patterns):
            return "academic_paper"
        
        # News article patterns
        news_patterns = [
            r'\b(published|reported|according to|news|article)\b',
            r'\b(daily|weekly|times|post|gazette|herald|chronicle)\b'
        ]
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in news_patterns) or \
           any(domain == site for site in ["nytimes.com", "bbc.com", "reuters.com"]):
            return "news_article"
        
        # Blog patterns
        blog_patterns = [
            r'\b(blog|posted by|posted on|author:|written by)\b',
        ]
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in blog_patterns) or \
           "blog" in url or domain == "medium.com":
            return "blog_post"
        
        # Documentation patterns
        doc_patterns = [
            r'\b(documentation|user guide|developer guide|api reference|manual)\b'
        ]
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in doc_patterns) or \
           any(domain == site for site in ["docs.python.org", "developer.mozilla.org"]):
            return "documentation"
        
        # Tutorial/How-to patterns
        tutorial_patterns = [
            r'\b(tutorial|how to|step by step|guide|learn how)\b'
        ]
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in tutorial_patterns):
            return "tutorial"
        
        # Q&A patterns
        if domain in ["stackoverflow.com", "quora.com", "reddit.com"]:
            return "q_and_a"
        
        # Default to general article
        return "article"
    
    def _calculate_quality(self, source):
        """Calculate content quality score."""
        text = source.get("text", "")
        
        # Base quality score
        score = 50
        
        # Length-based scoring
        word_count = len(text.split())
        if word_count > 2000:
            score += 20
        elif word_count > 1000:
            score += 15
        elif word_count > 500:
            score += 10
        elif word_count < 200:
            score -= 20
        
        # Check for features of high-quality content
        if re.search(r'\b(study|research|survey|analysis)\b', text, re.IGNORECASE):
            score += 5
        
        if re.search(r'\b(according to|cited|reference|source)\b', text, re.IGNORECASE):
            score += 10
        
        # Check for data richness (numbers, statistics)
        if len(re.findall(r'\d+%|\d+\.\d+|\b\d+\s+\w+\b', text)) > 5:
            score += 5
        
        # Penalize promotional content
        promo_terms = ['click here', 'buy now', 'sign up', 'subscribe', 'limited time offer']
        promo_count = sum(1 for term in promo_terms if term in text.lower())
        if promo_count > 0:
            score -= min(promo_count * 10, 40)
        
        # Cap at 0-100
        return max(0, min(100, score))
    
    def _calculate_overall_score(self, source):
        """Calculate overall source score combining all factors."""
        relevance = source.get("relevance", 50)
        credibility = source.get("credibility", 50)
        quality = source.get("quality", 50)
        
        # Weight factors by importance
        overall = (relevance * 0.5) + (credibility * 0.3) + (quality * 0.2)
        
        # Apply boost based on content type (academic papers and documentation are trusted more)
        content_type = source.get("content_type", "article")
        
        if content_type == "academic_paper":
            overall *= 1.2
        elif content_type == "documentation":
            overall *= 1.1
        
        # Cap at 100
        return min(100, overall)
    
    def _apply_domain_diversity(self, sources):
        """Ensure diversity of domains in the results."""
        if not sources:
            return []
        
        # Count domains
        domain_count = Counter()
        for source in sources:
            domain_count[source.get("domain", "")] += 1
        
        # Calculate diversity penalty for each source
        for source in sources:
            domain = source.get("domain", "")
            count = domain_count[domain]
            
            # Apply penalty to duplicate domains
            if count > 1:
                penalty = min(30, (count - 1) * 10)  # 10% per duplicate, max 30%
                source["diversity_penalty"] = penalty
                source["score"] = max(0, source["score"] - penalty)
        
        # Re-sort by adjusted score
        sources.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return sources
    
    def deduplicate_sources(self, sources, threshold=0.8):
        """Remove duplicate or near-duplicate sources."""
        if not sources:
            return []
        
        unique_sources = []
        content_hashes = set()
        
        for source in sources:
            # Skip sources without text
            if not source.get("text"):
                continue
            
            # Get content hash
            content_hash = source.get("content_hash") or fingerprint_text(source.get("text", ""))
            
            # Check for duplicates
            if content_hash in content_hashes:
                continue
            
            # Add to unique sources
            content_hashes.add(content_hash)
            unique_sources.append(source)
        
        logger.info(f"Deduplicated {len(sources)} sources to {len(unique_sources)} unique sources")
        return unique_sources

    def extract_key_points(self, sources, query):
        """
        Extract key points from multiple sources.
        
        Args:
            sources: List of analyzed sources
            query: Research query
            
        Returns:
            Dictionary of source IDs mapped to key points
        """
        if not sources:
            return {}
        
        key_points = {}
        query_keywords = set(extract_keywords(query))
        
        for i, source in enumerate(sources):
            text = source.get("text", "")
            if not text:
                continue
                
            # Generate a source ID
            source_id = f"S{i+1}"
            source["source_id"] = source_id
            
            # Extract sentences
            sentences = []
            try:
                if NLTK_AVAILABLE:
                    sentences = nltk.sent_tokenize(text)
                else:
                    # Fallback to simple regex-based sentence splitting
                    sentences = re.split(r'(?<=[.!?])\s+', text)
            except Exception:
                sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # Score sentences by relevance to query
            scored_sentences = []
            for sentence in sentences:
                # Skip short sentences
                if len(sentence.split()) < 5:
                    continue
                
                # Calculate relevance to query
                sentence_keywords = set(extract_keywords(sentence))
                keyword_overlap = len(query_keywords.intersection(sentence_keywords))
                
                # Score based on keyword overlap
                score = keyword_overlap * 10
                
                # Boost sentences with numbers or statistics
                if re.search(r'\d+%|\d+\.\d+|\b\d+\s+\w+\b', sentence):
                    score += 5
                
                # Boost sentences with key phrases
                key_phrases = ['significant', 'important', 'key finding', 'shows that', 'demonstrates']
                if any(phrase in sentence.lower() for phrase in key_phrases):
                    score += 3
                
                # Only keep sentences with meaningful scores
                if score > 0:
                    scored_sentences.append((sentence, score))
            
            # Sort by score and select top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = scored_sentences[:5]  # Limit to 5 key points per source
            
            # Store key points for this source
            points = []
            for sentence, score in top_sentences:
                points.append({
                    "text": sentence,
                    "relevance": score
                })
            
            key_points[source_id] = points
        
        return key_points

#=======================================================#
# Content Summarizer Class                              #
#=======================================================#

class ContentSummarizer:
    """Generate summaries from research content using various methods."""
    
    def __init__(self, config):
        """Initialize ContentSummarizer with configuration."""
        self.config = config
        self.method = config.get("summary_method", "textrank")
        self.model = config.get("summary_model", "gemma3:12b")
        self.depth = config.get("summary_depth", "medium")
        self.timeout = config.get("timeout", 60)
        self.max_content_length = 8000  # Maximum characters for summarization
    
    def summarize(self, source):
        """
        Generate a summary for a single source.
        
        Args:
            source: Source dictionary with text content
            
        Returns:
            Summary text
        """
        text = source.get("text", "")
        title = source.get("title", "")
        
        if not text:
            return "No content to summarize."
        
        # Truncate text if too long
        if len(text) > self.max_content_length:
            text = text[:self.max_content_length] + "... [text truncated]"
        
        # Choose appropriate summarization method
        if self.method == "textrank":
            return self._summarize_with_textrank(text, title)
        elif self.method == "ollama":
            return self._summarize_with_ollama(text, title, self.model)
        elif self.method == "openai":
            return self._summarize_with_openai(text, title)
        else:
            # Fallback to TextRank if method not recognized
            return self._summarize_with_textrank(text, title)
    
    def summarize_multiple(self, sources, query):
        """
        Generate summaries for multiple sources with batch processing.
        
        Args:
            sources: List of sources to summarize
            query: Research query for context
            
        Returns:
            Dictionary of source IDs mapped to summaries
        """
        if not sources:
            return {}
        
        # Convert to dict keyed by source_id
        source_dict = {s.get("source_id", f"S{i+1}"): s for i, s in enumerate(sources)}
        
        # Create results structure
        summaries = {}
        
        # Process in batches to avoid overloading
        batch_size = self.config.get("batch_size", 5)
        source_ids = list(source_dict.keys())
        
        for i in range(0, len(source_ids), batch_size):
            batch_ids = source_ids[i:i+batch_size]
            batch_sources = [source_dict[source_id] for source_id in batch_ids]
            
            # Summarize batch
            for source_id, source in zip(batch_ids, batch_sources):
                try:
                    summary = self.summarize(source)
                    summaries[source_id] = summary
                except Exception as e:
                    logger.error(f"Error summarizing source {source_id}: {str(e)}")
                    summaries[source_id] = f"Error generating summary: {str(e)[:100]}"
        
        return summaries
    
    def generate_research_summary(self, sources, key_points, query):
        """
        Generate a comprehensive research summary from multiple sources.
        
        Args:
            sources: List of analyzed sources
            key_points: Dictionary of key points by source ID
            query: Research query
            
        Returns:
            Comprehensive research summary
        """
        if not sources:
            return "No sources available to generate a research summary."
        
        # Extract top key points from all sources
        all_key_points = []
        for source_id, points in key_points.items():
            for point in points:
                source = next((s for s in sources if s.get("source_id") == source_id), None)
                if source:
                    all_key_points.append({
                        "text": point["text"],
                        "relevance": point["relevance"],
                        "source_id": source_id,
                        "source_type": source.get("content_type", "article"),
                        "source_title": source.get("title", "")
                    })
        
        # Sort key points by relevance
        all_key_points.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Create aggregated content for summarization
        if self.method == "textrank":
            return self._generate_aggregate_summary_textrank(all_key_points, sources, query)
        elif self.method == "ollama":
            return self._generate_aggregate_summary_ollama(all_key_points, sources, query)
        elif self.method == "openai":
            return self._generate_aggregate_summary_openai(all_key_points, sources, query)
        else:
            # Fallback to TextRank
            return self._generate_aggregate_summary_textrank(all_key_points, sources, query)
    
    def _summarize_with_textrank(self, text, title):
        """Summarize content using TextRank algorithm."""
        from textrankr import TextRank
        
        try:
            # Configure parameters based on depth
            ratio = 0.2  # Default ratio for medium depth
            if self.depth == "short":
                ratio = 0.1
            elif self.depth == "detailed":
                ratio = 0.3
            
            # Create the TextRank object
            textrank = TextRank()
            
            # Generate summary
            summarized = textrank.summarize(text, ratio=ratio)
            
            return summarized
        except Exception as e:
            logger.error(f"TextRank summarization error: {str(e)}")
            
            # Fallback to simple sentence extraction
            return self._fallback_summarize(text, title)
    
    def _fallback_summarize(self, text, title):
        """Simple fallback summarization when other methods fail."""
        # Split text into sentences
        try:
            if NLTK_AVAILABLE:
                sentences = nltk.sent_tokenize(text)
            else:
                sentences = re.split(r'(?<=[.!?])\s+', text)
        except:
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Skip if too few sentences
        if len(sentences) <= 3:
            return text
        
        # Select sentences based on position
        selected = []
        
        # Add first sentence (often contains key information)
        if sentences:
            selected.append(sentences[0])
        
        # Add some sentences from the middle (skipping first 20%)
        middle_start = max(1, int(len(sentences) * 0.2))
        middle_end = int(len(sentences) * 0.7)
        middle_sentences = sentences[middle_start:middle_end]
        
        # Take every Nth sentence based on desired length
        n = max(1, len(middle_sentences) // 3)
        selected.extend(middle_sentences[::n][:3])  # Take up to 3 sentences
        
        # Add a sentence from near the end (often contains conclusions)
        if len(sentences) > 5:
            selected.append(sentences[-2])  # Second-to-last often better than last
        
        # Join selected sentences
        summary = ' '.join(selected)
        
        return summary
    
    def _summarize_with_ollama(self, text, title, model):
        """Summarize content using Ollama API."""
        # Create prompt based on depth
        if self.depth == "short":
            prompt = f"""Create a concise 1-2 sentence summary of the following content.
Title: {title}

Content:
{text}

Summary:"""
        elif self.depth == "medium":
            prompt = f"""Create a clear, informative summary in 3-5 sentences of the following content.
Capture the main points and key details.

Title: {title}

Content:
{text}

Summary:"""
        else:  # detailed
            prompt = f"""Create a comprehensive summary of the following content.
Include main ideas, key supporting points, and important details.
The summary should be detailed but concise, focusing on the most important information.

Title: {title}

Content:
{text}

Detailed Summary:"""
        
        try:
            # Prepare API request
            api_url = self.config.get("ollama_api_url", "http://localhost:11434/api/generate")
            
            response = requests.post(
                api_url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result.get("response", "")
                return summary.strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return self._fallback_summarize(text, title)
                
        except Exception as e:
            logger.error(f"Ollama summarization error: {str(e)}")
            return self._fallback_summarize(text, title)
    
    def _summarize_with_openai(self, text, title):
        """Summarize content using OpenAI API."""
        api_key = self.config.get("openai_api_key", "")
        if not api_key:
            logger.warning("OpenAI API key not provided, falling back to TextRank")
            return self._summarize_with_textrank(text, title)
        
        # Construct system message based on depth
        if self.depth == "short":
            system_message = "You are a precise summarizer. Create a concise 1-2 sentence summary capturing the essential information."
        elif self.depth == "medium":
            system_message = "You are a skilled summarizer. Create a clear 3-5 sentence summary capturing main points and key context."
        else:  # detailed
            system_message = "You are an expert summarizer. Create a comprehensive summary including main ideas, important details, and key findings."
        
        try:
            # Prepare API request
            api_url = self.config.get("openai_api_url", "https://api.openai.com/v1/chat/completions")
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Title: {title}\n\nText to summarize:\n{text}"}
                ],
                "temperature": 0.3,
                "max_tokens": 800 if self.depth == "detailed" else 400
            }
            
            response = requests.post(
                api_url,
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result["choices"][0]["message"]["content"]
                return summary.strip()
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return self._fallback_summarize(text, title)
                
        except Exception as e:
            logger.error(f"OpenAI summarization error: {str(e)}")
            return self._fallback_summarize(text, title)
    
    def _generate_aggregate_summary_textrank(self, key_points, sources, query):
        """Generate aggregate summary using TextRank."""
        # Create a cohesive text combining key points
        aggregate_text = f"Research on: {query}\n\n"
        
        # Add content type breakdown
        content_types = Counter(s.get("content_type", "article") for s in sources)
        source_types = ", ".join(f"{count} {content_type}s" for content_type, count in content_types.most_common())
        aggregate_text += f"Based on {len(sources)} sources ({source_types}):\n\n"
        
        # Add top key points (limited by depth)
        point_limit = 10 if self.depth == "short" else (20 if self.depth == "medium" else 30)
        added_points = []
        
        for point in key_points[:point_limit]:
            point_text = point["text"]
            source_id = point["source_id"]
            
            # Avoid duplicate points
            if any(self._text_similarity(point_text, existing) > 0.7 for existing in added_points):
                continue
                
            aggregate_text += f"- {point_text} [{source_id}]\n"
            added_points.append(point_text)
        
        # Use TextRank on the aggregate text
        from textrankr import TextRank
        
        try:
            # Generate summary with parameters based on depth
            ratio = 0.3 if self.depth == "short" else (0.5 if self.depth == "medium" else 0.7)
            
            textrank = TextRank()
            summary = textrank.summarize(aggregate_text, ratio=ratio)
            
            # Add source reference
            summary += f"\n\nBased on {len(sources)} sources."
            
            return summary
            
        except Exception as e:
            logger.error(f"Aggregate TextRank error: {str(e)}")
            
            # Fallback: just return the aggregated text
            return aggregate_text
    
    def _generate_aggregate_summary_ollama(self, key_points, sources, query):
        """Generate aggregate summary using Ollama."""
        # Create prompt combining key points and source information
        prompt = f"""Generate a comprehensive research summary on: "{query}"

Based on {len(sources)} sources including:
"""
        
        # Add content type breakdown
        content_types = Counter(s.get("content_type", "article") for s in sources)
        for content_type, count in content_types.most_common():
            prompt += f"- {count} {content_type}s\n"
        
        prompt += "\nKey points from sources:\n"
        
        # Add top key points, grouped by source
        source_points = defaultdict(list)
        for point in key_points:
            source_id = point["source_id"]
            source_points[source_id].append(point)
        
        # Add points by source (limited by depth)
        source_limit = 5 if self.depth == "short" else (8 if self.depth == "medium" else 12)
        point_limit = 3 if self.depth == "short" else (5 if self.depth == "medium" else 8)
        
        for i, (source_id, points) in enumerate(source_points.items()):
            if i >= source_limit:
                break
                
            source = next((s for s in sources if s.get("source_id") == source_id), None)
            if source:
                prompt += f"\nFrom {source_id} ({source.get('content_type', 'article')}): {source.get('title', '')}\n"
                
                for j, point in enumerate(points):
                    if j >= point_limit:
                        break
                    prompt += f"- {point['text']}\n"
        
        # Add summary instructions based on depth
        if self.depth == "short":
            prompt += "\nCreate a brief, focused summary (3-5 sentences) highlighting the most important findings."
        elif self.depth == "medium":
            prompt += "\nCreate a comprehensive summary (1-2 paragraphs) synthesizing the key information and main conclusions."
        else:  # detailed
            prompt += "\nCreate a detailed research summary (2-3 paragraphs) that thoroughly covers the main findings, supporting evidence, and different perspectives."
        
        prompt += "\nInclude source references like [S1], [S2], etc. where appropriate."
        
        try:
            # Call Ollama API
            api_url = self.config.get("ollama_api_url", "http://localhost:11434/api/generate")
            
            response = requests.post(
                api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3
                    }
                },
                timeout=self.timeout * 2  # Give more time for aggregate summary
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return self._generate_aggregate_summary_textrank(key_points, sources, query)
                
        except Exception as e:
            logger.error(f"Ollama aggregate summary error: {str(e)}")
            return self._generate_aggregate_summary_textrank(key_points, sources, query)
    
    def _generate_aggregate_summary_openai(self, key_points, sources, query):
        """Generate aggregate summary using OpenAI."""
        api_key = self.config.get("openai_api_key", "")
        if not api_key:
            logger.warning("OpenAI API key not provided, falling back to TextRank")
            return self._generate_aggregate_summary_textrank(key_points, sources, query)
        
        # Create messages combining key points and source information
        system_message = f"""You are an expert research assistant preparing a comprehensive summary on: "{query}"
Your task is to synthesize information from multiple sources into a coherent research summary.
Include appropriate source citations [S1], [S2], etc.
"""
        
        user_message = f"Research query: {query}\n\n"
        user_message += f"Based on {len(sources)} sources including:\n"
        
        # Add content type breakdown
        content_types = Counter(s.get("content_type", "article") for s in sources)
        for content_type, count in content_types.most_common():
            user_message += f"- {count} {content_type}s\n"
        
        user_message += "\nKey points from sources:\n"
        
        # Add top key points, grouped by source
        source_points = defaultdict(list)
        for point in key_points:
            source_id = point["source_id"]
            source_points[source_id].append(point)
        
        # Add points by source (limited by depth)
        source_limit = 5 if self.depth == "short" else (8 if self.depth == "medium" else 12)
        point_limit = 3 if self.depth == "short" else (5 if self.depth == "medium" else 8)
        
        for i, (source_id, points) in enumerate(source_points.items()):
            if i >= source_limit:
                break
                
            source = next((s for s in sources if s.get("source_id") == source_id), None)
            if source:
                user_message += f"\nFrom {source_id} ({source.get('content_type', 'article')}): {source.get('title', '')}\n"
                
                for j, point in enumerate(points):
                    if j >= point_limit:
                        break
                    user_message += f"- {point['text']}\n"
        
        # Add summary instructions based on depth
        if self.depth == "short":
            user_message += "\nCreate a brief, focused summary (3-5 sentences) highlighting the most important findings."
        elif self.depth == "medium":
            user_message += "\nCreate a comprehensive summary (1-2 paragraphs) synthesizing the key information and main conclusions."
        else:  # detailed
            user_message += "\nCreate a detailed research summary (2-3 paragraphs) that thoroughly covers the main findings, supporting evidence, and different perspectives."
        
        try:
            # Prepare API request
            api_url = self.config.get("openai_api_url", "https://api.openai.com/v1/chat/completions")
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.3,
                "max_tokens": 1000 if self.depth == "detailed" else 500
            }
            
            response = requests.post(
                api_url,
                headers=headers,
                json=data,
                timeout=self.timeout * 2  # Give more time for aggregate summary
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return self._generate_aggregate_summary_textrank(key_points, sources, query)
                
        except Exception as e:
            logger.error(f"OpenAI aggregate summary error: {str(e)}")
            return self._generate_aggregate_summary_textrank(key_points, sources, query)
    
    def _text_similarity(self, text1, text2):
        """Calculate similarity between two texts (simplified)."""
        # Tokenize texts and create sets of words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0
        
        return intersection / union

#=======================================================#
# Research Report Generator Class                       #
#=======================================================#

class ReportGenerator:
    """Generate structured research reports."""
    
    def __init__(self, config):
        """Initialize the ReportGenerator with configuration."""
        self.config = config
        self.output_format = config.get("output_format", "markdown")
    
    def generate_report(self, query, sources, summaries, overall_summary, hierarchy=None):
        """
        Generate a comprehensive research report.
        
        Args:
            query: Research query
            sources: List of analyzed sources
            summaries: Dictionary of summaries by source ID
            overall_summary: Overall research summary
            hierarchy: Source hierarchy information
            
        Returns:
            Formatted research report
        """
        if self.output_format == "markdown":
            return self._generate_markdown_report(query, sources, summaries, overall_summary, hierarchy)
        elif self.output_format == "text":
            return self._generate_text_report(query, sources, summaries, overall_summary, hierarchy)
        elif self.output_format == "json":
            return self._generate_json_report(query, sources, summaries, overall_summary, hierarchy)
        else:
            # Default to markdown
            return self._generate_markdown_report(query, sources, summaries, overall_summary, hierarchy)
    
    def _generate_markdown_report(self, query, sources, summaries, overall_summary, hierarchy):
        """Generate a markdown-formatted research report."""
        # Get current date/time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Initialize report
        report = f"# Research Report: {query}\n\n"
        report += f"*Generated on: {timestamp}*\n\n"
        
        # Add metadata section
        report += "## Research Metadata\n\n"
        report += f"- **Query:** {query}\n"
        report += f"- **Sources:** {len(sources)}\n"
        
        # Count source types
        content_types = Counter(s.get("content_type", "article") for s in sources)
        source_types = ", ".join(f"{count} {content_type}s" for content_type, count in content_types.most_common())
        report += f"- **Source Types:** {source_types}\n"
        
        # Calculate average quality and credibility
        avg_quality = sum(s.get("quality", 0) for s in sources) / len(sources) if sources else 0
        avg_credibility = sum(s.get("credibility", 0) for s in sources) / len(sources) if sources else 0
        report += f"- **Average Source Quality:** {avg_quality:.1f}/100\n"
        report += f"- **Average Source Credibility:** {avg_credibility:.1f}/100\n\n"
        
        # Add executive summary
        report += "## Executive Summary\n\n"
        report += overall_summary + "\n\n"
        
        # Add key findings section if there are sources
        if sources:
            report += "## Key Findings\n\n"
            
            # Extract key points from high-quality sources
            top_sources = sorted(sources, key=lambda s: s.get("score", 0), reverse=True)[:5]
            
            for source in top_sources:
                source_id = source.get("source_id", "")
                if not source_id or source_id not in summaries:
                    continue
                
                # Get first sentence of summary as key finding
                summary = summaries[source_id]
                first_sentence = re.split(r'(?<=[.!?])\s+', summary)[0] if summary else ""
                
                if first_sentence:
                    report += f"- {first_sentence} [{source_id}]\n"
            
            report += "\n"
        
        # Add detailed source summaries
        report += "## Source Summaries\n\n"
        
        # Group sources by type for better organization
        sources_by_type = defaultdict(list)
        for source in sources:
            content_type = source.get("content_type", "article")
            sources_by_type[content_type].append(source)
        
        # Add summaries by source type
        for content_type, type_sources in sorted(sources_by_type.items()):
            report += f"### {content_type.replace('_', ' ').title()}s\n\n"
            
            for source in sorted(type_sources, key=lambda s: s.get("score", 0), reverse=True):
                source_id = source.get("source_id", "")
                if not source_id or source_id not in summaries:
                    continue
                
                title = source.get("title", "Untitled")
                url = source.get("url", "")
                summary = summaries[source_id]
                
                report += f"#### [{source_id}] {title}\n\n"
                report += f"**URL:** [{url}]({url})\n\n"
                report += f"**Summary:**\n{summary}\n\n"
                
                # Add source hierarchy if available
                if hierarchy and self.config.get("include_source_hierarchy", True):
                    parent_url = source.get("source_info", {}).get("parent")
                    if parent_url:
                        parent_source = next((s for s in sources if s.get("url") == parent_url), None)
                        if parent_source:
                            parent_id = parent_source.get("source_id", "Unknown")
                            report += f"**Discovered from:** [{parent_id}] {parent_source.get('title', 'Unknown')}\n\n"
                
                report += "---\n\n"
        
        # Add source hierarchy section if available
        if hierarchy and self.config.get("include_source_hierarchy", True):
            report += "## Source Hierarchy\n\n"
            report += "This diagram shows how sources were discovered during research:\n\n"
            
            # Generate a simple hierarchy visualization
            report += "```\n"
            for url, children in hierarchy.items():
                source = next((s for s in sources if s.get("url") == url), None)
                if source:
                    source_id = source.get("source_id", "Unknown")
                    report += f"{source_id}\n"
                    
                    for child_url in children:
                        child_source = next((s for s in sources if s.get("url") == child_url), None)
                        if child_source:
                            child_id = child_source.get("source_id", "Unknown")
                            report += f"└── {child_id}\n"
            report += "```\n\n"
        
        # Add research methodology
        report += "## Research Methodology\n\n"
        report += "This report was generated using Deep Research Assistant, which:\n\n"
        report += "1. Performed a comprehensive search for information related to the query\n"
        report += "2. Analyzed and prioritized sources based on relevance, credibility, and quality\n"
        report += f"3. Extracted and processed content from {len(sources)} distinct sources\n"
        report += "4. Generated summaries for each source using " + self._get_method_description()
        report += "5. Synthesized information into a comprehensive research report\n\n"
        
        # Add footer
        report += "---\n"
        report += f"*Generated by Deep Research Assistant v{__version__} on {timestamp}*"
        
        return report
    
    def _generate_text_report(self, query, sources, summaries, overall_summary, hierarchy):
        """Generate a plain text research report."""
        # Get current date/time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Initialize report
        report = f"RESEARCH REPORT: {query}\n"
        report += "=" * 80 + "\n\n"
        report += f"Generated on: {timestamp}\n\n"
        
        # Add metadata section
        report += "RESEARCH METADATA\n"
        report += "-" * 80 + "\n"
        report += f"Query: {query}\n"
        report += f"Sources: {len(sources)}\n"
        
        # Count source types
        content_types = Counter(s.get("content_type", "article") for s in sources)
        source_types = ", ".join(f"{count} {content_type}s" for content_type, count in content_types.most_common())
        report += f"Source Types: {source_types}\n\n"
        
        # Add executive summary
        report += "EXECUTIVE SUMMARY\n"
        report += "-" * 80 + "\n"
        report += overall_summary + "\n\n"
        
        # Add key findings section if there are sources
        if sources:
            report += "KEY FINDINGS\n"
            report += "-" * 80 + "\n"
            
            # Extract key points from high-quality sources
            top_sources = sorted(sources, key=lambda s: s.get("score", 0), reverse=True)[:5]
            
            for source in top_sources:
                source_id = source.get("source_id", "")
                if not source_id or source_id not in summaries:
                    continue
                
                # Get first sentence of summary as key finding
                summary = summaries[source_id]
                first_sentence = re.split(r'(?<=[.!?])\s+', summary)[0] if summary else ""
                
                if first_sentence:
                    report += f"- {first_sentence} [{source_id}]\n"
            
            report += "\n"
        
        # Add detailed source summaries
        report += "SOURCE SUMMARIES\n"
        report += "-" * 80 + "\n\n"
        
        # Group sources by type for better organization
        sources_by_type = defaultdict(list)
        for source in sources:
            content_type = source.get("content_type", "article")
            sources_by_type[content_type].append(source)
        
        # Add summaries by source type
        for content_type, type_sources in sorted(sources_by_type.items()):
            report += f"{content_type.replace('_', ' ').upper()}S\n"
            report += "-" * 40 + "\n\n"
            
            for source in sorted(type_sources, key=lambda s: s.get("score", 0), reverse=True):
                source_id = source.get("source_id", "")
                if not source_id or source_id not in summaries:
                    continue
                
                title = source.get("title", "Untitled")
                url = source.get("url", "")
                summary = summaries[source_id]
                
                report += f"[{source_id}] {title}\n"
                report += f"URL: {url}\n\n"
                report += f"Summary:\n{summary}\n\n"
                
                # Add source hierarchy if available
                if hierarchy and self.config.get("include_source_hierarchy", True):
                    parent_url = source.get("source_info", {}).get("parent")
                    if parent_url:
                        parent_source = next((s for s in sources if s.get("url") == parent_url), None)
                        if parent_source:
                            parent_id = parent_source.get("source_id", "Unknown")
                            report += f"Discovered from: [{parent_id}] {parent_source.get('title', 'Unknown')}\n"
                
                report += "-" * 40 + "\n\n"
        
        # Add research methodology
        report += "RESEARCH METHODOLOGY\n"
        report += "-" * 80 + "\n\n"
        report += "This report was generated using Deep Research Assistant, which:\n\n"
        report += "1. Performed a comprehensive search for information related to the query\n"
        report += "2. Analyzed and prioritized sources based on relevance, credibility, and quality\n"
        report += f"3. Extracted and processed content from {len(sources)} distinct sources\n"
        report += "4. Generated summaries for each source using " + self._get_method_description() + "\n"
        report += "5. Synthesized information into a comprehensive research report\n\n"
        
        # Add footer
        report += "-" * 80 + "\n"
        report += f"Generated by Deep Research Assistant v{__version__} on {timestamp}"
        
        return report
    
    def _generate_json_report(self, query, sources, summaries, overall_summary, hierarchy):
        """Generate a JSON-formatted research report."""
        # Get current date/time
        timestamp = datetime.now().isoformat()
        
        # Initialize report structure
        report = {
            "metadata": {
                "query": query,
                "timestamp": timestamp,
                "version": __version__,
                "source_count": len(sources)
            },
            "executive_summary": overall_summary,
            "sources": [],
            "methodology": {
                "steps": [
                    "Performed a comprehensive search for information related to the query",
                    "Analyzed and prioritized sources based on relevance, credibility, and quality",
                    f"Extracted and processed content from {len(sources)} distinct sources",
                    "Generated summaries for each source using " + self._get_method_description(),
                    "Synthesized information into a comprehensive research report"
                ],
                "summarization_method": self.config.get("summary_method", "textrank"),
                "summarization_model": self.config.get("summary_model", "")
            }
        }
        
        # Add source data
        for source in sources:
            source_id = source.get("source_id", "")
            if not source_id or source_id not in summaries:
                continue
                
            source_data = {
                "id": source_id,
                "url": source.get("url", ""),
                "title": source.get("title", ""),
                "domain": source.get("domain", ""),
                "content_type": source.get("content_type", "article"),
                "quality": source.get("quality", 0),
                "credibility": source.get("credibility", 0),
                "relevance": source.get("relevance", 0),
                "score": source.get("score", 0),
                "summary": summaries.get(source_id, "")
            }
            
            # Add source hierarchy if available
            if hierarchy and self.config.get("include_source_hierarchy", True):
                parent_url = source.get("source_info", {}).get("parent")
                if parent_url:
                    parent_source = next((s for s in sources if s.get("url") == parent_url), None)
                    if parent_source:
                        parent_id = parent_source.get("source_id", "Unknown")
                        source_data["parent_source"] = parent_id
            
            report["sources"].append(source_data)
        
        # Add hierarchy data if available
        if hierarchy and self.config.get("include_source_hierarchy", True):
            report["source_hierarchy"] = {}
            for url, children in hierarchy.items():
                source = next((s for s in sources if s.get("url") == url), None)
                if source:
                    source_id = source.get("source_id", "Unknown")
                    children_ids = []
                    
                    for child_url in children:
                        child_source = next((s for s in sources if s.get("url") == child_url), None)
                        if child_source:
                            child_id = child_source.get("source_id", "Unknown")
                            children_ids.append(child_id)
                    
                    report["source_hierarchy"][source_id] = children_ids
        
        # Convert to JSON string with nice formatting
        return json.dumps(report, indent=2)
    
    def _get_method_description(self):
        """Get a description of the summarization method used."""
        method = self.config.get("summary_method", "textrank")
        model = self.config.get("summary_model", "")
        
        if method == "textrank":
            return "TextRank algorithm for natural language processing"
        elif method == "ollama":
            return f"Ollama AI model ({model})"
        elif method == "openai":
            return f"OpenAI model ({model})"
        else:
            return "advanced AI-powered summarization"

#=======================================================#
# Main Research Assistant Class                         #
#=======================================================#

class DeepResearchAssistant:
    """Main class that orchestrates the entire research process."""
    
    def __init__(self, config=None):
        """Initialize the research assistant with configuration."""
        # Use provided config or defaults
        self.config = config or DEFAULT_CONFIG.copy()
        
        # Set up logging level based on config
        log_level = logging.DEBUG if self.config.get("debug", False) else (
            logging.INFO if self.config.get("verbose", False) else logging.WARNING)
        logger.setLevel(log_level)
        
        # Create output directory if it doesn't exist
        output_dir = self.config.get("output_dir", "research_results")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def research(self, query):
        """
        Execute the full research process for a query.
        
        Args:
            query: The research query
            
        Returns:
            Dictionary with research results and output file path
        """
        print_header(f"DEEP RESEARCH ASSISTANT - {query}")
        
        start_time = time.time()
        
        # Scale parameters based on depth
        depth = self.config.get("depth", 3)
        self.config["max_sources"] = self.config.get("source_count_by_depth", {}).get(
            depth, self.config.get("max_sources", 25))
        self.config["crawl_depth"] = self.config.get("crawl_depth_by_depth", {}).get(
            depth, self.config.get("crawl_depth", 2))
        
        # Initialize components
        searcher = WebSearch(self.config)
        crawler = WebCrawler(self.config)
        extractor = ContentExtractor(self.config)
        analyzer = SourceAnalyzer(self.config)
        summarizer = ContentSummarizer(self.config)
        report_generator = ReportGenerator(self.config)
        
        # Step 1: Search for information
        print("\n[1/5] Searching for relevant information...")
        search_results = searcher.search(query, depth)
        
        if not search_results:
            print("No search results found. Please try a different query.")
            return {"success": False, "error": "No search results found."}
        
        print(f"Found {len(search_results)} search results.")
        
        # Step 2: Crawl search results to discover additional sources
        print("\n[2/5] Discovering additional sources through crawling...")
        
        # Extract URLs from search results
        initial_urls = [result.get("url") for result in search_results if result.get("url")]
        
        # Limit initial URLs based on depth
        max_initial = min(10, depth * 2)
        initial_urls = initial_urls[:max_initial]
        
        # Crawl URLs to discover more sources
        crawl_results = crawler.crawl_urls_with_hierarchy(
            initial_urls, 
            max_depth=self.config.get("crawl_depth", 2),
            max_per_source=self.config.get("max_links_per_source", 3)
        )
        
        # Combine initial and discovered URLs (with deduplication)
        all_urls = set(initial_urls)
        if "crawled_urls" in crawl_results:
            all_urls.update(crawl_results["crawled_urls"])
        
        # Get source hierarchy
        source_hierarchy = crawl_results.get("source_hierarchy", {})
        
        print(f"Discovered {len(all_urls)} total sources through search and crawling.")
        
        # Step 3: Extract content from sources
        print("\n[3/5] Extracting content from sources...")
        
        # Limit the number of sources to extract based on config
        max_sources = self.config.get("max_sources", 25)
        if len(all_urls) > max_sources:
            print(f"Limiting to {max_sources} sources based on configuration.")
            # Prioritize initial URLs, then add discovered URLs
            extract_urls = list(initial_urls)
            for url in all_urls:
                if url not in extract_urls:
                    extract_urls.append(url)
                    if len(extract_urls) >= max_sources:
                        break
        else:
            extract_urls = list(all_urls)
        
        # Extract content from selected URLs
        print(f"Extracting content from {len(extract_urls)} sources...")
        extracted_contents = extractor.extract_batch(extract_urls, source_hierarchy)
        
        # Count successful extractions
        successful_extractions = [c for c in extracted_contents if c.get("text") and not c.get("error")]
        print(f"Successfully extracted content from {len(successful_extractions)} sources.")
        
        # Step 4: Analyze and organize content
        print("\n[4/5] Analyzing and organizing content...")
        analyzed_sources = analyzer.analyze_sources(successful_extractions, query)
        
        # Deduplicate sources
        unique_sources = analyzer.deduplicate_sources(analyzed_sources)
        
        # Extract key points from sources
        key_points = analyzer.extract_key_points(unique_sources, query)
        
        # Step 5: Generate summaries and research report
        print("\n[5/5] Generating summaries and research report...")
        
        # Generate individual source summaries
        source_summaries = summarizer.summarize_multiple(unique_sources, query)
        
        # Generate overall research summary
        overall_summary = summarizer.generate_research_summary(unique_sources, key_points, query)
        
        # Generate research report
        print("Creating final research report...")
        report = report_generator.generate_report(
            query, unique_sources, source_summaries, overall_summary, source_hierarchy
        )
        
        # Generate output filename and save report
        output_format = self.config.get("output_format", "markdown")
        output_dir = self.config.get("output_dir", "research_results")
        output_file = generate_output_filename(query, output_format, output_dir)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        minutes, seconds = divmod(int(execution_time), 60)
        
        print_header("RESEARCH COMPLETE!")
        print(f"Research results saved to: {output_file}")
        print(f"Total research time: {minutes} minutes, {seconds} seconds")
        print(f"Sources analyzed: {len(unique_sources)}")
        
        return {
            "success": True,
            "query": query,
            "output_file": output_file,
            "execution_time": execution_time,
            "source_count": len(unique_sources)
        }

#=======================================================#
# Command-line interface                                #
#=======================================================#

def setup_argparse():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deep Research Assistant - A comprehensive research tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("query", type=str, nargs="?", 
                        help="Research query or topic to investigate")
    
    # Optional arguments
    parser.add_argument("--depth", type=int, choices=range(1, 6), default=3,
                        help="Research depth (1-5): 1=quick overview, 5=comprehensive deep dive")
    parser.add_argument("--method", choices=["textrank", "ollama", "openai"], default="textrank",
                        help="Summarization method to use")
    parser.add_argument("--model", type=str, default="gemma3:12b",
                        help="Model to use with Ollama or OpenAI")
    parser.add_argument("--summary-depth", choices=["short", "medium", "detailed"], default="medium",
                        help="Level of detail in summaries")
    parser.add_argument("--format", choices=["markdown", "text", "json"], default="markdown",
                        help="Output format for the research report")
    parser.add_argument("--output-dir", type=str, default="research_results",
                        help="Directory for output files")
    parser.add_argument("--max-sources", type=int, default=0,
                        help="Maximum number of sources to analyze (0 for auto-scaling based on depth)")
    parser.add_argument("--exclude-domains", type=str, nargs="+", default=[],
                        help="Domains to exclude from search/crawl results")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434/api/generate",
                        help="URL for Ollama API (if using Ollama)")
    parser.add_argument("--openai-key", type=str, default="",
                        help="OpenAI API key (if using OpenAI)")
    parser.add_argument("--photon-path", type=str, default="./photon.py",
                        help="Path to photon.py script")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--version", action="version", version=f"Deep Research Assistant v{__version__}")
    
    return parser.parse_args()

def main():
    """Main entry point for the Deep Research Assistant."""
    # Parse command line arguments
    args = setup_argparse()
    
    # If no query provided, prompt the user
    if not args.query:
        print("Welcome to Deep Research Assistant!")
        args.query = input("Enter your research query: ").strip()
        if not args.query:
            print("No query provided. Exiting.")
            return 1
    
    # Create configuration from arguments
    config = DEFAULT_CONFIG.copy()
    config.update({
        "depth": args.depth,
        "summary_method": args.method,
        "summary_model": args.model,
        "summary_depth": args.summary_depth,
        "output_format": args.format,
        "output_dir": args.output_dir,
        "exclude_domains": args.exclude_domains,
        "photon_path": args.photon_path,
        "ollama_api_url": args.ollama_url,
        "openai_api_key": args.openai_key,
        "verbose": args.verbose,
        "debug": args.debug
    })
    
    # Set max_sources if specified
    if args.max_sources > 0:
        config["max_sources"] = args.max_sources
    
    # Initialize the research assistant
    assistant = DeepResearchAssistant(config)
    
    # Execute research
    try:
        result = assistant.research(args.query)
        if result["success"]:
            return 0
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 1
    except KeyboardInterrupt:
        print("\nResearch interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error during research: {str(e)}")
        if config.get("debug", False):
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())