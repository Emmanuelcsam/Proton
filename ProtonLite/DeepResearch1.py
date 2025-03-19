#!/usr/bin/env python3
"""
DeepResearch1.py - Core CLI and Orchestration Module

This script provides the command-line interface, configuration management,
and orchestration for the DeepResearch tool - an advanced research assistant
that combines enhanced search, content extraction, analysis, and AI summarization.

Usage:
  python DeepResearch1.py "your research query"
  python DeepResearch1.py --depth 4 --focus academic "quantum computing applications"
"""

import os
import sys
import json
import time
import argparse
import logging
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, unquote_plus

# Import the other DeepResearch modules
# Use explicit imports to avoid module attribute errors
from DeepResearch2 import WebSearch, WebCrawler, ContentExtractor
from DeepResearch3 import ContentAnalyzer, OllamaSummarizer, ReportGenerator

__version__ = "1.0.0"

# Configuration constants
CONFIG = {
    # Research parameters (1-5 scale)
    "depth": 3,
    "focus": "general",  # academic, news, technical, general
    "max_sources": 15,
    "exclude_domains": [],
    "include_urls": [],
    
    # Search settings
    "search_engine": "duckduckgo",
    "max_search_results": 25,
    "search_time_limit": "year",
    
    # Crawler settings
    "crawl_depth": 1,
    "max_links_per_page": 5,
    "respect_robots_txt": True,
    
    # Content extraction settings
    "extract_images": False,
    "min_content_length": 500,
    "max_content_length": 25000,
    
    # Analysis settings
    "detect_language": True,
    "calculate_readability": True,
    "credibility_check": True,
    "duplicate_threshold": 0.85,
    
    # Summarization settings
    "ollama_model": "gemma3:12b",
    "ollama_api_url": "http://localhost:11434/api/generate",
    "use_openai": False,
    "openai_api_key": "",
    "openai_api_url": "https://api.openai.com/v1/chat/completions",
    "openai_model": "gpt-3.5-turbo",
    
    # Output settings
    "output_format": "markdown",
    "output_file": "",
    "include_sources": True,
    "include_snippets": True,
    
    # Performance settings
    "timeout": 30,
    "max_retries": 3,
    "max_threads": 8,
    "max_time": 1800,  # 30 minutes max research time
    
    # UI settings
    "show_progress": True,
    "debug": False,
    
    # Scaling parameters based on depth
    "source_count_by_depth": {
        1: 5,    # Quick overview
        2: 10,   # Basic research
        3: 15,   # Comprehensive
        4: 25,   # In-depth
        5: 40    # Exhaustive
    },
    "estimated_time_by_depth": {
        1: 300,   # 5 minutes
        2: 600,   # 10 minutes
        3: 900,   # 15 minutes
        4: 1500,  # 25 minutes
        5: 2700   # 45 minutes
    },
    "summary_length_by_depth": {
        1: 500,   # Short summary
        2: 1000,  # Medium summary
        3: 2000,  # Detailed summary
        4: 3500,  # Comprehensive report
        5: 5000   # Extensive report
    }
}

# Domain credibility database - simplified version
DOMAIN_CREDIBILITY = {
    "edu": 90, "gov": 85, "org": 75, "ac.uk": 90, 
    "nytimes.com": 85, "bbc.com": 85, "reuters.com": 90,
    "nature.com": 95, "science.org": 95, "ieee.org": 90,
    "github.com": 80, "stackoverflow.com": 75,
    "wikipedia.org": 75
}

# Content type priorities
CONTENT_TYPE_PRIORITY = {
    "research_paper": 100, "news_article": 85, "documentation": 85,
    "tutorial": 80, "blog_post": 75, "q_and_a": 70
}

def setup_logging(level=logging.INFO):
    """Set up logging with consistent formatting."""
    logger = logging.getLogger("DeepResearch")
    logger.setLevel(level)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add formatter to console handler
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    return logger

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DeepResearch: Advanced Research Assistant for comprehensive topic exploration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("query", type=str, nargs="?", help="Research query or topic to investigate")
    
    # Optional arguments
    parser.add_argument("--depth", type=int, default=3, choices=range(1, 6),
                        help="Research depth (1-5): 1=quick overview, 5=comprehensive deep dive")
    parser.add_argument("--sources", type=int, default=0,
                        help="Maximum number of sources to analyze (0 for automatic based on depth)")
    parser.add_argument("--format", choices=["text", "markdown", "json"], default="markdown",
                        help="Output format for the research report")
    parser.add_argument("--focus", choices=["academic", "news", "technical", "general"],
                        default="general", help="Research focus area to prioritize sources")
    parser.add_argument("--model", type=str, default="gemma3:12b", 
                        help="Ollama model to use for summarization")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Timeout for individual web requests in seconds")
    parser.add_argument("--max-time", type=int, default=1800,
                        help="Maximum total research time in seconds (0 for unlimited)")
    parser.add_argument("--include-urls", type=str, nargs="+", default=[],
                        help="List of specific URLs to analyze")
    parser.add_argument("--exclude-domains", type=str, nargs="+", default=[],
                        help="Domains to exclude from search results")
    parser.add_argument("--output", type=str, default="",
                        help="Output file for the research report (default: auto-generated)")
    parser.add_argument("--openai-key", type=str, default="",
                        help="OpenAI API key for enhanced summarization (optional)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-progress", action="store_true", 
                        help="Disable progress reporting during research")
    parser.add_argument("--version", action="version", version=f"DeepResearch v{__version__}")
    
    return parser.parse_args()

def generate_output_filename(query, format="markdown"):
    """Generate a filename for the research output based on the query."""
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
    
    return f"research_{clean_query[:40]}_{timestamp}.{ext}"

def print_research_plan(query, config, depth):
    """Print the research plan based on configuration and depth."""
    depth_descriptions = {
        1: "Quick overview with essential information",
        2: "Basic research with key details and context",
        3: "Comprehensive analysis with multiple perspectives",
        4: "In-depth investigation with detailed analysis",
        5: "Exhaustive deep dive with extensive source exploration"
    }
    
    source_count = config['source_count_by_depth'][depth]
    estimated_time = config['estimated_time_by_depth'][depth]
    
    print("\n" + "=" * 80)
    print(f"DeepResearch: Advanced Research Plan for '{query}'")
    print("=" * 80)
    print(f"Research depth:     {depth}/5 - {depth_descriptions[depth]}")
    print(f"Sources to analyze: Up to {source_count} relevant sources")
    print(f"Research focus:     {config['focus'].capitalize()}")
    print(f"Estimated time:     {estimated_time//60}-{(estimated_time+120)//60} minutes")
    print(f"Output format:      {config['output_format'].capitalize()}")
    print(f"AI model:           {config['ollama_model']}")
    print(f"Output file:        {config['output_file']}")
    print("=" * 80 + "\n")

def perform_research(query, config):
    """Execute the complete research pipeline."""
    start_time = time.time()
    max_time = config['max_time']
    
    # Initialize components
    search = WebSearch(config)
    crawler = WebCrawler(config)
    extractor = ContentExtractor(config)
    analyzer = ContentAnalyzer(config)
    summarizer = OllamaSummarizer(config)
    report_generator = ReportGenerator(config)
    
    # Research stages
    try:
        # Stage 1: Initial search and source discovery
        print(f"[1/5] Searching for information on: {query}")
        search_results = search.execute_search(query)
        
        if not search_results:
            print("No search results found. Please try a different query.")
            return False
        
        # Stage 2: Prioritize and filter sources
        print(f"[2/5] Analyzing and prioritizing {len(search_results)} potential sources")
        prioritized_sources = analyzer.prioritize_sources(search_results, query)
        
        max_sources = min(config['max_sources'], len(prioritized_sources))
        selected_sources = prioritized_sources[:max_sources]
        print(f"Selected {len(selected_sources)} most relevant sources for in-depth analysis")
        
        # Stage 3: Content extraction and link discovery
        print("[3/5] Extracting detailed content from sources")
        contents = []
        
        # Handle included URLs if specified
        if config.get('include_urls'):
            print(f"Adding {len(config['include_urls'])} user-specified URLs")
            for url in config['include_urls']:
                if dr2.is_valid_url(url):
                    selected_sources.append({
                        'url': url,
                        'title': 'User Specified URL',
                        'snippet': '',
                        'source': 'user_input'
                    })
        
        with ThreadPoolExecutor(max_workers=config.get('max_threads', 5)) as executor:
            future_to_url = {
                executor.submit(extractor.extract_content, source['url']): source 
                for source in selected_sources
            }
            
            for i, future in enumerate(as_completed(future_to_url), 1):
                source = future_to_url[future]
                try:
                    content = future.result()
                    if content and content.get('text'):
                        # Merge the content with the source information
                        merged_content = {**source, **content}
                        contents.append(merged_content)
                        print(f"  ✓ [{i}/{len(selected_sources)}] Extracted content from: {merged_content.get('title', 'Unknown')}")
                    else:
                        print(f"  ✗ [{i}/{len(selected_sources)}] Failed to extract usable content from: {source.get('url', 'Unknown URL')}")
                        
                except Exception as e:
                    print(f"  ✗ [{i}/{len(selected_sources)}] Error extracting content from {source.get('url', 'Unknown URL')}: {str(e)}")
                
                # Check time limit
                if max_time > 0 and (time.time() - start_time) > max_time:
                    print("Reached maximum research time limit. Proceeding with collected information.")
                    break
        
        # Check if we have enough content to proceed
        if len(contents) < 2:
            if len(contents) == 0:
                print("Failed to extract content from any sources. Cannot generate a research report.")
                return False
            print("Warning: Limited content sources available. Research report may be incomplete.")
        
        # Stage 4: Content analysis, deduplication, and organization
        print("[4/5] Analyzing and organizing extracted information")
        processed_content = analyzer.process_content(contents, query)
        
        # Stage 5: Generate insights and create research report
        print("[5/5] Generating comprehensive research report")
        summary = summarizer.generate_summary(processed_content, query)
        
        report = report_generator.create_report(
            query=query,
            summary=summary,
            contents=processed_content,
            execution_time=time.time() - start_time
        )
        
        # Save the report
        with open(config['output_file'], 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nResearch complete! Report saved to: {config['output_file']}")
        print(f"Total research time: {(time.time() - start_time)/60:.1f} minutes")
        
        return True
    
    except KeyboardInterrupt:
        print("\nResearch interrupted by user. Saving partial results...")
        # Try to save partial results if available
        try:
            if 'processed_content' in locals() and processed_content:
                partial_report = report_generator.create_report(
                    query=query,
                    summary={"text": "Research was interrupted before completion. Results may be incomplete."},
                    contents=processed_content,
                    execution_time=time.time() - start_time,
                    is_complete=False
                )
                
                partial_output = config['output_file'].replace('.', '_partial.')
                with open(partial_output, 'w', encoding='utf-8') as f:
                    f.write(partial_report)
                print(f"Partial results saved to: {partial_output}")
        except Exception as e:
            print(f"Unable to save partial results: {str(e)}")
        
        return False
    
    except Exception as e:
        print(f"An error occurred during research: {str(e)}")
        if config['debug']:
            import traceback
            traceback.print_exc()
        return False

# Text and URL utility functions
def clean_text(text):
    """Clean text by removing extra whitespace, HTML tags, etc."""
    if not text:
        return ""
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&\w+;', ' ', text)  # Simple HTML entity removal
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_keywords(text):
    """Extract keywords from text."""
    if not text:
        return []
    text = clean_text(text).lower()
    # Basic stopwords
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by'}
    words = re.findall(r'\b\w+\b', text)
    return [w for w in words if w not in stopwords and len(w) > 2]

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
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return ""

def main():
    """Main entry point for the DeepResearch tool."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    setup_logging(log_level)
    
    # Load and validate configuration
    config = CONFIG.copy()
    config.update({
        'depth': args.depth,
        'max_sources': args.sources if args.sources > 0 else CONFIG['source_count_by_depth'][args.depth],
        'output_format': args.format,
        'focus': args.focus,
        'ollama_model': args.model,
        'timeout': args.timeout,
        'max_time': args.max_time,
        'exclude_domains': args.exclude_domains,
        'include_urls': args.include_urls,
        'output_file': args.output if args.output else generate_output_filename(args.query or "", args.format),
        'debug': args.debug,
        'show_progress': not args.no_progress,
    })
    
    # Handle OpenAI API key if provided
    if args.openai_key:
        # More permissive check for OpenAI API key format
        if "sk-" in args.openai_key:
            config['use_openai'] = True
            config['openai_api_key'] = args.openai_key
            print("OpenAI API key detected. Will use OpenAI for summarization if available.")
        else:
            print("Warning: The OpenAI API key format doesn't look right. Falling back to Ollama only.")
    
    # If no query provided, prompt the user
    if not args.query:
        print("Welcome to DeepResearch: Advanced Research Assistant")
        args.query = input("Enter your research query: ").strip()
        if not args.query:
            print("No query provided. Exiting.")
            return 1
    
    # Print research plan
    print_research_plan(args.query, config, args.depth)
    
    # Execute research pipeline
    success = perform_research(args.query, config)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
