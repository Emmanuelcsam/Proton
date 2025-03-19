#!/usr/bin/env python3
"""
DeepResearch2.py - Search and Content Extraction Module

This module handles web search, content extraction, and web crawling functionality
for the DeepResearch tool. It provides classes for discovering, fetching, and 
extracting content from web pages.
"""

import os
import sys
import json
import time
import logging
import re
import random
import urllib.request
import urllib.parse
import urllib.error
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urljoin

# Try to import BeautifulSoup, handle gracefully if not available
try:
    from bs4 import BeautifulSoup, Comment
except ImportError:
    BeautifulSoup = None

# Import utility functions directly to ensure they're available
# These are duplicated from DeepResearch1 to prevent circular imports
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

# Set up module logger
logger = logging.getLogger("DeepResearch.SearchExtraction")

# Default HTTP headers for web requests
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive"
}

#-------------------------------------------------------------------------
# WebSearch Class - Handles search functionality
#-------------------------------------------------------------------------
class WebSearch:
    """
    Enhanced web search functionality using DuckDuckGo with fallback mechanisms
    and improved result diversity.
    """
    
    def __init__(self, config):
        """Initialize the WebSearch module with configuration."""
        self.config = config
        self.debug = config.get('debug', False)
        self.timeout = config.get('timeout', 30)
        self.max_results = config.get('max_search_results', 25)
        self.focus = config.get('focus', 'general')
        self.time_limit = config.get('search_time_limit', 'year')
        self.exclude_domains = config.get('exclude_domains', [])
        
        # Check if ddgr command-line tool is available
        try:
            subprocess.run(['ddgr', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            self.ddgr_available = True
            logger.info("ddgr command-line tool found, will use it for better search results")
        except (subprocess.SubprocessError, FileNotFoundError):
            self.ddgr_available = False
            logger.info("ddgr not found, using direct API calls instead")
    
    def execute_search(self, query):
        """Execute a search for the given query and return prioritized results."""
        logger.info(f"Executing search for: {query}")
        
        # Add focus-specific terms if needed
        enhanced_query = self._enhance_query(query)
        
        # Define the maximum number of results based on depth
        depth = self.config.get('depth', 3)
        max_total_results = self.config.get('source_count_by_depth', {}).get(depth, 15) * 2
        
        # Execute search with appropriate method
        if self.ddgr_available:
            results = self._search_with_ddgr(enhanced_query, max_total_results)
        else:
            results = self._search_with_api(enhanced_query, max_total_results)
        
        if not results:
            logger.warning("No search results found")
            return []
            
        logger.info(f"Found {len(results)} search results")
        
        # Filter results
        filtered_results = self._filter_results(results)
        logger.info(f"After filtering: {len(filtered_results)} results")
        
        # Enhance results with metadata
        enhanced_results = self._enhance_results(filtered_results, query)
        
        # Prioritize for diversity
        prioritized_results = self._prioritize_for_diversity(enhanced_results)
        
        return prioritized_results
    
    def _enhance_query(self, query):
        """Enhance the query based on the research focus."""
        focus_terms = {
            "academic": ["research", "study", "paper", "journal", "scholarly"],
            "news": ["recent", "news", "article", "report"],
            "technical": ["tutorial", "documentation", "guide", "how-to", "github"],
        }
        
        # Only enhance if specific focus is requested (not "general")
        if self.focus in focus_terms and self.focus != "general":
            # Don't add terms if they're already in the query
            additional_terms = [term for term in focus_terms[self.focus] 
                               if term.lower() not in query.lower()]
            
            # Only add one random term to avoid overwhelming the query
            if additional_terms:
                selected_term = random.choice(additional_terms)
                enhanced_query = f"{query} {selected_term}"
                logger.debug(f"Enhanced query: '{query}' -> '{enhanced_query}'")
                return enhanced_query
        
        return query
    
    def _search_with_ddgr(self, query, max_results):
        """Use ddgr command-line tool to perform DuckDuckGo search."""
        results = []
        
        try:
            # Build ddgr command
            cmd = ['ddgr', '--json', '--num', str(min(max_results, 25))]
            
            # Add time limit if specified
            if self.time_limit:
                time_map = {'day': 'd', 'week': 'w', 'month': 'm', 'year': 'y'}
                if self.time_limit in time_map:
                    cmd.extend(['--time', time_map[self.time_limit]])
            
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
                return self._search_with_api(query, max_results)  # Fallback to API
                
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
                return self._search_with_api(query, max_results)  # Fallback to API
                
        except Exception as e:
            logger.error(f"Error using ddgr: {str(e)}")
            return self._search_with_api(query, max_results)  # Fallback to API
            
        return results
    
    def _search_with_api(self, query, max_results):
        """Use direct API calls to DuckDuckGo for search results."""
        results = []
        
        # Direct DuckDuckGo API implementation
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
            # Try backup search method
            return self._backup_search(query, max_results)
            
        if not results:
            logger.warning("No results from primary API, trying backup method")
            return self._backup_search(query, max_results)
            
        return results
    
    def _backup_search(self, query, max_results):
        """Backup search method using a different API endpoint."""
        results = []
        
        try:
            # Try using the DuckDuckGo lite API
            escaped_query = urllib.parse.quote_plus(query)
            api_url = f"https://lite.duckduckgo.com/lite/?q={escaped_query}"
            
            headers = DEFAULT_HEADERS.copy()
            request = urllib.request.Request(api_url, headers=headers)
            
            logger.debug(f"Using backup search API: {api_url}")
            
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
    
    def _filter_results(self, results):
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
                
            if any(excluded in domain for excluded in self.exclude_domains):
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
            r'linkedin\.com/in/', r'pinterest\.com/[^/]+$'
        ]
        
        for pattern in social_media_patterns:
            if re.search(pattern, url):
                return True
                
        # Skip shopping and product pages
        shopping_patterns = [r'amazon\.com/[^/]+/dp/', r'ebay\.com/itm/']
        
        for pattern in shopping_patterns:
            if re.search(pattern, url):
                return True
                
        # Skip unhelpful PDF files (small size indicated in snippet)
        if url.endswith('.pdf') and re.search(r'\b[1-9]\s*[kK][bB]\b', snippet):
            return True
            
        return False
    
    def _enhance_results(self, results, original_query):
        """Enhance search results with additional metadata for better prioritization."""
        enhanced_results = []
        query_terms = set(extract_keywords(original_query))
        
        for result in results:
            url = result.get('url', '')
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            
            domain = get_domain(url)
            
            # Calculate term overlap with query
            title_terms = set(extract_keywords(title))
            snippet_terms = set(extract_keywords(snippet))
            all_terms = title_terms.union(snippet_terms)
            
            term_overlap = len(query_terms.intersection(all_terms)) / max(1, len(query_terms))
            
            # Determine source type
            source_type = self._determine_source_type(url, title, snippet)
            
            # Prepare enhanced result
            enhanced_result = result.copy()
            enhanced_result.update({
                'domain': domain,
                'term_overlap': term_overlap,
                'source_type': source_type,
                'credibility_score': self._calculate_credibility(domain),
                'relevance_score': self._calculate_relevance(title, snippet, query_terms),
                'recency': self._estimate_recency(title, snippet)
            })
            
            enhanced_results.append(enhanced_result)
            
        return enhanced_results
    
    def _determine_source_type(self, url, title, snippet):
        """Determine the type of source based on URL, title, and snippet."""
        # Check for academic and research sources
        if re.search(r'\.(edu|ac\.[a-z]{2})/', url) or 'scholar.google.com' in url:
            return "academic"
            
        if any(domain in url for domain in [
            'arxiv.org', 'researchgate.net', 'academia.edu', 'jstor.org',
            'sciencedirect.com', 'springer.com', 'ncbi.nlm.nih.gov'
        ]):
            return "research_paper"
            
        # Check for news sources
        if any(domain in url for domain in [
            'nytimes.com', 'washingtonpost.com', 'bbc.com', 'reuters.com',
            'cnn.com', 'bloomberg.com', 'npr.org'
        ]):
            return "news_article"
            
        # Check for government sources
        if re.search(r'\.(gov|mil)/', url):
            return "government"
            
        # Check for documentation
        if any(domain in url for domain in [
            'docs.python.org', 'developer.mozilla.org', 'docs.microsoft.com'
        ]):
            return "documentation"
            
        # Look for clues in title
        title_lower = title.lower()
        
        if re.search(r'(tutorial|guide|how\s+to)', title_lower):
            return "tutorial"
            
        if re.search(r'(review|comparison)', title_lower):
            return "review"
            
        # Default to "article" if no specific type is determined
        return "article"
    
    def _calculate_credibility(self, domain):
        """Calculate a credibility score for the source."""
        # Domain credibility ranges (0-100)
        credibility_scores = {
            "edu": 90, "gov": 85, "org": 75,
            "nytimes.com": 85, "bbc.com": 85, "reuters.com": 90,
            "nature.com": 95, "science.org": 95, "wikipedia.org": 80
        }
        
        # Default credibility score
        base_score = 50
        
        # Check domains
        for credible_domain, score in credibility_scores.items():
            if credible_domain in domain:
                base_score = score
                break
        
        return base_score
    
    def _calculate_relevance(self, title, snippet, query_terms):
        """Calculate relevance score based on query term matching."""
        if not query_terms:
            return 50  # Default score
            
        title_terms = set(extract_keywords(title))
        snippet_terms = set(extract_keywords(snippet))
        
        # Calculate overlap with query terms
        title_overlap = len(query_terms.intersection(title_terms)) / len(query_terms)
        snippet_overlap = len(query_terms.intersection(snippet_terms)) / len(query_terms)
        
        # Title matches are more important than snippet matches
        relevance_score = (title_overlap * 0.7 + snippet_overlap * 0.3) * 100
        
        # Boost if exact phrase from query appears
        query_phrase = ' '.join(query_terms)
        if query_phrase.lower() in title.lower():
            relevance_score += 20
        elif query_phrase.lower() in snippet.lower():
            relevance_score += 10
            
        # Clamp to valid range
        return max(0, min(100, relevance_score))
    
    def _estimate_recency(self, title, snippet):
        """Estimate the recency of the content based on date patterns."""
        combined_text = f"{title} {snippet}"
        
        # Look for dates in the content
        if re.search(r'202[3-5]', combined_text):
            return 90  # Very recent (2023-2025)
            
        if re.search(r'2022', combined_text):
            return 80  # Recent year
            
        if re.search(r'202[0-1]', combined_text):
            return 70  # 2-3 years ago
            
        if re.search(r'201[5-9]', combined_text):
            return 60  # 4-10 years ago
            
        # Look for recency keywords
        if re.search(r'\b(new|latest|recent|update|current|today|this week|this month)\b', combined_text, re.IGNORECASE):
            return 85  # Likely recent based on keywords
            
        # Default - unknown recency
        return 50
    
    def _prioritize_for_diversity(self, results):
        """Prioritize results for diversity of sources and content types."""
        if not results:
            return []
            
        # Calculate a composite score for each result
        for result in results:
            credibility = result.get('credibility_score', 50)
            relevance = result.get('relevance_score', 50)
            recency = result.get('recency', 50)
            
            # Different weights based on research focus
            if self.focus == "academic":
                composite_score = (credibility * 0.5) + (relevance * 0.4) + (recency * 0.1)
            elif self.focus == "news":
                composite_score = (credibility * 0.3) + (relevance * 0.3) + (recency * 0.4)
            elif self.focus == "technical":
                composite_score = (credibility * 0.3) + (relevance * 0.6) + (recency * 0.1)
            else:  # general
                composite_score = (credibility * 0.4) + (relevance * 0.4) + (recency * 0.2)
                
            result['score'] = composite_score
            
        # Sort by composite score
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Now ensure diversity
        final_results = []
        domains_included = set()
        source_types_included = set()
        
        # First, include top results regardless of diversity
        top_results = results[:min(5, len(results))]
        final_results.extend(top_results)
        
        for result in top_results:
            domains_included.add(result.get('domain', ''))
            source_types_included.add(result.get('source_type', ''))
            
        # Then add remaining results with diversity consideration
        remaining = results[min(5, len(results)):]
        
        # Sort remaining by score but prioritize new domains and source types
        for result in sorted(remaining, key=lambda x: x.get('score', 0), reverse=True):
            domain = result.get('domain', '')
            source_type = result.get('source_type', '')
            
            # Prioritize new domains and source types
            diversity_bonus = 0
            if domain not in domains_included:
                diversity_bonus += 20
            if source_type not in source_types_included:
                diversity_bonus += 10
                
            result['score'] += diversity_bonus
            
        # Re-sort with diversity bonus and add to final results
        remaining.sort(key=lambda x: x.get('score', 0), reverse=True)
        final_results.extend(remaining)
        
        return final_results


#-------------------------------------------------------------------------
# ContentExtractor Class - Handles content extraction from web pages
#-------------------------------------------------------------------------
class ContentExtractor:
    """Extracts content from web pages with intelligent handling of various source types."""
    
    def __init__(self, config):
        """Initialize the ContentExtractor with configuration."""
        self.config = config
        self.debug = config.get('debug', False)
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.min_content_length = config.get('min_content_length', 500)
        self.max_content_length = config.get('max_content_length', 25000)
        
        # Check if BeautifulSoup is available
        if BeautifulSoup is None:
            logger.warning("BeautifulSoup is not available. Content extraction will be limited.")
    
    def extract_content(self, url):
        """Extract content from a web page."""
        result = {
            'url': url,
            'domain': get_domain(url),
            'title': '',
            'text': '',
            'metadata': {},
            'links': [],
            'timestamp': datetime.now().isoformat(),
            'error': None
        }
        
        # Skip unsupported URL schemes
        if not url.startswith(('http://', 'https://')):
            result['error'] = "Unsupported URL scheme"
            return result
            
        # Attempt to extract content with retries
        retries = 0
        while retries < self.max_retries:
            try:
                response_content, response_info = self._fetch_url(url)
                
                if not response_content:
                    retries += 1
                    time.sleep(1)
                    continue
                    
                # Store response info
                result['status_code'] = response_info.get('status', 0)
                result['content_type'] = response_info.get('content-type', '')
                
                # Process the content based on type
                if 'html' in result['content_type'].lower() or 'xhtml' in result['content_type'].lower():
                    extracted = self._extract_html_content(response_content, url)
                    result.update(extracted)
                elif 'json' in result['content_type'].lower():
                    extracted = self._extract_json_content(response_content)
                    result.update(extracted)
                elif 'text/plain' in result['content_type'].lower():
                    extracted = self._extract_text_content(response_content)
                    result.update(extracted)
                else:
                    # Attempt to treat as HTML as a fallback
                    extracted = self._extract_html_content(response_content, url)
                    result.update(extracted)
                
                # Check if we have enough content
                if len(result['text']) < self.min_content_length:
                    logger.debug(f"Content too short: {len(result['text'])} chars")
                    if retries < self.max_retries - 1:
                        retries += 1
                        time.sleep(1)
                        continue
                        
                break
                
            except Exception as e:
                logger.error(f"Extraction error: {str(e)}")
                retries += 1
                result['error'] = str(e)
                time.sleep(1)
        
        # Limit text length
        if len(result['text']) > self.max_content_length:
            result['text'] = result['text'][:self.max_content_length] + "... [truncated]"
                
        return result
    
    def _fetch_url(self, url):
        """Fetch content from a URL."""
        try:
            # Prepare request
            headers = DEFAULT_HEADERS.copy()
            
            # Create request with headers
            request = urllib.request.Request(url, headers=headers)
            
            # Send request with timeout
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                # Get response info
                response_info = dict(response.info())
                status = response.getcode()
                response_info['status'] = status
                
                # Check status
                if status != 200:
                    logger.warning(f"Non-200 status code: {status}")
                    if status >= 400:
                        return None, {'status': status}
                        
                # Get content type
                content_type = response_info.get('Content-Type', '').split(';')[0].lower()
                response_info['content-type'] = content_type
                
                # Read content
                content = response.read()
                
                # Handle content encoding
                charset = self._get_charset(response_info)
                if charset:
                    try:
                        content = content.decode(charset)
                    except UnicodeDecodeError:
                        # Fallback to utf-8
                        try:
                            content = content.decode('utf-8')
                        except UnicodeDecodeError:
                            # Last resort: decode with ignore
                            content = content.decode('utf-8', errors='ignore')
                else:
                    # Default to utf-8
                    try:
                        content = content.decode('utf-8')
                    except UnicodeDecodeError:
                        content = content.decode('utf-8', errors='ignore')
                        
                return content, response_info
                
        except urllib.error.HTTPError as e:
            logger.error(f"HTTP error: {e.code} - {url}")
            return None, {'status': e.code}
            
        except urllib.error.URLError as e:
            logger.error(f"URL error: {str(e)} - {url}")
            return None, {'status': 0, 'error': str(e)}
            
        except Exception as e:
            logger.error(f"Fetch error: {str(e)} - {url}")
            return None, {'status': 0, 'error': str(e)}
    
    def _get_charset(self, response_info):
        """Extract charset from response headers."""
        content_type = response_info.get('Content-Type', '')
        charset_match = re.search(r'charset=([^\s;]+)', content_type)
        
        if charset_match:
            return charset_match.group(1).strip()
            
        return None
    
    def _extract_html_content(self, html_content, url):
        """Extract content from HTML."""
        result = {
            'title': '',
            'text': '',
            'metadata': {},
            'links': []
        }
        
        # Skip if BeautifulSoup is not available
        if BeautifulSoup is None:
            result['text'] = self._extract_text_from_html_regex(html_content)
            result['title'] = self._extract_title_from_html_regex(html_content)
            return result
        
        try:
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                result['title'] = title_tag.get_text(strip=True)
                
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'form', 'iframe']):
                tag.decompose()
                
            # Extract content
            main_content = self._find_main_content(soup)
            
            if main_content:
                # Extract text from main content
                paragraphs = main_content.find_all('p')
                if paragraphs:
                    result['text'] = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                else:
                    result['text'] = main_content.get_text(strip=True)
            else:
                # If no main content identified, extract all paragraphs
                paragraphs = soup.find_all('p')
                result['text'] = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                
                # If still no text, extract all text from body
                if not result['text']:
                    body = soup.find('body')
                    if body:
                        result['text'] = body.get_text(strip=True)
            
            # Extract metadata
            meta_description = soup.find('meta', attrs={'name': 'description'})
            if meta_description:
                result['metadata']['description'] = meta_description.get('content', '')
                
            # Extract links if needed
            if self.config.get('extract_links', True):
                links = []
                for a in soup.find_all('a', href=True):
                    href = a.get('href', '').strip()
                    if href and not href.startswith('#'):
                        # Resolve relative URLs
                        absolute_url = urljoin(url, href)
                        link_text = a.get_text(strip=True)
                        links.append({
                            'url': absolute_url,
                            'text': link_text
                        })
                result['links'] = links
                
        except Exception as e:
            logger.error(f"Error parsing HTML: {str(e)}")
            # Fallback to regex extraction
            result['text'] = self._extract_text_from_html_regex(html_content)
            result['title'] = self._extract_title_from_html_regex(html_content)
            
        # Clean and normalize text
        result['text'] = self._normalize_text(result['text'])
            
        return result
    
    def _find_main_content(self, soup):
        """Find the main content container in a BeautifulSoup object."""
        # Try to find main content using common IDs and classes
        for container in [
            # Most likely content containers
            soup.find(id=re.compile(r'content|main|article|post|body', re.I)),
            soup.find(class_=re.compile(r'content|main|article|post|body', re.I)),
            
            # Main semantic elements
            soup.find('main'),
            soup.find('article')
        ]:
            if container:
                return container
                
        # If no specific container found, look for the div with most paragraphs
        divs = soup.find_all('div')
        if divs:
            return max(divs, key=lambda div: len(div.find_all('p')))
            
        return None
    
    def _extract_text_from_html_regex(self, html_content):
        """Extract text from HTML using regex (fallback when BeautifulSoup is not available)."""
        if not html_content:
            return ""
            
        # Remove script and style sections
        html_content = re.sub(r'<script[^>]*>.*?</script>', ' ', html_content, flags=re.DOTALL)
        html_content = re.sub(r'<style[^>]*>.*?</style>', ' ', html_content, flags=re.DOTALL)
        
        # Extract paragraphs
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html_content, re.DOTALL)
        
        # Remove HTML tags from paragraphs
        clean_paragraphs = []
        for p in paragraphs:
            # Remove HTML tags
            p = re.sub(r'<[^>]+>', ' ', p)
            # Clean up whitespace
            p = re.sub(r'\s+', ' ', p).strip()
            if p:
                clean_paragraphs.append(p)
                
        # Join paragraphs with newlines
        text = '\n\n'.join(clean_paragraphs)
        
        # If no paragraphs found, try to extract all text
        if not text:
            # Remove all HTML tags
            text = re.sub(r'<[^>]+>', ' ', html_content)
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
        return text
    
    def _extract_title_from_html_regex(self, html_content):
        """Extract title from HTML using regex (fallback when BeautifulSoup is not available)."""
        if not html_content:
            return ""
            
        # Extract title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.DOTALL)
        if title_match:
            title = title_match.group(1)
            # Remove HTML tags
            title = re.sub(r'<[^>]+>', ' ', title)
            # Clean up whitespace
            title = re.sub(r'\s+', ' ', title).strip()
            return title
            
        return ""
    
    def _extract_json_content(self, json_content):
        """Extract content from JSON."""
        result = {
            'title': '',
            'text': '',
            'metadata': {}
        }
        
        try:
            data = json.loads(json_content)
            
            # Handle different JSON structure types
            if isinstance(data, dict):
                # Look for common content fields
                for title_field in ['title', 'name', 'heading']:
                    if title_field in data and isinstance(data[title_field], str):
                        result['title'] = data[title_field]
                        break
                        
                # Look for content fields
                for content_field in ['content', 'text', 'body', 'description']:
                    if content_field in data and isinstance(data[content_field], str):
                        result['text'] = data[content_field]
                        break
                        
                # If no content field found, create a text representation of the data
                if not result['text']:
                    result['text'] = json.dumps(data, indent=2)
                    
            elif isinstance(data, list):
                # For lists, create a text representation
                result['title'] = "JSON Data"
                result['text'] = json.dumps(data, indent=2)
                
            else:
                result['title'] = "JSON Data"
                result['text'] = json_content
                
        except json.JSONDecodeError:
            result['title'] = "Invalid JSON"
            result['text'] = json_content
            
        return result
    
    def _extract_text_content(self, text_content):
        """Extract content from plain text."""
        result = {
            'title': '',
            'text': '',
            'metadata': {}
        }
        
        # Try to extract a title from the first line
        lines = text_content.splitlines()
        if lines:
            first_line = lines[0].strip()
            if first_line and len(first_line) <= 100:
                result['title'] = first_line
                
                # Use the rest as content
                result['text'] = '\n'.join(lines[1:])
            else:
                # No good title candidate, use a generic one
                result['title'] = "Text Document"
                result['text'] = text_content
        else:
            result['title'] = "Text Document"
            result['text'] = text_content
            
        return result
    
    def _normalize_text(self, text):
        """Normalize extracted text."""
        if not text:
            return ""
            
        # Fix common character encoding issues
        text = text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # Join paragraphs with double newlines
        return '\n\n'.join(paragraphs)


#-------------------------------------------------------------------------
# WebCrawler Class - Handles crawling operations
#-------------------------------------------------------------------------
class WebCrawler:
    """Crawls web pages to discover additional relevant content."""
    
    def __init__(self, config):
        """Initialize the WebCrawler with configuration."""
        self.config = config
        self.debug = config.get('debug', False)
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.crawl_depth = config.get('crawl_depth', 1)
        self.max_links_per_page = config.get('max_links_per_page', 5)
        
        # Track crawling state
        self.visited_urls = set()
    
    def crawl_url(self, url, depth=0, parent_url=None):
        """Crawl a URL and its outgoing links up to the specified depth."""
        # Skip if already visited
        if url in self.visited_urls:
            return None
            
        # Mark as visited
        self.visited_urls.add(url)
        
        logger.info(f"Crawling URL: {url} (depth {depth})")
        
        # Check depth limit
        if depth > self.crawl_depth:
            logger.debug(f"Reached maximum crawl depth ({self.crawl_depth}) for {url}")
            return None
            
        # Initialize result
        result = {
            'url': url,
            'parent_url': parent_url,
            'depth': depth,
            'links': [],
            'content': None
        }
        
        # Extract content and links
        try:
            # Create extractor
            extractor = ContentExtractor(self.config)
            
            # Extract content
            content = extractor.extract_content(url)
            
            # If extraction failed, skip further processing
            if not content or content.get('error'):
                if content and content.get('error'):
                    logger.warning(f"Error extracting content from {url}: {content.get('error')}")
                return result
                
            # Save the content
            result['content'] = content
            
            # Get links if we need to crawl deeper
            if depth < self.crawl_depth:
                links = content.get('links', [])
                
                # Prioritize and filter links
                prioritized_links = self._prioritize_links(links, url, depth)
                result['links'] = prioritized_links
                
                # Crawl prioritized links if configured to do so
                if self.crawl_depth > depth:
                    crawled_results = self._crawl_links(prioritized_links, depth + 1, url)
                    result['crawled_results'] = crawled_results
                    
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            
        return result
    
    def _prioritize_links(self, links, parent_url, depth):
        """Prioritize and filter links for crawling."""
        if not links:
            return []
            
        # Extract base URL components
        parent_domain = get_domain(parent_url)
        
        # Filter and score links
        scored_links = []
        for link in links:
            url = link.get('url', '')
            text = link.get('text', '')
            
            # Skip invalid or already visited URLs
            if not is_valid_url(url) or url in self.visited_urls:
                continue
                
            # Get link domain
            link_domain = get_domain(url)
            
            # Calculate link score based on various factors
            score = self._calculate_link_score(link, link_domain, parent_domain, depth)
            
            scored_links.append((link, score))
            
        # Sort by score and limit to max_links_per_page
        scored_links.sort(key=lambda x: x[1], reverse=True)
        prioritized_links = [link for link, score in scored_links[:self.max_links_per_page]]
        
        return prioritized_links
    
    def _calculate_link_score(self, link, link_domain, parent_domain, depth):
        """Calculate a priority score for a link."""
        url = link.get('url', '')
        text = link.get('text', '')
        
        # Initialize score
        score = 50  # Neutral starting point
        
        # Prefer links on the same domain (likely more relevant to the topic)
        if link_domain == parent_domain:
            score += 20
            
        # Adjust based on link text quality
        if text:
            # Longer text is likely more descriptive
            if len(text) > 10:
                score += 5
                
            # Keywords in text suggesting valuable content
            if re.search(r'\b(details|information|facts|data|research|study|analysis)\b', text, re.I):
                score += 10
                
        # Adjust based on URL characteristics
        if re.search(r'\b(about|research|study|paper|report|guide|tutorial)\b', url, re.I):
            score += 10
            
        # Penalize likely low-value pages
        if re.search(r'\b(login|signup|register|contact|privacy|terms)\b', url, re.I):
            score -= 30
            
        # Depth penalty (prefer shallower links)
        score -= depth * 10
        
        return max(0, score)
    
    def _crawl_links(self, links, depth, parent_url):
        """Crawl a list of links in parallel."""
        if not links:
            return []
            
        # Extract just the URLs
        urls = [link.get('url', '') for link in links]
        
        # Crawl in parallel
        results = []
        with ThreadPoolExecutor(max_workers=min(len(urls), 5)) as executor:
            future_to_url = {executor.submit(self.crawl_url, url, depth, parent_url): url for url in urls}
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error crawling {url}: {str(e)}")
                    
        return results


# Simple test function if script is run directly
if __name__ == "__main__":
    print("DeepResearch2.py - Search and Content Extraction Module")
    print("This module should be imported by DeepResearch1.py")
    
    # Simple test if arguments provided
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
        print(f"\nTesting content extraction on: {test_url}")
        
        # Basic logging setup for test
        logging.basicConfig(level=logging.INFO)
        
        # Simple config for test
        test_config = {
            'timeout': 30,
            'max_retries': 3,
            'min_content_length': 500,
            'max_content_length': 25000
        }
        
        # Extract content
        extractor = ContentExtractor(test_config)
        content = extractor.extract_content(test_url)
        
        print(f"\nTitle: {content.get('title', 'No title')}")
        print(f"Domain: {content.get('domain', 'Unknown')}")
        print(f"Content length: {len(content.get('text', ''))}")
        print("\nExcerpt:")
        print("-" * 60)
        print(content.get('text', 'No content')[:500] + "...")
        print("-" * 60)
