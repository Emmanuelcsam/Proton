#!/usr/bin/env python3
"""
DeepResearch3.py - Analysis and Summarization Module

This module provides content analysis, AI summarization, and report generation
capabilities for the DeepResearch tool. It handles analyzing and organizing
web content, generating comprehensive summaries using AI, and creating
structured research reports.
"""

import os
import sys
import json
import time
import logging
import re
import math
import random
import requests
from datetime import datetime
from collections import Counter
from urllib.parse import urlparse

# Define utility functions directly to prevent circular imports
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

# Domain credibility scoring
DOMAIN_CREDIBILITY = {
    "edu": 90, "gov": 85, "org": 75, "ac.uk": 90,
    "nytimes.com": 85, "bbc.com": 85, "reuters.com": 90,
    "nature.com": 95, "science.org": 95, "wikipedia.org": 80
}

# Content type priorities
CONTENT_TYPE_PRIORITY = {
    "research_paper": 100, "news_article": 85, "documentation": 85,
    "tutorial": 80, "blog_post": 75, "q_and_a": 70
}

# Set up module logger
logger = logging.getLogger("DeepResearch.AnalysisSummarization")

#-------------------------------------------------------------------------
# ContentAnalyzer Class - Handles content analysis and organization
#-------------------------------------------------------------------------
class ContentAnalyzer:
    """
    Advanced content analyzer for research with information prioritization,
    deduplication, and extraction of key insights.
    """
    
    def __init__(self, config):
        """Initialize the ContentAnalyzer with configuration."""
        self.config = config
        self.debug = config.get('debug', False)
        self.focus = config.get('focus', 'general')
        self.duplicate_threshold = config.get('duplicate_threshold', 0.85)
        self.depth = config.get('depth', 3)
    
    def prioritize_sources(self, sources, query):
        """Prioritize sources based on relevance, credibility, and diversity."""
        if not sources:
            return []
            
        # Extract query keywords for matching
        query_keywords = set(extract_keywords(query))
        
        # Analyze and score each source
        for source in sources:
            # Extract source attributes
            url = source.get('url', '')
            title = source.get('title', '')
            snippet = source.get('snippet', '')
            domain = source.get('domain', get_domain(url))
            
            # Calculate relevance score
            relevance = self._calculate_relevance(title, snippet, query_keywords)
            
            # Calculate credibility score
            credibility = self._calculate_credibility(domain, url)
            
            # Calculate freshness score
            freshness = source.get('recency', 50)
            
            # Calculate composite score based on research focus
            composite_score = self._calculate_composite_score(relevance, credibility, freshness)
            
            # Update source with scores
            source.update({
                'relevance': relevance,
                'credibility': credibility,
                'freshness': freshness,
                'composite_score': composite_score
            })
            
        # Apply domain diversity boosting
        prioritized = self._apply_domain_diversity(sources)
        
        # Sort by composite score
        prioritized.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        return prioritized
    
    def _calculate_relevance(self, title, snippet, query_keywords):
        """Calculate relevance of a source to the query."""
        if not query_keywords:
            return 50  # Default medium relevance
            
        title_keywords = set(extract_keywords(title))
        snippet_keywords = set(extract_keywords(snippet))
        
        # Calculate keyword overlap
        title_overlap = len(query_keywords.intersection(title_keywords))
        snippet_overlap = len(query_keywords.intersection(snippet_keywords))
        
        # Calculate term overlap ratios
        title_ratio = title_overlap / len(query_keywords) if query_keywords else 0
        snippet_ratio = snippet_overlap / len(query_keywords) if query_keywords else 0
        
        # Title matches are weighted more heavily
        relevance_score = (title_ratio * 0.7 + snippet_ratio * 0.3) * 100
        
        # Boost exact phrase matches
        query_phrase = ' '.join(query_keywords)
        title_lower = title.lower()
        snippet_lower = snippet.lower()
        
        if query_phrase in title_lower:
            relevance_score += 20
        elif query_phrase in snippet_lower:
            relevance_score += 10
            
        # Cap at 100
        return min(100, relevance_score)
    
    def _calculate_credibility(self, domain, url):
        """Calculate credibility of a source based on domain."""
        # Start with a neutral score
        credibility = 50
        
        # Check known credible domains
        for credible_domain, score in DOMAIN_CREDIBILITY.items():
            if credible_domain in domain:
                credibility = score
                break
                
        # Boost for academic and research domains
        if re.search(r'\.(edu|ac\.[a-z]{2})$', domain):
            credibility += 10
            
        # Boost for government domains
        if re.search(r'\.(gov|mil)$', domain):
            credibility += 10
            
        # Penalize domains with many subdomains (often content farms)
        subdomain_count = domain.count('.')
        if subdomain_count > 2:
            credibility -= (subdomain_count - 2) * 5
            
        # Clamp to valid range
        return max(0, min(100, credibility))
    
    def _calculate_composite_score(self, relevance, credibility, freshness):
        """Calculate composite score based on multiple factors and research focus."""
        # Adjust weights based on research focus
        if self.focus == 'academic':
            weights = {
                'relevance': 0.4,
                'credibility': 0.5,
                'freshness': 0.1
            }
        elif self.focus == 'news':
            weights = {
                'relevance': 0.3,
                'credibility': 0.4,
                'freshness': 0.3
            }
        elif self.focus == 'technical':
            weights = {
                'relevance': 0.5,
                'credibility': 0.3,
                'freshness': 0.2
            }
        else:  # general
            weights = {
                'relevance': 0.4,
                'credibility': 0.4,
                'freshness': 0.2
            }
            
        # Calculate weighted score
        composite = (
            relevance * weights['relevance'] +
            credibility * weights['credibility'] +
            freshness * weights['freshness']
        )
        
        return composite
    
    def _apply_domain_diversity(self, sources):
        """Apply domain diversity boosting to ensure varied sources."""
        # Count occurrences of each domain
        domain_counts = Counter()
        for source in sources:
            domain = source.get('domain', '')
            domain_counts[domain] += 1
            
        # Apply diversity boosting
        for source in sources:
            domain = source.get('domain', '')
            count = domain_counts[domain]
            
            # Calculate diversity penalty based on count
            if count > 1:
                # Penalize duplicate domains, but more so for highly repeated domains
                diversity_penalty = math.log(count + 1) * 5
                source['composite_score'] = max(0, source.get('composite_score', 0) - diversity_penalty)
                
        return sources
    
    def process_content(self, contents, query):
        """Process and analyze extracted content."""
        if not contents:
            return []
            
        # Step 1: Deduplicate content
        unique_contents = self._deduplicate_content(contents)
        
        # Step 2: Extract key information and annotate
        for content in unique_contents:
            # Skip if no text
            if not content.get('text'):
                continue
                
            # Extract key points
            content['key_points'] = self._extract_key_points(content, query)
                
            # Calculate content quality score
            content['quality_score'] = self._calculate_quality_score(content)
            
            # Determine content type if not already set
            if not content.get('source_type'):
                content['source_type'] = self._determine_content_type(content)
            
            # Create snippet if not available
            if not content.get('snippet'):
                content['snippet'] = self._create_snippet(content.get('text', ''), query)
            
        # Step 3: Prioritize for research report
        prioritized = self._prioritize_for_report(unique_contents, query)
        
        logger.info(f"Processed {len(prioritized)} unique content sources")
        return prioritized
    
    def _deduplicate_content(self, contents):
        """Remove duplicate or near-duplicate content."""
        if not contents:
            return []
            
        # Create a list for unique content
        unique_contents = []
        seen_fingerprints = {}
        
        for content in contents:
            # Skip if no text
            if not content.get('text'):
                continue
                
            # Generate content fingerprint
            text = content.get('text', '')
            fingerprint = ' '.join(extract_keywords(text)[:100])
            
            # Check for similar content
            duplicate_found = False
            for existing_idx, existing_fp in seen_fingerprints.items():
                similarity = self._calculate_similarity(fingerprint, existing_fp)
                if similarity >= self.duplicate_threshold:
                    duplicate_found = True
                    # If duplicate has higher quality, replace it
                    if content.get('quality_score', 0) > unique_contents[existing_idx].get('quality_score', 0):
                        unique_contents[existing_idx] = content
                    break
                    
            # Add to unique list if not a duplicate
            if not duplicate_found:
                seen_fingerprints[len(unique_contents)] = fingerprint
                unique_contents.append(content)
                
        logger.info(f"Deduplicated content: {len(contents)} -> {len(unique_contents)}")
        return unique_contents
    
    def _calculate_similarity(self, text1, text2):
        """Calculate Jaccard similarity between two texts."""
        if not text1 or not text2:
            return 0
            
        # Extract words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0
            
        return intersection / union
    
    def _extract_key_points(self, content, query):
        """Extract key points from content based on query relevance."""
        # Initialize key points
        key_points = []
        
        # Skip if no text
        text = content.get('text', '')
        if not text:
            return key_points
            
        # Extract query keywords
        query_keywords = set(extract_keywords(query))
            
        # Extract sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            
        # Score sentences by relevance to query
        scored_sentences = []
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.split()) < 5:
                continue
                
            # Calculate relevance to query
            sentence_keywords = set(extract_keywords(sentence))
            keyword_overlap = len(query_keywords.intersection(sentence_keywords))
            
            # Calculate score based on overlap
            score = keyword_overlap * 5
            
            # Boost sentences that mention numbers or dates
            if re.search(r'\d+(\.\d+)?', sentence):
                score += 2
                
            # Boost sentences that cite sources
            if re.search(r'\b(according to|reported|study|research)\b', sentence, re.I):
                score += 3
                
            scored_sentences.append((sentence, score))
            
        # Sort by score and select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = scored_sentences[:10]  # Limit to 10 key points
        
        # Convert to list of key points
        for sentence, score in top_sentences:
            if score > 0:  # Only include if relevant
                key_points.append({
                    'text': sentence,
                    'relevance': score
                })
                
        return key_points
    
    def _calculate_quality_score(self, content):
        """Calculate a quality score for content."""
        # Initialize score
        quality_score = 50  # Start with neutral score
        
        # Get content attributes
        text = content.get('text', '')
        title = content.get('title', '')
        domain = content.get('domain', '')
        source_type = content.get('source_type', '')
        
        # Skip if no text
        if not text:
            return 0
            
        # Calculate text length score
        word_count = len(text.split())
        
        # Reward substantial content
        if word_count > 1000:
            quality_score += 15
        elif word_count > 500:
            quality_score += 10
        elif word_count > 200:
            quality_score += 5

        # Check for sources and citations
        if re.search(r'\b(according to|cited|reference|source)', text, re.I):
            quality_score += 10
            
        # Check for data-rich content
        if len(re.findall(r'\d+(\.\d+)?', text)) > 5:
            quality_score += 5
            
        # Penalize content with excessive promotional content
        promo_terms = ['buy', 'discount', 'offer', 'sale', 'click', 'order', 'sign up']
        promo_count = sum(1 for term in promo_terms if term in text.lower())
        if promo_count > 2:
            quality_score -= min(30, promo_count * 5)
            
        # Clamp to valid range
        return max(0, min(100, quality_score))
    
    def _determine_content_type(self, content):
        """Determine the type of content based on URL, title, and text."""
        url = content.get('url', '')
        title = content.get('title', '')
        text = content.get('text', '')
        domain = content.get('domain', '')
        
        # Check for academic paper
        if (re.search(r'\b(abstract|introduction|methodology|conclusion|references)\b', text, re.I) and
            re.search(r'\b(study|research|analysis|experiment)\b', text, re.I)):
            return 'research_paper'
            
        # Check for news article
        if (re.search(r'\b(news|article|published|reported)\b', text, re.I) or
            domain in ['nytimes.com', 'cnn.com', 'bbc.com', 'reuters.com']):
            return 'news_article'
            
        # Check for blog post
        if ('blog' in url.lower() or domain in ['medium.com', 'wordpress.com']):
            return 'blog_post'
            
        # Check for documentation
        if (re.search(r'\b(documentation|manual|guide|tutorial|how-to)\b', title, re.I) or
            domain in ['docs.python.org', 'developer.mozilla.org']):
            return 'documentation'
            
        # Check for Q&A
        if (domain in ['stackoverflow.com', 'quora.com', 'reddit.com']):
            return 'q_and_a'
            
        # Default to article
        return 'article'
    
    def _create_snippet(self, text, query):
        """Create a representative snippet from text that is relevant to the query."""
        if not text:
            return ""
            
        # Clean and normalize text
        clean_text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_text) if s.strip()]
        
        if not sentences:
            return ""
            
        # Extract query terms
        query_terms = [term.lower() for term in query.split() if len(term) > 3]
        
        # Find sentences containing query terms
        relevant_sentences = []
        for sentence in sentences:
            if any(term in sentence.lower() for term in query_terms):
                relevant_sentences.append(sentence)
                
        # Select the best sentences
        if relevant_sentences:
            # Use the first relevant sentence that's not too short
            for sentence in relevant_sentences:
                if len(sentence) > 100:
                    return sentence
                    
            # If all are short, use the first one
            return relevant_sentences[0]
        else:
            # No relevant sentences found, use the first sentence
            return sentences[0]
    
    def _prioritize_for_report(self, contents, query):
        """Prioritize content for inclusion in the research report."""
        if not contents:
            return []
            
        # Calculate priority score for each content
        for content in contents:
            # Get content attributes
            quality = content.get('quality_score', 50)
            key_points = content.get('key_points', [])
            source_type = content.get('source_type', 'article')
            
            # Calculate key point relevance
            key_point_score = sum(p.get('relevance', 0) for p in key_points)
            if key_points:
                key_point_score /= len(key_points)
                
            # Calculate type priority based on content type
            type_priority = CONTENT_TYPE_PRIORITY.get(source_type, 50)
            
            # Adjust based on research focus
            if self.focus == 'academic' and source_type == 'research_paper':
                type_priority += 20
            elif self.focus == 'news' and source_type == 'news_article':
                type_priority += 20
            elif self.focus == 'technical' and source_type in ['documentation', 'tutorial']:
                type_priority += 20
                
            # Calculate final priority score
            priority_score = (quality * 0.4) + (key_point_score * 0.4) + (type_priority * 0.2)
            content['priority_score'] = priority_score
            
        # Sort by priority score
        contents.sort(key=lambda x: x.get('priority_score', 0), reverse=True)
        
        return contents


#-------------------------------------------------------------------------
# OllamaSummarizer Class - Handles AI-powered summarization
#-------------------------------------------------------------------------
class OllamaSummarizer:
    """
    Uses Ollama models for AI-powered summarization with context-awareness,
    source attribution, and structured output.
    """
    
    def __init__(self, config):
        """Initialize the OllamaSummarizer with configuration."""
        self.config = config
        self.debug = config.get('debug', False)
        self.model = config.get('ollama_model', 'gemma3:12b')
        self.api_url = config.get('ollama_api_url', 'http://localhost:11434/api/generate')
        self.depth = config.get('depth', 3)
        self.max_retries = config.get('max_retries', 3)
        
        # OpenAI configuration (if available)
        self.use_openai = config.get('use_openai', False)
        self.openai_api_key = config.get('openai_api_key', '')
        self.openai_api_url = config.get('openai_api_url', 'https://api.openai.com/v1/chat/completions')
        self.openai_model = config.get('openai_model', 'gpt-3.5-turbo')
        
        # Summary configuration
        self.max_content_length = 80000  # maximum content to send to LLM
    
    def generate_summary(self, contents, query):
        """Generate a comprehensive summary of the research content."""
        logger.info(f"Generating summary for query: {query}")
        
        # Check if we have content to summarize
        if not contents:
            logger.warning("No content to summarize")
            return {"text": "No content available to summarize."}
            
        # Prepare content for summarization
        prepared_content = self._prepare_content(contents)
        
        # Prepare sources reference
        sources_text = self._prepare_sources(contents)
        
        # Generate summary using appropriate method
        summary = {}
        
        # Determine the best summarization method based on config
        if self.use_openai and self.openai_api_key:
            logger.info("Using OpenAI for summarization")
            summary_text = self._generate_summary_with_openai(prepared_content, sources_text, query)
        else:
            logger.info(f"Using Ollama model {self.model} for summarization")
            summary_text = self._generate_summary_with_ollama(prepared_content, sources_text, query)
            
        if not summary_text:
            logger.warning("Failed to generate summary")
            summary_text = "Unable to generate a comprehensive summary due to technical issues."
            
        # Process the summary to add structure if needed
        summary = {
            "text": summary_text,
            "query": query,
            "source_count": len(contents),
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
    
    def _prepare_content(self, contents):
        """Prepare content for summarization."""
        if not contents:
            return ""
            
        # Sort by priority score if available
        contents = sorted(contents, key=lambda x: x.get('priority_score', 0), reverse=True)
        
        # Limit to the most relevant content based on depth
        depth_limits = {
            1: min(3, len(contents)),
            2: min(5, len(contents)),
            3: min(8, len(contents)),
            4: min(12, len(contents)),
            5: min(20, len(contents))
        }
        
        top_contents = contents[:depth_limits.get(self.depth, 5)]
        
        # Prepare content for summarization
        prepared_text = ""
        
        for i, content in enumerate(top_contents, 1):
            # Add content with source marker
            title = content.get('title', 'Untitled')
            text = content.get('text', '')
            
            # Limit text length for each source
            max_length_per_source = 5000 if self.depth >= 4 else 3000
            if len(text) > max_length_per_source:
                text = text[:max_length_per_source] + "..."
                
            # Add key points if available
            key_points = []
            for point in content.get('key_points', []):
                key_points.append(point.get('text', ''))
                
            prepared_text += f"SOURCE [S{i}]: {title}\n\n"
            
            if key_points:
                prepared_text += "KEY POINTS:\n"
                for point in key_points[:5]:  # Limit to top 5 key points
                    prepared_text += f"- {point}\n"
                prepared_text += "\n"
                
            prepared_text += f"CONTENT:\n{text}\n\n"
            prepared_text += "-" * 40 + "\n\n"
            
        return prepared_text
    
    def _prepare_sources(self, contents):
        """Prepare source references for the summary."""
        sources_text = ""
        
        for i, content in enumerate(contents, 1):
            title = content.get('title', 'Untitled')
            url = content.get('url', '')
            domain = content.get('domain', '')
            
            sources_text += f"[S{i}] {title} ({domain})\n"
            
        return sources_text
    
    def _generate_summary_with_ollama(self, content, sources, query):
        """Generate summary using Ollama API."""
        # Prepare prompt using template
        prompt = self._create_summary_prompt(query, sources, content)
        
        # Trim content if it's too long
        if len(prompt) > self.max_content_length:
            logger.warning(f"Prompt too long ({len(prompt)} chars), trimming...")
            content_template = self._create_summary_prompt(query, sources, "")
            max_content_len = self.max_content_length - len(content_template)
            content_trimmed = content[:max_content_len] + "...[additional content trimmed for length]"
            prompt = self._create_summary_prompt(query, sources, content_trimmed)
            
        # Prepare API request
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        # Attempt to generate summary with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling Ollama API (attempt {attempt + 1}/{self.max_retries})")
                
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=120  # Long timeout for complex summarization
                )
                
                if response.status_code == 200:
                    result = response.json()
                    summary = result.get('response', '')
                    
                    if summary:
                        logger.info(f"Successfully generated summary ({len(summary)} chars)")
                        return summary
                    else:
                        logger.warning("Empty summary returned from Ollama API")
                else:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logger.error(f"Error calling Ollama API: {str(e)}")
                
            # Wait before retry
            wait_time = 2 ** attempt  # Exponential backoff
            time.sleep(wait_time)
            
        # If all retries failed, return a simple error summary
        logger.error("All Ollama API attempts failed")
        return self._generate_fallback_summary(content, sources, query)
    
    def _generate_summary_with_openai(self, content, sources, query):
        """Generate summary using OpenAI API."""
        if not self.openai_api_key:
            logger.warning("No OpenAI API key provided, falling back to Ollama")
            return self._generate_summary_with_ollama(content, sources, query)
            
        # Prepare prompt using template
        prompt = self._create_summary_prompt(query, sources, content)
        
        # Trim prompt if it's too long
        if len(prompt) > self.max_content_length:
            logger.warning(f"Prompt too long ({len(prompt)} chars), trimming...")
            content_template = self._create_summary_prompt(query, sources, "")
            max_content_len = self.max_content_length - len(content_template)
            content_trimmed = content[:max_content_len] + "...[additional content trimmed for length]"
            prompt = self._create_summary_prompt(query, sources, content_trimmed)
            
        # Prepare API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        
        payload = {
            "model": self.openai_model,
            "messages": [
                {"role": "system", "content": "You are an expert research assistant tasked with synthesizing information from multiple sources."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 4000
        }
        
        # Attempt to generate summary with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling OpenAI API (attempt {attempt + 1}/{self.max_retries})")
                
                response = requests.post(
                    self.openai_api_url,
                    headers=headers,
                    json=payload,
                    timeout=120  # Long timeout for complex summarization
                )
                
                if response.status_code == 200:
                    result = response.json()
                    summary = result['choices'][0]['message']['content']
                    
                    if summary:
                        logger.info(f"Successfully generated summary ({len(summary)} chars)")
                        return summary
                    else:
                        logger.warning("Empty summary returned from OpenAI API")
                else:
                    logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logger.error(f"Error calling OpenAI API: {str(e)}")
                
            # Wait before retry
            wait_time = 2 ** attempt  # Exponential backoff
            time.sleep(wait_time)
            
        # If all retries failed, fall back to Ollama
        logger.error("All OpenAI API attempts failed, falling back to Ollama")
        return self._generate_summary_with_ollama(content, sources, query)
    
    def _create_summary_prompt(self, query, sources, content):
        """Create a prompt for the AI model to generate a summary."""
        prompt = f"""
You are an expert researcher tasked with analyzing and synthesizing information from multiple sources.
Based on the information provided, create a comprehensive, accurate, and insightful research report
on the topic: "{query}"

The report should:
1. Synthesize key information from all provided sources
2. Present a balanced and comprehensive view of the topic
3. Highlight important facts, statistics, and expert opinions
4. Identify areas of consensus and disagreement among sources
5. Include clear citations to attribute information to specific sources
6. Be well-structured with logical organization of ideas
7. Use an academic, neutral tone appropriate for a research document

Make sure to cite sources using [S1], [S2], etc. where appropriate. Provide a balanced perspective
and avoid emphasizing one viewpoint unless the evidence clearly supports it.

SOURCES:
{sources}

INFORMATION TO SYNTHESIZE:
{content}

FORMAT YOUR RESPONSE AS A STRUCTURED RESEARCH REPORT. BE THOROUGH, INFORMATIVE, AND OBJECTIVE.
"""
        return prompt
    
    def _generate_fallback_summary(self, content, sources, query):
        """Generate a simple fallback summary when AI summarization fails."""
        logger.info("Generating fallback summary")
        
        # Extract content from the prepared text
        source_matches = re.findall(r'SOURCE \[S(\d+)\]: (.+?)\n\n(?:KEY POINTS:\n)?(?:(?:- .*?\n)*\n)?CONTENT:\n(.*?)\n\n-{40}',
                                  content, re.DOTALL)
        
        if not source_matches:
            return f"Research on '{query}' found limited information. Please try again later."
            
        # Create a simple summary from key sentences in each source
        summary = f"# Research Summary: {query}\n\n"
        summary += "## Overview\n\n"
        summary += f"This research summary compiles information on '{query}' from {len(source_matches)} sources.\n\n"
        
        summary += "## Key Findings\n\n"
        
        # Extract key sentences from each source
        for source_num, title, source_content in source_matches:
            # Clean and split the content into sentences
            clean_content = re.sub(r'\s+', ' ', source_content).strip()
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_content) if s.strip()]
            
            # Select a few representative sentences
            if sentences:
                num_sentences = min(3, len(sentences))
                selected_sentences = []
                
                # Try to select sentences with the query terms
                query_terms = query.lower().split()
                for sentence in sentences:
                    if any(term in sentence.lower() for term in query_terms):
                        selected_sentences.append(sentence)
                        if len(selected_sentences) >= num_sentences:
                            break
                            
                # If not enough sentences with query terms, add some from the beginning
                if len(selected_sentences) < num_sentences:
                    for sentence in sentences:
                        if sentence not in selected_sentences:
                            selected_sentences.append(sentence)
                            if len(selected_sentences) >= num_sentences:
                                break
                                
                # Add the key findings from this source
                summary += f"### From Source [S{source_num}]: {title}\n\n"
                for sentence in selected_sentences:
                    summary += f"- {sentence}\n"
                summary += "\n"
                
        summary += "## Sources\n\n"
        summary += sources
        
        return summary


#-------------------------------------------------------------------------
# ReportGenerator Class - Handles report generation
#-------------------------------------------------------------------------
class ReportGenerator:
    """
    Generate structured research reports with proper formatting and citations
    in various output formats.
    """
    
    def __init__(self, config):
        """Initialize the ReportGenerator with configuration."""
        self.config = config
        self.debug = config.get('debug', False)
        self.output_format = config.get('output_format', 'markdown')
        self.output_file = config.get('output_file', '')
        self.depth = config.get('depth', 3)
    
    def create_report(self, query, summary, contents, execution_time, is_complete=True):
        """Create a comprehensive research report."""
        logger.info(f"Creating {self.output_format} report for: {query}")
        
        # Generate report in requested format
        if self.output_format == 'markdown':
            report = self._create_markdown_report(query, summary, contents, execution_time, is_complete)
        elif self.output_format == 'text':
            report = self._create_text_report(query, summary, contents, execution_time, is_complete)
        elif self.output_format == 'json':
            report = self._create_json_report(query, summary, contents, execution_time, is_complete)
        else:
            # Default to markdown
            report = self._create_markdown_report(query, summary, contents, execution_time, is_complete)
            
        return report
    
    def _create_markdown_report(self, query, summary, contents, execution_time, is_complete):
        """Create a markdown-formatted research report."""
        # Format datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start building the report
        report = f"# Research Report: {query}\n\n"
        
        # Add metadata
        report += "## Research Metadata\n\n"
        report += f"- **Query:** {query}\n"
        report += f"- **Date:** {now}\n"
        report += f"- **Sources Analyzed:** {len(contents)}\n"
        
        minutes, seconds = divmod(int(execution_time), 60)
        report += f"- **Research Time:** {minutes} minutes, {seconds} seconds\n"
        
        if not is_complete:
            report += "- **Status:** Incomplete (Research was interrupted)\n"
            
        report += "\n"
        
        # Add executive summary
        report += "## Executive Summary\n\n"
        
        # Process the summary text
        summary_text = summary.get('text', '')
        if summary_text:
            # If summary already has markdown headings, adjust them
            if re.search(r'^#+ ', summary_text, re.MULTILINE):
                # Adjust heading levels to fit within our document structure
                summary_text = re.sub(r'^# ', '### ', summary_text, flags=re.MULTILINE)
                summary_text = re.sub(r'^## ', '#### ', summary_text, flags=re.MULTILINE)
                summary_text = re.sub(r'^### ', '##### ', summary_text, flags=re.MULTILINE)
                
            report += summary_text
        else:
            report += "*No summary available.*\n"
            
        report += "\n"
        
        # Add key findings (if not already in summary)
        if not re.search(r'key findings|main points|key points', summary_text, re.IGNORECASE):
            report += "## Key Findings\n\n"
            
            # Extract key points from all contents
            all_key_points = []
            for content in contents:
                for point in content.get('key_points', []):
                    text = point.get('text', '')
                    source_index = contents.index(content) + 1
                    all_key_points.append((text, source_index))
                    
            # Sort by source index and add to report
            if all_key_points:
                for text, source_index in all_key_points[:10]:  # Limit to top 10
                    report += f"- {text} [S{source_index}]\n"
            else:
                report += "*No specific key points identified.*\n"
                
            report += "\n"
            
        # Add sources section with detailed information
        if self.config.get('include_sources', True):
            report += "## Sources\n\n"
            
            for i, content in enumerate(contents, 1):
                title = content.get('title', 'Untitled')
                url = content.get('url', '')
                domain = content.get('domain', '')
                source_type = content.get('source_type', 'article')
                
                report += f"### [S{i}] {title}\n\n"
                report += f"- **URL:** [{url}]({url})\n"
                report += f"- **Source:** {domain}\n"
                report += f"- **Type:** {source_type.capitalize()}\n"
                
                # Add snippets if enabled
                if self.config.get('include_snippets', True) and self.depth >= 3:
                    report += "\n**Key Excerpts:**\n\n"
                    
                    # Add key points if available
                    key_points = content.get('key_points', [])
                    if key_points:
                        for point in key_points[:3]:  # Limit to top 3
                            text = point.get('text', '')
                            report += f"> {text}\n\n"
                    else:
                        # Extract a representative snippet
                        text = content.get('text', '')
                        if text:
                            snippet = self._extract_representative_snippet(text, query)
                            report += f"> {snippet}\n\n"
                            
                report += "\n"
                
        # Add methodology section if depth is high enough
        if self.depth >= 3:
            report += "## Research Methodology\n\n"
            report += "This report was generated using DeepResearch, an advanced research assistant that:\n\n"
            report += "1. Performed a comprehensive search for information related to the query\n"
            report += "2. Analyzed and prioritized sources based on relevance, credibility, and information value\n"
            report += f"3. Extracted and processed content from {len(contents)} distinct sources\n"
            report += "4. Identified key points and insights across all sources\n"
            report += "5. Generated a synthesized report with proper source attribution\n\n"
            
        # Add footer
        report += "---\n"
        report += f"*Generated by DeepResearch on {now}*\n"
        
        return report
    
    def _create_text_report(self, query, summary, contents, execution_time, is_complete):
        """Create a plain text research report."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start building the report
        report = f"RESEARCH REPORT: {query}\n"
        report += "=" * 80 + "\n\n"
        
        # Add metadata
        report += "RESEARCH METADATA\n"
        report += "-" * 80 + "\n"
        report += f"Query: {query}\n"
        report += f"Date: {now}\n"
        report += f"Sources Analyzed: {len(contents)}\n"
        
        minutes, seconds = divmod(int(execution_time), 60)
        report += f"Research Time: {minutes} minutes, {seconds} seconds\n"
        
        if not is_complete:
            report += "Status: Incomplete (Research was interrupted)\n"
            
        report += "\n"
        
        # Add executive summary
        report += "EXECUTIVE SUMMARY\n"
        report += "-" * 80 + "\n"
        
        # Process the summary text
        summary_text = summary.get('text', '')
        if summary_text:
            # Remove markdown formatting for plain text
            summary_text = re.sub(r'^#+\s+', '', summary_text, flags=re.MULTILINE)
            summary_text = re.sub(r'\*\*(.*?)\*\*', r'\1', summary_text)
            summary_text = re.sub(r'\*(.*?)\*', r'\1', summary_text)
            
            report += summary_text
        else:
            report += "No summary available.\n"
            
        report += "\n"
        
        # Add sources section
        report += "SOURCES\n"
        report += "-" * 80 + "\n"
        
        for i, content in enumerate(contents, 1):
            title = content.get('title', 'Untitled')
            url = content.get('url', '')
            domain = content.get('domain', '')
            
            report += f"[S{i}] {title}\n"
            report += f"URL: {url}\n"
            report += f"Source: {domain}\n\n"
            
            # Add key excerpts
            key_points = content.get('key_points', [])
            if key_points:
                report += "Key points:\n"
                for point in key_points[:3]:  # Limit to top 3
                    report += f"- {point.get('text', '')}\n"
                report += "\n"
            
            report += "-" * 80 + "\n\n"
        
        # Add footer
        report += f"Generated by DeepResearch on {now}\n"
        
        return report
    
    def _create_json_report(self, query, summary, contents, execution_time, is_complete):
        """Create a JSON research report."""
        now = datetime.now().isoformat()
        
        # Build JSON object
        report_obj = {
            "metadata": {
                "query": query,
                "timestamp": now,
                "source_count": len(contents),
                "execution_time_seconds": execution_time,
                "is_complete": is_complete,
                "depth": self.depth
            },
            "summary": {
                "text": summary.get('text', ''),
                "query": query,
                "source_count": len(contents),
                "timestamp": now
            },
            "sources": []
        }
        
        # Add sources
        for i, content in enumerate(contents, 1):
            source_obj = {
                "index": i,
                "title": content.get('title', 'Untitled'),
                "url": content.get('url', ''),
                "domain": content.get('domain', ''),
                "source_type": content.get('source_type', 'article'),
                "quality_score": content.get('quality_score', 0),
                "priority_score": content.get('priority_score', 0),
                "key_points": content.get('key_points', [])
            }
            
            report_obj["sources"].append(source_obj)
            
        # Create JSON string
        return json.dumps(report_obj, indent=2)
    
    def _extract_representative_snippet(self, text, query):
        """Extract a representative snippet from text that is relevant to the query."""
        # Clean the text
        clean_text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_text) if s.strip()]
        
        if not sentences:
            return ""
            
        # Extract query terms
        query_terms = [term.lower() for term in query.split() if len(term) > 3]
        
        # Find sentences containing query terms
        relevant_sentences = []
        for sentence in sentences:
            if any(term in sentence.lower() for term in query_terms):
                relevant_sentences.append(sentence)
                
        # Select the best sentences
        if relevant_sentences:
            return relevant_sentences[0]
        else:
            # No relevant sentences found, use the first sentence
            return sentences[0]


# Simple test function if script is run directly
if __name__ == "__main__":
    print("DeepResearch3.py - Analysis and Summarization Module")
    print("This module should be imported by DeepResearch1.py")
    
    # Simple test if arguments provided
    if len(sys.argv) > 1:
        test_query = sys.argv[1]
        print(f"\nTesting summarization for query: {test_query}")
        
        # Basic logging setup for test
        logging.basicConfig(level=logging.INFO)
        
        # Simple config for test
        test_config = {
            'depth': 3,
            'focus': 'general',
            'ollama_model': 'gemma3:12b',
            'ollama_api_url': 'http://localhost:11434/api/generate'
        }
        
        # Create dummy content for testing
        test_content = [{
            'title': 'Test Document',
            'url': 'https://example.com/test',
            'domain': 'example.com',
            'text': f'This is a test document about {test_query}. ' * 10,
            'source_type': 'article',
            'key_points': [{'text': f'Key point about {test_query}', 'relevance': 10}]
        }]
        
        # Test summarizer
        summarizer = OllamaSummarizer(test_config)
        
        print("Testing Ollama connection...")
        try:
            response = requests.get(test_config['ollama_api_url'].replace('/api/generate', '/api/tags'), timeout=5)
            if response.status_code == 200:
                print("Ollama connection successful!")
                summary = summarizer.generate_summary(test_content, test_query)
                print("\nSummary:")
                print("-" * 60)
                print(summary.get('text', 'No summary generated'))
                print("-" * 60)
            else:
                print("Failed to connect to Ollama. Make sure it's running.")
        except Exception as e:
            print(f"Failed to connect to Ollama: {str(e)}")
