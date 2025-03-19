#!/usr/bin/env python3
"""
WebSum - Enhanced Photon Results Summarizer

This script:
1. Traverses through /home/jarvis/photon_results and its subfolders
2. Extracts URLs from text files in these folders
3. Fetches and extracts content from these URLs using BeautifulSoup
4. Summarizes content using improved algorithms:
   - Enhanced TextRank summarizer with NLP capabilities
   - Ollama models (local)
   - ChatGPT (OpenAI API)
   - Hugging Face models
5. Processes URLs in batches with intelligent theme recognition
6. Creates structured summaries with key themes, entities and metadata
7. Supports hierarchical summarization for better batch analysis
8. Saves comprehensive summary reports to summary.txt in each subfolder
"""

import os
import re
import json
import time
import argparse
import requests
import sys
import logging
import math
import heapq
import string
import nltk
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
import hashlib
from datetime import datetime
from textwrap import dedent

# Try to download nltk data if it's not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("Downloading NLTK POS tagger...")
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

__version__ = "2.0"

# Default configuration
PHOTON_ROOT = "/home/jarvis/photon_results"
BATCH_SIZE = 10
DEFAULT_OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_HF_API_URL = "https://api-inference.huggingface.co/models/"
DEFAULT_TIMEOUT = 90  # seconds
MAX_CONTENT_LENGTH = 8000  # characters

# Default API keys (can be overridden by environment variables or user input)
DEFAULT_OPENAI_API_KEY = "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"
DEFAULT_HF_API_KEY = "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"

# Regular expression for extracting URLs
URL_PATTERN = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[^)\s]*)?')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedTextRankSummarizer:
    """Enhanced implementation of TextRank algorithm for text summarization with advanced NLP features"""
    
    def __init__(self):
        # Enhanced stop words list with more common English stopwords
        self.stop_words = self._load_stop_words()
        
        # Initialize NLP components
        self._init_nlp()
        
        # Keep track of extracted entities for summarization
        self.entities = []
        
        # Track content topics and categories
        self.content_category = None
        self.topic_keywords = []
        
    def _load_stop_words(self):
        """Load an expanded set of stop words for better filtering"""
        # Core stop words from NLTK if available
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
        except (ImportError, LookupError):
            # Fallback to basic stop words
            stop_words = set([
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'were', 'will', 'with', 'about', 'above', 'after',
                'again', 'all', 'am', 'any', 'because', 'been', 'before', 'being',
                'below', 'between', 'both', 'but', 'can', 'did', 'do', 'does',
                'doing', 'down', 'during', 'each', 'few', 'further', 'had', 'has',
                'have', 'having', 'here', 'how', 'i', 'if', 'into', 'just', 'me',
                'more', 'most', 'my', 'no', 'nor', 'not', 'now', 'of', 'off', 
                'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'out', 'over',
                'own', 'same', 'so', 'some', 'such', 'than', 'that', 'the', 'their',
                'theirs', 'them', 'then', 'there', 'these', 'they', 'this', 'those',
                'through', 'to', 'too', 'under', 'until', 'up', 'very', 'we', 'what',
                'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will',
                'with', 'would', 'you', 'your', 'yours'
            ])
            
        # Add common web content words that aren't useful for summarization
        web_specific_stops = {
            'click', 'subscribe', 'comment', 'share', 'like', 'follow', 'website',
            'cookie', 'privacy', 'policy', 'terms', 'conditions', 'rights', 
            'reserved', 'copyright', 'site', 'http', 'https', 'www', 'html',
            'login', 'sign', 'menu', 'search', 'home', 'page', 'contact'
        }
        
        stop_words.update(web_specific_stops)
        return stop_words
    
    def _init_nlp(self):
        """Initialize NLP components for advanced analysis"""
        self.use_advanced_nlp = False
        
        try:
            # Check for additional NLTK packages
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
                
            try:
                nltk.data.find('chunkers/maxent_ne_chunker')
                nltk.data.find('corpora/words')
            except LookupError:
                nltk.download('maxent_ne_chunker', quiet=True)
                nltk.download('words', quiet=True)
            
            # Try to import NLTK components for sentiment analysis
            try:
                nltk.data.find('sentiment/vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
            
            from nltk.stem import WordNetLemmatizer
            self.lemmatizer = WordNetLemmatizer()
            
            self.use_advanced_nlp = True
            logger.debug("Advanced NLP features enabled")
        except Exception as e:
            logger.debug(f"Advanced NLP features not available: {e}")
            self.use_advanced_nlp = False
    
    def _clean_text(self, text):
        """Enhanced text cleaning with lemmatization and normalization"""
        if not text:
            return ""
            
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = text.lower().strip()
        
        if self.use_advanced_nlp:
            try:
                # Tokenize and lemmatize words
                words = nltk.word_tokenize(text)
                pos_tags = nltk.pos_tag(words)
                
                # Convert POS tags to WordNet format for lemmatization
                cleaned_words = []
                for word, tag in pos_tags:
                    # Skip punctuation and stopwords
                    if word in string.punctuation or word in self.stop_words:
                        continue
                    
                    # Convert POS tag to WordNet format
                    if tag.startswith('J'):
                        wordnet_pos = 'a'  # adjective
                    elif tag.startswith('V'):
                        wordnet_pos = 'v'  # verb
                    elif tag.startswith('N'):
                        wordnet_pos = 'n'  # noun
                    elif tag.startswith('R'):
                        wordnet_pos = 'r'  # adverb
                    else:
                        wordnet_pos = 'n'  # default to noun
                    
                    # Lemmatize the word
                    lemma = self.lemmatizer.lemmatize(word, pos=wordnet_pos)
                    
                    # Only add words longer than 2 characters
                    if len(lemma) > 2:
                        cleaned_words.append(lemma)
                
                return ' '.join(cleaned_words)
            except Exception as e:
                logger.debug(f"Advanced text cleaning failed: {e}")
                # Fall back to basic cleaning
                return ' '.join([word for word in text.split() 
                                if word not in self.stop_words 
                                and word not in string.punctuation
                                and len(word) > 2])
        else:
            # Basic cleaning without advanced NLP
            return ' '.join([word for word in text.split() 
                            if word not in self.stop_words 
                            and word not in string.punctuation
                            and len(word) > 2])
    
    def _extract_sentences(self, text):
        """Extract sentences with improved boundary detection"""
        if not text:
            return []
            
        # Handle common abbreviations to prevent incorrect sentence splits
        text = re.sub(r'(?i)(\s|^)(mr\.|mrs\.|ms\.|dr\.|prof\.|inc\.|ltd\.|sr\.|jr\.)', 
                      lambda m: m.group(1) + m.group(2).replace('.', '[DOT]'), text)
        
        # Extract sentences using NLTK if available
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            # Fallback to regex-based sentence extraction
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        
        # Restore the original periods in abbreviations
        sentences = [s.replace('[DOT]', '.') for s in sentences]
        
        # Filter out empty sentences and those that are too short
        filtered_sentences = []
        for s in sentences:
            s = s.strip()
            if s and len(s.split()) > 3:  # Skip sentences with 3 or fewer words
                filtered_sentences.append(s)
        
        return filtered_sentences
    
    def _extract_entities(self, text):
        """Extract named entities for identifying important content elements"""
        entities = {"PERSON": [], "ORGANIZATION": [], "LOCATION": [], "DATE": [], "OTHER": []}
        
        if not self.use_advanced_nlp or not text:
            return entities
            
        try:
            sentences = nltk.sent_tokenize(text)
            for sentence in sentences:
                words = nltk.word_tokenize(sentence)
                tagged = nltk.pos_tag(words)
                named_entities = nltk.ne_chunk(tagged)
                
                for chunk in named_entities:
                    if hasattr(chunk, 'label'):
                        entity_text = ' '.join([word for word, tag in chunk.leaves()])
                        entity_type = chunk.label()
                        
                        # Map to simplified entity types
                        if entity_type in ['PERSON']:
                            entities["PERSON"].append(entity_text)
                        elif entity_type in ['ORGANIZATION', 'GPE']:
                            entities["ORGANIZATION"].append(entity_text)
                        elif entity_type in ['LOCATION', 'GPE']:
                            entities["LOCATION"].append(entity_text)
                        elif entity_type in ['DATE', 'TIME']:
                            entities["DATE"].append(entity_text)
                        else:
                            entities["OTHER"].append(entity_text)
        except Exception as e:
            logger.debug(f"Entity extraction failed: {e}")
        
        # Remove duplicates and keep the most frequent entities
        for entity_type in entities:
            if entities[entity_type]:
                counter = Counter(entities[entity_type])
                entities[entity_type] = [item for item, count in counter.most_common(5)]
        
        # Store for later use in summarization
        self.entities = entities
        return entities
    
    def _compute_word_frequencies(self, text):
        """Calculate word importance using TF-IDF-like weighting"""
        if not text:
            return Counter()
            
        # Clean the text first
        clean_text = self._clean_text(text)
        
        # Split into words and count frequencies
        word_freq = Counter(clean_text.split())
        
        # Apply TF-IDF-like weighting to reduce common word importance
        total_words = sum(word_freq.values())
        unique_words = len(word_freq)
        
        if unique_words == 0:
            return Counter()
            
        # Calculate average frequency
        avg_freq = total_words / unique_words
        
        # Apply dampening to very common words to prevent them from dominating
        for word in list(word_freq.keys()):
            if word_freq[word] > 2 * avg_freq:
                word_freq[word] = 2 * avg_freq + math.log(word_freq[word] - 2 * avg_freq + 1)
        
        # Normalize frequencies
        max_freq = max(word_freq.values()) if word_freq else 1
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
        
        return word_freq
    
    def _build_sentence_graph(self, sentences, word_freq):
        """Build a graph of sentence similarities for PageRank algorithm"""
        n = len(sentences)
        similarity_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            # Skip very short sentences
            if len(sentences[i].split()) <= 3:
                continue
                
            # Clean and extract words from sentence i
            sentence_i_words = set(self._clean_text(sentences[i]).split())
            
            for j in range(i+1, n):  # Only calculate upper triangle to avoid duplication
                # Skip very short sentences
                if len(sentences[j].split()) <= 3:
                    continue
                    
                # Clean and extract words from sentence j
                sentence_j_words = set(self._clean_text(sentences[j]).split())
                
                # Calculate intersection
                intersection = sentence_i_words.intersection(sentence_j_words)
                
                # Skip if no common words
                if not intersection:
                    continue
                
                # Calculate weighted similarity based on word importance
                similarity = 0.0
                for word in intersection:
                    if word in word_freq:
                        similarity += word_freq[word]
                
                # Normalize by the lengths to favor sentences with more information overlap
                norm = math.log(1 + len(sentence_i_words) + len(sentence_j_words))
                if norm > 0:
                    similarity = similarity / norm
                
                # Apply a threshold to remove very weak connections
                if similarity > 0.05:
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity  # Make the matrix symmetric
        
        return similarity_matrix
    
    def _apply_page_rank(self, similarity_matrix, damping=0.85, max_iterations=50, tolerance=1e-6):
        """Apply the PageRank algorithm to rank sentences by importance"""
        n = len(similarity_matrix)
        
        if n == 0:
            return {}
        
        # Initialize PageRank scores
        scores = [1.0 / n] * n
        
        # Create column-normalized transition matrix for faster convergence
        transition_matrix = []
        for j in range(n):
            column_sum = sum(similarity_matrix[i][j] for i in range(n))
            if column_sum == 0:
                transition_matrix.append([0.0] * n)
            else:
                transition_matrix.append([similarity_matrix[i][j] / column_sum for i in range(n)])
        
        # Iterative PageRank computation
        for _ in range(max_iterations):
            new_scores = [(1 - damping) / n] * n
            
            for i in range(n):
                for j in range(n):
                    if similarity_matrix[j][i] > 0:
                        new_scores[i] += damping * scores[j] * transition_matrix[i][j]
            
            # Check for convergence
            score_diff = sum(abs(new_scores[i] - scores[i]) for i in range(n))
            if score_diff < tolerance:
                break
            
            scores = new_scores
        
        # Return scores as a dictionary
        return {i: scores[i] for i in range(n)}
    
    def _score_sentences(self, sentences, word_freq):
        """Score sentences using PageRank and semantic enhancements"""
        # For very short texts, use a simpler approach
        if len(sentences) <= 3:
            return {i: 1.0 for i in range(len(sentences))}
        
        # Build sentence similarity graph
        similarity_matrix = self._build_sentence_graph(sentences, word_freq)
        
        # Apply PageRank algorithm
        scores = self._apply_page_rank(similarity_matrix)
        
        # Apply position-based heuristics
        if len(sentences) > 2:
            # First paragraph sentences often contain important context
            for i in range(min(3, len(sentences))):
                if i in scores:
                    scores[i] *= 1.25
            
            # Last few sentences often contain conclusions
            for i in range(max(0, len(sentences) - 3), len(sentences)):
                if i in scores:
                    scores[i] *= 1.15
        
        # Apply entity-based importance
        all_entities = []
        for entity_type in self.entities:
            all_entities.extend(self.entities[entity_type])
        
        if all_entities:
            for i, sentence in enumerate(sentences):
                # Count entities in this sentence
                entity_matches = sum(1 for entity in all_entities if entity.lower() in sentence.lower())
                if entity_matches > 0 and i in scores:
                    scores[i] *= (1 + 0.1 * min(entity_matches, 3))  # Boost based on entity count
        
        # Apply length-based normalization (penalize very short or very long sentences)
        for i, sentence in enumerate(sentences):
            words = sentence.split()
            word_count = len(words)
            
            if i in scores:
                # Penalize extremely short sentences
                if word_count <= 5:
                    scores[i] *= 0.7
                # Penalize overly long sentences that might be hard to read
                elif word_count > 40:
                    scores[i] *= 0.8
        
        return scores
    
    def _analyze_sentiment(self, text):
        """Analyze the sentiment of text to provide tone information"""
        if not text or not self.use_advanced_nlp:
            return {"sentiment": "neutral", "score": 0.0}
        
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            sentiment_scores = sia.polarity_scores(text)
            
            # Determine overall sentiment
            compound_score = sentiment_scores['compound']
            if compound_score >= 0.05:
                sentiment = "positive"
            elif compound_score <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"
                
            return {
                "sentiment": sentiment,
                "score": compound_score,
                "details": sentiment_scores
            }
        except Exception as e:
            logger.debug(f"Sentiment analysis failed: {e}")
            return {"sentiment": "neutral", "score": 0.0}
    
    def _detect_category(self, text):
        """Detect the category of content for topical organization"""
        if not text:
            return {"category": "unknown", "confidence": 0.0}
        
        # Define category keywords
        categories = {
            "technology": ['software', 'hardware', 'programming', 'code', 'algorithm', 'data', 'internet', 
                         'website', 'app', 'digital', 'tech', 'computer', 'artificial intelligence', 
                         'machine learning', 'developers', 'cybersecurity', 'cloud', 'automation'],
            
            "business": ['company', 'business', 'market', 'stock', 'investment', 'profit', 'revenue', 
                        'sales', 'customer', 'strategy', 'startup', 'entrepreneur', 'industry', 
                        'commerce', 'corporate', 'management', 'finance', 'economic'],
            
            "health": ['health', 'medical', 'doctor', 'patient', 'hospital', 'disease', 'treatment', 
                      'therapy', 'medicine', 'wellness', 'fitness', 'nutrition', 'mental health', 
                      'healthcare', 'diagnosis', 'symptoms', 'surgery', 'clinical'],
            
            "news": ['news', 'report', 'update', 'latest', 'breaking', 'headlines', 'current events', 
                    'world', 'national', 'announcement', 'press release', 'journalist', 'media'],
            
            "politics": ['government', 'political', 'policy', 'election', 'vote', 'campaign', 'democrat', 
                        'republican', 'law', 'legislation', 'congress', 'senate', 'parliament', 
                        'president', 'politician', 'party', 'administration'],
            
            "education": ['education', 'school', 'university', 'college', 'student', 'teacher', 
                         'learning', 'academic', 'study', 'research', 'knowledge', 'curriculum', 
                         'classroom', 'professor', 'course', 'degree', 'lecture'],
            
            "science": ['science', 'scientific', 'research', 'study', 'experiment', 'theory', 
                       'discovery', 'biology', 'chemistry', 'physics', 'astronomy', 'psychology', 
                       'neuroscience', 'environmental', 'ecology', 'laboratory'],
            
            "entertainment": ['entertainment', 'movie', 'film', 'television', 'music', 'celebrity', 
                            'actor', 'actress', 'director', 'artist', 'game', 'show', 'performance', 
                            'theater', 'concert', 'streaming', 'media']
        }
        
        # Count category keywords in the text
        text_lower = text.lower()
        category_scores = {category: 0 for category in categories}
        
        for category, keywords in categories.items():
            for keyword in keywords:
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
                category_scores[category] += count
        
        # Get total keyword matches
        total_matches = sum(category_scores.values())
        
        # If no matches, return unknown
        if total_matches == 0:
            return {"category": "unknown", "confidence": 0.0}
        
        # Find category with highest count
        top_category = max(category_scores.items(), key=lambda x: x[1])
        
        # Calculate confidence score (ratio of top category matches to total matches)
        confidence = top_category[1] / total_matches
        
        self.content_category = {
            "category": top_category[0],
            "confidence": confidence,
            "distribution": {k: v/total_matches for k, v in category_scores.items() if v > 0}
        }
        
        return self.content_category
    
    def extract_keywords(self, text, max_keywords=8):
        """Extract the most important keywords and keyphrases from text"""
        if not text:
            return []
        
        # Get word frequencies
        word_freq = self._compute_word_frequencies(text)
        
        # Extract single-word keywords
        keywords = [(word, score) for word, score in word_freq.items() 
                   if len(word) > 3 and score > 0.3]
        
        # Extract multi-word keyphrases if advanced NLP is available
        keyphrases = []
        if self.use_advanced_nlp:
            try:
                # Extract noun phrases using POS tagging
                sentences = nltk.sent_tokenize(text)
                for sentence in sentences:
                    words = nltk.word_tokenize(sentence)
                    tagged = nltk.pos_tag(words)
                    
                    # Find noun phrases (adjective + noun or noun + noun)
                    i = 0
                    while i < len(tagged) - 1:
                        # Adjective + Noun pattern
                        if tagged[i][1].startswith('JJ') and tagged[i+1][1].startswith('NN'):
                            phrase = f"{tagged[i][0]} {tagged[i+1][0]}".lower()
                            if all(word not in self.stop_words for word in phrase.split()):
                                # Calculate score based on constituent words
                                score = sum(word_freq.get(word, 0) for word in phrase.split()) / 2
                                keyphrases.append((phrase, score))
                                
                        # Noun + Noun pattern (compound nouns)
                        elif tagged[i][1].startswith('NN') and tagged[i+1][1].startswith('NN'):
                            phrase = f"{tagged[i][0]} {tagged[i+1][0]}".lower()
                            if all(word not in self.stop_words for word in phrase.split()):
                                # Calculate score based on constituent words
                                score = sum(word_freq.get(word, 0) for word in phrase.split()) / 2
                                keyphrases.append((phrase, score))
                        
                        i += 1
            except Exception as e:
                logger.debug(f"Keyphrase extraction failed: {e}")
        
        # Combine keywords and keyphrases, remove duplicates
        all_terms = {}
        for term, score in keywords + keyphrases:
            if term not in all_terms or score > all_terms[term]:
                all_terms[term] = score
        
        # Add named entities with high scores
        for entity_type in self.entities:
            for entity in self.entities[entity_type]:
                if entity.lower() not in all_terms:
                    all_terms[entity.lower()] = 0.9  # High score for named entities
        
        # Convert to list and sort by score
        terms_list = [(term, score) for term, score in all_terms.items()]
        terms_list.sort(key=lambda x: x[1], reverse=True)
        
        # Store for later use
        self.topic_keywords = [term for term, _ in terms_list[:max_keywords]]
        
        # Return the top terms
        return terms_list[:max_keywords]
    
    def summarize(self, text, ratio=0.2, min_sentences=None, max_sentences=None):
        """
        Generate a comprehensive summary with metadata
        
        Parameters:
        - text: Input text to summarize
        - ratio: Target summary length ratio (0.0 to 1.0)
        - min_sentences: Minimum number of sentences to include
        - max_sentences: Maximum number of sentences to include
        
        Returns:
        - Dictionary with summary text and metadata
        """
        if not text or len(text.strip()) == 0:
            return {
                "summary": "",
                "metadata": {"error": "No content to summarize"}
            }
        
        # Record original text length
        original_text_length = len(text.split())
        
        # Extract named entities for context
        self._extract_entities(text)
        
        # Extract sentences
        sentences = self._extract_sentences(text)
        
        # Handle very short texts
        if len(sentences) <= 3:
            return {
                "summary": text,
                "metadata": {
                    "original_length": original_text_length,
                    "summary_length": original_text_length,
                    "compression_ratio": 1.0,
                    "sentence_count": len(sentences),
                    "reading_time_min": max(1, round(original_text_length / 200))
                }
            }
        
        # Calculate word frequencies for importance weighting
        word_freq = self._compute_word_frequencies(text)
        
        # Score sentences using PageRank and enhancements
        sentence_scores = self._score_sentences(sentences, word_freq)
        
        # Calculate number of sentences to include in the summary
        if min_sentences is not None and max_sentences is not None:
            target_count = max(min_sentences, min(max_sentences, int(len(sentences) * ratio)))
        elif min_sentences is not None:
            target_count = max(min_sentences, int(len(sentences) * ratio))
        elif max_sentences is not None:
            target_count = min(max_sentences, int(len(sentences) * ratio))
        else:
            target_count = max(1, int(len(sentences) * ratio))
        
        # Get the highest-scoring sentences
        top_indices = heapq.nlargest(target_count, sentence_scores, key=sentence_scores.get)
        
        # Sort indices to maintain original sentence order
        top_indices.sort()
        
        # Create the summary from selected sentences
        summary_sentences = [sentences[i] for i in top_indices]
        summary_text = ' '.join(summary_sentences)
        
        # Calculate metadata
        summary_length = len(summary_text.split())
        compression_ratio = summary_length / original_text_length if original_text_length > 0 else 1.0
        
        # Get additional analytics
        sentiment = self._analyze_sentiment(text)
        category = self._detect_category(text)
        keywords = self.extract_keywords(text)
        
        # Create metadata object
        metadata = {
            "original_length": original_text_length,
            "summary_length": summary_length,
            "compression_ratio": compression_ratio,
            "sentence_count": len(sentences),
            "summary_sentence_count": len(summary_sentences),
            "reading_time_min": max(1, round(original_text_length / 200)),
            "summary_reading_time_min": max(1, round(summary_length / 200)),
            "sentiment": sentiment["sentiment"],
            "category": category["category"] if category["confidence"] > 0.3 else "general",
            "keywords": [kw for kw, _ in keywords[:5]],
            "entities": {k: v for k, v in self.entities.items() if v}
        }
        
        return {
            "summary": summary_text,
            "metadata": metadata
        }

def setup_argparse():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract, fetch, and summarize URLs from Photon results with enhanced NLP features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--root-dir", type=str, default=PHOTON_ROOT,
                        help="Root directory containing Photon results")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of URLs to process in each batch")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="Timeout for HTTP requests and API calls (seconds)")
    parser.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_API_URL,
                        help="Ollama API endpoint URL")
    parser.add_argument("--max-content", type=int, default=MAX_CONTENT_LENGTH,
                        help="Maximum content length to send for summarization")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with additional information")
    parser.add_argument("--auto", action="store_true",
                        help="Start in automatic mode (no prompts between batches)")
    parser.add_argument("--format", choices=["text", "markdown", "json"], default="markdown",
                        help="Output format for summaries")
    parser.add_argument("--depth", choices=["short", "medium", "detailed"], default="medium",
                        help="Depth of summaries to generate")
    
    return parser.parse_args()

def configure_logging(verbose, debug):
    """Configure logging level based on command line arguments."""
    if debug:
        logger.setLevel(logging.DEBUG)
    elif verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

def print_header(text):
    """Print a formatted header."""
    try:
        width = min(os.get_terminal_size().columns, 80)
    except OSError:
        width = 80
    print(f"\n\033[1;36m{'=' * width}\n{text.center(width)}\n{'=' * width}\033[0m\n")

def extract_urls_from_file(file_path):
    """Extract URLs from a file using regex."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        # Find all URLs in the content
        urls = URL_PATTERN.findall(content)
        
        # Clean and deduplicate URLs
        clean_urls = []
        seen = set()
        for url in urls:
            # Clean up the URL to remove trailing punctuation
            url = url.rstrip('.,;:\'\"!?)')
            
            # Skip duplicates
            if url not in seen:
                seen.add(url)
                clean_urls.append(url)
        
        return clean_urls
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return []

def get_domain(url):
    """Extract domain from URL."""
    parsed = urlparse(url)
    domain = parsed.netloc
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain

def fetch_webpage_content(url, timeout=DEFAULT_TIMEOUT):
    """Fetch and extract text content from a webpage with improved parsing."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.9"
        }
        
        # Make the request
        response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        
        # Handle common error cases
        if response.status_code == 404:
            logger.warning(f"404 Not Found: {url}")
            return {
                "title": "404 Not Found",
                "text": "",
                "url": url,
                "status_code": 404,
                "error": True
            }
        
        if response.status_code >= 400:
            logger.warning(f"HTTP Error {response.status_code}: {url}")
            return {
                "title": f"HTTP Error {response.status_code}",
                "text": "",
                "url": url,
                "status_code": response.status_code,
                "error": True
            }
        
        # Check content type
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type and 'application/json' not in content_type:
            return {
                "title": f"Unsupported content type: {content_type}",
                "text": "",
                "url": url,
                "status_code": response.status_code,
                "error": True
            }
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else "No title found"
        
        # Remove unwanted elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", 
                         "meta", "button", "svg", "iframe", "noscript"]):
            tag.extract()
        
        # Extract meta description for context
        meta_description = ""
        meta_desc_tag = soup.find("meta", attrs={"name": "description"})
        if meta_desc_tag:
            meta_description = meta_desc_tag.get("content", "")
        
        # Try different content extraction strategies
        content_text = ""
        
        # Strategy 1: Look for article or main content tag
        main_content = soup.find(["article", "main", "section", "div"], 
                                class_=lambda c: c and any(s in str(c).lower() for s in 
                                                     ["content", "article", "main", "body", "post"]))
        if main_content:
            # Extract paragraphs from main content
            paragraphs = main_content.find_all("p")
            if paragraphs:
                content_text = "\n\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        
        # Strategy 2: If no content found, try all paragraphs
        if not content_text:
            paragraphs = soup.find_all("p")
            if paragraphs:
                content_text = "\n\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        
        # Strategy 3: Fallback to general text extraction
        if not content_text:
            # Look for content in specific divs or sections
            content_containers = soup.select("article, .content, .main, #content, #main, .post, .entry")
            if content_containers:
                content_text = "\n\n".join([el.get_text(strip=True) for el in content_containers])
            else:
                # Last resort: extract all text
                content_text = soup.get_text(separator="\n\n", strip=True)
        
        # Clean up extracted text
        content_text = re.sub(r'\n{3,}', '\n\n', content_text)  # Remove excess newlines
        content_text = re.sub(r'\s{2,}', ' ', content_text)     # Remove excess spaces
        
        # Add meta description to the beginning if available
        if meta_description and meta_description not in content_text:
            content_text = meta_description + "\n\n" + content_text
        
        # Return content with success status
        return {
            "title": title,
            "text": content_text,
            "url": url,
            "status_code": response.status_code,
            "error": False
        }
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return {
            "title": f"Error: {str(e)[:100]}",
            "text": "",
            "url": url,
            "status_code": 0,  # Use 0 to indicate a connection/non-HTTP error
            "error": True
        }

def check_ollama_connection(url, timeout=5):
    """Check if Ollama server is running and reachable."""
    try:
        base_url = url
        if "/api/generate" in url:
            base_url = url.split("/api/generate")[0]
        
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        response.raise_for_status()
        
        # Check if we got a valid response with models
        data = response.json()
        if "models" in data and len(data["models"]) > 0:
            logger.debug(f"Available Ollama models: {[m['name'] for m in data['models']]}")
            return True
        else:
            logger.warning("No models found in Ollama")
            return False
    except Exception as e:
        logger.error(f"Ollama connection failed: {e}")
        return False

def create_spinner():
    """Create a simple spinner to indicate progress during API calls."""
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

def summarize_with_enhanced_textrank(content, depth, timeout=DEFAULT_TIMEOUT):
    """Generate comprehensive summary using enhanced TextRank algorithm."""
    text = content["text"]
    if not text:
        return "No content to summarize."
    
    print("Generating enhanced summary with TextRank...", end="", flush=True)
    stop_spinner = create_spinner()
    
    try:
        # Create enhanced TextRank summarizer
        summarizer = EnhancedTextRankSummarizer()
        
        # Configure parameters based on desired depth
        if depth == "short":
            summary_params = {
                "ratio": 0.05,  # Very concise summary
                "min_sentences": 1,
                "max_sentences": 3
            }
        elif depth == "medium":
            summary_params = {
                "ratio": 0.15,  # Balanced summary
                "min_sentences": 3,
                "max_sentences": 7
            }
        else:  # detailed
            summary_params = {
                "ratio": 0.25,  # Comprehensive summary
                "min_sentences": 5,
                "max_sentences": 15
            }
        
        # Generate the summary with metadata
        result = summarizer.summarize(text, **summary_params)
        summary = result["summary"]
        metadata = result["metadata"]
        
        # Handle empty summary (can happen with poor quality content)
        if not summary:
            if len(text) > 200:
                summary = text[:200] + "..."
            else:
                summary = text
            
            stop_spinner()
            print(" Done! (Used original text due to quality issues)")
            return summary
        
        # Add context and metadata based on depth
        if depth in ["medium", "detailed"]:
            # Extract keywords
            keywords = summarizer.extract_keywords(text)
            if keywords:
                key_terms = ", ".join([term for term, _ in keywords[:5]])
                summary += f"\n\nKey terms: {key_terms}"
            
            # Add category for medium and detailed summaries
            if "category" in metadata and metadata["category"] != "general":
                summary += f"\nTopic: {metadata['category'].capitalize()}"
            
            # Add additional metadata for detailed summaries
            if depth == "detailed":
                # Add sentiment information
                if "sentiment" in metadata and metadata["sentiment"] != "neutral":
                    summary += f"\nTone: {metadata['sentiment'].capitalize()}"
                
                # Add reading time
                if "reading_time_min" in metadata:
                    summary += f"\nOriginal reading time: ~{metadata['reading_time_min']} min"
                    
                # Add named entities if available
                entities = []
                if "entities" in metadata:
                    for entity_type, values in metadata["entities"].items():
                        if values and entity_type in ["PERSON", "ORGANIZATION", "LOCATION"]:
                            entity_str = ", ".join(values[:3])
                            entities.append(f"{entity_type.capitalize()}: {entity_str}")
                
                if entities:
                    summary += "\n\nNamed entities: " + "; ".join(entities)
        
        stop_spinner()
        print(" Done!")
        return summary
    except Exception as e:
        stop_spinner()
        print(f" Error: {str(e)[:100]}")
        logger.error(f"Enhanced TextRank summarization error: {e}")
        return f"Error generating summary: {str(e)[:100]}"

def summarize_with_ollama(content, model, depth, api_url=DEFAULT_OLLAMA_API_URL, timeout=DEFAULT_TIMEOUT):
    """Summarize content using Ollama API with improved prompt engineering."""
    # Truncate content if too long
    text = content["text"]
    if len(text) > MAX_CONTENT_LENGTH:
        text = text[:MAX_CONTENT_LENGTH] + "... [truncated]"
    
    # Build advanced prompt based on depth
    if depth == "short":
        prompt = dedent(f"""
        Create a concise 1-2 sentence summary of the following content. 
        Focus on the core information and key points only.
        Maintain factual accuracy and exclude your own opinions.

        CONTENT:
        {text}

        SUMMARY:
        """)
    elif depth == "medium":
        prompt = dedent(f"""
        Summarize the following content in a single coherent paragraph (3-5 sentences).
        Include the main points, key details, and important context.
        Be factual, objective, and precise. Synthesize the information rather than just
        shortening the text.

        CONTENT:
        {text}

        SUMMARY:
        """)
    else:  # detailed
        prompt = dedent(f"""
        Create a comprehensive summary of the following content. Include:
        1. Main ideas and key supporting points (5-7 sentences)
        2. Important details, facts, and figures
        3. Key terms/concepts with brief explanations
        4. Overall topic/theme
        5. Tone or perspective of the content

        Be factual, objective, and well-structured.

        CONTENT:
        {text}

        DETAILED SUMMARY:
        """)
    
    print("Generating summary with Ollama...", end="", flush=True)
    stop_spinner = create_spinner()
    
    try:
        response = requests.post(
            api_url,
            json={
                "model": model, 
                "prompt": prompt, 
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for factual summaries
                    "top_p": 0.95        # High precision
                }
            },
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()
        summary = result.get("response", "No summary generated.")
        
        stop_spinner()
        print(" Done!")
        return summary.strip()
    except Exception as e:
        stop_spinner()
        print(f" Error: {str(e)[:100]}")
        logger.error(f"Ollama API error: {e}")
        return f"Error generating summary: {str(e)[:100]}"

def summarize_with_openai(content, api_key, depth, model="gpt-3.5-turbo", timeout=DEFAULT_TIMEOUT):
    """Summarize content using OpenAI API with improved prompts."""
    text = content["text"]
    title = content["title"]
    
    if len(text) > MAX_CONTENT_LENGTH:
        text = text[:MAX_CONTENT_LENGTH] + "... [truncated]"
    
    # Create system prompt based on depth
    if depth == "short":
        system_prompt = "You are an expert summarizer. Create a concise 1-2 sentence summary capturing only the essential information. Be objective and factual."
    elif depth == "medium":
        system_prompt = "You are an expert summarizer. Create a single coherent paragraph (3-5 sentences) that captures the main points and key context. Be objective and factual."
    else:  # detailed
        system_prompt = "You are an expert summarizer. Create a comprehensive, well-structured summary that includes main ideas, important details, key concepts, and overall theme. Be objective, factual, and informative."
    
    # Create user prompt with title for context
    user_prompt = f"Title: {title}\n\nContent to summarize:\n{text}"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,  # Lower temperature for more focused summaries
        "max_tokens": 1024 if depth == "detailed" else 512  # Adjust token limit based on depth
    }
    
    print(f"Generating summary with {model}...", end="", flush=True)
    stop_spinner = create_spinner()
    
    try:
        response = requests.post(
            DEFAULT_OPENAI_API_URL,
            headers=headers,
            json=data,
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()
        summary = result["choices"][0]["message"]["content"]
        
        stop_spinner()
        print(" Done!")
        return summary.strip()
    except Exception as e:
        stop_spinner()
        print(f" Error: {str(e)[:100]}")
        logger.error(f"OpenAI API error: {e}")
        return f"Error generating summary: {str(e)[:100]}"

def summarize_with_huggingface(content, api_key, model=None, depth="medium", timeout=DEFAULT_TIMEOUT):
    """Summarize content using Hugging Face API with improved parameters."""
    if not model:
        if depth == "detailed":
            model = "facebook/bart-large-cnn"  # Better for longer summaries
        else:
            model = "sshleifer/distilbart-cnn-12-6"  # Faster for short/medium summaries
    
    text = content["text"]
    if len(text) > MAX_CONTENT_LENGTH:
        text = text[:MAX_CONTENT_LENGTH] + "... [truncated]"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Adjust parameters based on depth
    parameters = {}
    if depth == "short":
        parameters = {
            "max_length": 50, 
            "min_length": 20,
            "do_sample": False,
            "truncation": True
        }
    elif depth == "medium":
        parameters = {
            "max_length": 150, 
            "min_length": 80,
            "do_sample": False,
            "truncation": True
        }
    else:  # detailed
        parameters = {
            "max_length": 300, 
            "min_length": 150,
            "do_sample": False,
            "truncation": True
        }
    
    payload = {
        "inputs": text,
        "parameters": parameters
    }
    
    print(f"Generating summary with Hugging Face ({model})...", end="", flush=True)
    stop_spinner = create_spinner()
    
    try:
        response = requests.post(
            f"{DEFAULT_HF_API_URL}{model}",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()
        
        stop_spinner()
        print(" Done!")
        
        if isinstance(result, list) and len(result) > 0:
            summary = result[0].get("summary_text", "No summary generated.").strip()
            
            # Clean up and improve formatting
            summary = re.sub(r'\s+', ' ', summary)  # Fix spacing issues
            summary = summary.replace(" .", ".")     # Fix spacing before periods
            summary = summary.replace(" ,", ",")     # Fix spacing before commas
            
            return summary
        else:
            return "No summary generated."
    except Exception as e:
        stop_spinner()
        print(f" Error: {str(e)[:100]}")
        logger.error(f"Hugging Face API error: {e}")
        return f"Error generating summary: {str(e)[:100]}"

def summarize_content(content, provider, model, depth, api_keys, api_urls, timeout=DEFAULT_TIMEOUT):
    """Summarize content using the selected provider with enhanced capabilities."""
    if not content["text"]:
        return f"No content available for {content['url']}"
    
    if provider == "textrank":
        return summarize_with_enhanced_textrank(content, depth, timeout)
    elif provider == "ollama":
        return summarize_with_ollama(content, model, depth, api_urls["ollama"], timeout)
    elif provider == "openai":
        if not api_keys["openai"]:
            return "Error: OpenAI API key not provided."
        return summarize_with_openai(content, api_keys["openai"], depth, model, timeout)
    elif provider == "huggingface":
        if not api_keys["huggingface"]:
            return "Error: Hugging Face API key not provided."
        return summarize_with_huggingface(content, api_keys["huggingface"], model, depth, timeout)
    else:
        return f"Error: Unknown provider '{provider}'."

def generate_hierarchical_batch_summary(summaries, depth="medium"):
    """
    Generate a hierarchical batch summary that identifies common themes and topics
    
    Args:
        summaries: List of summary dictionaries
        depth: Desired summary depth
        
    Returns:
        Dictionary with batch summary info
    """
    if not summaries:
        return {
            "overview": "No valid summaries to create an overall summary.",
            "themes": [],
            "count": 0
        }
    
    # Combine all the individual summaries for text analysis
    combined_text = "\n\n".join([
        f"Title: {s['title']}\nSummary: {s['summary']}" 
        for s in summaries 
        if s["summary"] and not s["summary"].startswith("Error")
    ])
    
    if not combined_text:
        return {
            "overview": "No valid summaries to create an overall summary.",
            "themes": [],
            "count": 0
        }
    
    try:
        # Use the EnhancedTextRank for analysis
        summarizer = EnhancedTextRankSummarizer()
        
        # Extract common themes (keywords and topics)
        keywords = summarizer.extract_keywords(combined_text, max_keywords=10)
        
        # Detect category/topic
        category = summarizer._detect_category(combined_text)
        
        # Extract named entities
        entities = summarizer._extract_entities(combined_text)
        
        # Generate hierarchical summary based on depth
        if depth == "short":
            # Very concise batch overview
            overview = summarizer.summarize(
                combined_text, ratio=0.03, min_sentences=1, max_sentences=2
            )["summary"]
            
            # Include top keywords/themes only
            themes = [kw for kw, _ in keywords[:3]]
            
        elif depth == "medium":
            # Moderate batch overview
            overview = summarizer.summarize(
                combined_text, ratio=0.1, min_sentences=2, max_sentences=4
            )["summary"]
            
            # Include more themes/topics with categories
            themes = [kw for kw, _ in keywords[:5]]
            
            # Add category if available
            if category["category"] != "unknown" and category["confidence"] > 0.3:
                overview += f"\n\nPrimary topic: {category['category'].capitalize()}"
            
        else:  # detailed
            # Comprehensive batch overview
            overview = summarizer.summarize(
                combined_text, ratio=0.15, min_sentences=3, max_sentences=6
            )["summary"]
            
            # Include detailed themes and entities
            themes = [kw for kw, _ in keywords[:8]]
            
            # Add category breakdown if available
            if category["category"] != "unknown" and category["confidence"] > 0.3:
                overview += f"\n\nPrimary topic: {category['category'].capitalize()}"
                
                # Add distribution of topics if available
                if "distribution" in category:
                    topic_dist = sorted(
                        [(k, v) for k, v in category["distribution"].items() if v > 0.1],
                        key=lambda x: x[1],
                        reverse=True
                    )
                    if topic_dist:
                        topic_str = ", ".join([f"{k.capitalize()} ({int(v*100)}%)" for k, v in topic_dist[:3]])
                        overview += f"\nTopic distribution: {topic_str}"
            
            # Add named entities if any significant ones were found
            significant_entities = []
            for entity_type in ["ORGANIZATION", "PERSON", "LOCATION"]:
                if entities.get(entity_type):
                    entity_items = entities[entity_type][:3]  # Top 3 entities of each type
                    if entity_items:
                        significant_entities.append(f"{entity_type.capitalize()}: {', '.join(entity_items)}")
            
            if significant_entities:
                overview += "\n\nCommon entities: " + "; ".join(significant_entities)
        
        # Return structured batch summary
        batch_summary = {
            "overview": overview,
            "themes": themes,
            "count": len([s for s in summaries if s["summary"] and not s["summary"].startswith("Error")]),
            "category": category["category"] if category["confidence"] > 0.3 else "mixed"
        }
        
        # Add entity count if available for detailed summaries
        if depth == "detailed":
            entity_counts = {}
            for entity_type, values in entities.items():
                if values:
                    entity_counts[entity_type] = len(values)
            
            if entity_counts:
                batch_summary["entities"] = entity_counts
        
        return batch_summary
        
    except Exception as e:
        logger.error(f"Error generating hierarchical batch summary: {e}")
        return {
            "overview": "Could not generate an overall batch summary due to an error.",
            "themes": [],
            "count": len([s for s in summaries if s["summary"] and not s["summary"].startswith("Error")])
        }

def process_batch(urls, provider, model, depth, api_keys, api_urls, timeout=DEFAULT_TIMEOUT, automatic_mode=False):
    """Process a batch of URLs: fetch, summarize, and display results with enhanced analytics."""
    valid_contents = []
    invalid_urls = []
    summaries = []
    all_404 = True  # Flag to track if all URLs result in 404 errors
    
    print(f"\nProcessing {len(urls)} URLs...")
    
    # Fetch content from URLs using parallel processing
    contents = []
    with ThreadPoolExecutor(max_workers=min(len(urls), 5)) as executor:
        future_to_url = {executor.submit(fetch_webpage_content, url, timeout): url for url in urls}
        for i, future in enumerate(as_completed(future_to_url), 1):
            url = future_to_url[future]
            try:
                content = future.result()
                # Print appropriate message based on content status
                if content["error"]:
                    if content["status_code"] == 404:
                        # In automatic mode, don't show 404 messages
                        if not automatic_mode:
                            print(f"[{i}/{len(urls)}] Skipping 404 Not Found: {url}")
                        invalid_urls.append(url)
                    else:
                        print(f"[{i}/{len(urls)}] Error: {content['title']} for {url}")
                        contents.append(content)
                        all_404 = False  # Found an error that's not a 404
                else:
                    # Calculate content size metrics
                    char_count = len(content['text'])
                    word_count = len(content['text'].split())
                    reading_time = max(1, round(word_count / 200))  # Estimated reading time in minutes
                    
                    print(f"[{i}/{len(urls)}] Fetched: {url} ({word_count} words, ~{reading_time} min read)")
                    contents.append(content)
                    valid_contents.append(content)
                    all_404 = False  # Found at least one valid URL
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                print(f"[{i}/{len(urls)}] Error fetching {url}: {str(e)[:100]}")
                contents.append({
                    "title": f"Error fetching {url}",
                    "text": "",
                    "url": url,
                    "status_code": 0,
                    "error": True
                })
    
    # Log the number of URLs being skipped due to 404
    if invalid_urls:
        logger.info(f"Skipping {len(invalid_urls)} URLs due to 404 errors")
        if not automatic_mode:  # Only show in non-automatic mode
            print(f"\n\033[1;33mSkipping {len(invalid_urls)} URLs due to 404 errors\033[0m")
    
    # Track content categories for batch analysis
    content_categories = Counter()
    
    # Summarize content (excluding 404 pages)
    for i, content in enumerate([c for c in contents if not (c.get("status_code") == 404)], 1):
        if not content["text"]:
            print(f"\n\033[1;33m[{i}/{len(contents) - len(invalid_urls)}] Error: Could not extract content from {content['url']}\033[0m")
            summaries.append({
                "title": content["title"],
                "url": content["url"],
                "summary": "No content could be extracted from this URL."
            })
            continue
            
        print(f"\n\033[1;33m[{i}/{len(contents) - len(invalid_urls)}] Summarizing: {content['title']}\033[0m")
        summary = summarize_content(content, provider, model, depth, api_keys, api_urls, timeout)
        
        # Attempt to detect content category
        if provider == "textrank":
            # We can use the TextRank categorization directly
            try:
                summarizer = EnhancedTextRankSummarizer()
                category = summarizer._detect_category(content["text"])
                if category["category"] != "unknown" and category["confidence"] > 0.3:
                    content_categories[category["category"]] += 1
            except Exception:
                pass
        
        domain = get_domain(content["url"])
        print(f"\n\033[1;32m[{i}/{len(contents) - len(invalid_urls)}] Summary for {content['title']}\033[0m")
        print(f"\033[0;36m{content['url']}\033[0m \033[0;90m({domain})\033[0m")
        print(f"{summary}")
        print("-" * 60)
        
        summaries.append({
            "title": content["title"],
            "url": content["url"],
            "summary": summary
        })
    
    # Generate hierarchical batch summary
    if valid_contents and summaries and any(s["summary"] and not s["summary"].startswith("Error") for s in summaries):
        print("\nGenerating comprehensive batch analysis...")
        
        batch_summary = generate_hierarchical_batch_summary(summaries, depth)
        
        # Display batch overview
        print("\n\033[1;32mBATCH SUMMARY:\033[0m")
        print(batch_summary["overview"])
        
        # Display common themes/topics
        if batch_summary["themes"]:
            print("\n\033[1;32mCOMMON THEMES:\033[0m")
            print(", ".join(batch_summary["themes"]))
        
        # Display content category distribution for detailed summaries
        if depth == "detailed" and content_categories:
            top_categories = content_categories.most_common(3)
            if top_categories:
                print("\n\033[1;32mCONTENT CATEGORIES:\033[0m")
                for category, count in top_categories:
                    percentage = int((count / len(valid_contents)) * 100)
                    print(f"{category.capitalize()}: {count} URLs ({percentage}%)")
        
        print("=" * 60)
        
        return summaries, batch_summary, all_404
    
    return summaries, {"overview": "No valid summaries generated for this batch.", "themes": [], "count": 0}, all_404

def collect_urls_from_folders(root_dir):
    """Collect URLs from all subfolders in the root directory with improved detection."""
    folder_urls = {}
    
    if not os.path.isdir(root_dir):
        logger.error(f"Root directory {root_dir} does not exist.")
        return folder_urls
    
    # List all subfolders
    subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    if not subfolders:
        logger.warning(f"No subfolders found in {root_dir}.")
        return folder_urls
    
    # Process each subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(root_dir, subfolder)
        urls = []
        
        # Get all text files in the subfolder
        text_files = [f for f in os.listdir(subfolder_path) 
                      if (f.endswith('.txt') or f.endswith('.lst') or f.endswith('.log')) 
                      and f != 'summary.txt' and os.path.isfile(os.path.join(subfolder_path, f))]
        
        # Extract URLs from each text file
        for text_file in text_files:
            file_path = os.path.join(subfolder_path, text_file)
            file_urls = extract_urls_from_file(file_path)
            urls.extend(file_urls)
        
        # Remove duplicates while preserving order
        unique_urls = []
        seen = set()
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        if unique_urls:
            folder_urls[subfolder] = unique_urls
    
    return folder_urls

def choose_provider():
    """Let user choose the summarization provider."""
    print("\nChoose a summarization provider:")
    print("1) Enhanced TextRank (built-in, no API required)")
    print("2) Ollama (local)")
    print("3) OpenAI (ChatGPT)")
    print("4) Hugging Face")
    
    while True:
        choice = input("Enter your choice (1-4): ").strip()
        if choice == "1":
            return "textrank"
        elif choice == "2":
            return "ollama"
        elif choice == "3":
            return "openai"
        elif choice == "4":
            return "huggingface"
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

def choose_model(provider):
    """Let user choose the model for the selected provider."""
    if provider == "textrank":
        # TextRank has no model options
        return "textrank"
    
    elif provider == "ollama":
        print("\nChoose an Ollama model:")
        print("1) gemma3:12b (best for summarization)")
        print("2) llama3:latest (balanced)")
        print("3) deepseek-r1:8b (faster)")
        print("4) Other (enter name)")
        
        choice = input("Enter your choice (1-4): ").strip()
        if choice == "1":
            return "gemma3:12b"
        elif choice == "2":
            return "llama3:latest"
        elif choice == "3":
            return "deepseek-r1:8b"
        elif choice == "4":
            return input("Enter the model name: ").strip()
        else:
            print("Invalid choice. Using gemma3:12b.")
            return "gemma3:12b"
    
    elif provider == "openai":
        print("\nChoose an OpenAI model:")
        print("1) gpt-3.5-turbo (fast)")
        print("2) gpt-4 (more comprehensive)")
        
        choice = input("Enter your choice (1-2): ").strip()
        if choice == "1":
            return "gpt-3.5-turbo"
        elif choice == "2":
            return "gpt-4"
        else:
            print("Invalid choice. Using gpt-3.5-turbo.")
            return "gpt-3.5-turbo"
    
    elif provider == "huggingface":
        print("\nChoose a Hugging Face model:")
        print("1) facebook/bart-large-cnn (high quality summarization)")
        print("2) sshleifer/distilbart-cnn-12-6 (faster summarization)")
        print("3) google/pegasus-xsum (concise summaries)")
        print("4) Other (enter name)")
        
        choice = input("Enter your choice (1-4): ").strip()
        if choice == "1":
            return "facebook/bart-large-cnn"
        elif choice == "2":
            return "sshleifer/distilbart-cnn-12-6"
        elif choice == "3":
            return "google/pegasus-xsum"
        elif choice == "4":
            return input("Enter the model name: ").strip()
        else:
            print("Invalid choice. Using facebook/bart-large-cnn.")
            return "facebook/bart-large-cnn"
    
    return None

def choose_depth():
    """Let user choose the summarization depth."""
    print("\nChoose summarization depth:")
    print("1) Short (1-2 sentences, concise overview)")
    print("2) Medium (one paragraph, balanced summary)")
    print("3) Detailed (comprehensive with themes and metadata)")
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        if choice == "1":
            return "short"
        elif choice == "2":
            return "medium"
        elif choice == "3":
            return "detailed"
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def get_api_keys(provider):
    """Get API keys for the selected provider."""
    api_keys = {
        "openai": DEFAULT_OPENAI_API_KEY,
        "huggingface": DEFAULT_HF_API_KEY
    }
    
    # No API keys needed for textrank
    if provider == "textrank":
        return api_keys
    
    if provider == "openai":
        key_input = input(f"\nEnter your OpenAI API key (or press Enter to use default/env var): ").strip()
        if key_input:
            api_keys["openai"] = key_input
        elif os.environ.get("OPENAI_API_KEY"):
            api_keys["openai"] = os.environ.get("OPENAI_API_KEY")
    
    elif provider == "huggingface":
        key_input = input(f"\nEnter your Hugging Face API key (or press Enter to use default/env var): ").strip()
        if key_input:
            api_keys["huggingface"] = key_input
        elif os.environ.get("HF_API_KEY"):
            api_keys["huggingface"] = os.environ.get("HF_API_KEY")
    
    return api_keys

def save_enhanced_summary(folder_path, batch_summaries, batch_overall_summaries, format_type="markdown"):
    """Save enhanced summaries to a file in the folder."""
    summary_path = os.path.join(folder_path, "summary.txt")
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            if format_type == "markdown":
                f.write(f"# Website Summaries - {current_date}\n\n")
                
                # Global summary for all batches
                if batch_overall_summaries:
                    f.write("## Overall Themes\n\n")
                    
                    # Collect all themes from all batches
                    all_themes = []
                    for batch in batch_overall_summaries:
                        if "themes" in batch and batch["themes"]:
                            all_themes.extend(batch["themes"])
                    
                    # Count theme frequencies
                    theme_counter = Counter(all_themes)
                    top_themes = theme_counter.most_common(10)
                    
                    if top_themes:
                        f.write("Common themes across all content:\n\n")
                        for theme, count in top_themes:
                            f.write(f"- **{theme}** ({count} occurrences)\n")
                        f.write("\n")
                    
                    # Overall summary of all batch overviews
                    f.write("### Executive Summary\n\n")
                    all_overviews = "\n\n".join([b["overview"] for b in batch_overall_summaries 
                                                 if "overview" in b and b["overview"]])
                    
                    # Create a super-summary using TextRank
                    if all_overviews:
                        try:
                            summarizer = EnhancedTextRankSummarizer()
                            executive_summary = summarizer.summarize(
                                all_overviews, ratio=0.3, min_sentences=3, max_sentences=5
                            )["summary"]
                            f.write(f"{executive_summary}\n\n")
                        except Exception as e:
                            logger.error(f"Error generating executive summary: {e}")
                            f.write("This folder contains multiple batches of website summaries.\n\n")
                
                # Write each batch with its URLs and summaries
                for batch_idx, (summaries, overall) in enumerate(zip(batch_summaries, batch_overall_summaries), 1):
                    f.write(f"## Batch {batch_idx} ({len(summaries)} URLs)\n\n")
                    
                    # Write batch overview
                    if "overview" in overall and overall["overview"]:
                        f.write("### Batch Overview\n\n")
                        f.write(f"{overall['overview']}\n\n")
                    
                    # Write themes if available
                    if "themes" in overall and overall["themes"]:
                        f.write("### Themes\n\n")
                        for theme in overall["themes"]:
                            f.write(f"- {theme}\n")
                        f.write("\n")
                    
                    # Write individual summaries
                    f.write("### Individual Summaries\n\n")
                    for i, summary in enumerate(summaries, 1):
                        f.write(f"#### {i}. {summary['title']}\n")
                        f.write(f"*URL*: {summary['url']}\n\n")
                        f.write(f"{summary['summary']}\n\n")
                        f.write("---\n\n")
            
            elif format_type == "text":
                # Simple text format
                f.write(f"WEBSITE SUMMARIES - {current_date}\n")
                f.write("=" * 60 + "\n\n")
                
                for batch_idx, (summaries, overall) in enumerate(zip(batch_summaries, batch_overall_summaries), 1):
                    f.write(f"BATCH {batch_idx} ({len(summaries)} URLs)\n")
                    f.write("-" * 60 + "\n\n")
                    
                    if "overview" in overall and overall["overview"]:
                        f.write("Batch Overview:\n")
                        f.write(f"{overall['overview']}\n\n")
                    
                    if "themes" in overall and overall["themes"]:
                        f.write("Themes: ")
                        f.write(", ".join(overall["themes"]) + "\n\n")
                    
                    for i, summary in enumerate(summaries, 1):
                        f.write(f"{i}. {summary['title']}\n")
                        f.write(f"URL: {summary['url']}\n\n")
                        f.write(f"{summary['summary']}\n\n")
                        f.write("-" * 40 + "\n\n")
            
            elif format_type == "json":
                # JSON format for programmatic access
                summary_data = {
                    "metadata": {
                        "generated_date": current_date,
                        "total_batches": len(batch_summaries),
                        "total_urls": sum(len(s) for s in batch_summaries),
                        "total_successful": sum("overview" in b and b["count"] > 0 for b in batch_overall_summaries)
                    },
                    "batches": []
                }
                
                for batch_idx, (summaries, overall) in enumerate(zip(batch_summaries, batch_overall_summaries), 1):
                    batch_data = {
                        "batch_id": batch_idx,
                        "overview": overall.get("overview", ""),
                        "themes": overall.get("themes", []),
                        "count": overall.get("count", 0),
                        "summaries": []
                    }
                    
                    for summary in summaries:
                        batch_data["summaries"].append({
                            "title": summary["title"],
                            "url": summary["url"],
                            "summary": summary["summary"]
                        })
                    
                    summary_data["batches"].append(batch_data)
                
                json.dump(summary_data, f, indent=2)
        
        print(f"\nSummary saved to: {summary_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving summary to {summary_path}: {e}")
        print(f"Error saving summary: {str(e)}")
        return False

def check_for_key_press():
    """Check if a key has been pressed without blocking execution."""
    try:
        # Check if there's any input ready (non-blocking)
        import select
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if ready:
            # There's input ready, read it
            key = sys.stdin.readline().strip().lower()
            return key
        return None
    except Exception:
        return None

def main():
    """Main function with improved workflow."""
    args = setup_argparse()
    configure_logging(args.verbose, args.debug)
    
    print_header("WEBSUMMARIZER - ENHANCED PHOTON RESULTS SUMMARIZER")
    print(f"Version: {__version__}")
    
    # Choose summarization provider, model, and depth
    provider = choose_provider()
    model = choose_model(provider)
    depth = choose_depth()
    api_keys = get_api_keys(provider)
    
    api_urls = {
        "ollama": args.ollama_url,
        "openai": DEFAULT_OPENAI_API_URL,
        "huggingface": DEFAULT_HF_API_URL
    }
    
    # Check Ollama connection if needed
    if provider == "ollama" and not check_ollama_connection(args.ollama_url):
        print("\nError: Cannot connect to Ollama. Make sure Ollama is running.")
        print("You can start Ollama with: ollama serve")
        print(f"And pull the model with: ollama pull {model}")
        return 1
    
    # Collect URLs from folders
    print(f"\nCollecting URLs from subfolders in {args.root_dir}...")
    folder_urls = collect_urls_from_folders(args.root_dir)
    
    if not folder_urls:
        print("No URLs found in any subfolder.")
        return 1
    
    print(f"\nFound {len(folder_urls)} subfolders with URLs:")
    for folder, urls in folder_urls.items():
        print(f"- {folder}: {len(urls)} URLs")
    
    # Track automatic mode state
    automatic_mode = args.auto
    if automatic_mode:
        print("\nAutomatic mode enabled. Processing will continue until complete or interrupted.")
        print(f"Using batch size of 20 for automatic mode (overrides default of {args.batch_size}).")
        print("Press 'n' then Enter at any time to exit automatic mode.")
        print("Or press Ctrl+C at any time to stop processing and save results.")
    
    # Process each subfolder
    for folder, urls in folder_urls.items():
        folder_path = os.path.join(args.root_dir, folder)
        print_header(f"Processing folder: {folder}")
        print(f"Found {len(urls)} URLs to process")
        
        # Process URLs in batches
        batch_summaries = []
        batch_overall_summaries = []
        
        i = 0
        while i < len(urls):
            # Use batch size of 20 when in automatic mode, otherwise use the specified batch size
            current_batch_size = 20 if automatic_mode else args.batch_size
            batch = urls[i:i + current_batch_size]
            
            batch_number = i // current_batch_size + 1
            total_batches = (len(urls) + current_batch_size - 1) // current_batch_size
            print_header(f"Processing batch {batch_number}/{total_batches} ({len(batch)} URLs)")
            
            summaries, overall_summary, all_404 = process_batch(
                batch, provider, model, depth, api_keys, api_urls, args.timeout, automatic_mode
            )
            
            batch_summaries.append(summaries)
            batch_overall_summaries.append(overall_summary)
            
            # Move to next batch
            i += current_batch_size
            
            # Check if we should continue to the next batch
            if i < len(urls):
                if all_404:
                    # Automatically continue to next batch if all URLs were 404s
                    if not automatic_mode:  # Only show in non-automatic mode
                        print("\nAll URLs in this batch resulted in 404 errors. Automatically continuing to next batch...")
                    continue
                elif automatic_mode:
                    # In fully automatic mode, just continue without prompting
                    # Just print a brief status message
                    print(f"\nAutomatic mode active - continuing to next batch ({batch_number+1}/{total_batches}).")
                    print("(Press 'n' then Enter to exit automatic mode)")
                    # Brief pause to allow for reading messages
                    time.sleep(1.0)
                    
                    # Non-blocking check if 'n' was pressed to exit automatic mode
                    key = check_for_key_press()
                    if key == 'n':
                        automatic_mode = False
                        print("Automatic mode disabled. Will prompt before each batch.")
                        # Make sure to clearly ask about continuing and enforce input validation
                        while True:
                            choice = input("Continue with next batch? (y/n): ").strip().lower()
                            if choice == 'y':
                                break
                            elif choice == 'n':
                                print("Stopping at user request.")
                                i = len(urls)  # This will exit the outer while loop
                                break
                            else:
                                print("Please enter 'y' or 'n'.")
                else:
                    # Implement better input validation to ensure only valid inputs are accepted
                    valid_response = False
                    while not valid_response:
                        choice = input("\nContinue with next batch? (y/n/a for automatic mode): ").strip().lower()
                        if choice == 'a':
                            automatic_mode = True
                            print("Automatic mode enabled. Processing will continue until complete or interrupted.")
                            print("Using batch size of 20 for remaining batches.")
                            print("Press 'n' then Enter at any time to exit automatic mode.")
                            valid_response = True
                        elif choice == 'y':
                            valid_response = True
                        elif choice == 'n':
                            print("Stopping at user request.")
                            i = len(urls)  # This will exit the outer while loop
                            valid_response = True
                        else:
                            print("Invalid input. Please enter 'y', 'n', or 'a'.")
        
        # Save summaries to file in the specified format
        save_enhanced_summary(folder_path, batch_summaries, batch_overall_summaries, format_type=args.format)
    
    print_header("SUMMARIZATION COMPLETE")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user.")
        print("Saving any completed summaries...")
        # Note: The summaries for the completed batches are already saved by the main loop
        print("Summary files have been saved for completed batches.")
        sys.exit(1)
