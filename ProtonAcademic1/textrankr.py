#!/usr/bin/env python3
"""
TextRank Implementation

A Python implementation of the TextRank algorithm for automatic text summarization.
This is used by the Deep Research Assistant for generating summaries without external APIs.
"""

import re
import math
import numpy as np
from collections import Counter

class TextRank:
    """TextRank implementation for extractive text summarization."""
    
    def __init__(self, damping=0.85, convergence_threshold=1e-5, max_iterations=50):
        """
        Initialize the TextRank algorithm.
        
        Args:
            damping: Damping factor for PageRank algorithm (typically 0.85)
            convergence_threshold: Threshold for PageRank convergence
            max_iterations: Maximum iterations for PageRank algorithm
        """
        self.damping = damping
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.stop_words = self._load_stop_words()
    
    def _load_stop_words(self):
        """Load English stop words."""
        # Common English stop words
        return {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
            'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
            'to', 'from', 'in', 'out', 'on', 'off', 'at', 'by', 'with', 'into', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'can', 'will', 'should', 'now', 'i', 'me', 'my',
            'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
            'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
            'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'would', 'should',
            'could', 'ought', 'i\'m', 'you\'re', 'he\'s', 'she\'s', 'it\'s', 'we\'re',
            'they\'re', 'i\'ve', 'you\'ve', 'we\'ve', 'they\'ve', 'i\'d', 'you\'d',
            'he\'d', 'she\'d', 'we\'d', 'they\'d', 'i\'ll', 'you\'ll', 'he\'ll',
            'she\'ll', 'we\'ll', 'they\'ll', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t',
            'hasn\'t', 'haven\'t', 'hadn\'t', 'doesn\'t', 'don\'t', 'didn\'t', 'won\'t',
            'wouldn\'t', 'shan\'t', 'shouldn\'t', 'can\'t', 'cannot', 'couldn\'t',
            'mustn\'t', 'let\'s', 'that\'s', 'who\'s', 'what\'s', 'here\'s', 'there\'s',
            'when\'s', 'where\'s', 'why\'s', 'how\'s', 'click', 'site', 'page', 'also'
        }
    
    def _extract_sentences(self, text):
        """
        Extract sentences from text.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Handle common abbreviations to prevent incorrect sentence splits
        text = re.sub(r'(?i)(\s|^)(mr\.|mrs\.|ms\.|dr\.|prof\.|inc\.|ltd\.|sr\.|jr\.)', 
                      lambda m: m.group(1) + m.group(2).replace('.', '[DOT]'), text)
        
        # Simple sentence splitting based on common punctuation
        # This is a basic implementation; NLTK would be more accurate if available
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
    
    def _tokenize(self, sentence):
        """
        Tokenize a sentence into words.
        
        Args:
            sentence: Input sentence
            
        Returns:
            List of words
        """
        # Convert to lowercase and remove punctuation
        sentence = sentence.lower()
        words = re.findall(r'\b\w+\b', sentence)
        
        # Filter out stop words and short words
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        return filtered_words
    
    def _calculate_similarity(self, sentence1, sentence2):
        """
        Calculate similarity between two sentences using cosine similarity.
        
        Args:
            sentence1: First sentence
            sentence2: Second sentence
            
        Returns:
            Similarity score between 0 and 1
        """
        # Tokenize sentences
        words1 = self._tokenize(sentence1)
        words2 = self._tokenize(sentence2)
        
        # Skip if either sentence has no content words
        if not words1 or not words2:
            return 0.0
        
        # Create word frequency counters
        counter1 = Counter(words1)
        counter2 = Counter(words2)
        
        # Find all unique words
        all_words = set(counter1.keys()).union(set(counter2.keys()))
        
        # Calculate dot product and magnitudes
        dot_product = sum(counter1.get(word, 0) * counter2.get(word, 0) for word in all_words)
        magnitude1 = math.sqrt(sum(counter1.get(word, 0) ** 2 for word in all_words))
        magnitude2 = math.sqrt(sum(counter2.get(word, 0) ** 2 for word in all_words))
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _build_similarity_matrix(self, sentences):
        """
        Build a sentence similarity matrix.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Similarity matrix
        """
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i][j] = self._calculate_similarity(sentences[i], sentences[j])
        
        return similarity_matrix
    
    def _page_rank(self, similarity_matrix):
        """
        Apply PageRank algorithm to the similarity matrix.
        
        Args:
            similarity_matrix: Sentence similarity matrix
            
        Returns:
            Dictionary of sentence indices to PageRank scores
        """
        n = len(similarity_matrix)
        
        # Initialize scores
        scores = np.ones(n) / n
        
        # Normalize the similarity matrix (column-stochastic)
        column_sums = similarity_matrix.sum(axis=0)
        # Avoid division by zero
        column_sums[column_sums == 0] = 1
        stochastic_matrix = similarity_matrix / column_sums
        
        # PageRank algorithm
        for _ in range(self.max_iterations):
            prev_scores = scores.copy()
            
            # Calculate new scores
            scores = (1 - self.damping) / n + self.damping * (stochastic_matrix.T @ scores)
            
            # Check for convergence
            if np.abs(scores - prev_scores).sum() < self.convergence_threshold:
                break
        
        # Convert to dictionary
        return {i: scores[i] for i in range(n)}
    
    def summarize(self, text, ratio=0.2, min_sentences=None, max_sentences=None):
        """
        Generate a summary of the input text.
        
        Args:
            text: Input text to summarize
            ratio: Proportion of sentences to include in the summary (0.0 to 1.0)
            min_sentences: Minimum number of sentences to include
            max_sentences: Maximum number of sentences to include
            
        Returns:
            Generated summary
        """
        # Extract sentences
        sentences = self._extract_sentences(text)
        
        # Handle very short texts
        if len(sentences) <= 3:
            return text
        
        # Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(sentences)
        
        # Apply PageRank
        sentence_scores = self._page_rank(similarity_matrix)
        
        # Determine number of sentences for the summary
        target_count = max(1, int(len(sentences) * ratio))
        
        if min_sentences is not None:
            target_count = max(min_sentences, target_count)
        
        if max_sentences is not None:
            target_count = min(max_sentences, target_count)
        
        # Select top sentences
        ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in ranked_sentences[:target_count]]
        
        # Sort indices to maintain original sentence order
        selected_indices.sort()
        
        # Generate summary
        summary_sentences = [sentences[i] for i in selected_indices]
        summary = ' '.join(summary_sentences)
        
        return summary


# Simple example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Read from file if provided
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        # Example text
        text = """
        TextRank is an algorithm based on Google's PageRank, adapted for text processing.
        It's used for automatic summarization and keyword extraction from documents.
        The algorithm works by building a graph representation of sentences and applying
        a graph-based ranking algorithm to select the most important sentences.
        TextRank is an extractive summarization method, meaning it selects existing
        sentences from the original text rather than generating new ones.
        This approach makes it language-independent and doesn't require training on a corpus.
        """
    
    # Create TextRank instance and generate summary
    textrank = TextRank()
    summary = textrank.summarize(text, ratio=0.5)
    
    print("Original text:")
    print(text)
    print("\nGenerated summary:")
    print(summary)
