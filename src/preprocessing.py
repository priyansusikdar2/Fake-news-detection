"""
Advanced text preprocessing for fake news detection
Includes cleaning, normalization, and feature extraction
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing"""
    remove_special_chars: bool = True
    remove_numbers: bool = False
    remove_stopwords: bool = True
    lowercase: bool = True
    stemming: bool = False
    lemmatization: bool = True
    min_word_length: int = 2
    max_word_length: int = 30
    custom_stopwords: Optional[List[str]] = None


class AdvancedTextPreprocessor:
    """Advanced text preprocessor with multiple cleaning options"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.stop_words = set(stopwords.words('english'))
        
        # Add custom stopwords
        if self.config.custom_stopwords:
            self.stop_words.update(self.config.custom_stopwords)
        
        # Add domain-specific stopwords for news
        self.stop_words.update([
            'said', 'say', 'says', 'told', 'report', 'reports', 'according',
            'published', 'updated', 'posted', 'shared', 'liked', 'commented'
        ])
        
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Compile regex patterns for performance
        self.url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.special_chars_pattern = re.compile(r'[^a-zA-Z\s]')
        self.number_pattern = re.compile(r'\d+')
        self.extra_spaces_pattern = re.compile(r'\s+')
        
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove emails
        text = self.email_pattern.sub('', text)
        
        # Remove mentions and hashtags (but keep the words)
        text = self.mention_pattern.sub('', text)
        text = self.hashtag_pattern.sub('', text)
        
        # Remove special characters
        if self.config.remove_special_chars:
            text = self.special_chars_pattern.sub(' ', text)
        
        # Remove numbers
        if self.config.remove_numbers:
            text = self.number_pattern.sub('', text)
        
        # Convert to lowercase
        if self.config.lowercase:
            text = text.lower()
        
        # Remove extra spaces
        text = self.extra_spaces_pattern.sub(' ', text).strip()
        
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        tokens = word_tokenize(text)
        
        # Filter by word length
        tokens = [t for t in tokens if self.config.min_word_length <= len(t) <= self.config.max_word_length]
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list"""
        return [t for t in tokens if t not in self.stop_words]
    
    def apply_stemming(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens"""
        return [self.stemmer.stem(t) for t in tokens]
    
    def apply_lemmatization(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization to tokens"""
        return [self.lemmatizer.lemmatize(t) for t in tokens]
    
    def process(self, text: str) -> Dict[str, Any]:
        """Complete preprocessing pipeline"""
        # Step 1: Normalize
        text = self.normalize_unicode(text)
        
        # Step 2: Clean
        text = self.clean_text(text)
        
        # Step 3: Tokenize
        tokens = self.tokenize(text)
        
        # Step 4: Remove stopwords
        if self.config.remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Step 5: Stemming or Lemmatization
        if self.config.stemming:
            tokens = self.apply_stemming(tokens)
        elif self.config.lemmatization:
            tokens = self.apply_lemmatization(tokens)
        
        # Join back to text
        processed_text = ' '.join(tokens)
        
        return {
            'original_text': text,
            'processed_text': processed_text,
            'tokens': tokens,
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens))
        }


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom feature extractor for text data"""
    
    def __init__(self, ngram_range: Tuple[int, int] = (1, 3), max_features: int = 10000):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vocabulary_ = None
        
    def fit(self, X, y=None):
        # Build vocabulary
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            stop_words='english'
        )
        self.vectorizer.fit(X)
        self.vocabulary_ = self.vectorizer.vocabulary_
        return self
    
    def transform(self, X):
        return self.vectorizer.transform(X)


def extract_linguistic_features(text: str) -> Dict[str, float]:
    """Extract advanced linguistic features"""
    features = {}
    
    # Basic statistics
    words = text.split()
    sentences = sent_tokenize(text)
    
    features['word_count'] = len(words)
    features['sentence_count'] = len(sentences)
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    features['avg_sentence_length'] = features['word_count'] / max(1, features['sentence_count'])
    
    # Punctuation features
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['quotes_count'] = text.count('"') + text.count("'")
    features['ellipsis_count'] = text.count('...')
    features['comma_density'] = text.count(',') / max(1, features['word_count'])
    
    # Capitalization features
    features['caps_word_ratio'] = sum(1 for w in words if w.isupper()) / max(1, len(words))
    features['title_word_ratio'] = sum(1 for w in words if w.istitle()) / max(1, len(words))
    
    # Part-of-speech features
    pos_tags = nltk.pos_tag(words)
    pos_counts = Counter(tag for word, tag in pos_tags)
    
    features['noun_ratio'] = sum(pos_counts[tag] for tag in ['NN', 'NNS', 'NNP', 'NNPS']) / max(1, len(words))
    features['verb_ratio'] = sum(pos_counts[tag] for tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']) / max(1, len(words))
    features['adj_ratio'] = sum(pos_counts[tag] for tag in ['JJ', 'JJR', 'JJS']) / max(1, len(words))
    features['adv_ratio'] = sum(pos_counts[tag] for tag in ['RB', 'RBR', 'RBS']) / max(1, len(words))
    
    # Readability scores (simplified)
    features['flesch_score'] = 206.835 - 1.015 * (features['avg_sentence_length']) - 84.6 * (features['avg_word_length'])
    features['flesch_score'] = max(0, min(100, features['flesch_score']))  # Clamp to 0-100
    
    return features