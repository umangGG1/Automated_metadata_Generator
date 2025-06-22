# Import necessary libraries
import os
import re
import json
import datetime
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Document processing
import PyPDF2
from docx import Document
import pytesseract
from PIL import Image
import cv2

# Data processing
import pandas as pd
import numpy as np

# NLP and ML
import nltk
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import textstat
from langdetect import detect
from wordcloud import WordCloud

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class DocumentProcessor:
    """
    Handles document processing and content extraction from various file formats
    """
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF files"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        try:
            doc = Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return "\n".join(text)
        except Exception as e:
            print(f"Error extracting text from DOCX: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error extracting text from TXT: {e}")
            return ""
    
    def extract_text_from_image(self, file_path: str) -> str:
        """Extract text from images using OCR"""
        try:
            # Preprocess image for better OCR
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get image with only black and white
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Perform OCR
            text = pytesseract.image_to_string(thresh)
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from image: {e}")
            return ""
    
    def extract_content(self, file_path: str) -> Dict[str, Any]:
        """Main method to extract content from any supported file format"""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Extract text based on file type
        text = ""
        if file_extension == '.pdf':
            text = self.extract_text_from_pdf(str(file_path))
        elif file_extension == '.docx':
            text = self.extract_text_from_docx(str(file_path))
        elif file_extension == '.txt':
            text = self.extract_text_from_txt(str(file_path))
        elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            text = self.extract_text_from_image(str(file_path))
        
        # Get file statistics
        file_stats = file_path.stat()
        
        return {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_extension': file_extension,
            'file_size': file_stats.st_size,
            'creation_time': datetime.datetime.fromtimestamp(file_stats.st_ctime),
            'modification_time': datetime.datetime.fromtimestamp(file_stats.st_mtime),
            'extracted_text': text,
            'text_length': len(text),
            'word_count': len(text.split()) if text else 0
        }


class SemanticAnalyzer:
    """
    Performs semantic analysis on extracted text to identify key content
    """
    
    def __init__(self):
        # Initialize NLP models
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize classification pipeline
        self.classifier = pipeline("zero-shot-classification", 
                                 model="facebook/bart-large-mnli")
        
        # Define common document categories
        self.document_categories = [
            "academic paper", "business report", "legal document", "technical manual",
            "research paper", "financial report", "marketing material", "policy document",
            "instruction manual", "presentation", "article", "letter", "contract"
        ]
        
        # Initialize topic keywords
        self.topic_keywords = {
            "technology": ["software", "computer", "digital", "AI", "machine learning", "data", "algorithm"],
            "business": ["revenue", "profit", "market", "sales", "strategy", "finance", "investment"],
            "science": ["research", "experiment", "hypothesis", "analysis", "methodology", "results"],
            "legal": ["law", "regulation", "compliance", "contract", "agreement", "terms"],
            "medical": ["health", "patient", "treatment", "diagnosis", "medical", "clinical"],
            "education": ["student", "learning", "curriculum", "education", "teaching", "academic"]
        }
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the text"""
        try:
            return detect(text)
        except:
            return "unknown"
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics"""
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade_level": textstat.flesch_kincaid_grade(text),
            "automated_readability_index": textstat.automated_readability_index(text),
            "coleman_liau_index": textstat.coleman_liau_index(text),
            "gunning_fog": textstat.gunning_fog(text)
        }
    
    def extract_key_phrases(self, text: str, num_phrases: int = 10) -> List[str]:
        """Extract key phrases using frequency analysis"""
        # Simple implementation - can be enhanced with more sophisticated NLP
        from collections import Counter
        import re
        
        # Clean text and extract phrases
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = set(['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        words = [word for word in words if word not in stop_words]
        
        # Get most common words/phrases
        word_freq = Counter(words)
        return [word for word, count in word_freq.most_common(num_phrases)]
    
    def classify_document_type(self, text: str) -> Dict[str, float]:
        """Classify document type using zero-shot classification"""
        try:
            # Use first 1000 characters for classification to avoid token limits
            text_sample = text[:1000] if len(text) > 1000 else text
            result = self.classifier(text_sample, self.document_categories)
            
            return {
                "predicted_type": result['labels'][0],
                "confidence": result['scores'][0],
                "all_scores": dict(zip(result['labels'], result['scores']))
            }
        except Exception as e:
            print(f"Error in document classification: {e}")
            return {"predicted_type": "unknown", "confidence": 0.0, "all_scores": {}}
    
    def identify_topics(self, text: str) -> Dict[str, float]:
        """Identify topics based on keyword matching"""
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keywords in self.topic_keywords.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            topic_scores[topic] = score / len(text.split()) if text.split() else 0
        
        return topic_scores
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the text"""
        # Simple extractive summarization - first few sentences
        sentences = text.split('. ')
        summary = ""
        
        for sentence in sentences:
            if len(summary + sentence) < max_length:
                summary += sentence + ". "
            else:
                break
        
        return summary.strip()
    
    def analyze_content(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive semantic analysis"""
        if not text or len(text.strip()) == 0:
            return {
                "language": "unknown",
                "readability": {},
                "key_phrases": [],
                "document_type": {"predicted_type": "unknown", "confidence": 0.0},
                "topics": {},
                "summary": "",
                "sentiment": "neutral"
            }
        
        analysis = {
            "language": self.detect_language(text),
            "readability": self.calculate_readability(text),
            "key_phrases": self.extract_key_phrases(text),
            "document_type": self.classify_document_type(text),
            "topics": self.identify_topics(text),
            "summary": self.generate_summary(text)
        }
        
        return analysis


class MetadataGenerator:
    """
    Generates structured metadata from document content and semantic analysis
    """
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.analyzer = SemanticAnalyzer()
    
    def generate_file_hash(self, file_path: str) -> str:
        """Generate MD5 hash of the file for integrity checking"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Simple entity extraction (can be enhanced with spaCy NER)"""
        import re
        
        entities = {
            "emails": re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
            "urls": re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text),
            "dates": re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text),
            "phone_numbers": re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
        }
        
        return entities
    
    def calculate_content_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate various content metrics"""
        if not text:
            return {}
        
        words = text.split()
        sentences = text.split('.')
        paragraphs = text.split('\n\n')
        
        return {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "average_words_per_sentence": len(words) / len(sentences) if sentences else 0,
            "average_characters_per_word": len(text) / len(words) if words else 0,
            "unique_words": len(set(word.lower() for word in words)),
            "lexical_diversity": len(set(word.lower() for word in words)) / len(words) if words else 0
        }
    
    def generate_metadata(self, file_path: str) -> Dict[str, Any]:
        """Generate comprehensive metadata for a document"""
        try:
            # Extract content
            content_data = self.processor.extract_content(file_path)
            text = content_data['extracted_text']
            
            # Perform semantic analysis
            semantic_analysis = self.analyzer.analyze_content(text)
            
            # Extract entities
            entities = self.extract_entities(text)
            
            # Calculate metrics
            content_metrics = self.calculate_content_metrics(text)
            
            # Generate file hash
            file_hash = self.generate_file_hash(file_path)
            
            # Compile comprehensive metadata
            metadata = {
                # File Information
                "file_metadata": {
                    "file_name": content_data['file_name'],
                    "file_path": content_data['file_path'],
                    "file_extension": content_data['file_extension'],
                    "file_size_bytes": content_data['file_size'],
                    "file_size_human": self._format_file_size(content_data['file_size']),
                    "creation_time": content_data['creation_time'].isoformat(),
                    "modification_time": content_data['modification_time'].isoformat(),
                    "file_hash_md5": file_hash
                },
                
                # Content Information
                "content_metadata": {
                    "language": semantic_analysis['language'],
                    "document_type": semantic_analysis['document_type'],
                    "summary": semantic_analysis['summary'],
                    "key_phrases": semantic_analysis['key_phrases'],
                    "topics": semantic_analysis['topics'],
                    "entities": entities,
                    "content_metrics": content_metrics,
                    "readability_scores": semantic_analysis['readability']
                },
                
                # Processing Information
                "processing_metadata": {
                    "processing_timestamp": datetime.datetime.now().isoformat(),
                    "extraction_method": self._get_extraction_method(content_data['file_extension']),
                    "text_extracted": bool(text and text.strip()),
                    "analysis_completed": True,
                    "metadata_version": "1.0"
                },
                
                # Quality Metrics
                "quality_metrics": {
                    "text_extraction_confidence": 1.0 if text and len(text) > 100 else 0.5,
                    "classification_confidence": semantic_analysis['document_type'].get('confidence', 0.0),
                    "completeness_score": self._calculate_completeness_score(text, entities, semantic_analysis)
                }
            }
            
            return metadata
            
        except Exception as e:
            # Return error metadata if processing fails
            return {
                "error": str(e),
                "file_path": file_path,
                "processing_timestamp": datetime.datetime.now().isoformat(),
                "status": "failed"
            }
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format"""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f}{size_names[i]}"
    
    def _get_extraction_method(self, file_extension: str) -> str:
        """Get the extraction method used for the file type"""
        method_map = {
            '.pdf': 'PyPDF2',
            '.docx': 'python-docx',
            '.txt': 'direct_read',
            '.png': 'OCR_pytesseract',
            '.jpg': 'OCR_pytesseract',
            '.jpeg': 'OCR_pytesseract',
            '.tiff': 'OCR_pytesseract',
            '.bmp': 'OCR_pytesseract'
        }
        return method_map.get(file_extension, 'unknown')
    
    def _calculate_completeness_score(self, text: str, entities: Dict, semantic_analysis: Dict) -> float:
        """Calculate a completeness score for the metadata"""
        score = 0.0
        
        # Text extraction (40% of score)
        if text and len(text) > 50:
            score += 0.4
        
        # Entity extraction (20% of score)
        if any(entities.values()):
            score += 0.2
        
        # Document classification (20% of score)
        if semantic_analysis['document_type']['confidence'] > 0.5:
            score += 0.2
        
        # Key phrases extraction (10% of score)
        if len(semantic_analysis['key_phrases']) > 0:
            score += 0.1
        
        # Topic identification (10% of score)
        if any(score > 0 for score in semantic_analysis['topics'].values()):
            score += 0.1
        
        return min(score, 1.0)
    
    def save_metadata(self, metadata: Dict[str, Any], output_path: str = None) -> str:
        """Save metadata to JSON file"""
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"metadata_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        return output_path


if __name__ == "__main__":
    # Test the system
    print("Testing Automated Metadata Generation System...")
    
    # Initialize components
    generator = MetadataGenerator()
    
    # Test with sample file if available
    test_file = "MARS OPEN PROJECTS 2025.pdf"
    if os.path.exists(test_file):
        print(f"Processing test file: {test_file}")
        metadata = generator.generate_metadata(test_file)
        
        if 'error' not in metadata:
            print("✅ Processing completed successfully!")
            print(f"Document type: {metadata['content_metadata']['document_type']['predicted_type']}")
            print(f"Word count: {metadata['content_metadata']['content_metrics']['word_count']}")
            print(f"Language: {metadata['content_metadata']['language']}")
        else:
            print(f"❌ Processing failed: {metadata['error']}")
    else:
        print("No test file found. System is ready for use.")
    
    print("System test completed!")