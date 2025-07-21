#!/usr/bin/env python3
"""
EchoNexus Cloud Document Processor
High-performance document processing for large-scale ingestion
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import logging
from datetime import datetime

# Document processing libraries
try:
    import fitz  # PyMuPDF
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"Required library missing: {e}")
    sys.exit(1)

# Vector processing
import numpy as np
from openai import OpenAI

class CloudDocumentProcessor:
    """High-performance cloud document processor"""
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.batch_size = 50
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def process_documents_batch(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """Process all documents in batch mode"""
        
        results = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'total_vectors': 0,
            'processing_time': 0
        }
        
        start_time = datetime.now()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all documents
        input_path = Path(input_dir)
        documents = list(input_path.glob('**/*.pdf')) + list(input_path.glob('**/*.epub'))
        
        results['total_files'] = len(documents)
        
        self.logger.info(f"Processing {len(documents)} documents")
        
        for doc_path in documents:
            try:
                doc_result = self.process_single_document(doc_path, output_dir)
                
                if doc_result['success']:
                    results['processed_files'] += 1
                    results['total_chunks'] += doc_result['chunks_count']
                    results['total_vectors'] += doc_result['vectors_count']
                else:
                    results['failed_files'] += 1
                    self.logger.error(f"Failed to process {doc_path}: {doc_result.get('error')}")
                
            except Exception as e:
                results['failed_files'] += 1
                self.logger.error(f"Error processing {doc_path}: {e}")
        
        results['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        # Save processing summary
        summary_file = os.path.join(output_dir, 'processing_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def process_single_document(self, doc_path: Path, output_dir: str) -> Dict[str, Any]:
        """Process a single document"""
        
        result = {
            'success': False,
            'chunks_count': 0,
            'vectors_count': 0,
            'file_size': 0
        }
        
        try:
            # Extract text
            text_content = self.extract_text(doc_path)
            
            if not text_content:
                result['error'] = 'No text extracted'
                return result
            
            # Create chunks
            chunks = self.create_chunks(text_content)
            result['chunks_count'] = len(chunks)
            
            # Generate vectors
            vectors = self.generate_vectors_batch(chunks)
            result['vectors_count'] = len(vectors)
            
            # Save results
            doc_id = self.generate_doc_id(doc_path.name)
            self.save_document_data(doc_id, doc_path.name, chunks, vectors, output_dir)
            
            result['success'] = True
            result['file_size'] = doc_path.stat().st_size
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def extract_text(self, doc_path: Path) -> str:
        """Extract text from document"""
        
        if doc_path.suffix.lower() == '.pdf':
            return self.extract_pdf_text(doc_path)
        elif doc_path.suffix.lower() in ['.epub', '.epub3']:
            return self.extract_epub_text(doc_path)
        else:
            return ""
    
    def extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF"""
        
        try:
            doc = fitz.open(pdf_path)
            text_content = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text_content += page.get_text() + "\n"
            
            doc.close()
            return text_content
        
        except Exception as e:
            self.logger.error(f"PDF extraction error: {e}")
            return ""
    
    def extract_epub_text(self, epub_path: Path) -> str:
        """Extract text from EPUB"""
        
        try:
            book = epub.read_epub(epub_path)
            text_content = ""
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    content = item.get_content().decode('utf-8')
                    soup = BeautifulSoup(content, 'html.parser')
                    text_content += soup.get_text() + "\n"
            
            return text_content
        
        except Exception as e:
            self.logger.error(f"EPUB extraction error: {e}")
            return ""
    
    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create text chunks"""
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + 100:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'start_pos': start,
                    'end_pos': end
                })
                chunk_id += 1
            
            start = end - self.chunk_overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def generate_vectors_batch(self, chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Generate vectors in batches"""
        
        vectors = []
        
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_texts = [chunk['text'] for chunk in batch_chunks]
            
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch_texts
                )
                
                for embedding_data in response.data:
                    vector = np.array(embedding_data.embedding)
                    vectors.append(vector)
                
            except Exception as e:
                self.logger.error(f"Vector generation error: {e}")
                # Add zero vectors for failed batch
                for _ in batch_chunks:
                    vectors.append(np.zeros(1536))
        
        return vectors
    
    def generate_doc_id(self, filename: str) -> str:
        """Generate document ID"""
        import hashlib
        return hashlib.md5(filename.encode()).hexdigest()
    
    def save_document_data(self, doc_id: str, filename: str, chunks: List[Dict], 
                          vectors: List[np.ndarray], output_dir: str):
        """Save processed document data"""
        
        # Save vectors
        vectors_file = os.path.join(output_dir, f"{doc_id}_vectors.npy")
        np.save(vectors_file, np.array(vectors))
        
        # Save chunks
        chunks_file = os.path.join(output_dir, f"{doc_id}_chunks.json")
        with open(chunks_file, 'w') as f:
            json.dump(chunks, f, indent=2)
        
        # Save metadata
        metadata = {
            'doc_id': doc_id,
            'filename': filename,
            'chunks_count': len(chunks),
            'vectors_count': len(vectors),
            'processed_at': datetime.now().isoformat(),
            'vectors_file': vectors_file,
            'chunks_file': chunks_file
        }
        
        metadata_file = os.path.join(output_dir, f"{doc_id}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='EchoNexus Cloud Document Processor')
    parser.add_argument('--input-dir', required=True, help='Input directory containing documents')
    parser.add_argument('--output-dir', required=True, help='Output directory for processed data')
    parser.add_argument('--mode', default='batch', choices=['batch', 'streaming'], help='Processing mode')
    
    args = parser.parse_args()
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable required")
        sys.exit(1)
    
    processor = CloudDocumentProcessor(openai_api_key)
    
    print(f"Starting document processing...")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Mode: {args.mode}")
    
    results = processor.process_documents_batch(args.input_dir, args.output_dir)
    
    print(f"\nProcessing completed:")
    print(f"  Total files: {results['total_files']}")
    print(f"  Processed: {results['processed_files']}")
    print(f"  Failed: {results['failed_files']}")
    print(f"  Total chunks: {results['total_chunks']}")
    print(f"  Total vectors: {results['total_vectors']}")
    print(f"  Processing time: {results['processing_time']:.2f} seconds")

if __name__ == '__main__':
    main()
