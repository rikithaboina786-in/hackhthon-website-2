from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from pathlib import Path

class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def process_pdf(self, pdf_path):
        """Load and chunk PDF files"""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    def process_csv(self, csv_path):
        """Load and chunk CSV files"""
        df = pd.read_csv(csv_path)
        documents = []
        
        for idx, row in df.iterrows():
            # Convert each row to a document
            content = " | ".join([f"{col}: {val}" for col, val in row.items()])
            documents.append({
                'page_content': content,
                'metadata': {'source': csv_path, 'row': idx}
            })
        
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    def process_documents(self, file_paths):
        """Process multiple files"""
        all_chunks = []
        
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                chunks = self.process_pdf(file_path)
            elif file_path.endswith('.csv'):
                chunks = self.process_csv(file_path)
            all_chunks.extend(chunks)
        
        return all_chunks