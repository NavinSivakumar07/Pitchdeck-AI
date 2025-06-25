"""
Vector Database Module for Pinecone Integration
Handles vector storage, retrieval, and similarity search for pitch deck documents.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import pinecone
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """Handles Pinecone vector database operations for pitch deck analysis."""
    
    def __init__(self):
        """Initialize the vector database with Pinecone and OpenAI embeddings."""
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "1pitchdeck")
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        # Initialize Pinecone
        pinecone.init(api_key=self.api_key, environment=self.environment)
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # More cost-effective option
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.index = None

        # Initialize the index
        self._setup_index()
    
    def _setup_index(self):
        """Set up the Pinecone index with proper configuration."""
        try:
            # Check if index exists
            existing_indexes = pinecone.list_indexes()

            if self.index_name not in existing_indexes:
                logger.info(f"Creating new index: {self.index_name}")

                # Create index
                pinecone.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI text-embedding-3-small dimension
                    metric="cosine"
                )

                # Wait for index to be ready
                logger.info("Waiting for index to be ready...")
                time.sleep(10)
            else:
                logger.info(f"Using existing index: {self.index_name}")

            # Connect to the index
            self.index = pinecone.Index(self.index_name)

            logger.info("Vector database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Pinecone index: {e}")
            raise
    
    def add_documents(self, documents: List[Document], batch_size: int = 100) -> bool:
        """
        Add documents to the vector database in batches.

        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to process in each batch

        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents to add")
                return False

            logger.info(f"Adding {len(documents)} documents to vector database...")

            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                vectors_to_upsert = []

                try:
                    # Prepare vectors for this batch
                    for j, doc in enumerate(batch):
                        # Generate embedding for the document
                        embedding = self.embeddings.embed_query(doc.page_content)

                        # Create unique ID
                        doc_id = f"{doc.metadata.get('document_id', f'doc_{i}_{j}')}"

                        # Prepare metadata (Pinecone has limits on metadata size)
                        metadata = {
                            "text": doc.page_content[:1000],  # Limit text size
                            "company_name": doc.metadata.get('company_name', ''),
                            "file_name": doc.metadata.get('file_name', ''),
                            "file_type": doc.metadata.get('file_type', ''),
                            "chunk_index": doc.metadata.get('chunk_index', 0)
                        }

                        vectors_to_upsert.append({
                            "id": doc_id,
                            "values": embedding,
                            "metadata": metadata
                        })

                    # Upsert to Pinecone
                    self.index.upsert(vectors=vectors_to_upsert)
                    logger.info(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")

                    # Small delay to avoid rate limits
                    time.sleep(2)

                except Exception as e:
                    logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                    continue

            logger.info("Successfully added all documents to vector database")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def search_similar_documents(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Document]:
        """
        Search for similar documents using semantic similarity.

        Args:
            query: Search query text
            k: Number of similar documents to return
            filter_dict: Optional metadata filters

        Returns:
            List of similar Document objects
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings.embed_query(query)

            # Perform similarity search
            search_results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
                filter=filter_dict
            )

            # Convert results to Document objects
            documents = []
            for match in search_results.matches:
                metadata = match.metadata
                doc = Document(
                    page_content=metadata.get('text', ''),
                    metadata={
                        'company_name': metadata.get('company_name', ''),
                        'file_name': metadata.get('file_name', ''),
                        'file_type': metadata.get('file_type', ''),
                        'chunk_index': metadata.get('chunk_index', 0),
                        'score': match.score
                    }
                )
                documents.append(doc)

            logger.info(f"Found {len(documents)} similar documents for query: {query[:50]}...")
            return documents

        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    def search_by_company(self, company_name: str, k: int = 10) -> List[Document]:
        """
        Search for documents by company name.

        Args:
            company_name: Name of the company to search for
            k: Number of documents to return

        Returns:
            List of Document objects for the company
        """
        try:
            # Create filter for company name
            filter_dict = {"company_name": {"$eq": company_name.lower()}}

            # Search with company filter using the company name as query
            return self.search_similar_documents(company_name, k=k, filter_dict=filter_dict)

        except Exception as e:
            logger.error(f"Error searching for company {company_name}: {e}")
            return []
    
    def get_companies_list(self) -> List[str]:
        """
        Get a list of all companies in the database.

        Returns:
            List of company names
        """
        try:
            # For now, return empty list as Pinecone doesn't provide easy metadata enumeration
            # This would need to be maintained separately or queried differently
            logger.info("Company list retrieval not implemented yet")
            return []

        except Exception as e:
            logger.error(f"Error getting companies list: {e}")
            return []
    
    def check_company_exists(self, company_name: str) -> bool:
        """
        Check if a company exists in the database.
        
        Args:
            company_name: Name of the company to check
            
        Returns:
            True if company exists, False otherwise
        """
        try:
            results = self.search_by_company(company_name, k=1)
            return len(results) > 0
            
        except Exception as e:
            logger.error(f"Error checking if company exists: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            stats = self.index.describe_index_stats()
            
            return {
                "total_vectors": stats.get("total_vector_count", 0),
                "index_fullness": stats.get("index_fullness", 0),
                "dimension": stats.get("dimension", 0),
                "namespaces": stats.get("namespaces", {})
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def delete_all_vectors(self) -> bool:
        """
        Delete all vectors from the index (use with caution).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.warning("Deleting all vectors from the index...")
            self.index.delete(delete_all=True)
            logger.info("All vectors deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            return False
