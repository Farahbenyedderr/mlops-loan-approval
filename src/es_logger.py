# src/es_logger.py
from datetime import datetime, timezone
from elasticsearch import Elasticsearch

# Connect to Elasticsearch
es = Elasticsearch(hosts=["http://127.0.0.1:9200"])


def log_to_es(index_name, doc):
    """Log a dictionary document to Elasticsearch"""
    # Check if index exists, if not create it
    if not es.indices.exists(index=index_name):
        # Create index with basic settings
        es.indices.create(
            index=index_name,
            body={
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            }
        )
    
    # Add timestamp
    doc["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    # Index the document
    es.index(index=index_name, document=doc)