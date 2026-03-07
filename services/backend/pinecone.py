import os

from models import MODELS
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "request_papers"

if not pc.has_index(index_name):
    # NOTE: Using BGE model for this index
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=MODELS["bge"]["embed_size"],
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="eu-south-2"),
        deletion_protection="disabled",
        tags={"environment": "development"},
    )
