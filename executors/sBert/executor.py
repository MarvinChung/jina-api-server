from jina import DocumentArray, Executor, requests, dynamic_batching
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
# from docarray import DocList, BaseDoc
from docarray import DocumentArray
from typing import Generator, Optional, Union, Dict, List, Any
from jina.serve.runtimes.gateway.http.fastapi import FastAPIBaseGateway
from jina import Document, DocumentArray, Flow, Executor, requests
from fastapi import FastAPI
import time
import asyncio


class sBert(Executor):
    """SBert embeds text into 384-dim vectors using all-MiniLM-L6-v2"""
    def __init__(self, device: str = 'cpu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.model.to(device)  # Move the model to device

    @requests(on='/embeddings')
    @dynamic_batching(preferred_batch_size=4, timeout=200)
    def embeddings(self, docs: DocumentArray, **kwargs):
        
        #"""Add text-based embeddings to all documents"""
        with torch.inference_mode():
            embeddings = self.model.encode(docs.texts)
        docs.embeddings = embeddings

        print(docs.embeddings.shape)

        # return EmbeddingsResponse(data=sentence_embeddings, model=docs.model)
