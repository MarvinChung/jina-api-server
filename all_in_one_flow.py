from jina import DocumentArray, Executor, requests, dynamic_batching, Flow
from sentence_transformers import SentenceTransformer
from docarray import DocumentArray, Document
from typing import Generator, Optional, Union, Dict, List, Any
from pydantic import BaseModel, Field
from jina.serve.runtimes.gateway.http.fastapi import FastAPIBaseGateway
from fastapi import FastAPI
import torch

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[Any]]
    user: Optional[str] = None

class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: UsageInfo


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


class MyGateway(FastAPIBaseGateway):
    @property
    def app(self):
        app = FastAPI()

        @app.post("/embeddings")
        async def create_embeddings(request: EmbeddingsRequest):
            token_num = 0
            "Creates embeddings for the text"""
            results = []
            errors = []

            if isinstance(request.input, str):
                request.input = [request.input]

            print(request.input)
            async for docs, error in self.streamer.stream(
                docs=DocumentArray([Document(text=text_input) for text_input in request.input]),
                exec_endpoint='/embeddings',
            ):
                print(docs)
                print(docs[0])
                if error:
                    errors.append(error)
                else:
                    for i, doc in enumerate(docs):
                        data = {
                            "object": "embedding",
                            "embedding": doc.embedding.tolist(),
                            "index": i,
                        }
                        results.append(data)

            print(results)

            # this number is not correct, should be return by embeddings executor since a word may be split to multiple tokens
            token_num += sum([len(text_input) for text_input in request.input])
            return EmbeddingsResponse(
                        data=results,
                        model=request.model,
                        usage=UsageInfo(
                        prompt_tokens=token_num,
                        total_tokens=token_num,
                        completion_tokens=None,
                        ),
                    ).dict(exclude_none=True)


        return app


with Flow().config_gateway(uses=MyGateway, port=57012, protocol='http').add(uses=sBert, name='sBert') as flow:
    flow.block()