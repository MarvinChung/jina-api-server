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
from executors import sBert
from openai_api_protocol import *

# class UsageInfo(BaseDoc):
#     prompt_tokens: int = 0
#     total_tokens: int = 0
#     completion_tokens: Optional[int] = 0

# class EmbeddingsRequest(BaseDoc):
#     model: Optional[str] = None
#     engine: Optional[str] = None
#     input: Union[str, List[Any]]
#     user: Optional[str] = None

# class EmbeddingsResponse(BaseDoc):
#     object: str = "list"
#     data: List[Dict[str, Any]]
#     model: str
#     usage: UsageInfo


class MyGateway(FastAPIBaseGateway):
    @property
    def app(self):
        app = FastAPI()

        @app.post("/embeddings")
        async def get(request: EmbeddingsRequest):
            toc = time.time()

            token_num = 0
            "Creates embeddings for the text"""
            results = []
            errors = []

            if isinstance(request.input, str):
                request.input = [request.input]

            # for i, text_input in enumerate(request.input):
            #     async for docs, error in self.streamer.stream(
            #         docs=DocumentArray([Document(text=text_input)]),
            #         exec_endpoint='/embeddings',
            #     ):
            #         print(docs)
            #         print(dir(docs))
            #         print(dir(docs[0]))
            #         print(len(docs))
            #         print(docs[0])
            #         if error:
            #             errors.append(error)
            #         else:
            #             data = {
            #                 "object": "embedding",
            #                 "embedding": docs[0].embedding.tolist(),
            #                 "index": i,
            #             }
            #             results.append(data)

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
