from jina import Executor, requests, Deployment, Flow
from docarray import DocList, BaseDoc
from docarray.documents import ImageDoc
from docarray.typing import AnyTensor

import numpy as np

class InputDoc(BaseDoc):
    img: ImageDoc

class OutputDoc(BaseDoc):
    embedding: AnyTensor

class MyExec(Executor):
    # @requests(
    #     on='/bar',
    #     request_schema=DocList[InputDoc],
    #     response_schema=DocList[OutputDoc],
    # )
    @requests(
        on='/bar',
        request_schema=DocList[InputDoc],
        response_schema=DocList[OutputDoc],
    )
    def bar(
            self, docs: DocList[InputDoc], **kwargs
    ) -> DocList[OutputDoc]:
        docs_return = DocList[OutputDoc](
            [OutputDoc(embedding=np.zeros((100, 1))) for _ in range(len(docs))]
        )
        return docs_return

# d = Deployment(uses=MyExec, protocol='http')

# with d:
#     d.block()


with Deployment(uses=MyExec) as dep:
    docs = dep.post(
        on='/bar',
        inputs=InputDoc(img=ImageDoc(tensor=np.zeros((3, 224, 224)))),
        return_type=DocList[OutputDoc],
    )
    assert docs[0].embedding.shape == (100, 1)
    # assert docs.__class__.document_type == OutputDoc

## Not yet support
# f = Flow().config_gateway(protocol='http').add(uses=MyExec)
# f.expose_endpoint('/bar', summary='my endpoint')
# with f:
#     f.block()