from jina import Client, Document

client = Client(host='0.0.0.0', port=12345, protocol='http')
docs = client.post(on='/embeddings', inputs=Document(text='Hi there!'))
print(docs)
print("embeddings shape:", docs.embeddings.shape)