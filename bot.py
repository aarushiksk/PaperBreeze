from sentence_transformers import SentenceTransformer

embeddings = SentenceTransformer("BAAI/bge-large-en-v1.5")

sentence=['hi how are you','how']
print(embeddings.encode(sentence).tolist())
