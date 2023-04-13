import numpy as np
import json
from openai import Embedding
from config import OPENAI_API_KEY
import openai

class DocumentEmbeddings(dict): # maps embedding vectors indecies to content objects

    def __init__(self, embedding_matrix, content_objects):
        self.embedding_matrix = embedding_matrix
        self.content_objects = content_objects
        for i, obj in enumerate(content_objects):
            self[obj['embedding_id']] = i

    @staticmethod
    def load(matrix_file, contents_file) -> 'DocumentEmbeddings':
        with open(matrix_file, 'rb') as f:
            embedding_matrix = np.load(f)
        
        with open(contents_file, 'r') as f:
            content_objects = json.load(f)

        return DocumentEmbeddings(embedding_matrix, content_objects)


    def search(self, query, n_results=20) -> list[dict]:
        openai.api_key = OPENAI_API_KEY

        # Get the query embedding
        query_embedding = Embedding.create(input=query, model='text-embedding-ada-002')['data'][0]['embedding']

        # Compute the cosine similarity between the query embedding and document embeddings
        # similarities = np.array([cosine_similarity(query_embedding, doc_embedding) for doc_embedding in self.embedding_matrix])
        similarities = np.dot(self.embedding_matrix, query_embedding)

        # Get the indices of the top n_results content objects sorted by cosine similarity
        top_indices = np.argsort(similarities)[-n_results:][::-1]

        # Get the top n_results content objects and their corresponding scores
        results = [{'score': similarities[i], **self.content_objects[i]} for i in top_indices]

        return results
