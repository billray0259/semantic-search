import argparse
import os
import json
import numpy as np
from util import chunk_document, read_pages
from openai import Embedding
import openai
from config import OPENAI_API_KEY



def process_txt_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return [text]

def process_pdf_file(file_path):
    return read_pages(file_path)

def process_md_file(file_path):
    return process_txt_file(file_path)

def get_file_processing_function(extension):
    file_processing_functions = {
        '.txt': process_txt_file,
        '.pdf': process_pdf_file,
        '.md': process_md_file,
    }
    return file_processing_functions.get(extension)

def main(source_directory):
    openai.api_key = OPENAI_API_KEY
    embeddings = []
    content_objects = []

    for root, _, files in os.walk(source_directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_path = os.path.abspath(file_path)
            _, extension = os.path.splitext(file_path)

            process_function = get_file_processing_function(extension)
            if process_function is None:
                continue

            print(f'Processing {file_path}...')

            document_pages = process_function(file_path)
            for page_number, page in enumerate(document_pages):
                chunks = chunk_document(page)
                for chunk_number, chunk in enumerate(chunks):
                    response = Embedding.create(input=chunk, model='text-embedding-ada-002')
                    embedding = response['data'][0]['embedding']
                    embeddings.append(embedding)
                    content_objects.append({
                        'embedding_id': len(embeddings) - 1,
                        'embedding_text': chunk,
                        'source_file': file_path,
                        'chunk_number': chunk_number,
                        'page_number': page_number,
                        'max_chunk_tokens': 8191,
                        'file_last_modified': os.path.getmtime(file_path)
                    })

    np.save('embeddings.npy', np.array(embeddings))

    with open('embeddings.json', 'w') as f:
        json.dump(content_objects, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate embeddings for files in a directory.')
    parser.add_argument('source_directory', help='Directory containing files to generate embeddings for.')
    args = parser.parse_args()
    main(args.source_directory)
