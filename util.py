from PyPDF2 import PdfReader
import tiktoken


def read_pages(file_name):
    with open(file_name, 'rb') as file:
        reader = PdfReader(file)
        pages = [page.extract_text() for page in reader.pages]
    return pages


def chunk_document(document: str, max_tokens=8191) -> list[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(document)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunks.append(enc.decode(tokens[i:i + max_tokens]))
    return chunks

