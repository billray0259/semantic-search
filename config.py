import os

# Flask app settings
PORT = 8000
SECRET_KEY = os.urandom(24)

# Supported file types
SUPPORTED_FILE_TYPES = ['.txt', '.pdf', '.md']

# Maximum tokens per chunk
MAX_CHUNK_TOKENS = 8191

NUM_RESULTS = 5

# OpenAI API key
with open("OPENAI_API_KEY.txt", "r") as f:
    OPENAI_API_KEY = f.read().strip()
