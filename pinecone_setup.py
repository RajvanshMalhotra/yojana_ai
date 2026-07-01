from pinecone import Pinecone
from dotenv import load_dotenv
import os
load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
models = pc.inference.list_models()
for m in models:
    print(m)