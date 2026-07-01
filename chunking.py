#load dataset - done 
#CREATE CHUNKS RECURSIVELY 
#EXTRACT THE TEXT FROM THE CHUNKS
#S
import fitz
import yaml
import re
import os
import requests
import textwrap
import random
import pickle
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# FIRST WE WILL LOAD THE DATASET AND THEN CREATE CHUNKS RECURSIVELY.
def load_economics_dataset(config):
    ds=load_dataset("ksrepo/investopedia-dataset", split='train')
    arts=[]
    # we will make a list of dictionar of title and text
    for i, item in enumerate(ds):
        arts.append({"title":item["title"],"text":item['clean_text']})
    return arts

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    load_economics_dataset(config)

# now we will create chunks recursively. We will use the textwrap module
def split_para(text):
    for para in re.split(r"\n\s*\n", text):
        if para.strip():
            return [para.strip()]
    return []

def split_sentences(text):
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if(sent.strip()):
            return [sent.strip()]
    return []


def download_pdf(pdf_path):
    debug_path=os.makedirs(config['data_dir'], exist_ok=True) # yaha par actually hamari data directory create ho rahi hai
    print("path",debug_path)
    local_path=os.path.join(config['data_dir'],"human_nutrition.pdf")
    print("local_path",local_path)
    if not os.path.exists(local_path):
        if(pdf_path.startswith("http")):
            print(f"Downloading {pdf_path} to {local_path}")
            response=requests.get(pdf_path)
            with open(local_path,"wb") as f:
                f.write(response.content)
            return local_path
        else:
            return ValueError("wrong url provided")
    return local_path


def recursive_split(text,max_words,level=0):
    '''level is the recursion-depth / granularity parameter for recursive_split(text, max_words, level=0)
        level==0: split by paragraphs.
        level==1: split by sentences.
        level>=2: split into fixed-size word chunks (chunks of max_words).
        It's incremented when a coarser split returns only one'''
    word_count=len(text.split())
    if word_count<=max_words:
        return [text]
    if level==0:
        parts=split_para(text)
    elif level==1:
        parts=split_sentences(text)
    else:
        words=text.split()
        return [ ' '.join(words[i:i+max_words]) for i in range(0,len(words),max_words)]
    chunks=[]
    current=''
    for part in parts:
        proposed_part=f'{current}{part}'.strip()
        if len(proposed_part.split())<=max_words:
            current=proposed_part
        else:
            if current.strip():
                chunks.append(current.srip())
            if len (part.split())>max_words:
                chunks.extend(recursive_split(part,max_words,level+1))
            else:
                current=part
    if(current.strip()):
        chunks.append(current.strip())
    return chunks



def extract_text_from_pdf(pdf_path):
    doc=fitz.open(pdf_path)
    pages=[]
    for i,page in enumerate(doc):
        pages.append({
            "page_num":i+1,
            "text":page.get_text()
        })
    return pages


# INcase we want to create chunks from the economics dataset, we will create a function for that as well.

def load_economics_dataset(config):
    dataset = load_dataset("ksrepo/investopedia-dataset", split='train')
    articles = []

    for i, item in enumerate(dataset):
        if len(articles) > config['max_articles']:
            break
        text = item["clean_text"]

        articles.append({
            "title": item["title"],
            "text": text
        })

    return articles


def create_econ_chunks(config):
    articles = load_economics_dataset(config)
    print('Loaded {} articles'.format(len(articles)))
    chunks = []
    chunk_id = 0

    for article in tqdm(articles):

        pieces = recursive_split(article["text"], config['chunk_size'])

        for piece in pieces:
            if piece.strip() and len(piece.split()) >= config['min_chunk_size']:
                piece = re.sub(r'(?<=\S)\s*\n\s*', ' ', piece)

                chunks.append({
                    "id": chunk_id,
                    "text": piece,
                    "start_page": None,
                    "source": "hf dataset",
                    "domain": "economics",
                    "title": article["title"]
                })

                chunk_id += 1

    return chunks

    
def create_chunks(config):
    if os.path.exists(os.path.join(config['data_dir'],config["chunks_path"])):
        with open(os.path.join(config['data_dir'],config['chunks_path']),"rb") as f:
            chunks=pickle.load(f)
            return chunks
    pdf_path=download_pdf(config["pdf_path"])
    pages=extract_text_from_pdf(pdf_path)   
     
    #   for Creating chunks, we would like to have the chunk ID as well as the text and start page and 
    chunks=[]
    chunk_id=0
  
    for page in pages:
        text=page["text"]
        page_number=page["page_num"]
        peices=recursive_split(text,config['chunk_size'])
        for peice in peices:
            if peice.strip() and len(peice.split())>=config['min_chunk_size']:
                normalized_piece = re.sub(r'(?<=\S)\s*\n\s*', ' ', peice)

                chunks.append({
                    "chunk_id":chunk_id,
                    "text":normalized_piece,
                    "start_page":page_number,
                    "source":"pdf",
                    "domain":"nutrition",
                    "title":"human_nutrition"
                })
                chunk_id+=1
    print("loading econ subset\n")
    if config.get("us_econ_subset",False):
        econ_chunks=create_econ_chunks(config)
        for econ_chunk in econ_chunks:
            econ_chunk["chunk_id"]=chunk_id
            chunk_id+=1
        chunks.extend(econ_chunks)
    with open(os.path.join(config['data_dir'],config['chunks_path']),"wb") as f:
        pickle.dump(chunks,f)
    return chunks

#  chunk stats

def get_chunk_stats(chunks):
    word_len=[]
    char_len=[]
    for chunk in chunks:
        text=chunk["text"]
        word_len.append(len(text.split()))
        char_len.append(len(text))
        print("stats for nerds baby 👾")
        print("\n no of chunks are :",len(chunks))
        print("\n word len stats are :\n")
        print("mean",np.mean(word_len))
        print("std dev",np.std(word_len))
        print("min",np.min(word_len))
        print("max",np.max(word_len))

        print("\nCharacter Length Stats")
        print("Average:", np.mean(char_len))
        print("Std Dev:", np.std(char_len))
        print("Min:", np.min(char_len))
        print("Max:", np.max(char_len))

def display_chunks(chunks, k=5, box_width=80):
    r"""
    Method to view a sample of chunks
    :param chunks:
    :param k:
    :param box_width:
    :return: None
    """
    sampled_chunks = random.sample(chunks, min(k, len(chunks)))
    horizontal_border = "+" + "-" * (box_width - 2) + "+"

    for idx, chunk in enumerate(sampled_chunks, 1):
        print(f"Chunk #{idx}")
        print(horizontal_border)

        # Metadata header
        meta = (
            f"Words: {len(chunk['text'].split())} | Title: {chunk['title'][:50]}"
        )
        meta_line = "| " + meta.ljust(box_width - 4) + " |"
        print(meta_line)

        print(horizontal_border)

        # Wrapped chunk text
        wrapped_text = textwrap.wrap(
            chunk["text"],
            width=box_width - 4
        )

        for line in wrapped_text:
            print("| " + line.ljust(box_width - 4) + " |")

        print(horizontal_border)
        print()  # spacing between boxes


if __name__ == '__main__':
    # Below code can be used to test out different chunking strategies
    # without going through entire retrieval process
    with open("config.yaml") as f:
        config= yaml.safe_load(f)
    chunks = create_chunks(config)
    get_chunk_stats(chunks)
    display_chunks(chunks)




            

        
        