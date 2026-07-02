import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

load_dotenv()

class Memory:
    def __init__(self, config, file_path="chat_history.json"):
        self.file_path = file_path
        self.memory_recent_k = config["memory_recent_k"]
        self.token_wall = config.get("token_wall", 200_000)

        # Use the small/fast query model for summarisation — 7B is plenty for this task
        summary_model = config.get("query_llm_model", config["llm_model"])
        self.client = InferenceClient(
            provider="featherless-ai",
            model=summary_model,
            api_key=os.environ["HF_TOKEN"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            summary_model,
            token=os.environ.get("HF_TOKEN"),
        )
        self.token_count = 0

    def load(self):
        if not os.path.exists(self.file_path):
            return []
        with open(self.file_path, "r") as f:
            data = json.load(f)
        self.token_count = data.get("tokens", 0)
        return data.get("messages", [])

    def save(self, messages):
        with open(self.file_path, "w") as f:
            json.dump({"messages": messages, "tokens": self.token_count}, f, indent=2)

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))

    def track_message(self, message):
        self.token_count += self.count_tokens(message['content'])

    def should_compress(self):
        return self.token_count >= self.token_wall

    def get_recent(self, messages):
        return messages[-self.memory_recent_k:]

    def summarize(self, messages):
        older = messages[:-self.memory_recent_k] if len(messages) > self.memory_recent_k else []
        if not older:
            return ""

        text = "\n".join(
            [f"{m['role']}: {m['content']}" for m in older]
        )

        prompt = f"""Summarize the following conversation briefly
        in not more than 50 words preserving
        key facts and context.

        {text}"""

        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
        )

        summary = response.choices[0].message.content.strip()

        self.token_count = self.count_tokens(summary) + sum(
            self.count_tokens(m['content']) for m in self.get_recent(messages)
        )

        return summary

    def clear(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            self.token_count = 0


# if __name__ == "__main__":
#     import yaml

#     with open("config.yaml", "r") as f:
#         config = yaml.safe_load(f)

#     mem = Memory(config, file_path="demo_chat_history.json")

#     conversation = [
#         {"role": "user", "content": "I'm building a RAG pipeline migrating from FAISS to Pinecone."},
#         {"role": "assistant", "content": "Got it — want help with the embedding step or the index migration itself?"},
#         {"role": "user", "content": "Embeddings, I'm using Gemini embeddings and hit some SDK issues."},
#         {"role": "assistant", "content": "What error are you seeing exactly?"},
#         {"role": "user", "content": "A dimension mismatch when upserting into the Pinecone index."},
#         {"role": "assistant", "content": "That usually means your index was created with a different dimension than your embedding model outputs."},
#     ]

#     messages = mem.load()

#     for msg in conversation:
#         messages.append(msg)
#         mem.track_message(msg)

#         if mem.should_compress():
#             print(f"\n[token wall hit at {mem.token_count} tokens, compressing...]")
#             summary = mem.summarize(messages)
#             print(f"[summary]: {summary}")
#             messages = mem.get_recent(messages)

#     mem.save(messages)
#     print(f"\nFinal stored messages: {len(messages)}")
#     print(f"Final token count: {mem.token_count}")