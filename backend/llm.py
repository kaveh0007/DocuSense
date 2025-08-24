import os
from typing import List, Dict
from textwrap import dedent

# Optional OpenAI
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Fallback to open-source
from transformers import pipeline


def _build_prompt(question: str, contexts: List[Dict]) -> str:
    context_block = "\n\n".join([
        f"[{i+1}] (source: {c['file_name']} p.{c['page']})\n{c['text']}"
        for i, c in enumerate(contexts)
    ])
    prompt = dedent(f"""
    You are a helpful assistant answering questions using ONLY the provided context.
    If the context is insufficient, say you don't know.

    Question:
    {question}

    Context:
    {context_block}

    Answer in 5-10 concise sentences, cite sources like [1], [2] where relevant.
    """).strip()
    return prompt


def generate_answer(question: str, contexts: List[Dict]) -> str:
    prompt = _build_prompt(question, contexts)

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if OPENAI_AVAILABLE and openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            msg = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a careful assistant that grounds every answer ONLY in the given context."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return msg.choices[0].message.content.strip()
        except Exception as e:
            # Fall through to local model
            pass

    # Local fallback: FLAN-T5 Base
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    out = generator(prompt, max_new_tokens=256)
    return out[0]["generated_text"].strip()
