# =============================================================================
# YOJAN AI — SCHEME-AWARE CHUNKING (TEMP / LEARNING FILE)
# =============================================================================
#
# WHY THIS IS DIFFERENT FROM chunking.py
# ----------------------------------------
# The old chunking.py treated text as a flat stream of words and split it into
# fixed-size word windows (recursive_split).  That works for a nutrition PDF
# where every paragraph is roughly equal in importance.
#
# Government schemes are STRUCTURED ENTITIES.  Each scheme has:
#   - a name
#   - eligibility rules
#   - benefits
#   - how to apply
#   - who it's for
#
# Splitting a scheme mid-sentence destroys that structure.  A user asking
# "what schemes exist for women farmers?" needs ONE chunk that contains ALL of
# the relevant scheme info — not three word-window fragments that each miss half
# the answer.
#
# THE STRATEGY
# ------------
#   chunk = one scheme  (not one page, not N words)
#
# If a scheme is very long we split it by SECTION (eligibility / benefits /
# application), not by arbitrary word count.
#
# On top of the text we add AGENTIC CONTEXT — LLM-generated metadata that
# makes retrieval smarter:
#
#   beneficiaries : ["women", "farmers", "BPL families"]
#   category      : "agriculture"
#   benefit_summary: "Monthly pension of ₹3000 after age 60"
#   eligibility_summary: "Indian farmer, age 18-40, <2 hectares land"
#   tags          : ["pension", "old age", "PM scheme"]
#
# This metadata travels with every chunk into Pinecone so we can filter on it
# (e.g. filter={"category": "agriculture"}) AND embed it alongside the text so
# semantic search picks up concepts the raw text might not mention.
#
# =============================================================================

import os
import json
import pickle
import yaml
from dotenv import load_dotenv
from mlx_lm import load, generate
from tqdm import tqdm

load_dotenv()


# =============================================================================
# STEP 1: LOAD SCHEME DATA
# =============================================================================
#
# scraper.py already handles data collection and writes data/schemes.json.
# Run it once:  python scraper.py
#
# load_schemes() reads that file.  If it doesn't exist yet, it falls back to
# SAMPLE_SCHEMES so you can still run this file without scraping first.
#
# Each scheme coming from scraper.py is a dict with these keys:
#   scheme_id, name, ministry, level, categories, beneficiary_tags,
#   description, eligibility, benefits, documents_required,
#   application_process, application_url, source_url
#
# The structured fields (eligibility, benefits, etc.) are used DIRECTLY in
# build_enriched_text — no LLM call needed for those.
# LLM extraction only runs as a fallback when those fields are empty.

def load_schemes(config: dict) -> list[dict]:
    path = os.path.join(config["data_dir"], "schemes.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            schemes = json.load(f)
        print(f"Loaded {len(schemes)} schemes from {path}")
        return schemes
    print(f"[WARN] {path} not found — run scraper.py first. Using SAMPLE_SCHEMES for now.")
    return SAMPLE_SCHEMES

SAMPLE_SCHEMES = [
    {
        "name": "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)",
        "description": (
            "PM-KISAN provides income support of ₹6,000 per year to all farmer "
            "families across the country with cultivable land, payable in three "
            "equal installments of ₹2,000 every four months.  The scheme targets "
            "small and marginal farmers.  Beneficiaries must be Indian citizens "
            "who own agricultural land.  The funds are transferred directly to "
            "the bank account of the farmer."
        ),
    },
    {
        "name": "Beti Bachao Beti Padhao",
        "description": (
            "The Beti Bachao Beti Padhao scheme aims to address the declining "
            "Child Sex Ratio and promote the welfare of girl children.  It focuses "
            "on preventing gender-biased sex selective elimination and ensuring "
            "survival, protection, and education of the girl child.  The scheme "
            "operates through district-level campaigns, awareness programs, and "
            "conditional cash transfers for families with girl children."
        ),
    },
    {
        "name": "Ayushman Bharat PM-JAY",
        "description": (
            "Ayushman Bharat Pradhan Mantri Jan Arogya Yojana is the world's "
            "largest health insurance scheme.  It provides health cover of ₹5 lakh "
            "per family per year for secondary and tertiary care hospitalisation.  "
            "The scheme targets the bottom 40% of the population identified through "
            "the Socio-Economic Caste Census (SECC).  Treatment is cashless and "
            "paperless at empanelled public and private hospitals."
        ),
    },
]


# =============================================================================
# STEP 2: AGENTIC CONTEXT EXTRACTION
# =============================================================================
#
# We call the LLM once per scheme and ask it to return a small JSON with the
# metadata fields we care about.
#
# WHY JSON?  Because we need to store each field separately in Pinecone metadata
# for filtering.  A free-text summary wouldn't let us do
#   filter={"beneficiaries": {"$in": ["farmers"]}}
#
# PROMPT DESIGN NOTES
# -------------------
# - We use a SYSTEM message to lock the output format.
# - We ask for a JSON object with fixed keys so parsing is reliable.
# - We keep it short (max_new_tokens=300) — we only need structured metadata,
#   not a long essay.

# This prompt is ONLY called for the two fields the API doesn't give us:
#   benefit_summary  — what does this scheme actually provide?
#   eligibility_summary — who can apply and under what conditions?
#
# PROMPT DESIGN PRINCIPLES
# -------------------------
# 1. Role + constraint up front: "You are X. Do Y. Never do Z."
#    The model locks onto the role and constraint before reading the scheme.
#
# 2. One task per field, with a concrete example of good vs bad output.
#    Without examples the model tends to copy-paste from the description
#    instead of summarising.
#
# 3. "If the description does not mention X, write Unknown."
#    Forces the model to be honest rather than hallucinating eligibility rules.
#
# 4. JSON-only output with no markdown fences requested explicitly.
#    Reduces the need for post-processing strip logic.

CONTEXT_EXTRACTION_PROMPT = """You are a plain-language summariser for Indian government schemes.

Given a scheme name and its brief description, return a JSON object with exactly two keys:

  "benefit_summary"    : One sentence (≤20 words) stating the CONCRETE benefit — what money,
                         service, or resource the beneficiary receives.
                         Good:  "Provides ₹6,000 per year directly to farmer bank accounts."
                         Bad:   "A scheme that aims to support farmers in India."

  "eligibility_summary": One sentence (≤20 words) stating WHO qualifies and any KEY condition.
                         Good:  "Indian farmers aged 18+ who own cultivable agricultural land."
                         Bad:   "Eligible citizens as per government norms."

Rules:
- If the description does not clearly state the benefit amount or type, write the best inference
  from context but do NOT invent specific numbers.
- If eligibility is not mentioned, write "Open to eligible Indian citizens as per scheme norms."
- Return ONLY a raw JSON object. No markdown, no explanation, no extra keys.

Scheme name: {name}
Description: {description}"""

def is_json(data):
    try:
        json.loads(data)
        return True
    except (ValueError, TypeError):
        return False


def llm_extract_summaries(scheme: dict, mlx_model, mlx_tokenizer) -> tuple[str, str]:
    """
    Uses local MLX model to infer benefit_summary and eligibility_summary.
    Returns (benefit_summary, eligibility_summary).
    """
    prompt = CONTEXT_EXTRACTION_PROMPT.format(
        name=scheme["name"],
        description=scheme.get("description", ""),
    )
    messages = [{"role": "user", "content": prompt}]
    formatted = mlx_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    raw = generate(mlx_model, mlx_tokenizer, prompt=formatted, max_tokens=300, verbose=False).strip()

    # Strip markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # Robust extraction: slice from first { to last } to strip trailing text
    start, end = raw.find('{'), raw.rfind('}')
    if start != -1 and end > start:
        raw = raw[start:end + 1]

    if is_json(raw):
        parsed = json.loads(raw)
        return parsed.get("benefit_summary", ""), parsed.get("eligibility_summary", "")

    tqdm.write(f"  [WARN] JSON parse failed for '{scheme['name']}' — raw: {repr(raw[:150])}")
    return "", ""


def extract_agentic_context(scheme: dict, mlx_model, mlx_tokenizer) -> dict:
    """
    Builds structured metadata for a scheme.

    Split strategy:
      - category, beneficiaries, tags  → taken directly from API structured fields (free)
      - benefit_summary, eligibility_summary → taken from scraped fields if available,
        otherwise inferred by LLM from briefDescription (one call per scheme)
    """
    cats = scheme.get("categories", [])
    category = cats[0].lower().replace(" ", "_").replace(",", "") if cats else "other"
    beneficiaries = scheme.get("beneficiary_tags", [])
    tags = scheme.get("beneficiary_tags", [])

    # Use scraped detail fields if available (Delhi seeds have these)
    benefit_summary = scheme.get("benefits", "")
    eligibility_summary = scheme.get("eligibility", "")

    # For central schemes the detail page was empty — ask the LLM to infer from description
    if not benefit_summary and not eligibility_summary and scheme.get("description"):
        benefit_summary, eligibility_summary = llm_extract_summaries(scheme, mlx_model, mlx_tokenizer)

    return {
        "beneficiaries": beneficiaries,
        "category": category,
        "benefit_summary": benefit_summary,
        "eligibility_summary": eligibility_summary,
        "tags": tags,
    }



# STEP 3: BUILD SCHEME CHUNKS

#
# Each chunk = one scheme.
# The chunk dict mirrors the structure from chunking.py so it can plug straight
# into indexing.py without changes:
#
#   chunk_id, text, domain, title, source, start_page
#
# We ADD the agentic context fields on top.
#
# WHAT GOES INTO "text"?
# ----------------------
# We embed TWO things together in the text field:
#
#   [metadata header]  — the structured fields serialised as readable text
#   [scheme description] — the original description
#
# This is called "context-enriched embedding".  The embedding model sees BOTH
# the raw description AND the extracted metadata, so a query like
# "schemes for poor women" can match "beneficiaries: women, BPL families" even
# if those exact words don't appear in the description.

def build_enriched_text(scheme: dict, context: dict) -> str:
    """
    Combines the agentic context with the full scheme details into one string
    for embedding.

    Priority for each field:
      - scraper.py fields (eligibility, benefits, etc.) are used when present
      - LLM-extracted context fields fill in what's missing
    """
    # scraper.py schemes have these directly; SAMPLE_SCHEMES won't
    eligibility = scheme.get("eligibility") or context.get("eligibility_summary", "")
    benefits    = scheme.get("benefits")    or context.get("benefit_summary", "")
    tags        = scheme.get("categories") or scheme.get("beneficiary_tags") or context.get("tags", [])
    ministry    = scheme.get("ministry", "")
    how_to_apply = scheme.get("application_process", "")

    lines = [
        f"Scheme: {scheme['name']}",
        f"Ministry: {ministry}",
        f"Category: {context.get('category', '')}",
        f"For: {', '.join(context.get('beneficiaries', []))}",
        f"Benefits: {benefits}",
        f"Eligibility: {eligibility}",
        f"How to apply: {how_to_apply}",
        f"Tags: {', '.join(tags) if isinstance(tags, list) else tags}",
    ]
    header = "\n".join(l for l in lines if l.split(": ", 1)[-1].strip())
    return f"{header}\n\n{scheme['description']}"


def create_scheme_chunks(schemes: list[dict], mlx_model, mlx_tokenizer) -> list[dict]:
    """
    Single-threaded chunking using a local MLX model — no rate limits, no threading needed.
    """
    chunks = []
    bar = tqdm(schemes, desc="Chunking schemes", unit="scheme")
    for chunk_id, scheme in enumerate(bar):
        bar.set_postfix_str(scheme["name"][:40])

        context = extract_agentic_context(scheme, mlx_model, mlx_tokenizer)
        enriched_text = build_enriched_text(scheme, context)

        chunks.append({
            "chunk_id": chunk_id,
            "text": enriched_text,
            "domain": "schemes",
            "title": scheme["name"],
            "source": "yojan_schemes",
            "start_page": None,
            "beneficiaries": context.get("beneficiaries", []),
            "category": context.get("category", "other"),
            "benefit_summary": context.get("benefit_summary", ""),
            "eligibility_summary": context.get("eligibility_summary", ""),
            "tags": context.get("tags", []),
        })

        if context.get("benefit_summary"):
            tqdm.write(f"  ✓ {scheme['name'][:50]} → {context['benefit_summary'][:60]}")

    return chunks

def save_chunks(chunks: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"\nSaved {len(chunks)} chunks to {path}")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    print("Loading MLX model (downloads on first run, ~2GB)...")
    mlx_model, mlx_tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
    print("Model loaded.\n")

    schemes = load_schemes(config)
    chunks = create_scheme_chunks(schemes, mlx_model, mlx_tokenizer)
    save_chunks(chunks, path=os.path.join(config["data_dir"], "mlx_enriched", "scheme_chunks.pkl"))
    print("\n=== SAMPLE ENRICHED TEXT (chunk 0) ===")
    print(chunks[0]["text"])