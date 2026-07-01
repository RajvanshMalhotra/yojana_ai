"""
Data capture script for YojanaAI.
Fetches central schemes from myscheme.gov.in and Delhi schemes from delhi.gov.in.
Output: data/schemes.json
"""

import os
import json
import time
import random
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "schemes.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/149.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8,hi;q=0.7",
    "Origin": "https://www.myscheme.gov.in",
    "Referer": "https://www.myscheme.gov.in/",
    "X-Api-Key": "tYTy5eEhlu9rFjyxuCr7ra7ACp4dv1RH8gWuHTDc",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ── Central schemes via myscheme.gov.in ──────────────────────────────────────

def fetch_scheme_list_page(page: int, size: int = 10) -> list[dict]:
    """
    Hits the hidden REST API that myscheme.gov.in's React frontend calls.
    If this endpoint changes, inspect Network > Fetch/XHR on myscheme.gov.in/search.
    """
    url = "https://api.myscheme.gov.in/search/v6/schemes"
    params = {
        "lang": "en",
        "q": "[]",
        "keyword": "",
        "sort": "",
        "from": (page - 1) * size,
        "size": size,
    }
    try:
        resp = SESSION.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("data", {}).get("hits", {}).get("items", [])
        return [item["fields"] | {"slug": item["fields"].get("slug", item["id"])} for item in items]
    except Exception as e:
        print(f"  [warn] list page {page} failed: {e}")
        return []


def fetch_scheme_detail(slug: str) -> dict:
    """Scrapes the detail page for eligibility, benefits, documents, application."""
    url = f"https://www.myscheme.gov.in/schemes/{slug}"
    try:
        resp = SESSION.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        def extract(selector):
            el = soup.select_one(selector)
            return el.get_text(separator=" ", strip=True) if el else ""

        eligibility = (
            extract("#eligibility")
            or extract("[data-section='eligibility']")
            or extract(".eligibility")
        )
        benefits = (
            extract("#benefits")
            or extract("[data-section='benefits']")
            or extract(".benefits")
        )
        documents_section = (
            soup.select_one("#documents-required")
            or soup.select_one(".documents-required")
        )
        documents = []
        if documents_section:
            documents = [li.get_text(strip=True) for li in documents_section.select("li") if li.get_text(strip=True)]

        application = (
            extract("#how-to-apply")
            or extract("[data-section='how-to-apply']")
            or extract(".how-to-apply")
        )

        return {
            "eligibility": eligibility,
            "benefits": benefits,
            "documents_required": documents,
            "application_process": application,
            "source_url": url,
        }
    except Exception as e:
        print(f"  [warn] detail fetch failed for {slug}: {e}")
        return {"eligibility": "", "benefits": "", "documents_required": [], "application_process": "", "source_url": url}


def parse_central_scheme(raw: dict, detail: dict) -> dict:
    """Merges list-API fields with scraped detail page into the canonical schema."""
    slug = raw.get("slug", "")
    name = raw.get("schemeName", slug)
    ministry = raw.get("nodalMinistryName", "")
    categories = raw.get("schemeCategory", [])
    tags = raw.get("tags", [])
    description = raw.get("briefDescription", "")
    level = raw.get("level", "central").lower()

    return {
        "scheme_id": slug,
        "name": name,
        "ministry": ministry,
        "level": level,
        "beneficiary_state": raw.get("beneficiaryState", []),
        "categories": categories,
        "beneficiary_tags": tags,
        "description": description,
        "eligibility": detail.get("eligibility", ""),
        "benefits": detail.get("benefits", ""),
        "documents_required": detail.get("documents_required", []),
        "application_process": detail.get("application_process", ""),
        "application_url": "",
        "source_url": detail.get("source_url", f"https://www.myscheme.gov.in/schemes/{slug}"),
    }


def scrape_central_schemes(max_pages: int = 20, size: int = 10) -> list[dict]:
    # Fetch total count first so tqdm can show % and ETA
    first_page = fetch_scheme_list_page(1, size)
    try:
        resp = SESSION.get("https://api.myscheme.gov.in/search/v6/schemes",
                           params={"lang": "en", "q": "[]", "keyword": "", "sort": "", "from": 0, "size": 1},
                           timeout=15)
        total = resp.json().get("data", {}).get("summary", {}).get("total", None)
    except Exception:
        total = None

    print(f"Fetching central schemes from myscheme.gov.in... (total: {total or '?'})")
    schemes = []
    bar = tqdm(total=total, desc="Scraping schemes", unit="scheme")

    for page in range(1, max_pages + 1):
        items = fetch_scheme_list_page(page, size) if page > 1 else first_page
        if not items:
            break
        for item in items:
            slug = item.get("slug") or item.get("schemeSlug") or item.get("id")
            if not slug:
                continue
            time.sleep(random.uniform(0.8, 1.5))
            detail = fetch_scheme_detail(str(slug))
            scheme = parse_central_scheme(item, detail)
            schemes.append(scheme)
            bar.set_postfix_str(scheme["name"][:45])
            bar.update(1)
        time.sleep(random.uniform(1.0, 2.0))

    bar.close()
    return schemes


# ── State-specific schemes via myscheme.gov.in filter ─────────────────────────

def scrape_state_schemes(state: str) -> list[dict]:
    """Fetches all schemes for a specific state using the beneficiaryState filter."""
    import json as _json
    print(f"Fetching {state} state schemes from myscheme.gov.in...")
    q = _json.dumps([{"identifier": "beneficiaryState", "value": state}])
    schemes = []
    page = 1
    size = 100
    while True:
        params = {"lang": "en", "q": q, "keyword": "", "sort": "", "from": (page - 1) * size, "size": size}
        try:
            resp = SESSION.get("https://api.myscheme.gov.in/search/v6/schemes", params=params, timeout=15)
            resp.raise_for_status()
            items = resp.json().get("data", {}).get("hits", {}).get("items", [])
            if not items:
                break
            for item in items:
                raw = item["fields"]
                scheme = parse_central_scheme(raw, {"eligibility": "", "benefits": "", "documents_required": [], "application_process": "", "source_url": f"https://www.myscheme.gov.in/schemes/{raw.get('slug', '')}"})
                schemes.append(scheme)
                print(f"  ✓ {scheme['name'][:70]}")
            page += 1
        except Exception as e:
            print(f"  [warn] page {page} failed: {e}")
            break
    print(f"  Total {state} schemes: {len(schemes)}")
    return schemes


# ── Delhi seeds (kept for schemes not on myscheme.gov.in) ─────────────────────

DELHI_SEEDS = [
    {
        "scheme_id": "delhi-ladli",
        "name": "Ladli Scheme",
        "ministry": "Department of Women & Child Development, Delhi",
        "level": "delhi",
        "categories": ["women", "girl_child"],
        "beneficiary_tags": ["girl", "family"],
        "description": "Financial assistance for families with girl children to promote their education and welfare.",
        "eligibility": "Families with girl children born after 01-01-2008 who are residents of Delhi. Family income below ₹1 lakh per annum.",
        "benefits": "₹5,000 at birth, ₹2,500 at class 1, ₹2,500 at class 6, ₹5,000 at class 9, ₹5,000 at class 10/11, ₹11,000 at class 12.",
        "documents_required": ["Birth certificate of girl child", "Aadhaar of parents", "Delhi domicile certificate", "Income certificate", "Bank account details"],
        "application_process": "Apply via Women & Child Development Department, Delhi or online at wcddel.in.",
        "application_url": "https://wcddel.in",
        "source_url": "https://delhi.gov.in/page/ladli",
    },
    {
        "scheme_id": "delhi-widow-pension",
        "name": "Widow Pension Scheme",
        "ministry": "Department of Social Welfare, Delhi",
        "level": "delhi",
        "categories": ["women", "pension"],
        "beneficiary_tags": ["widow", "senior"],
        "description": "Monthly pension for widows residing in Delhi to provide financial security.",
        "eligibility": "Widows aged 18 years and above who have been residents of Delhi for at least 5 years. Annual family income below ₹1 lakh.",
        "benefits": "₹2,500 per month",
        "documents_required": ["Death certificate of husband", "Aadhaar card", "Delhi domicile proof", "Income certificate", "Bank passbook"],
        "application_process": "Apply at the nearest District Social Welfare Office in Delhi.",
        "application_url": "https://edistrict.delhigovt.nic.in",
        "source_url": "https://delhi.gov.in/page/widow-pension",
    },
    {
        "scheme_id": "delhi-old-age-pension",
        "name": "Old Age Pension Scheme (Delhi)",
        "ministry": "Department of Social Welfare, Delhi",
        "level": "delhi",
        "categories": ["senior_citizen", "pension"],
        "beneficiary_tags": ["elderly", "senior"],
        "description": "Monthly pension for senior citizens in Delhi to support their financial needs.",
        "eligibility": "Delhi residents aged 60 years and above. Annual income below ₹1 lakh. Must not be receiving pension from any other government scheme.",
        "benefits": "₹2,000 per month (age 60-69), ₹2,500 per month (age 70+)",
        "documents_required": ["Aadhaar card", "Age proof", "Delhi domicile certificate", "Income certificate", "Bank account details"],
        "application_process": "Apply at the District Social Welfare Office or online at edistrict.delhigovt.nic.in.",
        "application_url": "https://edistrict.delhigovt.nic.in",
        "source_url": "https://delhi.gov.in/page/old-age-pension",
    },
    {
        "scheme_id": "delhi-disability-pension",
        "name": "Disability Pension Scheme (Delhi)",
        "ministry": "Department of Social Welfare, Delhi",
        "level": "delhi",
        "categories": ["disabled", "pension"],
        "beneficiary_tags": ["disabled", "differently_abled"],
        "description": "Monthly financial assistance for persons with disabilities residing in Delhi.",
        "eligibility": "Delhi residents with 40% or more disability. Annual income below ₹1 lakh.",
        "benefits": "₹2,500 per month",
        "documents_required": ["Disability certificate (40% or above)", "Aadhaar card", "Delhi domicile proof", "Income certificate", "Bank account details"],
        "application_process": "Apply at the District Social Welfare Office.",
        "application_url": "https://edistrict.delhigovt.nic.in",
        "source_url": "https://delhi.gov.in/page/disability-pension",
    },
    {
        "scheme_id": "delhi-tirth-yatra",
        "name": "Mukhyamantri Tirth Yatra Yojana",
        "ministry": "Revenue Department, Delhi",
        "level": "delhi",
        "categories": ["senior_citizen", "travel"],
        "beneficiary_tags": ["elderly", "pilgrim"],
        "description": "Free pilgrimage trips for senior citizens of Delhi to major religious sites across India.",
        "eligibility": "Delhi residents aged 70 years and above (one attendant allowed if above 60 years). Not availed the scheme in the last 3 years.",
        "benefits": "Free train journey (AC class), accommodation, meals, and local transport to pilgrimage sites.",
        "documents_required": ["Aadhaar card", "Age proof", "Delhi voter ID / domicile certificate", "Medical fitness certificate"],
        "application_process": "Apply online at tirth.delhi.gov.in or through MLAs.",
        "application_url": "https://tirth.delhi.gov.in",
        "source_url": "https://delhi.gov.in/page/tirth-yatra",
    },
    {
        "scheme_id": "delhi-muft-bijli",
        "name": "Mukhyamantri Muft Bijli Yojana",
        "ministry": "Power Department, Delhi",
        "level": "delhi",
        "categories": ["household", "subsidy"],
        "beneficiary_tags": ["household", "consumer"],
        "description": "Free electricity subsidy for Delhi households with low to moderate power consumption.",
        "eligibility": "Domestic electricity consumers in Delhi. Free for consumption up to 200 units/month; 50% subsidy for 201-400 units/month.",
        "benefits": "Up to 200 units free per month. 50% subsidy on 201-400 units. No subsidy above 400 units.",
        "documents_required": ["Electricity consumer number", "Aadhaar card"],
        "application_process": "Automatic for eligible consumers — no application needed. Subsidy is applied directly to the electricity bill.",
        "application_url": "https://bsesdelhi.com",
        "source_url": "https://delhi.gov.in/page/muft-bijli",
    },
]


def scrape_delhi_schemes() -> list[dict]:
    print("Collecting Delhi schemes (seeded list)...")
    for s in DELHI_SEEDS:
        print(f"  ✓ {s['name']}")
    return DELHI_SEEDS


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    central = scrape_central_schemes(max_pages=50, size=100)
    delhi_api = scrape_state_schemes("Delhi")
    delhi_seeds = scrape_delhi_schemes()

    # Deduplicate: prefer API schemes over seeds (seeds may overlap)
    seen_ids = {s["scheme_id"] for s in delhi_api}
    extra_seeds = [s for s in delhi_seeds if s["scheme_id"] not in seen_ids]

    all_schemes = central + delhi_api + extra_seeds
    print(f"\nTotal schemes collected: {len(all_schemes)} ({len(central)} central, {len(delhi_api)} Delhi API, {len(extra_seeds)} Delhi seeds)")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_schemes, f, ensure_ascii=False, indent=2)

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
