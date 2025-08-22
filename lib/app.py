import streamlit as st
import requests, json
import PyPDF2
import numpy as np
import re
from openai import OpenAI

from typing import Optional, List, Dict
import sys, platform
import openai as openai_pkg


client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------- PAGE ----------

st.set_page_config(page_title="Career Buddy 2.0", layout="wide")

# ---------- GLOBAL UI: Header & CSS ----------
st.markdown(
    """
    <div style="padding:14px 16px;border-radius:10px;background:linear-gradient(90deg,#0ea5e9, #22c55e);color:white;margin-bottom:6px;">
      <div style="font-size:18px;font-weight:700;letter-spacing:0.3px;">Career Buddy 2.0</div>
      <div style="opacity:0.9;font-size:13px;">Find adjacent roles, bold pathways, and a clear plan forward.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Global CSS for cards, badges, and chips
st.markdown(
    """
    <style>
      .cb-card{border:1px solid #e5e7eb;border-radius:12px;padding:14px;margin:10px 0;background:#ffffff;box-shadow:0 1px 2px rgba(0,0,0,0.04)}
      .cb-row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
      .cb-title{font-weight:700;font-size:16px;margin:0}
      .cb-sub{color:#6b7280;font-size:13px;margin-top:2px}
      .cb-meta{font-size:12px;color:#374151;margin-top:8px}
      .cb-sep{height:1px;background:#f3f4f6;margin:10px 0}
      .cb-badge{display:inline-block;border-radius:999px;padding:3px 8px;font-size:11px;font-weight:600;margin-right:6px;border:1px solid rgba(0,0,0,0.06)}
      .cb-badge.adj{background:#ecfeff;color:#0369a1;border-color:#a5f3fc}
      .cb-badge.path{background:#eef2ff;color:#3730a3;border-color:#c7d2fe}
      .cb-badge.stretch{background:#fffbeb;color:#92400e;border-color:#fde68a}
      .cb-badge.gen{background:#f3f4f6;color:#374151;border-color:#e5e7eb}
      .cb-chip{display:inline-block;padding:4px 10px;border-radius:999px;border:1px solid #e5e7eb;background:#fafafa;font-size:12px;margin-right:6px;margin-top:6px}
      .cb-chip:hover{background:#f5f5f5}
      .cb-why{font-size:12px;color:#4b5563}
      .cb-vision{font-size:12px;color:#0f766e}
      /* Expander polish */
      details {
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
        background: #fbfbfb !important;
        transition: box-shadow .2s ease;
      }
      details:hover { box-shadow: 0 1px 12px rgba(0,0,0,.05); }
      summary { padding: 10px 12px !important; }
      /* Primary button */
      .stButton>button {
        border-radius: 10px;
        border: 1px solid #0ea5e9;
        background: linear-gradient(180deg,#22c55e,#16a34a);
        color: #fff; font-weight: 600;
        box-shadow: 0 2px 6px rgba(16,185,129,.25);
        transition: transform .05s ease-in-out, box-shadow .2s ease;
      }
      .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 10px rgba(16,185,129,.35); }
      .stButton>button:active { transform: translateY(0); }
      /* Inputs */
      .stTextInput>div>div>input, .stNumberInput input, .stSelectbox div[data-baseweb="select"]>div {
        border-radius: 10px !important;
      }
      /* Toolbar and grid */
      .cb-toolbar{position:sticky;top:62px;z-index:10;background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:8px 12px;margin:8px 0 10px 0;display:flex;align-items:center;gap:10px;box-shadow:0 1px 2px rgba(0,0,0,0.03)}
      .cb-term{font-size:13px;color:#374151}
      .cb-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
      @media(max-width: 980px){ .cb-grid{grid-template-columns:1fr;} }
      .cb-section{font-size:12px;color:#6b7280;margin:6px 0 4px 0;text-transform:uppercase;letter-spacing:.06em}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- SECRETS ----------
# Put these in .streamlit/secrets.toml or your host's env vars
ADZUNA_APP_ID  = st.secrets["ADZUNA_APP_ID"]
ADZUNA_APP_KEY = st.secrets["ADZUNA_APP_KEY"]





EMBED_MODEL = "text-embedding-3-small"  # or "text-embedding-ada-002" if you prefer
CHAT_MODEL  = "gpt-4"                   # set your preferred chat model

# ---------- SESSION STATE ----------
defaults = {
    "essential_responses": {"career_vision":"", "unique_value":"", "growth_challenge":""},
    "optional_responses":  {"work_env":"", "personal_interests":""},
    "adaptive_responses":  {"leadership_insight":""},
    "detail_level":        "Quick",
    "synopsis_text":       "",
    "cv_text":             "",
    "extracted":           {},   # parsed CV facts (location/degree/subject/skills)
    "adjacent_titles":    [],
    "transitional_roles": [],
    "search_page":        1,
    "accum_results":     [],
    "last_used_term":    None,
    "last_term_category": None,
    "stretch_titles": [],
    "seniority":        {"level":"", "years": None, "evidence": ""},
}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ---------- HELPERS ----------
def extract_text(file):
    if not file: return ""
    if file.type == "application/pdf":
        text = ""
        try:
            reader = PyPDF2.PdfReader(file)
            for p in reader.pages:
                t = p.extract_text()
                if t: text += t + "\n"
        except Exception as e:
            st.error(f"PDF read error: {e}")
        return text.strip()
    elif file.type == "text/plain":
        return file.getvalue().decode("utf-8", errors="ignore")
    else:
        st.error("Please upload a PDF or TXT.")
        return ""

def parse_cv_structured(cv_text: str) -> dict:
    if not cv_text:
        return {
            "location":"Not found","degree":"Not found","subject":"Not found","skills":[],
            "work_history":[], "certifications":[], "languages":[]
        }
    prompt = f"""
Extract the following from the CV text and return valid JSON with keys exactly:
- location (string or "Not found")
- degree (string or "Not found")
- subject (string or "Not found")
- skills (array of strings)
- work_history (array of objects with keys: title, company, start_date, end_date, industry; dates as YYYY-MM or "Unknown")
- certifications (array of strings)
- languages (array of strings)

CV:
'''{cv_text[:12000]}'''
"""
    try:
        r = client.chat.completions.create(model=CHAT_MODEL,
        messages=[
            {"role":"system","content":"Return only valid JSON. No commentary."},
            {"role":"user","content":prompt},
        ],
        temperature=0.0)
        content = r.choices[0].message.content
        data = json.loads(content)
        # Normalise shapes/types
        data.setdefault("skills", [])
        data.setdefault("work_history", [])
        data.setdefault("certifications", [])
        data.setdefault("languages", [])
        if isinstance(data.get("skills"), str):
            data["skills"] = [s.strip() for s in data["skills"].split(",") if s.strip()]
        # coerce work_history items
        wh = []
        for item in (data.get("work_history") or []):
            if not isinstance(item, dict):
                continue
            wh.append({
                "title": str(item.get("title",""))[:80],
                "company": str(item.get("company",""))[:80],
                "start_date": str(item.get("start_date","Unknown"))[:10],
                "end_date": str(item.get("end_date","Unknown"))[:10],
                "industry": str(item.get("industry",""))[:80],
            })
        data["work_history"] = wh
        # coerce simple arrays
        data["certifications"] = [str(x)[:120] for x in (data.get("certifications") or []) if str(x).strip()]
        data["languages"] = [str(x)[:60] for x in (data.get("languages") or []) if str(x).strip()]
        return data
    except Exception as e:
        st.warning(f"CV parse fallback (JSON error): {e}")
        return {
            "location":"Not found","degree":"Not found","subject":"Not found","skills":[],
            "work_history":[], "certifications":[], "languages":[]
        }

def build_profile_synopsis(answers: dict, cv_text: str, detail_level: str) -> str:
    # answers = {essential_responses, optional_responses, adaptive_responses}
    prompt_detail = "detailed and bespoke" if detail_level == "Detailed" else "concise"
    combined_info = f"""
**Career Vision:** {answers['essential'].get('career_vision','')}
**Unique Value Proposition:** {answers['essential'].get('unique_value','')}
**Growth Through Challenges:** {answers['essential'].get('growth_challenge','')}

**Leadership Insight:** {answers['adaptive'].get('leadership_insight','(not provided)')}
**Work Environment:** {answers['optional'].get('work_env','(not provided)')}
**Personal Interests:** {answers['optional'].get('personal_interests','(not provided)')}

**CV Content (excerpt):**
{cv_text[:4000]}
"""
    prompt = f"""You are an expert career advisor.
Using the information below, write a {prompt_detail} **Profile Synopsis** that covers:
- career aspirations & target roles,
- strengths and proof points,
- working style, values, and leadership/collaboration style,
- a short 'roles you fit best' line,
- a short 'stretch roles to explore' line,
- 4–6 bullet achievements written crisply in first person.

Keep it professional, uplifting, and specific. Avoid repeating headings.

INPUT:
{combined_info}

Return only the final synopsis text (no markdown headings).
"""
    try:
        r = client.chat.completions.create(model=CHAT_MODEL,
        messages=[
            {"role":"system","content":"You write professional candidate summaries."},
            {"role":"user","content":prompt},
        ],
        temperature=0.7)
        return r.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating synopsis: {e}")
        # graceful fallback
        return (
            "Profile Synopsis (fallback)\n"
            f"- Vision: {answers['essential'].get('career_vision','')}\n"
            f"- Strengths: {answers['essential'].get('unique_value','')}\n"
            f"- Growth: {answers['essential'].get('growth_challenge','')}\n"
        )

@st.cache_data(ttl=300, show_spinner=False)
def adzuna_search(what=None, where=None, category=None, sort_by=None, distance=None,
                  salary_min=None, job_type="Any", permanent=False,
                  exclude=None, results_per_page=25, page=1) -> list:
    url = f"https://api.adzuna.com/v1/api/jobs/gb/search/{page}"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": results_per_page,
        "content-type": "application/json",
    }
    if what:        params["what"] = what
    if where:       params["where"] = where
    if distance is not None: params["distance"] = int(distance)
    if category:    params["category"] = category
    if sort_by and sort_by != "relevance": params["sort_by"] = sort_by
    if salary_min:  params["salary_min"] = int(salary_min)
    if job_type == "Full Time": params["full_time"] = 1
    if job_type == "Part Time": params["part_time"] = 1
    if permanent:   params["permanent"] = 1
    if exclude:     params["what_exclude"] = exclude

    try:
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code != 200:
            st.error(f"Adzuna error {resp.status_code}: {resp.text[:200]}")
            return []
        return resp.json().get("results", [])
    except Exception as e:
        st.error(f"Adzuna request failed: {e}")
        return []

def embed_texts(texts: list) -> np.ndarray:
    if not texts: return np.zeros((0, 1536), dtype="float32")
    r = client.embeddings.create(model=EMBED_MODEL, input=texts)
    embs = np.array([d.embedding for d in r.data], dtype="float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    return embs / norms  # normalize → cosine ready

def rank_jobs_by_fit(profile_text: str, jobs: list) -> list:
    job_blobs = []
    for j in jobs:
        title = j.get("title","")
        comp  = (j.get("company") or {}).get("display_name","")
        desc  = j.get("description","") or ""
        job_blobs.append(f"{title} at {comp}\n{desc[:2000]}")
    if not job_blobs: return []
    job_embs = embed_texts(job_blobs)
    prof_emb = embed_texts([profile_text]).reshape(1,-1)
    scores = (job_embs @ prof_emb.T).ravel()
    order = np.argsort(-scores)
    ranked = []
    for idx in order:
        j = jobs[idx].copy()
        j["_fit_score"] = float(scores[idx])
        ranked.append(j)
    return ranked

# ---------- LLM JSON helper (with auto-repair) ----------
def llm_json(system_prompt: str, user_prompt: str, temperature: float = 0.2, max_repair: int = 1):
    """Call chat model and parse JSON. If parsing fails, try a short repair pass."""
    def _call(uprompt):
        r = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": uprompt},
            ],
            temperature=temperature,
        )
        return r.choices[0].message.content
    raw = _call(user_prompt)
    for i in range(max_repair + 1):
        try:
            return json.loads(raw)
        except Exception:
            if i >= max_repair:
                raise
            repair_prompt = (
                "Your previous response was not valid JSON. Return ONLY valid JSON that matches the requested structure. "
                "Do not add commentary. Here is the invalid content to fix as JSON:\n\n" + (raw or "")
            )
            raw = _call(repair_prompt)

# ---------- Query expansion (LLM) ----------
def expand_query_terms(base_term: str, seniority: str, skills: list, degree: str, subject: str, top_k: int = 6) -> list:
    base_term = (base_term or "").strip()
    sys = "Return only valid JSON: an array of short strings (job search terms)."
    up = (
        "Given the base job search term, the user's seniority, top skills, degree and subject, "
        "suggest up to 6 alternative job title terms or common aliases that recruiters use. "
        "Keep each term under 40 chars. JSON array only.\n\n"
        f"BASE: {base_term}\nSENIORITY: {seniority}\nDEGREE: {degree}\nSUBJECT: {subject}\nSKILLS: {', '.join((skills or [])[:12])}"
    )
    try:
        data = llm_json(sys, up, temperature=0.2)
        out, seen = [], set()
        for t in (data or [])[:top_k]:
            t = (t or "").strip()
            if t and t.lower() not in seen and len(t) < 60:
                seen.add(t.lower()); out.append(t)
        return out
    except Exception:
        return []


# ---------- Career Ladder generator ----------
def generate_career_ladder(adjacent_roles: list, seniority_level: str, profile_text: str, work_history: list) -> dict:
    """Ask the LLM for next 1–2 logical steps for each adjacent role. Returns {role: [next1, next2]}"""
    try:
        hist = ", ".join([f"{(w.get('title') or '')}@{(w.get('company') or '')}" for w in (work_history or [])][:6])
        prompt = (
            "Create a short career ladder for each adjacent role. For each role, propose the next 1–2 logical steps "
            "(e.g., Senior/Lead/Manager variants), consistent with the user's seniority and background. Return JSON object mapping role -> [next1, next2].\n\n"
            f"SENIORITY: {seniority_level}\nPROFILE:\n{profile_text[:1200]}\n\nADJACENT ROLES: {', '.join((adjacent_roles or [])[:10])}\nWORK HISTORY: {hist}"
        )
        r = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Return only valid JSON: an object mapping role string -> array of 1–2 short next-step titles."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        data = json.loads(r.choices[0].message.content)
        if isinstance(data, dict):
            out = {}
            for k, v in data.items():
                steps = []
                for s in (v or [])[:2]:
                    s = (s or "").strip()
                    if s:
                        steps.append(s[:60])
                if steps:
                    out[str(k)[:60]] = steps
            return out
    except Exception:
        pass
    return {}

# ---------- Market skills miner ----------
def extract_market_skills_from_jobs(jobs: list, top_k: int = 20) -> list:
    """Use the LLM to extract a concise ordered list of required/desired skills from a set of job descriptions."""
    if not jobs:
        return []
    snippets = []
    for j in jobs[:8]:
        title = j.get("title", "")
        comp  = (j.get("company") or {}).get("display_name", "")
        desc  = (j.get("description") or "")[:1200]
        snippets.append(f"{title} — {comp}\n{desc}")
    sys = "Return only valid JSON: an array of unique skill phrases ordered by importance (max 20)."
    up = (
        "From these job descriptions, extract the most important skills/requirements as short phrases. "
        "Use consistent casing (e.g., 'SQL', 'Power BI', 'A/B testing', 'stakeholder management'). Return JSON array only.\n\n"
        + "\n\n---\n".join(snippets)
    )
    try:
        skills = llm_json(sys, up, temperature=0.1)
        clean = []
        seen = set()
        for s in (skills or [])[:top_k]:
            s = (s or "").strip()
            if not s:
                continue
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            clean.append(s[:60])
        return clean
    except Exception:
        return []

# ---------- Progression Plan generator ----------
def generate_progression_plan(
    target_role: str,
    extracted: Dict,
    synopsis: str,
    seniority: Dict,
    market_skills: Optional[List[str]] = None,
    preferred_providers: Optional[List[str]] = None,
) -> Dict:
    """Return structured plan with rationale, skill gaps, actions, courses, certs, projects."""
    sys = "Return only valid JSON with keys: rationale (string), skill_gaps (array of strings), actions (array of objects with keys: step, timeline), courses (array of objects with keys: title, provider, note), certifications (array of strings), projects (array of strings)."
    work_hist_str = "; ".join([f"{(w.get('title') or '')}@{(w.get('company') or '')}" for w in (extracted.get('work_history') or [])[:5]])
    up = (
        "Build a concise progression plan for the target role, grounded in BOTH the user's background AND current market demand. Include rationale, the top missing skills, "
        "concrete actions with rough timelines, 2–3 course suggestions (use generic titles + providers like Coursera/Udemy/edX), "
        "any relevant certifications, and 1–2 portfolio project ideas. Bias course suggestions toward PREFERRED_PROVIDERS when possible. JSON only.\n\n"
        f"TARGET ROLE: {target_role}\nSENIORITY: {seniority.get('level')}\nDEGREE: {extracted.get('degree')}\nSUBJECT: {extracted.get('subject')}\n"
        f"SKILLS: {', '.join((extracted.get('skills') or [])[:15])}\nCERTS: {', '.join((extracted.get('certifications') or [])[:10])}\n"
        f"LANGUAGES: {', '.join((extracted.get('languages') or [])[:10])}\nWORK HIST: {work_hist_str}\n"
        f"MARKET_SKILLS: {', '.join((market_skills or [])[:20])}\n"
        f"PREFERRED_PROVIDERS: {', '.join((preferred_providers or [])[:6])}\n"
        f"\nPROFILE:\n" + synopsis[:1200]
    )
    try:
        r = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
              {"role": "system", "content": sys},
              {"role": "user", "content": up},
            ],
            temperature=0.2,
        )
        data = json.loads(r.choices[0].message.content)
        if isinstance(data, dict):
            # light sanitation
            data.setdefault('rationale', '')
            data['skill_gaps'] = [str(x)[:80] for x in (data.get('skill_gaps') or []) if str(x).strip()][:8]
            def _arr_objs(arr, keys):
                out=[]
                for o in (arr or [])[:6]:
                    if not isinstance(o, dict):
                        continue
                    out.append({k: str(o.get(k,""))[:120] for k in keys})
                return out
            data['actions']  = _arr_objs(data.get('actions'),  ['step','timeline'])
            data['courses']  = _arr_objs(data.get('courses'),  ['title','provider','note'])
            data['certifications'] = [str(x)[:80] for x in (data.get('certifications') or []) if str(x).strip()][:8]
            data['projects'] = [str(x)[:120] for x in (data.get('projects') or []) if str(x).strip()][:6]
            return data
    except Exception:
        pass
    return {}

# ---------- Roadmap markdown helper ----------
def make_roadmap_markdown(synopsis: str, adj: list, stretch: list, pathways: list, ladder: dict, plan: dict) -> str:
    md = ["# Career Roadmap\n"]
    md.append("## Profile Synopsis\n" + (synopsis or "") + "\n")
    if adj:
        md.append("## Adjacent Roles\n- " + "\n- ".join(adj) + "\n")
    if stretch:
        md.append("## Stretch Roles\n- " + "\n- ".join(stretch) + "\n")
    if pathways:
        md.append("## Pathways Beyond Your Industry\n- " + "\n- ".join([ (p.get('title') or '') for p in (pathways or []) if p.get('title') ]) + "\n")
    if ladder:
        md.append("## Career Ladder\n")
        for k,v in ladder.items():
            md.append(f"- **{k}** → " + " → ".join(v))
        md.append("")
    if plan:
        md.append("## Progression Plan\n")
        if plan.get('rationale'): md.append("**Why this role**: " + plan['rationale'] + "\n")
        if plan.get('skill_gaps'): md.append("**Skill gaps**: " + ", ".join(plan['skill_gaps']) + "\n")
        if plan.get('actions'):
            md.append("**Actions & timeline**:\n" + "\n".join([f"- {a.get('step')} ({a.get('timeline')})" for a in plan['actions']]))
        if plan.get('courses'):
            md.append("\n**Courses**:\n" + "\n".join([f"- {c.get('title')} — {c.get('provider')} ({c.get('note')})" for c in plan['courses']]))
        if plan.get('certifications'):
            md.append("\n**Certifications**: " + ", ".join(plan['certifications']))
        if plan.get('projects'):
            md.append("\n**Project ideas**:\n" + "\n".join([f"- {p}" for p in plan['projects']]))
    return "\n".join(md)

# ---------- CATEGORY/STRETCH HELPERS ----------
def classify_term(term: str, adjacent_titles: list, pathways: list, seniority_level: str) -> str:
    t = (term or "").strip().lower()
    adj = {(x or "").strip().lower() for x in (adjacent_titles or [])}
    path = {((o or {}).get("title") or "").strip().lower() for o in (pathways or [])}
    stretch_prefixes = ("junior","senior","lead","manager","head","director","vp","executive")
    if t in adj:
        return "Adjacent Role"
    if t in path:
        return "Pathway Beyond Industry"
    if any(t.startswith(p + " ") for p in stretch_prefixes):
        return "Stretch Role"
    return "General"

def make_stretch_titles(adjacent_titles: list, seniority_level: str) -> list:
    level = (seniority_level or "").lower().strip()
    pref_map = {
        "entry": ["junior"],
        "junior": ["junior"],
        "mid": ["senior"],
        "senior": ["lead"],
        "lead": ["manager"],
        "manager": ["head"],
        "director": ["vp", "director"],
        "executive": ["executive"],
    }
    prefixes = pref_map.get(level, ["senior"])  # default progression
    out = []
    for title in (adjacent_titles or [])[:8]:
        base = (title or "").strip()
        if not base:
            continue
        for p in prefixes:
            out.append(f"{p.title()} {base}")
    # de-duplicate while preserving order
    seen = set(); ded = []
    for t in out:
        k = t.lower()
        if k in seen: continue
        seen.add(k); ded.append(t)
    return ded[:12]

# ---------- SENIORITY INFERENCE ----------
def infer_seniority(profile_text: str, cv_text: str) -> dict:
    """Infer years of experience and seniority level using the LLM. Returns {level, years, evidence}."""
    try:
        prompt = (
            "From the following candidate materials, infer (1) total professional years of experience, "
            "and (2) a single seniority level from this set: entry, junior, mid, senior, lead, manager, director, executive. "
            "Return strict JSON with keys: years (number or null), level (string), evidence (short phrase).")
        content = (
            f"PROFILE SYNOPSIS:\n{profile_text[:2000]}\n\n"
            f"CV EXCERPT:\n{cv_text[:3000]}\n\n"
            f"WORK HISTORY (compact):\n" + " | ".join([
                f"{(w.get('title') or '')} @ {(w.get('company') or '')} ({(w.get('start_date') or 'Unknown')}–{(w.get('end_date') or 'Unknown')})"
                for w in (st.session_state.get('extracted', {}).get('work_history') or [])[:6]
            ])
        )
        r = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt + "\n\n" + content},
            ],
            temperature=0.0,
        )
        data = json.loads(r.choices[0].message.content)
        level = (data.get("level") or "").strip().lower()
        years = data.get("years")
        evidence = (data.get("evidence") or "").strip()
        # basic sanitation
        allowed = {"entry","junior","mid","senior","lead","manager","director","executive"}
        if level not in allowed:
            level = "mid"
        try:
            years = float(years) if years is not None else None
        except Exception:
            years = None
        return {"level": level, "years": years, "evidence": evidence[:160]}
    except Exception:
        return {"level": "mid", "years": None, "evidence": "defaulted"}

# ---------- ADJACENT ROLES SUGGESTION ----------
def suggest_adjacent_roles(profile_text: str, skills: list, degree: str = "", subject: str = "", extras: dict = None, work_history: list = None, certs: list = None, languages: list = None, top_k: int = 8) -> list:
    """Suggest adjacent job titles using synopsis, skills, degree/subject, broader Q&A context, and history/certs/languages."""
    try:
        wh_lines = []
        for w in (work_history or [])[:6]:
            wh_lines.append(f"{(w.get('title') or '')} @ {(w.get('company') or '')} ({(w.get('start_date') or 'Unknown')}–{(w.get('end_date') or 'Unknown')}) [{(w.get('industry') or '')}]")
        prompt = (
            "Suggest 8 adjacent job titles that are realistic next steps **within the same industry or closely related industries**. "
            "Focus on incremental growth, natural progressions, and specialisations that build directly on the candidate’s degree, skills, and aspirations. "
            "Exclude radical pivots or unrelated industries. "
            "Return a JSON array of short job titles only (no explanations).\n\n"
            f"PROFILE SYNOPSIS:\n{profile_text[:2000]}\n\n"
            f"TOP SKILLS:\n{', '.join((skills or [])[:15])}\n\n"
            f"DEGREE: {degree or 'Not provided'}\n"
            f"SUBJECT: {subject or 'Not provided'}\n"
            f"WORK HISTORY:\n{'; '.join(wh_lines)}\n"
            f"CERTIFICATIONS: {', '.join((certs or [])[:10])}\n"
            f"LANGUAGES: {', '.join((languages or [])[:10])}\n"
            f"CAREER VISION: {(extras or {}).get('career_vision','')}\n"
            f"UNIQUE VALUE: {(extras or {}).get('unique_value','')}\n"
            f"GROWTH CHALLENGE: {(extras or {}).get('growth_challenge','')}\n"
            f"LEADERSHIP INSIGHT: {(extras or {}).get('leadership_insight','')}\n"
            f"WORK ENVIRONMENT: {(extras or {}).get('work_env','')}\n"
            f"PERSONAL INTERESTS: {(extras or {}).get('personal_interests','')}\n"
            f"SENIORITY: {((extras or {}).get('seniority_level') or (extras or {}).get('seniority') or '')}\n"
        )
        r = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Return only valid JSON (an array of strings)."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        payload = r.choices[0].message.content
        data = json.loads(payload)
        out = []
        seen = set()
        for t in data:
            t = (t or "").strip()
            if t and len(t) < 60 and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        return out[:top_k]
    except Exception:
        return [
            "Data Analyst", "Business Analyst", "Insights Analyst",
            "Analytics Consultant", "Operations Analyst", "Reporting Analyst",
            "Market Research Analyst", "Revenue Analyst",
        ][:top_k]

# ---------- TRANSITIONAL ROLES SUGGESTION ----------
def suggest_transitional_roles(profile_text: str, skills: list, degree: str = "", subject: str = "", extras: dict = None, work_history: list = None, certs: list = None, languages: list = None, top_k: int = 8) -> list:
    """Suggest cross-industry roles leveraging transferable skills, academic background, broader profile context, and history/certs/languages."""
    try:
        wh_lines = []
        for w in (work_history or [])[:6]:
            wh_lines.append(f"{(w.get('title') or '')} @ {(w.get('company') or '')} ({(w.get('start_date') or 'Unknown')}–{(w.get('end_date') or 'Unknown')}) [{(w.get('industry') or '')}]")
        prompt = (
            "Suggest 8 realistic career pathways **outside the candidate's current or neighboring industries**. "
            "Focus on bold but plausible cross-industry pivots that leverage transferable skills, academic background, and personal aspirations. "
            "Do not include direct extensions of their current field. "
            "Return JSON as an array of objects with keys: 'title' and 'note'. 'title' is a short job title; 'note' is a 6–12 word reason mentioning overlapping skills or degree/subject relevance.\n\n"
            f"PROFILE SYNOPSIS:\n{profile_text[:1800]}\n\n"
            f"TOP SKILLS:\n{', '.join((skills or [])[:15])}\n\n"
            f"DEGREE: {degree or 'Not provided'}\n"
            f"SUBJECT: {subject or 'Not provided'}\n"
            f"WORK HISTORY:\n{' ; '.join(wh_lines)}\n"
            f"CERTIFICATIONS: {', '.join((certs or [])[:10])}\n"
            f"LANGUAGES: {', '.join((languages or [])[:10])}\n"
            f"CAREER VISION: {(extras or {}).get('career_vision','')}\n"
            f"UNIQUE VALUE: {(extras or {}).get('unique_value','')}\n"
            f"GROWTH CHALLENGE: {(extras or {}).get('growth_challenge','')}\n"
            f"LEADERSHIP INSIGHT: {(extras or {}).get('leadership_insight','')}\n"
            f"WORK ENVIRONMENT: {(extras or {}).get('work_env','')}\n"
            f"PERSONAL INTERESTS: {(extras or {}).get('personal_interests','')}\n"
            f"SENIORITY: {((extras or {}).get('seniority_level') or (extras or {}).get('seniority') or '')}\n"
        )
        r = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Return only valid JSON (array of objects with 'title' and 'note')."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )
        data = json.loads(r.choices[0].message.content)
        out = []
        seen = set()
        for obj in data:
            title = (obj.get("title") or "").strip()
            note  = (obj.get("note")  or "").strip()
            if not title:
                continue
            key = title.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append({"title": title[:60], "note": note[:140]})
        return out[:top_k]
    except Exception:
        return [
            {"title": "Digital Marketing Analyst", "note": "Analytical skills + SQL applied to campaigns"},
            {"title": "Supply Chain Analyst", "note": "Forecasting & reporting for logistics"},
            {"title": "Product Analyst", "note": "User/event data, stakeholder reporting"},
            {"title": "People Analytics Specialist", "note": "HR insights using dashboards & stats"},
            {"title": "Customer Insights Analyst", "note": "Survey and behavioural analysis"},
            {"title": "Sales Operations Analyst", "note": "Pipeline reporting, revenue metrics"},
            {"title": "Risk & Compliance Analyst", "note": "Controls, documentation, trend analysis"},
            {"title": "Operations Research Analyst", "note": "Optimisation & scenario modelling"},
        ][:top_k]

STOPWORDS = set("""
a an and are as at be by for from has have i in is it its of on or that the to with you your our we they this those these their
role roles experience experienced strong skills skill ability abilities deliver delivering delivered responsible responsibility
team teams working work works passion passionate proven track record knowledge understanding understanding of excellent good great
""".split())

def match_reasons(profile_text: str, skills: list, job_text: str, extra_keywords: list = None, top_k: int = 3) -> list:
    reasons = []
    job_lc = job_text.lower()
    # 1) skills first
    for s in (skills or [])[:15]:
        tok = (s or "").strip().lower()
        if not tok or tok in STOPWORDS: continue
        if tok in job_lc and tok not in reasons:
            reasons.append(tok)
            if len(reasons) >= top_k: return reasons
    # 1b) extra keywords (titles, certs, languages)
    for s in (extra_keywords or [])[:15]:
        tok = (s or "").strip().lower()
        if not tok or tok in STOPWORDS: continue
        if tok in job_lc and tok not in reasons:
            reasons.append(tok)
            if len(reasons) >= top_k: return reasons
    # 2) fallback: profile keywords
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z\-+/&0-9]{1,}", profile_text.lower()) if w not in STOPWORDS]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    for w,_ in sorted(freq.items(), key=lambda x: -x[1]):
        if w in job_lc and w not in reasons:
            reasons.append(w)
            if len(reasons) >= top_k: break
    return reasons[:top_k]

 # ---------- SIDEBAR: Status & Tips ----------
with st.sidebar:
    st.title("🚀 Ready to search")
    syn_ready = bool(st.session_state.get("synopsis_text"))
    adj_n  = len(st.session_state.get("adjacent_titles") or [])
    path_n = len(st.session_state.get("transitional_roles") or [])
    sen    = st.session_state.get("seniority", {}).get("level") or "?"
    st.metric("Profile synopsis", "Ready" if syn_ready else "Missing")
    st.metric("Adjacent roles", adj_n)
    st.metric("Pathways", path_n)
    st.metric("Seniority", sen.title() if isinstance(sen, str) else "?")
    st.caption("Tip: You can edit extracted CV facts in Step 2 → then click ‘Apply edits’. Suggestions refresh instantly.")
    with st.expander("Deployment checks", expanded=False):
        try:
            st.write(f"Python: {sys.version.split()[0]} on {platform.system()} {platform.release()}")
        except Exception:
            st.write("Python: unknown")
        try:
            import streamlit as _st
            st.write(f"Streamlit: {_st.__version__}")
        except Exception:
            pass
        try:
            st.write(f"OpenAI SDK: {getattr(openai_pkg, '__version__', 'unknown')}")
        except Exception:
            pass
        ok_openai = bool(st.secrets.get("OPENAI_API_KEY"))
        ok_adzuna = bool(st.secrets.get("ADZUNA_APP_ID")) and bool(st.secrets.get("ADZUNA_APP_KEY"))
        st.write("OPENAI_API_KEY: " + ("✅ set" if ok_openai else "❌ missing"))
        st.write("ADZUNA keys: " + ("✅ set" if ok_adzuna else "❌ missing"))

# ---------- UI: TABS ----------
# ---------- Reusable card renderer ----------
def render_job_card(j: dict, fit: float, category: str, term: str, why: list, vision_snip: str = ""):
    title = j.get("title","")
    company = (j.get("company") or {}).get("display_name","")
    loc = (j.get("location") or {}).get("display_name","")
    url = j.get("redirect_url","#")
    smin = j.get("salary_min"); smax = j.get("salary_max")
    # badge class
    cls = {
        "Adjacent Role": "adj",
        "Pathway Beyond Industry": "path",
        "Stretch Role": "stretch",
        "General": "gen",
    }.get(category or "General", "gen")
    badge = f"<span class='cb-badge {cls}'>{category or 'General'}</span>"
    salary = f"£{int(smin):,}–£{int(smax):,}" if smin and smax else (f"£{int(smin):,}+" if smin else (f"£{int(smax):,}" if smax else ""))
    why_txt = (" · ".join(why)) if why else ""
    st.markdown(
        f"""
        <div class='cb-card'>
          <div class='cb-row'>
            {badge}
            <div class='cb-title'>{title}</div>
          </div>
          <div class='cb-sub'>{company} · {loc}</div>
          <div class='cb-meta'>Fit score: <b>{fit:.3f}</b> · {salary} · term: <code>{term}</code></div>
          <div class='cb-sep'></div>
          {f"<div class='cb-why'>Why this matches: {why_txt}</div>" if why_txt else ''}
          {f"<div class='cb-vision'>Aligned with your vision: {vision_snip}…</div>" if vision_snip else ''}
          <div style='margin-top:10px;'>
            <a href='{url}' target='_blank'>View job ↗</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
tab1, tab2, tab3 = st.tabs(["Profile Q&A", "Upload CV & Synopsis", "Job Matches"])

# --- TAB 1: Questionnaire ---
with tab1:
    st.header("Step 1 · Tell us about yourself")
    st.caption("Your answers stay in this session only. Nothing is stored server-side.")

    colA, colB = st.columns(2)
    with colA:
        st.text_area(
            "What are your primary career goals and aspirations? How does your ideal role support your long-term vision?",
            key="career_vision",
            value=st.session_state.essential_responses["career_vision"],
            height=120
        )
        st.text_area(
            "Describe a significant project or challenge you’ve faced and what you learned from it.",
            key="growth_challenge",
            value=st.session_state.essential_responses["growth_challenge"],
            height=120
        )
    with colB:
        st.text_area(
            "Which skills and experiences uniquely distinguish you? (e.g., leadership, teamwork, innovation)",
            key="unique_value",
            value=st.session_state.essential_responses["unique_value"],
            height=120
        )
        st.selectbox(
            "Profile detail level",
            ["Quick","Detailed"],
            key="detail_level",
            index=0 if st.session_state.detail_level=="Quick" else 1
        )

    # Save essential
    if st.button("Save responses"):
        st.session_state.essential_responses = {
            "career_vision": st.session_state.career_vision,
            "unique_value":  st.session_state.unique_value,
            "growth_challenge": st.session_state.growth_challenge,
        }
        st.success("Saved!")

    # Adaptive follow-up if leadership/team is mentioned
    uv = (st.session_state.essential_responses["unique_value"] or "").lower()
    if ("leadership" in uv or "team" in uv or "mentorship" in uv):
        st.subheader("Adaptive follow-up (leadership)")
        st.text_area(
            "Describe your leadership style or collaboration approach:",
            key="leadership_insight",
            value=st.session_state.adaptive_responses["leadership_insight"]
        )
        if st.button("Save leadership insight"):
            st.session_state.adaptive_responses["leadership_insight"] = st.session_state.leadership_insight
            st.success("Saved leadership insight")

    with st.expander("Optional insights"):
        st.text_area(
            "What work environment and culture do you thrive in?",
            key="work_env",
            value=st.session_state.optional_responses["work_env"]
        )
        st.text_area(
            "How do your personal interests contribute to your professional growth?",
            key="personal_interests",
            value=st.session_state.optional_responses["personal_interests"]
        )
        if st.button("Save optional insights"):
            st.session_state.optional_responses = {
                "work_env": st.session_state.work_env,
                "personal_interests": st.session_state.personal_interests
            }
            st.success("Saved optional insights")

# --- TAB 2: Upload CV & Generate Synopsis ---
with tab2:
    st.header("Step 2 · Upload your CV & generate your profile")
    uploaded = st.file_uploader("Upload CV (PDF/TXT)", type=["pdf","txt"])
    if uploaded:
        st.session_state.cv_text = extract_text(uploaded)
        st.info("CV loaded. (Preview below)")
        st.text_area("CV preview (first 1,000 chars)", value=st.session_state.cv_text[:1000], height=200)
    c_demo1, c_demo2 = st.columns([1,1])
    if c_demo1.button("✨ Load demo profile"):
        demo_cv = (
            "Mia Taylor\n\n"
            "Education\nBSc (Hons) Economics, University of Manchester, 2019\n\n"
            "Experience\nData Analyst, RetailCo (2022–Present) — SQL, Python, Power BI; built weekly sales dashboards; A/B testing for promos.\n"
            "Business Analyst, FinServe (2019–2022) — Stakeholder management, requirements, reporting, Excel, Tableau.\n\n"
            "Skills\nSQL, Python (pandas), Power BI, Tableau, Excel, A/B testing, Forecasting, Stakeholder management\n"
            "Certifications\nGoogle Data Analytics Professional Certificate\n"
        )
        st.session_state.cv_text = demo_cv
        st.session_state.essential_responses = {
            "career_vision": "Grow into an analytics leadership role driving decisions with data.",
            "unique_value": "Blend of business partnering and hands-on analytics; strong stakeholder skills.",
            "growth_challenge": "Scaling impact from dashboards to experimentation and causal inference.",
        }
        st.session_state.optional_responses = {
            "work_env": "Collaborative, product-led teams with clear outcomes.",
            "personal_interests": "Behavioural economics, fitness, mentoring juniors.",
        }
        st.success("Demo profile loaded. You can generate a synopsis now.")
    if c_demo2.button("🗑️ Clear demo/profile"):
        st.session_state.cv_text = ""
        st.session_state.synopsis_text = ""
        st.session_state.extracted = {}
        st.session_state.adjacent_titles = []
        st.session_state.transitional_roles = []
        st.session_state.progression_plan = None
        st.session_state.career_ladder = {}
        st.success("Cleared. Upload your CV or type answers to proceed.")

    if st.button("Generate Profile Synopsis", disabled=not (st.session_state.cv_text or any(st.session_state.essential_responses.values()))):
        with st.status("Generating your profile…", expanded=True) as status:
            progress = st.progress(0)
            # 1) Parse CV
            status.update(label="Parsing CV…")
            st.session_state.extracted = parse_cv_structured(st.session_state.cv_text)
            progress.progress(15)

            # 2) Build synopsis
            status.update(label="Writing your synopsis…")
            answers = {
                "essential": st.session_state.essential_responses,
                "optional":  st.session_state.optional_responses,
                "adaptive":  st.session_state.adaptive_responses,
            }
            st.session_state.synopsis_text = build_profile_synopsis(
                answers=answers,
                cv_text=st.session_state.cv_text,
                detail_level=st.session_state.detail_level
            )
            progress.progress(55)

            # 3) Infer seniority
            status.update(label="Inferring seniority…")
            try:
                st.session_state.seniority = infer_seniority(
                    st.session_state.synopsis_text,
                    st.session_state.cv_text,
                )
            except Exception:
                st.session_state.seniority = {"level":"mid","years": None, "evidence":"defaulted"}
            progress.progress(70)

            # 4) Suggestions: adjacent and pathways
            status.update(label="Suggesting adjacent roles…")
            try:
                extras_ctx = {
                    **st.session_state.essential_responses,
                    **st.session_state.adaptive_responses,
                    **st.session_state.optional_responses,
                    "seniority_level": st.session_state.get("seniority", {}).get("level", ""),
                }
                st.session_state.adjacent_titles = suggest_adjacent_roles(
                    st.session_state.synopsis_text,
                    st.session_state.extracted.get("skills", []),
                    st.session_state.extracted.get("degree"),
                    st.session_state.extracted.get("subject"),
                    extras_ctx,
                    st.session_state.extracted.get("work_history"),
                    st.session_state.extracted.get("certifications"),
                    st.session_state.extracted.get("languages"),
                )
            except Exception:
                st.session_state.adjacent_titles = []
            progress.progress(82)

            status.update(label="Mapping pathways beyond your industry…")
            try:
                roles = suggest_transitional_roles(
                    st.session_state.synopsis_text,
                    st.session_state.extracted.get("skills", []),
                    st.session_state.extracted.get("degree"),
                    st.session_state.extracted.get("subject"),
                    extras_ctx,
                    st.session_state.extracted.get("work_history"),
                    st.session_state.extracted.get("certifications"),
                    st.session_state.extracted.get("languages"),
                )
                adj = {t.strip().lower() for t in (st.session_state.adjacent_titles or [])}
                st.session_state.transitional_roles = [o for o in (roles or []) if (o.get("title"," ").strip().lower() not in adj)]
            except Exception:
                st.session_state.transitional_roles = []
            progress.progress(100)
            status.update(label="Profile ready!", state="complete")
        st.toast("Profile generated — jump to Step 3 to find matches ✨")

    if st.session_state.synopsis_text:
        st.subheader("Profile Synopsis")
        sen = st.session_state.get("seniority", {})
        if sen:
            yrs = (f" ~{int(sen['years'])} yrs" if isinstance(sen.get('years'), (int,float)) else "")
            st.caption(f"🌟 Estimated seniority: **{sen.get('level','mid').title()}**{yrs} — {sen.get('evidence','').strip()}")
        st.write(st.session_state.synopsis_text)
        st.download_button(
            "Download synopsis (TXT)",
            data=st.session_state.synopsis_text.encode(),
            file_name="profile_synopsis.txt",
            mime="text/plain"
        )
        with st.expander("🔎 Extracted facts from your CV (edit/confirm)", expanded=False):
            st.caption("You can edit these before searching. Click ‘Apply edits’ to refresh suggestions.")
            current_facts = {
                "location": st.session_state.extracted.get("location"),
                "degree": st.session_state.extracted.get("degree"),
                "subject": st.session_state.extracted.get("subject"),
                "skills": st.session_state.extracted.get("skills", []),
                "work_history": st.session_state.extracted.get("work_history", []),
                "certifications": st.session_state.extracted.get("certifications", []),
                "languages": st.session_state.extracted.get("languages", []),
            }
            facts_json = st.text_area("Edit JSON", value=json.dumps(current_facts, ensure_ascii=False, indent=2), height=260)
            c1, c2 = st.columns([1,1])
            if c1.button("Apply edits & refresh suggestions"):
                try:
                    new_data = json.loads(facts_json)
                    st.session_state.extracted.update(new_data)
                    # Recompute suggestions with edits
                    extras_ctx = {
                        **st.session_state.essential_responses,
                        **st.session_state.adaptive_responses,
                        **st.session_state.optional_responses,
                        "seniority_level": st.session_state.get("seniority", {}).get("level", ""),
                    }
                    st.session_state.adjacent_titles = suggest_adjacent_roles(
                        st.session_state.synopsis_text,
                        st.session_state.extracted.get("skills", []),
                        st.session_state.extracted.get("degree"),
                        st.session_state.extracted.get("subject"),
                        extras_ctx,
                        st.session_state.extracted.get("work_history"),
                        st.session_state.extracted.get("certifications"),
                        st.session_state.extracted.get("languages"),
                    )
                    roles = suggest_transitional_roles(
                        st.session_state.synopsis_text,
                        st.session_state.extracted.get("skills", []),
                        st.session_state.extracted.get("degree"),
                        st.session_state.extracted.get("subject"),
                        extras_ctx,
                        st.session_state.extracted.get("work_history"),
                        st.session_state.extracted.get("certifications"),
                        st.session_state.extracted.get("languages"),
                    )
                    adj = {t.strip().lower() for t in (st.session_state.adjacent_titles or [])}
                    st.session_state.transitional_roles = [o for o in (roles or []) if (o.get("title"," ").strip().lower() not in adj)]
                    st.success("Applied edits. Suggestions refreshed.")
                except Exception as e:
                    st.error(f"JSON error: {e}")
            if c2.button("Reset to auto-extracted"):
                st.session_state.extracted = parse_cv_structured(st.session_state.cv_text)
                st.success("Facts reset from CV. Click ‘Apply edits’ if you want to tweak again.")

# --- TAB 3: Job Matches ---
with tab3:
    st.header("Step 3 · Find matching jobs")
    st.caption("We don’t store your CV or answers. Searches are live against the jobs API.")

    left, right = st.columns([2,1])
    with right:
        st.markdown("<div class='cb-section'>Filters</div>", unsafe_allow_html=True)
        min_salary    = st.number_input("Min salary", 0, 500000, 0, step=1000)
        job_type      = st.selectbox("Job type", ["Any","Full Time","Part Time"])
        permanent     = st.checkbox("Permanent only")
        st.markdown("<div class='cb-section'>Keywords & sorting</div>", unsafe_allow_html=True)
        exclude_terms = st.text_input("Exclude keywords")
        sort_by       = st.selectbox("Sort by", ["relevance","date","salary"])
        st.markdown("<div class='cb-section'>Location</div>", unsafe_allow_html=True)
        location_pref = st.text_input("Location (optional)", value=(st.session_state.extracted.get("location") or "London"))
        remote_only  = st.checkbox("Remote/Hybrid only", value=False)
        radius_miles = st.slider("Search radius (miles)", 0, 50, 25)
        st.markdown("<div class='cb-section'>Discovery</div>", unsafe_allow_html=True)
        broaden = st.checkbox("Broaden search with adjacent roles", value=True)

    with left:
        st.write("**Search term** prioritises your subject, then skills, then a generic fallback.")
        subject = st.session_state.extracted.get("subject","Not found")
        skills  = st.session_state.extracted.get("skills",[])
        search_term = subject if subject and subject.lower()!="not found" else (skills[0] if skills else "analyst")
        st.code(f"Search: {search_term} | Location: {location_pref}")

        # Sticky control bar with active term/category and quick reset
        _sel = st.session_state.get("selected_term") or search_term
        _cat = st.session_state.get("_selected_category") or classify_term(
            _sel,
            st.session_state.get("adjacent_titles"),
            st.session_state.get("transitional_roles"),
            st.session_state.get("seniority", {}).get("level")
        )
        _cls = 'adj' if _cat=='Adjacent Role' else ('path' if _cat=='Pathway Beyond Industry' else ('stretch' if _cat=='Stretch Role' else 'gen'))
        st.markdown(f"<div class='cb-toolbar'><span class='cb-badge {_cls}'>{_cat}</span><span class='cb-term'>Searching: <code>{_sel}</code> in <b>{location_pref or 'Anywhere'}</b></span></div>", unsafe_allow_html=True)
        c_reset = st.columns([6,1])
        with c_reset[1]:
            if st.button("Reset selection"):
                st.session_state["selected_term"] = None
                st.session_state["_selected_category"] = None

        # ⭐ Stretch roles: slightly more senior variants of adjacent roles
        stretch = make_stretch_titles(
            st.session_state.get("adjacent_titles"),
            st.session_state.get("seniority", {}).get("level")
        )
        st.session_state["stretch_titles"] = stretch
        if stretch:
            with st.expander("⭐ Stretch roles (one level up)"):
                cols = st.columns(3)
                for i, title in enumerate(stretch):
                    if cols[i % 3].button(title, key=f"stretch_{i}"):
                        st.session_state["selected_term"] = title
                        st.session_state["_selected_category"] = "Stretch Role"
                        st.info(f"Stretch role selected: {title}")

        if st.session_state.adjacent_titles:
            with st.expander("✨ You could also explore these adjacent roles"):
                cols = st.columns(3)
                for i, title in enumerate(st.session_state.adjacent_titles):
                    if cols[i % 3].button(title, key=f"adjrole_{i}"):
                        st.session_state["selected_term"] = title
                        st.session_state["_selected_category"] = "Adjacent Role"
                        st.info(f"Quick role selected: {title}")

        # Cross-industry pathways expander (always visible)
        # If synopsis exists but no pathways yet, try to generate them on the fly
        if st.session_state.synopsis_text and not st.session_state.get("transitional_roles"):
            try:
                extras_ctx = {
                    **st.session_state.essential_responses,
                    **st.session_state.adaptive_responses,
                    **st.session_state.optional_responses,
                    "seniority_level": st.session_state.get("seniority", {}).get("level", ""),
                }
                roles = suggest_transitional_roles(
                    st.session_state.synopsis_text,
                    st.session_state.extracted.get("skills", []),
                    st.session_state.extracted.get("degree"),
                    st.session_state.extracted.get("subject"),
                    extras_ctx,
                    st.session_state.extracted.get("work_history"),
                    st.session_state.extracted.get("certifications"),
                    st.session_state.extracted.get("languages"),
                )
                adj = {t.strip().lower() for t in (st.session_state.adjacent_titles or [])}
                st.session_state.transitional_roles = [o for o in (roles or []) if (o.get("title"," ").strip().lower() not in adj)]
            except Exception:
                st.session_state.transitional_roles = []

        with st.expander("🌱 Pathways Beyond Your Industry", expanded=False):
            if not st.session_state.synopsis_text:
                st.caption("Generate your profile synopsis first to see cross‑industry pathways.")
            else:
                roles = st.session_state.get("transitional_roles") or []
                if not roles:
                    if st.button("↻ Refresh pathways"):
                        try:
                            extras_ctx = {
                                **st.session_state.essential_responses,
                                **st.session_state.adaptive_responses,
                                **st.session_state.optional_responses,
                                "seniority_level": st.session_state.get("seniority", {}).get("level", ""),
                            }
                            roles = suggest_transitional_roles(
                                st.session_state.synopsis_text,
                                st.session_state.extracted.get("skills", []),
                                st.session_state.extracted.get("degree"),
                                st.session_state.extracted.get("subject"),
                                extras_ctx,
                                st.session_state.extracted.get("work_history"),
                                st.session_state.extracted.get("certifications"),
                                st.session_state.extracted.get("languages"),
                            )
                            adj = {t.strip().lower() for t in (st.session_state.adjacent_titles or [])}
                            st.session_state.transitional_roles = [o for o in (roles or []) if (o.get("title"," ").strip().lower() not in adj)]
                        except Exception:
                            st.session_state.transitional_roles = []
                        roles = st.session_state.get("transitional_roles") or []
                if roles:
                    cols = st.columns(3)
                    for i, obj in enumerate(roles):
                        title = (obj.get("title") or "").strip()
                        note  = (obj.get("note")  or "").strip()
                        if not title:
                            continue
                        if cols[i % 3].button(title, key=f"transrole_{i}"):
                            st.session_state["selected_term"] = title
                            st.session_state["_selected_category"] = "Pathway Beyond Industry"
                            st.info(f"Selected pathway: {title}")
                        if note:
                            cols[i % 3].caption(note)

        # 📈 Career Ladder (AI)
        with st.expander("📈 Career Ladder", expanded=False):
            if st.button("Generate ladder", key="gen_ladder_btn"):
                st.session_state.career_ladder = generate_career_ladder(
                    st.session_state.get("adjacent_titles"),
                    st.session_state.get("seniority", {}).get("level"),
                    st.session_state.synopsis_text,
                    st.session_state.extracted.get("work_history")
                )
            ladder = st.session_state.get("career_ladder") or {}
            if ladder:
                for base, steps in ladder.items():
                    st.markdown(f"**{base}** → " + " → ".join(steps))
                    # chips for next steps
                    ccols = st.columns(min(3, max(1, len(steps))))
                    for i, nx in enumerate(steps):
                        if ccols[i % len(ccols)].button(nx, key=f"ladder_{base}_{i}"):
                            st.session_state["selected_term"] = nx
                            # set category badge based on classification
                            st.session_state["_selected_category"] = classify_term(
                                nx,
                                st.session_state.get("adjacent_titles"),
                                st.session_state.get("transitional_roles"),
                                st.session_state.get("seniority", {}).get("level")
                            )
                            st.info(f"Selected next step: {nx}")
            else:
                st.caption("Click ‘Generate ladder’ to see your next steps.")

        # 🧭 Progression Plan (AI)
        with st.expander("🧭 Progression Plan (AI)", expanded=False):
            adj = st.session_state.get("adjacent_titles") or []
            stretch = st.session_state.get("stretch_titles") or []
            paths = [o.get("title") for o in (st.session_state.get("transitional_roles") or []) if o.get("title")]
            choices = [t for t in (adj + stretch + paths) if t]
            target = st.selectbox("Target role", options=["(select)"] + choices, index=0, key="plan_target")
            providers_str = st.text_input("Preferred providers (comma‑separated)", value="Coursera, edX, Udemy, Google, AWS, Microsoft")
            providers = [p.strip() for p in providers_str.split(",") if p.strip()]
            if st.button("Generate plan", disabled=(target == "(select)")):
                # Mine live market skills from top job postings for the chosen target
                market_jobs = adzuna_search(
                    what=target,
                    where=location_pref or None,
                    sort_by=sort_by,
                    salary_min=min_salary if min_salary>0 else None,
                    job_type=job_type,
                    permanent=permanent,
                    exclude=exclude_terms or None,
                    distance=radius_miles,
                    results_per_page=25,
                    page=1,
                )
                market_skills = extract_market_skills_from_jobs(market_jobs[:5])
                st.session_state.progression_plan = generate_progression_plan(
                    target,
                    st.session_state.extracted,
                    st.session_state.synopsis_text,
                    st.session_state.get("seniority", {}),
                    market_skills=market_skills,
                    preferred_providers=providers,
                )
            plan = st.session_state.get("progression_plan") or {}
            if plan:
                if plan.get("rationale"): st.markdown("**Why this role**: " + plan["rationale"])
                if plan.get("skill_gaps"): st.markdown("**Skill gaps**: " + ", ".join(plan["skill_gaps"]))
                if plan.get("actions"):
                    st.markdown("**Actions & timeline**:")
                    for a in plan["actions"]:
                        st.markdown(f"- {a.get('step')} ({a.get('timeline')})")
                if plan.get("courses"):
                    st.markdown("**Courses**:")
                    for c in plan["courses"]:
                        st.markdown(f"- {c.get('title')} — {c.get('provider')} ({c.get('note')})")
                if plan.get("certifications"):
                    st.markdown("**Certifications**: " + ", ".join(plan["certifications"]))
                if plan.get("projects"):
                    st.markdown("**Project ideas**:")
                    for p in plan["projects"]:
                        st.markdown(f"- {p}")
                md = make_roadmap_markdown(
                    st.session_state.synopsis_text,
                    adj,
                    stretch,
                    st.session_state.get("transitional_roles") or [],
                    st.session_state.get("career_ladder") or {},
                    plan,
                )
                st.download_button("📄 Download Career Roadmap (Markdown)", data=md.encode("utf-8"), file_name="career_roadmap.md", mime="text/markdown")

        # If a role chip was selected earlier, prioritise it in the UI preview
        if st.session_state.get("selected_term"):
            search_term = st.session_state["selected_term"]
            st.caption(f"Prioritising selected role: {search_term}")

        st.markdown("<div style='position:sticky;top:0;z-index:5;height:0'></div>", unsafe_allow_html=True)
        if st.button("Search & Rank", disabled=not st.session_state.synopsis_text):
            # reset accumulated results and page on a fresh search
            st.session_state["accum_results"] = []
            st.session_state["search_page"] = 1
            with st.spinner("Querying Adzuna and ranking by fit…"):
                subject_raw = st.session_state.extracted.get("subject", "").strip()
                skills = st.session_state.extracted.get("skills", [])
                terms = []
                # If the user picked a role chip, try it first
                sel = st.session_state.get("selected_term")
                if sel:
                    terms.append(sel)
                # Seniority-prefixed variants to try (helps land the right level)
                level = st.session_state.get("seniority", {}).get("level")
                base_for_level = sel or search_term
                if level and base_for_level:
                    lvl_map = {
                        "entry": ["junior"],
                        "junior": ["junior"],
                        "mid": [],
                        "senior": ["senior"],
                        "lead": ["lead"],
                        "manager": ["manager", "lead"],
                        "director": ["head", "director"],
                        "executive": ["vp", "executive"],
                    }
                    for prefix in lvl_map.get(level, []):
                        terms.append(f"{prefix} {base_for_level}")

                if broaden and st.session_state.adjacent_titles:
                    terms.extend(st.session_state.adjacent_titles[:6])
                # also include cross-industry transitional roles (titles only)
                if broaden and st.session_state.get("transitional_roles"):
                    terms.extend([o.get("title") for o in st.session_state.transitional_roles[:6] if o.get("title")])
                terms.extend([t.strip() for t in re.split(r"[,;/|]", subject_raw) if t.strip()])
                terms.extend([s for s in skills if s][:2])
                terms.extend(["data analyst", "business analyst", "insight analyst", "analytics consultant"])
                seen = set()
                dedup_terms = []
                for t in terms:
                    t = t.strip()
                    if t and t.lower() not in seen:
                        seen.add(t.lower())
                        dedup_terms.append(t)
                terms = dedup_terms or ["analyst"]

                # LLM query expansion (smart synonyms/aliases)
                try:
                    base_for_expand = sel or search_term
                    exp = expand_query_terms(
                        base_for_expand,
                        st.session_state.get("seniority", {}).get("level"),
                        skills,
                        st.session_state.extracted.get("degree"),
                        st.session_state.extracted.get("subject"),
                    )
                    if exp:
                        terms.extend(exp)
                        # de-dup again after expansion
                        seen = set(); dedup_terms = []
                        for t in terms:
                            t = (t or "").strip()
                            if t and t.lower() not in seen:
                                seen.add(t.lower()); dedup_terms.append(t)
                        terms = dedup_terms
                except Exception:
                    pass

                results = []
                used_term = None
                for term in terms:
                    results = adzuna_search(
                        what=term,
                        where=location_pref or None,
                        sort_by=sort_by,
                        salary_min=min_salary if min_salary>0 else None,
                        job_type=job_type,
                        permanent=permanent,
                        exclude=exclude_terms or None,
                        distance=radius_miles,
                        results_per_page=25,
                        page=1
                    )
                    if results:
                        used_term = term
                        break

                if used_term is not None:
                    st.session_state["last_used_term"] = used_term
                    # Decide category: prefer explicit selection, otherwise classify
                    explicit_cat = st.session_state.pop("_selected_category", None)
                    if explicit_cat and st.session_state.get("selected_term") and used_term.lower() == st.session_state["selected_term"].lower():
                        st.session_state["last_term_category"] = explicit_cat
                    else:
                        st.session_state["last_term_category"] = classify_term(
                            used_term,
                            st.session_state.get("adjacent_titles"),
                            st.session_state.get("transitional_roles"),
                            st.session_state.get("seniority", {}).get("level")
                        )
                    sel = st.session_state.get("selected_term")
                    if sel and used_term.lower() != sel.lower():
                        st.warning(f"No results for '{sel}'. Fell back to '{used_term}'.")
                    else:
                        st.info(f"Used search term: '{used_term}' – showing top matches.")

                # Optional client-side filter for remote/hybrid
                if remote_only and results:
                    kk = (" remote " , " hybrid ")
                    filtered = []
                    for r in results:
                        txt = (r.get("title","") + "\n" + (r.get("description") or "")).lower()
                        if any(k.strip() in txt for k in kk):
                            filtered.append(r)
                    results = filtered

                # Always clear any selected adjacent role after each search so next run reverts to base term
                st.session_state["selected_term"] = None

                # accumulate results for pagination
                st.session_state["accum_results"].extend(results)

        # --- Results rendering (always shows when we have accumulated results) ---
        if st.session_state.get("accum_results"):
            ranked = rank_jobs_by_fit(st.session_state.synopsis_text, st.session_state["accum_results"])[:50]
            # Compact summary + badge header
            cat = st.session_state.get("last_term_category") or "Results"
            term = st.session_state.get("last_used_term") or ""
            st.markdown(f"<div class='cb-row'><div class='cb-badge {'adj' if cat=='Adjacent Role' else ('path' if cat=='Pathway Beyond Industry' else ('stretch' if cat=='Stretch Role' else 'gen'))}'>{cat}</div><div style='font-size:12px;color:#6b7280;'>term: <code>{term}</code></div></div>", unsafe_allow_html=True)
            st.caption(f"{len(ranked)} jobs ranked by fit")

            # Grid render
            st.markdown("<div class='cb-grid'>", unsafe_allow_html=True)
            cols = st.columns(2)
            for i, j in enumerate(ranked):
                with cols[i % 2]:
                    extra_kw = []
                    extra_kw.extend([(w.get("title") or "") for w in (st.session_state.extracted.get("work_history") or [])])
                    extra_kw.extend(st.session_state.extracted.get("certifications") or [])
                    extra_kw.extend(st.session_state.extracted.get("languages") or [])
                    why = match_reasons(
                        st.session_state.synopsis_text,
                        st.session_state.extracted.get("skills", []),
                        (j.get("title","") + "\n" + (j.get("description") or "")),
                        extra_keywords=extra_kw,
                    )
                    deg = st.session_state.extracted.get("degree")
                    subj = st.session_state.extracted.get("subject")
                    if deg and str(deg).lower() not in STOPWORDS and deg.lower() not in [w.lower() for w in why]:
                        if len(why) < 3 and (deg.lower() in (j.get("description") or "").lower()):
                            why.append(deg)
                    if subj and str(subj).lower() not in STOPWORDS and subj.lower() not in [w.lower() for w in why]:
                        if len(why) < 3 and (subj.lower() in (j.get("description") or "").lower()):
                            why.append(subj)
                    vision = st.session_state.essential_responses.get("career_vision")
                    vision_snip = (vision.strip().split(".")[0][:80]) if vision else ""
                    render_job_card(
                        j,
                        j.get("_fit_score", 0.0),
                        st.session_state.get("last_term_category") or "General",
                        st.session_state.get("last_used_term") or "",
                        why,
                        vision_snip,
                    )
            st.markdown("</div>", unsafe_allow_html=True)

            # Pagination: Load more results (works for reruns too)
            if st.session_state.get("last_used_term"):
                if st.button("Load more"):
                    with st.spinner("Loading more jobs…"):
                        st.session_state["search_page"] += 1
                        more = adzuna_search(
                            what=st.session_state["last_used_term"],
                            where=location_pref or None,
                            sort_by=sort_by,
                            salary_min=min_salary if min_salary>0 else None,
                            job_type=job_type,
                            permanent=permanent,
                            exclude=exclude_terms or None,
                            distance=radius_miles,
                            results_per_page=25,
                            page=st.session_state["search_page"],
                        )
                        # optional remote filter again
                        if remote_only and more:
                            kk = (" remote ", " hybrid ")
                            more = [r for r in more if any(k.strip() in (r.get("title","") + "\n" + (r.get("description") or "")).lower() for k in kk)]
                        st.session_state["accum_results"].extend(more or [])