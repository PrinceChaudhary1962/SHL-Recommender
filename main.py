
"""
SHL Assessment Recommender - FastAPI Service (Lightweight Version)
No sentence-transformers required - uses keyword + constraint scoring only.
Faster deployment, lower memory usage.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================================
# DATA MODELS
# ===========================================================================

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str


# ===========================================================================
# LOAD CATALOG
# ===========================================================================

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "shl_catalog.json")

with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    CATALOG: List[Dict[str, Any]] = json.load(f)

CATALOG_BY_NAME = {item["name"].lower(): item for item in CATALOG}
CATALOG_BY_URL = {item["url"]: item for item in CATALOG}

ALL_SKILLS: Set[str] = set()
ALL_CATEGORIES: Set[str] = set()
for item in CATALOG:
    ALL_SKILLS.update(s.lower() for s in item.get("skills", []))
    ALL_CATEGORIES.update(c.lower() for c in item.get("categories", []))


# ===========================================================================
# CONVERSATION STATE
# ===========================================================================

@dataclass
class ConversationState:
    turn_count: int = 0
    has_role_info: bool = False
    has_seniority: bool = False
    has_skills_info: bool = False
    has_test_type_preference: bool = False
    is_vague: bool = True
    is_comparison_mode: bool = False
    is_off_topic: bool = False
    user_refused_to_answer: bool = False
    comparison_items: List[str] = field(default_factory=list)

    def update(self, messages: List[Message]):
        self.turn_count = len(messages)
        all_text = " ".join(m.content.lower() for m in messages)

        role_kw = [
            "developer", "engineer", "manager", "analyst", "designer",
            "consultant", "sales", "marketing", "hr", "finance", "accountant",
            "nurse", "teacher", "driver", "clerk", "admin", "executive",
            "director", "lead", "architect", "scientist", "researcher",
            "programmer", "coder", "devops", "data scientist", "product manager",
            "project manager", "business analyst", "quality assurance", "qa",
            "support", "operations", "recruiter", "writer", "editor",
            "ux", "ui", "frontend", "backend", "full stack"
        ]
        self.has_role_info = any(k in all_text for k in role_kw)

        seniority_kw = [
            "junior", "senior", "mid", "entry", "graduate", "experienced",
            "lead", "principal", "chief", "head of", "manager", "director",
            "4 years", "5 years", "6 years", "7 years", "8 years", "9 years",
            "10 years", "fresh", "intern", "fresher", "0-2", "3-5", "6-10",
            "entry level", "mid level", "senior level", "executive level"
        ]
        self.has_seniority = any(k in all_text for k in seniority_kw)
        self.has_skills_info = any(s in all_text for s in ALL_SKILLS)

        test_type_kw = [
            "cognitive", "personality", "behavioral", "skill", "simulation",
            "coding", "technical", "aptitude", "video interview", "360",
            "feedback", "sjt", "situational judgment"
        ]
        self.has_test_type_preference = any(k in all_text for k in test_type_kw)
        self.is_vague = not (self.has_role_info or self.has_skills_info)

        comp_kw = [
            "difference", "compare", "versus", "vs", "better", "which is",
            "what is the difference", "how does", "contrast", "difference between"
        ]
        self.is_comparison_mode = any(k in all_text for k in comp_kw)

        if self.is_comparison_mode:
            self.comparison_items = []
            alias_map = {}
            for it in CATALOG:
                name = it["name"]
                name_lower = name.lower()
                alias_map[name_lower] = name
                words = [w for w in name_lower.split() if len(w) > 2 and w not in {'the', 'and', 'for', 'new', 'with', 'our', 'from', 'view'}]
                for i in range(min(3, len(words))):
                    fragment = ' '.join(words[:i+1])
                    if fragment not in alias_map:
                        alias_map[fragment] = name

            alias_map['opq'] = 'OPQ32 - Occupational Personality Questionnaire'
            alias_map['opq32'] = 'OPQ32 - Occupational Personality Questionnaire'
            alias_map['opq32r'] = 'OPQ32r - Occupational Personality Questionnaire Revised'
            alias_map['verify g+'] = 'Verify G+ - General Ability'
            alias_map['verify g'] = 'Verify G+ - General Ability'
            alias_map['mq'] = 'MQ - Motivation Questionnaire'
            alias_map['sjt'] = 'Situational Judgement Test (SJT)'
            alias_map['java'] = 'Java (New)'
            alias_map['python'] = 'Python (New)'
            alias_map['sql'] = 'SQL (New)'
            alias_map['c#'] = 'C# (New)'
            alias_map['csharp'] = 'C# (New)'
            alias_map['javascript'] = 'JavaScript (New)'
            alias_map['360'] = '360 Feedback'

            last_user_msg = messages[-1].content.lower() if messages[-1].role == "user" else ""
            sorted_aliases = sorted(alias_map.keys(), key=len, reverse=True)

            for alias in sorted_aliases:
                if len(alias) <= 4:
                    pattern = r'\b' + re.escape(alias) + r'\b'
                else:
                    pattern = re.escape(alias)
                if re.search(pattern, last_user_msg):
                    full_name = alias_map[alias]
                    if full_name not in self.comparison_items:
                        self.comparison_items.append(full_name)
            self.comparison_items = self.comparison_items[:2]

        off_topic_kw = [
            "salary", "negotiate", "interview tips", "resume", "cv",
            "legal advice", "law", "visa", "immigration", "contract",
            "how to hire", "hiring process", "onboarding", "termination",
            "ignore previous", "disregard", "prompt injection", "jailbreak",
            "system prompt", "you are now", "act as", "pretend to be",
            "ignore all", "forget", "new instructions", "override",
            "do not follow", "bypass", "hack", "exploit"
        ]
        self.is_off_topic = any(k in all_text for k in off_topic_kw)

        if messages and messages[-1].role == "user":
            last = messages[-1].content.lower()
            refusal_kw = [
                "i don't know", "not sure", "no preference", "doesn't matter",
                "anything is fine", "up to you", "you decide", "whatever",
                "don't care", "any", "does not matter", "no idea"
            ]
            self.user_refused_to_answer = any(k in last for k in refusal_kw)


# ===========================================================================
# CONSTRAINT EXTRACTION
# ===========================================================================

def extract_constraints(query: str) -> Dict[str, Any]:
    q = query.lower()
    constraints: Dict[str, Any] = {
        "role": None, "seniority": None, "skills": [],
        "test_types": [], "job_levels": [], "categories": [], "exclude": [],
    }

    role_patterns = [
        r"(?:hiring|looking for|need|want|recruiting)\s+(?:a|an)\s+([a-z\s]+?)(?:\s+who|\s+that|\s+with|\s+for|\s+to|\s+and|\.|,|$)",
        r"([a-z\s]+?)\s+(?:developer|engineer|manager|analyst|designer|consultant|specialist)",
        r"(?:java|python|javascript|sql|c#|frontend|backend|full[\s\-]?stack|data)\s+(?:developer|engineer)",
        r"(?:devops|cloud|security|network|systems)\s+(?:engineer|architect|specialist)",
    ]
    for pat in role_patterns:
        m = re.search(pat, q)
        if m:
            constraints["role"] = m.group(1).strip()
            break

    seniority_map = {
        "entry": ["entry", "junior", "fresh", "intern", "0-2 years", "graduate", "fresher", "entry level"],
        "mid": ["mid", "intermediate", "3-5 years", "4 years", "5 years", "experienced", "mid level"],
        "senior": ["senior", "lead", "principal", "6-10 years", "experienced", "senior level"],
        "executive": ["executive", "director", "chief", "vp", "head of", "c-level", "executive level"],
    }
    for level, kws in seniority_map.items():
        if any(k in q for k in kws):
            constraints["seniority"] = level
            constraints["job_levels"].append(level)
            break

    for skill in ALL_SKILLS:
        if skill in q:
            constraints["skills"].append(skill)

    test_type_map = {
        "C": ["cognitive", "aptitude", "reasoning", "mental ability", "general ability", "iq"],
        "P": ["personality", "behavioral traits", "work style", "motivation", "opq", "values", "culture"],
        "B": ["behavioral", "situational judgment", "sjt", "workplace behavior"],
        "K": ["skills", "technical", "knowledge", "coding", "programming", "language", "microsoft"],
        "S": ["simulation", "hands-on", "practical", "coding simulation", "real-world"],
        "V": ["video interview", "ai interview", "smart interview", "on-demand interview"],
        "D": ["360", "feedback", "multi-rater", "development"],
    }
    for tt, kws in test_type_map.items():
        if any(k in q for k in kws):
            if tt not in constraints["test_types"]:
                constraints["test_types"].append(tt)

    for cat in ALL_CATEGORIES:
        if cat in q:
            constraints["categories"].append(cat)

    exclusion_pats = [
        r"not\s+([a-z\s,]+?)(?:\.|,|$|\s+and|\s+but)",
        r"exclude\s+([a-z\s,]+?)(?:\.|,|$|\s+and|\s+but)",
        r"without\s+([a-z\s,]+?)(?:\.|,|$|\s+and|\s+but)",
        r"no\s+([a-z\s,]+?)(?:\.|,|$|\s+and|\s+but)",
    ]
    for pat in exclusion_pats:
        for match in re.findall(pat, q):
            constraints["exclude"].extend(s.strip() for s in match.split(",") if s.strip())

    return constraints


# ===========================================================================
# SCORING & RETRIEVAL (Keyword-based, no embeddings)
# ===========================================================================

def score_assessment(item: Dict[str, Any], constraints: Dict[str, Any]) -> float:
    score = 0.0

    item_skills = set(s.lower() for s in item.get("skills", []))
    query_skills = set(s.lower() for s in constraints.get("skills", []))
    if query_skills:
        overlap = len(item_skills & query_skills)
        score += (overlap / max(len(query_skills), 1)) * 4.0

    item_cats = set(c.lower() for c in item.get("categories", []))
    query_cats = set(c.lower() for c in constraints.get("categories", []))
    if query_cats:
        overlap = len(item_cats & query_cats)
        score += (overlap / max(len(query_cats), 1)) * 2.0

    if constraints.get("test_types"):
        if item.get("test_type") in constraints["test_types"]:
            score += 2.5

    item_levels = set(l.lower() for l in item.get("job_levels", []))
    query_levels = set(l.lower() for l in constraints.get("job_levels", []))
    if query_levels:
        if item_levels & query_levels:
            score += 1.5

    if constraints.get("role"):
        role = constraints["role"].lower()
        combined = f"{item['name']} {item['description']}".lower()
        if role in combined:
            score += 2.0

    for exc in constraints.get("exclude", []):
        exc_l = exc.lower()
        if exc_l in item["name"].lower() or exc_l in item["description"].lower():
            score -= 3.0
        if any(exc_l in s for s in item.get("skills", [])):
            score -= 2.0
        if any(exc_l in c for c in item.get("categories", [])):
            score -= 1.5

    if not constraints.get("test_types") and item.get("test_type") == "C":
        score += 0.2

    return score


def keyword_search(query: str, top_k: int = 10) -> List[tuple]:
    constraints = extract_constraints(query)
    scored = []
    for it in CATALOG:
        cons_score = score_assessment(it, constraints)
        q_words = set(query.lower().split())
        it_text = f"{it['name']} {it['description']} {' '.join(it.get('skills', []))}"
        it_words = set(it_text.lower().split())
        overlap = len(q_words & it_words)
        scored.append((cons_score + overlap * 0.3, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(it, sc) for sc, it in scored[:top_k]]


def get_recommendations(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    results = keyword_search(query, top_k=max_results)
    out = []
    for it, sc in results:
        if sc > 0:
            out.append({
                "name": it["name"],
                "url": it["url"],
                "test_type": it["test_type"],
                "score": sc,
                "description": it["description"],
            })
    return out


# ===========================================================================
# RESPONSE GENERATION
# ===========================================================================

CLARIFY_QUESTIONS = {
    "role": [
        "Could you tell me more about the role you're hiring for? For example, what job title or key skills are you looking for?",
        "I'd be happy to help! What specific position are you trying to fill, and what are the main responsibilities?",
        "To recommend the right assessments, could you share the job title or the key skills required for this role?",
    ],
    "seniority": [
        "What level of experience are you looking for? (e.g., entry-level, mid-level, senior)",
        "Could you tell me about the seniority level? Is this for a junior, mid-level, or senior position?",
        "How many years of experience should the ideal candidate have?",
    ],
    "test_type": [
        "Are you looking for any specific type of assessment? For example: cognitive ability tests, personality questionnaires, technical skills tests, or coding simulations?",
        "Do you have a preference for the assessment type — such as aptitude tests, personality assessments, or hands-on technical evaluations?",
        "Would you like me to include personality/behavioral assessments along with cognitive and technical ones?",
    ],
    "default": [
        "Is there anything else I should know about the role? For example, specific technologies, soft skills, or industry requirements?",
        "Any other requirements I should consider — such as specific programming languages, tools, or behavioral competencies?",
        "Would you like assessments for any particular skills or competencies that we haven't covered yet?",
    ],
}


def pick_clarification(state: ConversationState, messages: List[Message]) -> str:
    if not state.has_role_info and not state.has_skills_info:
        pool = CLARIFY_QUESTIONS["role"]
    elif not state.has_seniority:
        pool = CLARIFY_QUESTIONS["seniority"]
    elif not state.has_test_type_preference:
        pool = CLARIFY_QUESTIONS["test_type"]
    else:
        pool = CLARIFY_QUESTIONS["default"]
    return pool[state.turn_count % len(pool)]


def build_recommendation_reply(recs: List[Dict[str, Any]], state: ConversationState) -> str:
    if not recs:
        return "I couldn't find specific assessments matching your criteria. Could you provide more details or adjust your requirements?"
    count = len(recs)
    types = set(r["test_type"] for r in recs)
    type_names = {
        "C": "cognitive ability", "P": "personality", "B": "behavioral",
        "K": "skills/knowledge", "S": "simulation", "V": "video interview", "D": "360° feedback",
    }
    type_desc = ", ".join(type_names.get(t, t) for t in types)
    if state.turn_count <= 2:
        intro = f"Based on what you've shared, here are {count} assessments that could be a good fit:"
    else:
        intro = f"Here are {count} assessments that match your updated requirements:"
    return f"{intro} I've included a mix of {type_desc} assessments."


def build_comparison_reply(items: List[str]) -> str:
    found = []
    for name in items:
        for it in CATALOG:
            if name.lower() in it["name"].lower() or it["name"].lower() in name.lower():
                found.append(it)
                break
    if len(found) < 2:
        return "I need a bit more clarity. Could you tell me the exact names of the two assessments you'd like me to compare?"
    a, b = found[0], found[1]
    reply = f"Here's a comparison between **{a['name']}** and **{b['name']}**:\n\n"
    reply += f"**{a['name']}**: {a['description']}\n"
    reply += f"- Type: {a['test_type']} | Duration: {a['duration']}\n"
    reply += f"- Skills: {', '.join(a.get('skills', [])[:5])}\n\n"
    reply += f"**{b['name']}**: {b['description']}\n"
    reply += f"- Type: {b['test_type']} | Duration: {b['duration']}\n"
    reply += f"- Skills: {', '.join(b.get('skills', [])[:5])}\n\n"
    reply += "**Key Differences:**\n"
    if a["test_type"] != b["test_type"]:
        reply += f"- {a['name']} is a {a['test_type']}-type assessment, while {b['name']} is {b['test_type']}-type.\n"
    sa, sb = set(a.get("skills", [])), set(b.get("skills", []))
    ua, ub = sa - sb, sb - sa
    if ua:
        reply += f"- {a['name']} uniquely covers: {', '.join(list(ua)[:3])}\n"
    if ub:
        reply += f"- {b['name']} uniquely covers: {', '.join(list(ub)[:3])}\n"
    reply += "\nWould you like me to recommend which one fits your specific role better?"
    return reply


def build_refinement_reply(recs: List[Dict[str, Any]], added_types: List[str]) -> str:
    type_str = ", ".join(added_types)
    return f"I've updated your recommendations to include {type_str} assessments. Here is the revised shortlist:"


def build_refusal_reply() -> str:
    return (
        "I'm here to help you find the right SHL assessments for your hiring needs. "
        "I can only discuss SHL's assessment catalog and make recommendations based on job requirements. "
        "How can I help you with assessment selection today?"
    )


# ===========================================================================
# ENDPOINTS
# ===========================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    messages = request.messages
    if not messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")
    if len(messages) >= 8:
        return ChatResponse(
            reply="I've reached the maximum number of turns for this conversation. Please start a new conversation if you need further assistance.",
            recommendations=[],
            end_of_conversation=True,
        )

    state = ConversationState()
    state.update(messages)

    last_msg = messages[-1]
    if last_msg.role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    user_query = last_msg.content

    if state.is_off_topic:
        return ChatResponse(reply=build_refusal_reply(), recommendations=[], end_of_conversation=False)

    if state.is_comparison_mode and len(state.comparison_items) >= 2:
        return ChatResponse(reply=build_comparison_reply(state.comparison_items), recommendations=[], end_of_conversation=False)

    enough = (state.has_role_info or state.has_skills_info) and (state.has_seniority or state.turn_count >= 3)
    if state.turn_count == 1 and state.is_vague:
        enough = False
    if state.user_refused_to_answer and (state.has_role_info or state.has_skills_info):
        enough = True

    refinement_kw = [
        "add", "also", "include", "more", "additionally", "extra",
        "refine", "update", "change", "instead", "rather", "actually",
        "modify", "replace", "switch", "different", "other", "new",
    ]
    is_refinement = any(k in user_query.lower() for k in refinement_kw)

    if not enough:
        return ChatResponse(reply=pick_clarification(state, messages), recommendations=[], end_of_conversation=False)

    full_query = " ".join(m.content for m in messages if m.role == "user")
    recs = get_recommendations(full_query, max_results=10)

    if is_refinement:
        prev_names: Set[str] = set()
        for i, m in enumerate(messages):
            if m.role == "assistant" and i > 0:
                for it in CATALOG:
                    if it["name"].lower() in m.content.lower():
                        prev_names.add(it["name"])
        new_recs = [r for r in recs if r["name"] not in prev_names]
        if len(new_recs) >= 3:
            recs = new_recs

    recs = recs[:10]

    if is_refinement:
        added = list(set(r["test_type"] for r in recs))
        reply = build_refinement_reply(recs, added)
    else:
        reply = build_recommendation_reply(recs, state)

    formatted = [Recommendation(name=r["name"], url=r["url"], test_type=r["test_type"]) for r in recs]

    end_conv = False
    if len(recs) >= 3 and state.turn_count >= 4:
        sat_kw = ["thanks", "thank you", "perfect", "great", "good", "fine", "ok", "okay", "that works", "sounds good", "appreciate", "done"]
        if any(k in user_query.lower() for k in sat_kw):
            end_conv = True

    return ChatResponse(reply=reply, recommendations=formatted, end_of_conversation=end_conv)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
