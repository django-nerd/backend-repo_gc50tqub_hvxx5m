import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import create_document, get_documents, db
from schemas import Entry, Feedback

app = FastAPI(title="Journaling API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- Utilities ---------

def serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    d = {**doc}
    _id = d.pop("_id", None)
    if _id is not None:
        d["id"] = str(_id)
    # Convert datetimes to isoformat
    for k, v in list(d.items()):
        if isinstance(v, datetime):
            d[k] = v.astimezone(timezone.utc).isoformat()
    return d


def period_bounds(period: str, reference: Optional[datetime] = None):
    if reference is None:
        reference = datetime.now(timezone.utc)
    else:
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        else:
            reference = reference.astimezone(timezone.utc)

    # Normalize to start of day
    day_start = reference.replace(hour=0, minute=0, second=0, microsecond=0)

    if period == "weekly":
        # ISO week: Monday start
        weekday = day_start.weekday()  # Monday=0
        start = day_start - timedelta(days=weekday)
        end = start + timedelta(days=7)
    elif period == "monthly":
        start = day_start.replace(day=1)
        # Next month start
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1)
        else:
            end = start.replace(month=start.month + 1)
    elif period == "yearly":
        start = day_start.replace(month=1, day=1)
        end = start.replace(year=start.year + 1)
    else:
        raise HTTPException(status_code=400, detail="Invalid period. Use weekly, monthly or yearly.")

    return start, end


def previous_period_bounds(period: str, ref: Optional[datetime] = None):
    start, end = period_bounds(period, ref)
    duration = end - start
    prev_end = start
    prev_start = start - duration
    return prev_start, prev_end


def analyze_entries(entries: List[Dict[str, Any]]):
    if not entries:
        return {
            "count": 0,
            "avg_mood": None,
            "top_tags": [],
            "top_words": [],
        }

    # Average mood
    moods = [e.get("mood") for e in entries if isinstance(e.get("mood"), int)]
    avg_mood = round(sum(moods) / len(moods), 2) if moods else None

    # Tags frequency
    from collections import Counter
    tags = []
    for e in entries:
        if isinstance(e.get("tags"), list):
            tags.extend([t.lower() for t in e["tags"] if isinstance(t, str)])
    tag_counts = Counter(tags)
    top_tags = tag_counts.most_common(5)

    # Simple word frequency from content
    stopwords = set(
        "a al algo alguna alguno algunas algunos ante antes como con contra de del desde donde durante e el ella ellas ellos en entre era eramos eran es esa ese eso esta estaba estaban estamos estan estar este esto fue fueron ha han hasta hay la las le les lo los mas me mi mis mucho muy no nos o para pero por porque que se sin sobre su sus te tienen tuvo un una uno y ya yo tus tus".split()
    )
    words = []
    for e in entries:
        content = (e.get("content") or "") + " " + (e.get("title") or "")
        for w in content.lower().replace("\n", " ").split():
            w = ''.join([c for c in w if c.isalpha()])
            if len(w) >= 4 and w not in stopwords:
                words.append(w)
    word_counts = Counter(words)
    top_words = word_counts.most_common(10)

    return {
        "count": len(entries),
        "avg_mood": avg_mood,
        "top_tags": top_tags,
        "top_words": top_words,
    }


def build_ai_summary(period: str, current_stats, prev_stats):
    parts = []
    count = current_stats["count"]
    parts.append(f"Registraste {count} entradas.")

    # Mood
    cm = current_stats.get("avg_mood")
    pm = prev_stats.get("avg_mood") if prev_stats else None
    if cm is not None:
        if pm is not None:
            diff = round(cm - pm, 2)
            if diff > 0:
                parts.append(f"Tu estado de ánimo promedio subió {diff} puntos respecto al periodo anterior (ahora {cm}/5).")
            elif diff < 0:
                parts.append(f"Tu estado de ánimo promedio bajó {abs(diff)} puntos respecto al periodo anterior (ahora {cm}/5).")
            else:
                parts.append(f"Tu estado de ánimo promedio se mantuvo estable en {cm}/5.")
        else:
            parts.append(f"Tu estado de ánimo promedio fue {cm}/5.")

    # Tags
    if current_stats["top_tags"]:
        top = ", ".join([f"{t} ({c})" for t, c in current_stats["top_tags"][:5]])
        parts.append(f"Temas más presentes: {top}.")

    # Words
    if current_stats["top_words"]:
        topw = ", ".join([w for w, _ in current_stats["top_words"][:5]])
        parts.append(f"Palabras clave destacadas: {topw}.")

    # Guidance
    guidance = []
    if cm is not None and cm < 3:
        guidance.append("Prueba a registrar pequeños logros diarios y actividades que te den energía.")
    if current_stats["top_tags"]:
        top_tag = current_stats["top_tags"][0][0]
        guidance.append(f"Explora más sobre '{top_tag}' o define objetivos concretos relacionados.")
    if not guidance:
        guidance.append("Sigue manteniendo el hábito: constancia > intensidad.")

    parts.append("Recomendación: " + " ".join(guidance))

    return " ".join(parts)


# --------- Routes ----------

@app.get("/")
def read_root():
    return {"message": "Journaling API running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    return response


@app.get("/schema")
def get_schema():
    from schemas import Entry, Feedback
    return {
        "entry": Entry.model_json_schema(),
        "feedback": Feedback.model_json_schema(),
    }


@app.post("/entries")
def create_entry(entry: Entry):
    # Default date to now if not provided
    data = entry.model_dump()
    if not data.get("date"):
        data["date"] = datetime.now(timezone.utc)
    new_id = create_document("entry", data)
    return {"id": new_id}


@app.get("/entries")
def list_entries(
    start: Optional[datetime] = Query(None, description="Start datetime (UTC)"),
    end: Optional[datetime] = Query(None, description="End datetime (UTC)"),
    limit: Optional[int] = Query(None, ge=1, le=500)
):
    flt: Dict[str, Any] = {}
    if start or end:
        rng: Dict[str, Any] = {}
        if start:
            rng["$gte"] = start
        if end:
            rng["$lt"] = end
        flt["date"] = rng
    docs = get_documents("entry", flt, limit)
    docs_sorted = sorted(docs, key=lambda d: d.get("date") or d.get("created_at") or datetime.now(), reverse=True)
    return [serialize_doc(d) for d in docs_sorted]


class SummaryResponse(BaseModel):
    period: str
    start: datetime
    end: datetime
    total_entries: int
    avg_mood: Optional[float]
    summary: str
    stats: Dict[str, Any]


@app.get("/summary", response_model=SummaryResponse)
def get_summary(
    period: str = Query(..., pattern="^(weekly|monthly|yearly)$"),
    reference: Optional[datetime] = Query(None, description="Reference datetime (UTC) for the period")
):
    start, end = period_bounds(period, reference)

    # Current period entries
    curr_docs = get_documents("entry", {"date": {"$gte": start, "$lt": end}})

    # Previous period for comparison
    pstart, pend = previous_period_bounds(period, reference)
    prev_docs = get_documents("entry", {"date": {"$gte": pstart, "$lt": pend}})

    curr_stats = analyze_entries(curr_docs)
    prev_stats = analyze_entries(prev_docs)

    summary_text = build_ai_summary(period, curr_stats, prev_stats)

    return SummaryResponse(
        period=period,
        start=start,
        end=end,
        total_entries=curr_stats["count"],
        avg_mood=curr_stats.get("avg_mood"),
        summary=summary_text,
        stats={
            "current": curr_stats,
            "previous": prev_stats,
        },
    )


@app.post("/feedback")
def send_feedback(feedback: Feedback):
    new_id = create_document("feedback", feedback)
    return {"id": new_id}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
