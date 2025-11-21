import os, re
from fastapi import APIRouter
from pydantic import BaseModel
from .config import settings

router = APIRouter(prefix="/interpret", tags=["interpretation"])

# Input & output models
class StoryRequest(BaseModel):
    story: str

class IntentResponse(BaseModel):
    persona: str | None = None
    topic: list[str] | None = None
    geography: list[str] | None = None
    time: dict | None = None
    insight_goal: str | None = None
    used_llm: bool = False


@router.post("/", response_model=IntentResponse)
async def interpret_story(req: StoryRequest):
    story = req.story.strip()

    # --- If API key available → use LLM ---
    if settings.OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL
            )

            prompt = f"""
            You are an AI that extracts structured intent from a user story.
            Example output JSON:
            {{
              "persona": "Insurance Analyst",
              "topic": ["opioid", "claims"],
              "geography": ["Virginia"],
              "time": {{"from": 2018, "to": 2024}},
              "insight_goal": "forecast premium impacts"
            }}
            Now extract structured JSON for this story:
            {story}
            """

            resp = client.responses.create(
                model=settings.OPENAI_MODEL,
                input=prompt,
                response_format={"type": "json_object"}
            )
            out = resp.output[0].content[0].text
            import json
            parsed = json.loads(out)
            parsed["used_llm"] = True
            return parsed
        except Exception as e:
            print("LLM fallback due to error:", e)

    # --- Simple fallback parser (no LLM) ---
    persona_match = re.search(r"As an ([^,\.]+)", story, re.I)
    persona = persona_match.group(1).strip() if persona_match else None

    topics = re.findall(r"(opioid|depression|anxiety|mental health|suicide|treatment|insurance|cost|claims)",
                        story, re.I)
    geos = re.findall(r"\b(Alabama|Alaska|Arizona|California|Virginia|New York|Texas|US|United States)\b",
                      story, re.I)
    years = [int(y) for y in re.findall(r"(20\d{2})", story)]
    time_range = {"from": min(years)} if years else None

    return {
        "persona": persona,
        "topic": [t.lower() for t in topics] or None,
        "geography": geos or None,
        "time": time_range,
        "insight_goal": None,
        "used_llm": False
    }
