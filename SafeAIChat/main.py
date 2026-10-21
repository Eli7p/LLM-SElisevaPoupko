
# ספריות Python: ניהול משתני סביבה, JSON, עבודה אסינכרונית ותקשורת HTTP.
import os
import json
import asyncio
import ssl
from pathlib import Path
from typing import Dict, List

import httpx
import truststore
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from openai import OpenAI


# טעינת משתני הסביבה מקובץ .env בזמן הפיתוח המקומי.
from dotenv import load_dotenv
import os

load_dotenv()

# =========== הגדרות חיבור לשירות SafeAI ===========
# קורא את מפתח ה-API ואת כתובת השירות; השרת לא עולה ללא מפתח.

API_KEY = os.getenv("SAFEAI_API_KEY")
BASE_URL = "https://safeai613.com/v1"

if not API_KEY:
    raise RuntimeError(
        "Missing SAFEAI_API_KEY environment variable"
    )

# =========== יצירת אפליקציית ה-API והגדרות CORS ===========
# CORS מאפשר לדף המקומי שבדפדפן לשלוח בקשות לשרת.

app = FastAPI(
    title="SafeAI Chat API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?",
    allow_origins=[
        "null",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =========== מבני הנתונים של בקשות ותשובות ===========
# Pydantic בודק ומגדיר את השדות שמתקבלים ונשלחים דרך ה-API.

class RequestModel(BaseModel):
    model: str = "gpt-4o-mini"
    input: str
    session_id: str = "default"


class ResponseModel(BaseModel):
    output_text: str
    model: str


# =========== יצירת לקוח המודל ===========
# HTTPX ו-truststore מטפלים בתקשורת מאובטחת אל שירות SafeAI.

httpx_client = httpx.Client(
    timeout=60.0,
    verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
)

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    http_client=httpx_client
)

# =========== זיכרון שיחות בזיכרון השרת ===========
# כל מזהה שיחה מקבל היסטוריה נפרדת; הנתונים נעלמים כשהשרת נסגר.

# Each session stores its own conversation history.
conversation_memory: Dict[str, List[dict]] = {}

# =========== נקודות הקצה של השרת ===========

@app.get("/", include_in_schema=False)
async def serve_chat_page():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/health")
async def health_check():
    # בדיקת זמינות פשוטה למעקב ולבדיקות חיבור.
    return {"status": "healthy"}


@app.post("/generate", response_model=ResponseModel)
async def generate_content(request: RequestModel):

    try:
        history = conversation_memory.setdefault(
            request.session_id, []
        )

        messages = history + [
            {
                "role": "user",
                "content": request.input
            }
        ]

        response = await asyncio.to_thread(
            client.responses.create,
            model=request.model,
            input=messages
        )

        answer = response.output_text

        # Save both sides of the conversation.
        history.extend([
            {
                "role": "user",
                "content": request.input
            },
            {
                "role": "assistant",
                "content": answer
            }
        ])

        return ResponseModel(
            output_text=answer,
            model=request.model
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/chat/stream")
async def chat_stream(request: RequestModel):
    """
    שולח את התשובה בהדרגה ללקוח באמצעות Server-Sent Events (SSE).
    """

    # יוצר אירועי זרם, שומר את השיחה, ומשדר גם שגיאות אם התרחשו.
    async def event_generator():
        history = conversation_memory.setdefault(
            request.session_id, []
        )

        messages = history + [
            {
                "role": "user",
                "content": request.input
            }
        ]

        try:
            response = await asyncio.to_thread(
                lambda: client.responses.create(
                    model=request.model,
                    input=messages,
                    stream=True
                )
            )

            answer_parts = []

            # Read model output as it arrives.
            for event in response:
                if event.type == "response.output_text.delta":
                    delta = event.delta
                    answer_parts.append(delta)

                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "type": "delta",
                                "text": delta
                            },
                            ensure_ascii=False
                        )
                        + "\n\n"
                    )

                elif event.type == "response.completed":
                    usage = event.response.usage
                    if usage is not None:
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "type": "usage",
                                    "input_tokens": usage.input_tokens,
                                    "output_tokens": usage.output_tokens
                                },
                                ensure_ascii=False
                            )
                            + "\n\n"
                        )

            answer = "".join(answer_parts)

            history.extend([
                {
                    "role": "user",
                    "content": request.input
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ])

            yield (
                "data: "
                + json.dumps(
                    {"type": "done"},
                    ensure_ascii=False
                )
                + "\n\n"
            )

        except Exception as e:
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "error",
                        "message": str(e)
                    },
                    ensure_ascii=False
                )
                + "\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.delete("/chat/{session_id}")
async def clear_conversation(session_id: str):
    """
    מוחק את היסטוריית השיחה של מזהה נתון.
    """
    conversation_memory.pop(session_id, None)

    return {
        "status": "cleared",
        "session_id": session_id
    }


# =========== ניקוי משאבים בסגירת השרת ===========
# סוגר את חיבור HTTPX כדי לשחרר משאבי רשת.

@app.on_event("shutdown")
async def shutdown_event():
    httpx_client.close()


if __name__ == "__main__":
    # מאפשר להריץ את השרת ישירות דרך python main.py.
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )

