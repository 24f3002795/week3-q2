import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("AIPIPE_TOKEN")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Request(BaseModel):
    image_base64: str
    question: str


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/answer-image")
async def answer_image(req: Request):
    if not TOKEN:
        raise HTTPException(500, "AIPIPE_TOKEN not configured")

    image_url = f"data:image/png;base64,{req.image_base64}"

    payload = {
        "model": "openai/gpt-4.1-nano",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer ONLY the question from the image. "
                    "Return only the answer. "
                    "If numeric return only the number as a string."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": req.question,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        },
                    },
                ],
            },
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                "https://aipipe.org/openrouter/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {TOKEN}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        print(r.status_code)
        print(r.text)

        r.raise_for_status()

        data = r.json()

        answer = data["choices"][0]["message"]["content"].strip()

        return {"answer": answer}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
