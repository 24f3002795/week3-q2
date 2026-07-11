import os
import re
import httpx

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")

if not AIPIPE_TOKEN:
    raise RuntimeError("AIPIPE_TOKEN not found")

app = FastAPI(title="Multimodal Image QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    image_base64: str
    question: str


@app.get("/")
def home():
    return {
        "status": "running"
    }


def clean_answer(answer: str) -> str:
    answer = answer.strip()

    # remove markdown formatting
    answer = answer.replace("```", "")
    answer = answer.replace("`", "")

    # remove common currency symbols
    answer = answer.replace("$", "")
    answer = answer.replace("₹", "")
    answer = answer.replace("€", "")
    answer = answer.replace("£", "")

    # remove commas from numbers
    answer = answer.replace(",", "")

    return answer.strip()


@app.post("/answer-image")
async def answer_image(req: ImageRequest):

    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert at reading charts, receipts, invoices, "
                    "tables, scanned documents and graphs.\n\n"
                    "Answer ONLY the user's question.\n"
                    "If the answer is numeric return ONLY the number.\n"
                    "Do NOT include units.\n"
                    "Do NOT include currency symbols.\n"
                    "Do NOT explain."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": req.question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{req.image_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0
    }

    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json"
    }

    try:

        async with httpx.AsyncClient(timeout=90) as client:

            response = await client.post(
                "https://aipipe.org/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )

        if response.status_code != 200:
            print(response.text)
            raise HTTPException(
                status_code=500,
                detail=response.text
            )

        result = response.json()

        answer = result["choices"][0]["message"]["content"]

        return {
            "answer": clean_answer(answer)
        }

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
