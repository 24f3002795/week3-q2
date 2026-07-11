import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# -----------------------------
# AI Pipe Configuration
# -----------------------------
client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="YOUR_AIPIPE_BASE_URL"   # <-- Replace with the AI Pipe base URL from the course
)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Multimodal Image QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request Model
# -----------------------------
class ImageQuestion(BaseModel):
    image_base64: str
    question: str


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/answer-image")
def answer_image(req: ImageQuestion):
    try:
        image_url = f"data:image/png;base64,{req.image_base64}"

        response = client.chat.completions.create(
            model="gpt-4.1-mini",   # Use the model specified by AI Pipe if different
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions about images. "
                        "Return ONLY the answer. "
                        "If the answer is numeric, return only the number as a string "
                        "without units, currency symbols, or extra words."
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
            temperature=0,
        )

        answer = response.choices[0].message.content.strip()

        return {"answer": answer}

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))
