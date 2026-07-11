import base64

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI
from fastapi import HTTPException
client = OpenAI()

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
def answer(req: Request):
    try:
        image_url = f"data:image/png;base64,{req.image_base64}"

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Answer ONLY the question from the image.\n"
                                "Return only the answer.\n"
                                "If numeric return only the number.\n"
                                f"Question: {req.question}"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            },
                        },
                    ],
                }
            ],
        )

        return {
            "answer": response.choices[0].message.content.strip()
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
