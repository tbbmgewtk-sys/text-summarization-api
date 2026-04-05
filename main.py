from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextInput(BaseModel):
    text: str
    max_sentences: int = 2

top_sentences = sorted(scored, reverse=True)[:input.max_sentences]

if input.max_sentences <= 0:
    raise HTTPException(status_code=400, detail="max_sentences must be > 0")

@app.get("/info")
def info():
    return {
        "project": "Text Summarization API",
        "version": "1.0",
        "author": "Ngo Thanh Khang"
    }

@app.post("/predict")
def summarize(input: TextInput):
    if not input.text:
        raise HTTPException(status_code=400, detail="Text is required")

    if len(input.text) < 20:
        raise HTTPException(status_code=400, detail="Text too short")

    # Tách câu
    sentences = [s.strip() for s in input.text.split(".") if s.strip()]

    # Tính điểm câu (dựa vào độ dài)
    scored = [(len(s), s) for s in sentences]

    # Lấy 2 câu dài nhất (coi như quan trọng nhất)
    top_sentences = sorted(scored, reverse=True)[:2]

    # Sắp xếp lại đúng thứ tự ban đầu
    top_sentences = [s for _, s in sorted(top_sentences, key=lambda x: sentences.index(x[1]))]

    summary = ". ".join(top_sentences)

    return {
    "input_length": len(input.text),
    "num_sentences": len(sentences),
    "summary": summary,
    "summary_length": len(summary)
}