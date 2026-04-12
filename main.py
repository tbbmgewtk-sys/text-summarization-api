from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class TextInput(BaseModel):
    text: str
    max_sentences: int = 2

@app.get("/info")
def info():
    return {
        "project": "Text Summarization API",
        "version": "1.0",
        "author": "Ngo Thanh Khang"
    }

@app.post("/predict")
def summarize(input: TextInput):
    # 1. Kiểm tra đầu vào cơ bản
    if not input.text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    if input.max_sentences <= 0:
        raise HTTPException(status_code=400, detail="max_sentences must be > 0")

    if len(input.text) < 20:
        raise HTTPException(status_code=400, detail="Text too short")

    # 2. Tách câu
    sentences = [s.strip() for s in input.text.split(".") if s.strip()]
    
    if not sentences:
        raise HTTPException(status_code=400, detail="No valid sentences found")

    # 3. Tính điểm câu (dựa vào độ dài)
    scored = [(len(s), s) for s in sentences]

    # 4. Lấy các câu quan trọng nhất dựa trên max_sentences
    # Sử dụng input.max_sentences thay vì số 2 cố định
    top_scored = sorted(scored, reverse=True)[:input.max_sentences]

    # 5. Sắp xếp lại đúng thứ tự xuất hiện ban đầu trong văn bản
    # Lưu ý: Dùng list.index() có thể chậm với văn bản cực lớn, nhưng ổn với app nhỏ
    top_sentences = [s for _, s in sorted(top_scored, key=lambda x: sentences.index(x[1]))]

    summary = ". ".join(top_sentences) + "."

    return {
        "input_length": len(input.text),
        "num_sentences": len(sentences),
        "summary": summary,
        "summary_length": len(summary)
    }
