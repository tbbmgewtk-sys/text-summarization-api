# 🚀 Text Summarization API

## 📌 Giới thiệu

Đây là một **RESTful API tóm tắt văn bản** được xây dựng bằng FastAPI.
API cho phép người dùng nhập vào một đoạn văn bản và trả về bản tóm tắt ngắn gọn.

Do vấn đề tương thích giữa Python 3.14 và thư viện AI, hệ thống sử dụng phương pháp **heuristic-based summarization** (dựa trên chấm điểm câu) để mô phỏng quá trình tóm tắt văn bản.

---

## 🧠 Ý tưởng & Thuật toán

API thực hiện tóm tắt theo các bước:

1. Tách văn bản thành các câu
2. Chấm điểm mỗi câu dựa trên độ dài
3. Chọn các câu có điểm cao nhất (quan trọng nhất)
4. Ghép lại thành bản tóm tắt

👉 Đây là cách tiếp cận đơn giản nhưng hiệu quả để mô phỏng NLP.

---

## ⚙️ Công nghệ sử dụng

* Python 3.14
* FastAPI
* Uvicorn
* Pydantic

---

## 📂 Cấu trúc project

```
bart-text-summarization-main/
│── main.py
│── requirements.txt
│── README.md
│── api_service_colab.ipynb
│── test_api.py
```

---

## 🚀 Cài đặt & chạy project

### 1. Cài thư viện

```bash
python -m pip install fastapi uvicorn
```

---

### 2. Chạy server

```bash
python -m uvicorn main:app --reload
```

---

### 3. Mở API Docs

Truy cập:

```
http://127.0.0.1:8000/docs
```

---

## 📡 API Endpoints

### 🔹 GET `/`

Kiểm tra API hoạt động

**Response:**

```json
{
  "message": "Text Summarization API"
}
```

---

### 🔹 GET `/health`

Kiểm tra trạng thái hệ thống

**Response:**

```json
{
  "status": "OK"
}
```

---

### 🔹 GET `/info`

Thông tin project

**Response:**

```json
{
  "project": "Text Summarization API",
  "version": "1.0",
  "author": "Ngo Thanh Khang"
}
```

---

### 🔹 POST `/predict`

Tóm tắt văn bản

**Request:**

```json
{
  "text": "Artificial intelligence is transforming industries. It enables automation and smarter decisions. Many companies adopt AI to improve productivity.",
  "max_sentences": 2
}
```

**Response:**

```json
{
  "input_length": 150,
  "num_sentences": 3,
  "summary": "Artificial intelligence is transforming industries. Many companies adopt AI to improve productivity",
  "summary_length": 110
}
```

---

## 🧪 Ví dụ sử dụng

Sử dụng Swagger UI tại `/docs` để test trực tiếp API.

---

## ⚠️ Hạn chế

* Không sử dụng mô hình AI thực (do hạn chế môi trường Python 3.14)
* Chỉ sử dụng heuristic (độ dài câu) để đánh giá

---

## 🚀 Hướng phát triển

* Tích hợp mô hình NLP như BART hoặc T5
* Cải thiện thuật toán chấm điểm câu (TF-IDF, TextRank)
* Deploy API lên cloud (Render, Railway)
* Xây dựng giao diện frontend

---

## 👨‍💻 Tác giả

**Ngô Thành Khang**
**24120066**

---

## 📄 License

Dự án phục vụ mục đích học tập.
