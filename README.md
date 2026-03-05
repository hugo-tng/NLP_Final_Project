# NLP Final Project - Fine-tuning Pre-trained Models for Vietnamese NLP Tasks

> _Dự án Cuối kì môn Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing)_

---

## 👥 Thành Viên Nhóm

- Tăng Duy Hào
- Nguyễn Thành Phát
- Hồ Bảo Ngân

---

## 📋 Tổng Quan Dự Án

Dự án này triển khai ba bài toán NLP khác nhau trên ngôn ngữ Tiếng Việt, sử dụng các kiến trúc Transformer được pre-train trước đó. Mục tiêu là fine-tune các mô hình để giải quyết ba vấn đề cụ thể trong xử lý ngôn ngữ tự nhiên: phân loại độc hại, trả lời câu hỏi sản phẩm, và tóm tắt văn bản.

---

## 🎯 Mục tiêu Dự án

- Hiểu sâu kiến trúc Transformer và ba loại mô hình chính: encoder-only, encoder-decoder, decoder-only.
- Thực hành fine-tuning mô hình ngôn ngữ pretrained trên dữ liệu tự thu thập và tự xử lý .
- Áp dụng data augmentation bằng synthetic data sinh ra từ LLMs.
- Đánh giá mô hình bằng cả các metric truyền thống lẫn đánh giá bằng LLMs khác (LLM-based evaluation).
- So sánh, phân tích và rút ra nhận xét về hiệu năng, tính tổng quát và khả năng học của từng loại kiến trúc.

---

## 📁 Cấu Trúc Dự Án

```
NLP_Final_Project/
│
├── Toxic_text_classification_vi/       # Bài toán: Phân loại văn bản độc hại
│   └── (Sử dụng kiến trúc Encoder-only như PhoBERT/RoBERTa)
│
├── Vietnamese_text_sumarization/       # Bài toán: Sinh tóm tắt văn bản
│   └── (Sử dụng kiến trúc Encoder-decoder như ViT5/T5)
│
├── Chatbot_ProductQA/                  # Bài toán: Chatbot hỏi đáp sản phẩm
│   └── (Sử dụng kiến trúc Decoder-only như Qwen3-0.6B)
│
└── README.md
```

---

## 🎯 Ba Bài Toán Chính

### 1. 🚨 Toxic Text Classification (Encoder-Only)

**Folder:** `Toxic_text_classification_vi/`

**Mục tiêu:** Fine-tune mô hình Encoder-only để phân loại các bình luận độc hại trên Tiếng Việt.

**Mô hình:** `PhoBERT-large` (Encoder-only)

**Kiến trúc:**

- Mô hình Encoder-only sử dụng Masked Language Modeling
- Xử lý toàn bộ chuỗi văn bản qua Self-Attention
- Đầu ra là embedding của [CLS] token để phân loại

[📖 Chi tiết](Toxic_text_classification_vi/README.md)

---

### 2. 📝 Vietnamese Text Summarization (Encoder-Decoder)

**Folder:** `Vietnamese_text_sumarization/`

**Mục tiêu:** Fine-tune mô hình Encoder-Decoder để tóm tắt các văn bản (tin tức, bài báo) Tiếng Việt.

**Mô hình:** `ViT5-base` (Encoder-Decoder)

**Kiến trúc:**

- **Encoder:** Xử lý văn bản gốc, tạo biểu diễn ngữ cảnh sâu
- **Decoder:** Sinh tóm tắt từng token, sử dụng Cross-Attention tham chiếu Encoder
- Kiến trúc Seq2Seq cổ điển phù hợp cho Seq2Seq tasks

[📖 Chi tiết](Vietnamese_text_sumarization/README.md)

---

### 3. 🤖 Product Q&A Chatbot (Decoder-Only)

**Folder:** `Chatbot_ProductQA/`

**Mục tiêu:** Fine-tune mô hình Decoder-only cho các bài toán sinh văn bản tự do hoặc hội thoại, xây dựng chatbot đóng vai trò là trợ lý ảo giải đáp về sản phẩm

**Mô hình:** `Qwen3-0.6B` (Decoder-only)

**Kiến trúc:**

- Mô hình Decoder-only sử dụng Causal Attention
- Dự đoán token tiếp theo một cách autoregressive
- Phù hợp cho các tác vụ sinh văn bản tuần tự

[📖 Chi tiết](Chatbot_ProductQA/README.md)

---

## 🔧 Các Phương Pháp Kỹ Thuật Chính

### 1. Fine-tuning Strategy

- **Learning Rate:** 2e-5 (phổ biến cho fine-tuning)
- **Batch Size:** 4-8 (tùy GPU memory)
- **Epochs:** 3-5
- **Optimizer:** AdamW
- **Mix Precision:** fp16 để tăng tốc độ

### 2. Data Augmentation

- Sử dụng LLM (Gemini) để tạo synthetic data
- Cải thiện độ tổng quát (generalization) và giảm overfitting
- Tăng ~5-10% hiệu suất trên các metrics

### 3. Evaluation Metrics

- **Toxic Classification:** Precision, Recall, F1-score, Accuracy
- **Text Summarization:** ROUGE-1, ROUGE-2, ROUGE-L
- **LLM-based Evaluation:** Coherence, Relevance, Fluency, Factuality

---

## 🚀 Hướng Dẫn Sử Dụng

### Chuẩn bị môi trường

1. Clone repository và cài đặt dependencies cơ bản
2. Cấu hình credentials cần thiết (Hugging Face, Gemini API nếu sử dụng)

### Chạy từng dự án

Mỗi folder con là một dự án độc lập với hướng dẫn chi tiết riêng. Xem README trong từng thư mục để biết cách chạy cụ thể.

---

## 📚 Kiến Thức Nền Tảng

### Transformer Architecture

- **Attention Mechanism:** Self-attention và Cross-attention
- **Positional Encoding:** Thêm thông tin vị trí token
- **Multi-head Attention:** Học nhiều biểu diễn khác nhau

### Transfer Learning

- Sử dụng mô hình pre-trained trên dữ liệu lớn
- Fine-tuning trên dữ liệu task-specific
- Giảm thời gian huấn luyện và cải thiện hiệu suất

### Vietnamese NLP

- Xử lý đặc biệt của dấu thanh, từ ghép Tiếng Việt
- Tokenizer chuyên biệt: `PhoBERT`, `ViT5`
- Công cụ: Underthesea cho tiền xử lý

---

## 🔗 Tài Nguyên Bổ Trợ

### Models & Datasets

- [Hugging Face Models](https://huggingface.co/models)
- [HuggingFace Datasets](https://huggingface.co/datasets)
- [VietAI Models](https://github.com/VietAI)

### Papers & References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [T5: Text-to-Text Transfer Transformer](https://arxiv.org/abs/1910.10683)
- [PhoBERT: Pre-trained Language Models for Vietnamese](https://arxiv.org/abs/2003.00196)

### Tools

- [Transformers Library](https://huggingface.co/transformers/)
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [Underthesea](https://github.com/undertheseanlp/underthesea) - Vietnamese NLP

---

## 📝 Ghi Chú Quan Trọng

- Đảm bảo GPU driver và CUDA tương thích
- Lưu checkpoints thường xuyên trong quá trình fine-tuning
- Sử dụng Weights & Biases để track experiments
- Kiểm tra dữ liệu cẩn thận trước fine-tuning
- Để ý đến overfitting khi dữ liệu nhỏ

---

## 📞 Liên Hệ & Support

Nếu có câu hỏi hoặc vấn đề kỹ thuật, vui lòng tạo Issue hoặc liên hệ với thành viên nhóm.
