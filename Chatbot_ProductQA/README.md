# Product Q&A Chatbot - Decoder-Only Fine-tuning

**Fine-tuning mô hình Decoder-only cho bài toán Product Q&A**

---

## 📋 Tổng Quan

Dự án này implements một chatbot trả lời câu hỏi (Q&A) về sản phẩm trên nền tảng Sendo, dựa trên kiến trúc Decoder-only và mô hình pre-trained `Qwen3-0.6B`.

---

## 🎯 Mục Tiêu

- Fine-tune mô hình Decoder-only nhẹ nhàng (0.6B parameters) cho tác vụ Q&A
- Xây dựng chatbot có thể:
  - Hiểu câu hỏi về sản phẩm từ khách hàng
  - Sinh ra câu trả lời tự nhiên, relevant và hữu ích
  - Xử lý các cuộc hội thoại multiturn (multiple turns)
- Tăng cường dữ liệu với synthetic data từ LLM

---

## 🏗️ Kiến Trúc Mô Hình

### Decoder-Only Architecture

```
Input Tokens → Embedding Layer → Transformer Blocks
              ↓
         Self-Attention (Causal Masking)
              ↓
         Feed-Forward Networks
              ↓
         x Repeat N times
              ↓
         Output Projection → Logits → Next Token Prediction
```

**Đặc điểm:**

- Sử dụng **Causal Attention**: Chỉ nhìn các token phía trước, không nhìn future tokens
- Dự đoán token tiếp theo một cách **autoregressive**
- Phù hợp cho sinh văn bản: câu trả lời, code generation, summarization
- Nhẹ nhàng (0.6B params) → Inference nhanh, tiêu điện năng thấp

---

## 📊 Dữ Liệu

### Dataset Gốc

- **Nguồn:** [Sendo Vietnamese Multiturn Dataset](https://huggingface.co/datasets/5CD-AI/sendo_vietnamese_multiturn_gemini_50k)
- **Kích thước:** ~50,000 cuộc hội thoại
- **Nội dung:** Câu hỏi-câu trả lời thực tế về sản phẩm từ nền tảng Sendo
- **Định dạng:** Multiturn conversations (history + current question + answer)

### Dữ Liệu Tổng Hợp

- **File:** `Datasets/synthetic_final.csv`
- **Phương pháp:** Sinh dữ liệu bằng LLM (Gemini)
- **Mục tiêu:** Tăng độ đa dạng, giảm overfitting, cải thiện generalization

### Tiền Xử Lý Dữ Liệu

```python
# Format chuyển đoạn hội thoại thành format Decoder-only
# Input: [CLS] context [SEP] question [SEP]
# Output: answer sequence
```

---

## 🔧 Cấu Hình Fine-tuning

| Tham số         | Giá trị    | Mô tả                   |
| --------------- | ---------- | ----------------------- |
| Model           | Qwen3-0.6B | Base model              |
| Learning Rate   | 2e-5       | Optimizer learning rate |
| Batch Size      | 4-8        | Tùy GPU memory          |
| Epochs          | 3-5        | Số epoch training       |
| Max Seq Length  | 512-1024   | Độ dài tối đa sequence  |
| Optimizer       | AdamW      | Adam with weight decay  |
| Scheduler       | Linear     | Learning rate scheduler |
| Mixed Precision | fp16       | Để tăng tốc độ          |
| Warmup Steps    | ~500-1000  | Warmup scheduler        |

---

## 📁 Cấu Trúc Thư Mục

```
Chatbot_ProductQA/
├── README.md                              # Tệp này
├── Datasets/
│   ├── Dataset03.csv                      # Dataset gốc
│   └── synthetic_final.csv                # Dữ liệu tổng hợp
└── Source/
    ├── Decoder_Only_Finetune.ipynb        # Main fine-tuning notebook
    └── Synthetic_Data_DO.ipynb            # Synthetic data generation
```

---

## 📖 Notebooks

### 1. `Decoder_Only_Finetune.ipynb`

**Notebook chính cho fine-tuning**

Các bước:

1. Load dataset từ CSV
2. Preprocess và tokenize data
3. Load pre-trained Qwen3-0.6B
4. Fine-tuning với Trainer API
5. Inference và testing
6. Save model to Hugging Face Hub

### 2. `Synthetic_Data_DO.ipynb`

**Tạo dữ liệu tổng hợp**

Các bước:

1. Load original dataset
2. Gọi Gemini API để sinh variations
3. Validate generated data
4. Save to CSV
5. Merge với original data

---

## 🚀 Cách Chạy

### 1. Chuẩn Bị Môi Trường

```bash
# Install dependencies
pip install torch transformers datasets pandas numpy jupyter wandb

# Login to Hugging Face (nếu cần push model)
huggingface-cli login

# Setup Gemini API key (nếu sinh synthetic data)
export GEMINI_API_KEY="your-key-here"
```

### 2. Chạy Fine-tuning

```bash
# Mở Jupyter và chạy Decoder_Only_Finetune.ipynb
jupyter notebook Source/Decoder_Only_Finetune.ipynb
```

### 3. (Optional) Generate Synthetic Data

```bash
jupyter notebook Source/Synthetic_Data_DO.ipynb
```

---

## 📈 Kết Quả Dự Kiến

| Metric           | Baseline  | +Augmentation |
| ---------------- | --------- | ------------- |
| Perplexity       | ~15-20    | ~14-18        |
| BLEU Score       | ~25-30    | ~28-35        |
| Response Quality | Fair      | Good          |
| Response Latency | ~50-100ms | ~50-100ms     |

**Cải thiện sau augmentation:** +5-10% trên quality metrics

---

## 💡 Kỹ Thuật & Best Practices

### 1. Causal Masking

- Đảm bảo model chỉ dự đoán dựa trên tokens trước nó
- Tự động applied trong Decoder-only models

### 2. Context Window

- Lưu conversation history trong context
- Cho phép model trả lời với hiểu biết về cuộc hội thoại trước

### 3. Temperature & Top-k Sampling

- Điều chỉnh độ "creativity" của responses
- Temperature cao → Diverse nhưng less consistent
- Temperature thấp → Consistent nhưng repetitive

### 4. Beam Search

```python
# Generate output với beam search
outputs = model.generate(
    input_ids,
    max_length=256,
    num_beams=5,
    temperature=0.7,
    top_p=0.95
)
```

---

## 🔍 Debugging & Troubleshooting

| Vấn đề               | Giải pháp                                  |
| -------------------- | ------------------------------------------ |
| Out of Memory        | Giảm batch size, max sequence length       |
| Model overfitting    | Thêm more synthetic data, regularization   |
| Responses bị lặp lại | Giảm temperature, tăng diversity penalties |
| Slow inference       | Sử dụng quantization, smaller batch size   |

---

## 📚 Tài Nguyên Tham Khảo

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Qwen Model Cards](https://huggingface.co/Qwen)
- [Being Modest: Architectural Design Lessons Learned](https://arxiv.org/abs/2305.11206)

---

## ✅ Checklist

- [ ] Dataset đã chuẩn bị (gốc + synthetic)
- [ ] Dependencies cài đặt
- [ ] GPU memory đủ
- [ ] Hugging Face credentials configured
- [ ] Training completed & saved
- [ ] Model pushed to Hub (optional)
- [ ] Evaluation metrics recorded

---

**Last Updated:** Tháng 3, 2026
