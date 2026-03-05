# Toxic Text Classification - Encoder-Only Fine-tuning

**Fine-tuning mô hình Encoder-only cho bài toán phân loại văn bản độc hại Tiếng Việt**

---

## 📋 Tổng Quan

Dự án này xây dựng một hệ thống phân loại nhận xét/bình luận độc hại trên Tiếng Việt, sử dụng kiến trúc Encoder-only và mô hình pre-trained `PhoBERT-large`.

---

## 🎯 Mục Tiêu

- Fine-tune mô hình Encoder-only `PhoBERT-large` cho tác vụ text classification
- Phát hiện các bình luận/nhận xét độc hại (toxic comments)
- Xây dựng hệ thống tự động moderating cho nền tảng social media/comment
- Cải thiện hiệu suất với synthetic data augmentation
- Đánh giá model với các metrics phù hợp (Accuracy, Precision, Recall, F1)

---

## 🏗️ Kiến Trúc Mô Hình

### Encoder-Only Architecture

```
Input Text → Embedding Layer → Transformer Blocks
            ↓
        Self-Attention (Bidirectional)
            ↓
        Feed-Forward Networks
            ↓
        x Repeat N times
            ↓
        [CLS] Token Representation → Classification Head → Logits → Label
```

**Đặc điểm:**

- Sử dụng **Bidirectional Self-Attention**: Nhìn cả forward và backward
- Được pre-trained với **Masked Language Modeling (MLM)**
- Ideal cho text classification, semantic similarity, named entity recognition
- `PhoBERT-large`: Điều chỉnh cho Tiếng Việt, tốt hơn mô hình BERT tiêu chuẩn

---

## 📊 Dữ Liệu

### Dataset Gốc

- **File:** `Datasets/Dataset01.csv`
- **Kích thước:** Tùy định sẵn trong project
- **Nội dung:** Nhận xét/bình luận Tiếng Việt được gắn nhãn Toxic/Non-toxic
- **Nhãn:**
  - **0 (Non-toxic):** Bình luận bình thường, lành mạnh
  - **1 (Toxic):** Chứa ngôn ngữ độc hại, lăng mạ, xúc phạm

### Dữ Liệu Tổng Hợp

- **File:** `Datasets/synthetic_final.csv`
- **Phương pháp:** Sinh dữ liệu bằng Paraphrasing và LLM
- **Mục tiêu:** Tăng data diversity, giảm overfitting

### Tiền Xử Lý Dữ Liệu

```python
# Xử lý đặc biệt cho Tiếng Việt
from underthesea import word_tokenize

text = "Đây là một bình luận độc hại"
tokens = word_tokenize(text)
# → ["Đây", "là", "một", "bình", "luận", "độc", "hại"]

# PhoBERT tokenizer xử lý tốt dấu thanh, từ ghép
```

---

## 🔧 Cấu Hình Fine-tuning

| Tham số         | Giá trị       | Mô tả                             |
| --------------- | ------------- | --------------------------------- |
| Model           | PhoBERT-large | Base model pre-trained            |
| Learning Rate   | 2e-5          | Optimizer learning rate           |
| Batch Size      | 8-16          | Tùy GPU memory                    |
| Epochs          | 3-5           | Số epoch training                 |
| Max Seq Length  | 256           | Độ dài tối đa sequence            |
| Optimizer       | AdamW         | Adam with weight decay            |
| Scheduler       | Linear        | Learning rate scheduler           |
| Mixed Precision | fp16          | Để tăng tốc độ training           |
| Warmup Steps    | ~500          | Linear warmup                     |
| Weight Decay    | 0.01          | Regularization                    |
| Dropout         | 0.1           | Dropout trong classification head |

---

## 📁 Cấu Trúc Thư Mục

```
Toxic_text_classification_vi/
├── README.md                              # Tệp này
├── Datasets/
│   ├── Dataset01.csv                      # Dataset gốc được gắn nhãn
│   └── synthetic_final.csv                # Dữ liệu tổng hợp
└── Source/
    ├── Encoder_Only_Finetune.ipynb        # Main fine-tuning notebook
    └── Synthetic_Data_EncodeOnly.ipynb    # Synthetic data generation
```

---

## 📖 Notebooks

### 1. `Encoder_Only_Finetune.ipynb`

**Notebook chính cho fine-tuning**

Các bước:

1. Load và explore dataset
2. Data preprocessing & tokenization (với Underthesea)
3. Create train/val/test splits
4. Load pre-trained PhoBERT-large
5. Add classification head
6. Fine-tuning với Trainer API
7. Evaluation trên test set
8. Inference examples
9. Save and push model to Hugging Face

### 2. `Synthetic_Data_EncodeOnly.ipynb`

**Tạo dữ liệu tổng hợp**

Các bước:

1. Load original toxic/non-toxic comments
2. Paraphrase sentences (giữ ngữ nghĩa, đổi từ ngữ)
3. Sinh variations bằng LLM hoặc rule-based
4. Validate data quality
5. Save to CSV
6. Combine với original dataset

---

## 🚀 Cách Chạy

### 1. Chuẩn Bị Môi Trường

```bash
# Install dependencies
pip install torch transformers datasets pandas numpy jupyter wandb underthesea scikit-learn

# Login to Hugging Face
huggingface-cli login

# (Optional) Setup Weights & Biases
wandb login
```

### 2. Chạy Fine-tuning

```bash
# Mở Jupyter
jupyter notebook Source/Encoder_Only_Finetune.ipynb
```

### 3. (Optional) Generate Synthetic Data

```bash
jupyter notebook Source/Synthetic_Data_EncodeOnly.ipynb
```

---

## 📈 Kết Quả Dự Kiến

### Metrics

| Metric    | Baseline   | +Augmentation |
| --------- | ---------- | ------------- |
| Accuracy  | ~82-88%    | ~85-92%       |
| Precision | ~80-86%    | ~83-90%       |
| Recall    | ~78-86%    | ~80-88%       |
| F1-Score  | ~79-86%    | ~82-89%       |
| ROC-AUC   | ~0.88-0.94 | ~0.90-0.96    |

**Cải thiện sau augmentation:** +3-5% trên các metrics

### Confusion Matrix

```
                 Predicted
            Negative    Positive
Actual  Negative   TN        FP
        Positive   FN        TP
```

---

## 💡 Kỹ Thuật & Best Practices

### 1. Xử Lý Tiếng Việt

- Sử dụng `Underthesea` hoặc `PyVi` để tokenization
- PhoBERT tokenizer đã tối ưu cho Tiếng Việt
- Cẩn thận với dấu thanh, từ ghép

### 2. Class Imbalance

Nếu data imbalanced (một class nhiều hơn):

```python
# Weighted loss
from torch import nn
weights = torch.tensor([weight_neg, weight_pos])
loss = nn.CrossEntropyLoss(weight=weights)
```

### 3. Evaluation Strategy

```python
# Confusing matrix & detailed metrics
from sklearn.metrics import classification_report, confusion_matrix

report = classification_report(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
```

### 4. Inference với Confidence

```python
# Get prediction confidence
logits = model(input_ids)
probs = torch.softmax(logits, dim=-1)
confidence, prediction = torch.max(probs, dim=1)
```

---

## 🔍 Debugging & Troubleshooting

| Vấn đề           | Giải pháp                                       |
| ---------------- | ----------------------------------------------- |
| Poor performance | Kiểm tra data quality, tăng training epochs     |
| Overfitting      | Thêm augmentation, tăng dropout, regularization |
| Class imbalance  | Dùng weighted loss, balanced sampling           |
| Slow training    | Dùng mixed precision (fp16), tăng batch size    |
| Out of memory    | Giảm batch size, seq length                     |

---

## 🎯 Evaluation Metrics Chi Tiết

- **Accuracy:** (TP + TN) / Total - Tổng số dự đoán đúng
- **Precision:** TP / (TP + FP) - Độ chính xác của toxic predictions
- **Recall:** TP / (TP + FN) - Độ đầy đủ phát hiện toxic
- **F1-Score:** 2 _ (Precision _ Recall) / (Precision + Recall)
- **ROC-AUC:** Area under ROC curve - Hiệu năng tổng thể

---

## 📚 Tài Nguyên Tham Khảo

- [PhoBERT: Pre-trained Language Models for Vietnamese](https://arxiv.org/abs/2003.00196)
- [Hugging Face Text Classification Guide](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- [Underthesea Documentation](https://github.com/undertheseanlp/underthesea)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

---

## ✅ Checklist

- [ ] Dataset preprocessed & split
- [ ] Synthetic data generated
- [ ] Dependencies installed
- [ ] PhoBERT model downloaded
- [ ] GPU/compute resources ready
- [ ] Training completed
- [ ] Evaluation metrics recorded
- [ ] Model saved & pushed to Hub (optional)

---

**Last Updated:** Tháng 3, 2026
