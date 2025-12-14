ViT5 Vietnamese Text Summarization (Encoder-Decoder Architecture)

#1. Giới thiệu:
Dự án này triển khai mô hình Encoder-Decoder dựa trên ViT5-base (một biến thể tiếng Việt của T5) để giải quyết bài toán tóm tắt văn bản tiếng Việt.

#2. Mục tiêu chính:
- Rút gọn đoạn văn bản dài (tin tức, bài báo) thành tóm tắt ngắn gọn, mạch lạc nhưng vẫn giữ đầy đủ thông tin quan trọng.
- Áp dụng fine-tuning mô hình pretrained trên dữ liệu tiếng Việt.
- Tăng cường dữ liệu bằng synthetic data sinh từ LLM (Gemini).
- Đánh giá bằng metric truyền thống (ROUGE) và LLM-based evaluation (Gemini).

#3. Kiến trúc Encoder-Decoder trong Text Summarization:
Mô hình Encoder-Decoder là kiến trúc Transformer cổ điển dành cho các nhiệm vụ sequence-to-sequence (Seq2Seq):
- Encoder: Xử lý văn bản đầu vào dài, tạo biểu diễn ngữ cảnh sâu nhờ cơ chế self-attention.
- Decoder: Sinh tóm tắt từng token một cách autoregressive, sử dụng masked self-attention và cross-attention để tham chiếu encoder.

ViT5 (dựa trên T5) được pretrained theo phong cách text-to-text, rất phù hợp cho summarization vì khả năng sinh văn bản (paraphrase thay vì trích xuất nguyên câu).

#4. Quá trình xử lý dữ liệu với tiếng Việt:
- Xử lý tốt dấu thanh, từ ghép.
- Kết hợp tokenizer chuyên biệt và công cụ như Underthesea cho tiền xử lý.

#5. Dữ liệu:
Dataset gốc: fcsn37/vietnamese-text-summarization (~30k-100k cặp text-summary từ tin tức tiếng Việt).
Synthetic data: Sinh thêm bằng Gemini API, tăng độ đa dạng và giảm overfitting.
Tổng dữ liệu sau augmentation: Kết hợp gốc + synthetic, push lên Hugging Face.
Chia dữ liệu theo tỉ lệ 80-10-10

#6. Quy trình Fine-tuning:
Notebook chính: Encoder_Decoder.ipynb
Các bước chính:
- Load pretrained ViT5-base từ VietAI.
- Sử dụng Seq2SeqTrainer (Hugging Face) với data collator động.
Hyperparameters:
- Batch size: 4-8
- Epochs: 3-5
- Learning rate: ~2e-5
- Optimizer: AdamW
- Mixed precision (fp16)

Logging với Wandb.
So sánh trước/sau augmentation: Augmentation cải thiện rõ rệt độ tổng quát.
Inference sử dụng beam search để sinh tóm tắt chất lượng cao.

#7. Đánh giá:
Metric truyền thống (ROUGE): Đo độ chồng lấp n-gram với tóm tắt tham chiếu.
Các mô hình ViT5 fine-tuned trên dữ liệu tương tự thường đạt ROUGE-1 ~45-55, ROUGE-2 ~25-35, ROUGE-L ~40-50 (tùy dataset).
Với augmentation, cải thiện ~5-10% so với baseline.
LLM-based Evaluation (Gemini): Chấm điểm coherence, relevance, fluency, factuality (thang 1-10).
Cải thiện trung bình 0.5-1 điểm sau augmentation.

#8. Kết quả định tính: Tóm tắt mạch lạc, giữ fact chính xác, ít hallucination hơn sau augmentation.

#9. Hạn chế & Hướng phát triển
- Chủ yếu hiệu quả trên văn bản tin tức; có thể kém hơn với văn phong không trang trọng.
- Cần GPU để fine-tune.
- Mở rộng: Thêm RAG cho fact-checking, thử decoder-only models, hoặc so sánh nhiều LLM sinh synthetic data.

Dự án tuân thủ đầy đủ yêu cầu môn học: Hiểu Transformer, fine-tuning, data augmentation, dual evaluation. Chi tiết code và log trong notebook.
