# ENCODER-DECODER VÀ BÀI TOÁN TÓM TẮT VĂN BẢN
# Chi tiết bài toán: https://colab.research.google.com/drive/1NQ3kWLSxm0j9xkvRDRFUgtjZTpGipHyW 
'''
Mục đích: Bài toán đặt ra là xây dựng mô hình có khả năng rút gọn một đoạn văn dài thành bản tóm tắt ngắn, cô đọng nhưng vẫn giữ đầy đủ thông tin quan trọng. Dữ liệu gồm các cặp “text–summary” tiếng Việt dùng để huấn luyện mô hình học quan hệ giữa nội dung và tóm tắt. Mục tiêu là tạo ra mô hình sinh tóm tắt chính xác, mạch lạc và gần nhất với bản tóm tắt tham chiếu.
Model: ViT5-base + Synthetic Data
Đường dẫn mô hình:[model](https://huggingface.co/fcsn37/vit5-vietnamese-summarization-final)
Dataset: fcsn37/vietnamese-text-summarization + fcsn37/vietnamese-text-summarization-synthetic-dataset
Đường dẫn tập dữ liệu sử dụng:
[Dữ liệu nguyên mẫu](fcsn37/vietnamese-text-summarization-30k)
[Dữ liệu tạo sinh](https://huggingface.co/datasets/fcsn37/vietnamese-text-summarization-synthetic-dataset)
'''
## Giới thiệu: 
"""
Encoder-Decoder và bài toán Tóm tắt văn bản:
- Fine-tune mô hình pretrained trên dữ liệu tự thu thập/lấy từ nguồn mở (dataset vietnamese-text-summarization từ Hugging Face).
- Áp dụng data augmentation bằng synthetic data sinh từ LLM để tăng cường dataset.
- Đánh giá mô hình bằng metrics truyền thống (ROUGE) và LLM-based evaluation (sử dụng Gemini để chấm điểm coherence, relevance, v.v.).
- So sánh hiệu suất trước/sau augmentation và rút ra nhận xét về tính tổng quát của mô hình.
"""
## Thiết lập:
#### Cài đặt thư viện:

# ! pip install -q transformers datasets sentencepiece rouge-score accelerate wandb google-generativeai underthesea tensorflow openai
# ! pip install evaluate underthesea

import torch
import os
import evaluate
import numpy as np
import pandas as pd
import google.generativeai as genai
import wandb
import re
import json
import kagglehub
import underthesea
import matplotlib.pyplot as plt
import time
import random
import seaborn as sns
from openai import OpenAI
from typing import List, Dict
from datetime import datetime
from huggingface_hub import login, HfApi
from rouge_score import rouge_scorer
from tqdm import tqdm
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq, pipeline,
    TrainerCallback
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from underthesea import word_tokenize, text_normalize
from sklearn.model_selection import train_test_split
from google.colab import files
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerateContentResponse
from collections import Counter

pd.set_option('display.max_colwidth', None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

#Kết nối Hugging Face:
login(token="HF_TOKEN")

#Kết nối GenAI API:
GEMINI_API_KEY = "GEMINI_API_KEY"
genai.configure(api_key=GEMINI_API_KEY)

#Kết nối WANDB:
WANDB_PROJECT = "vit5-text-summarization"
WANDB_API_KEY = "WANDB_API_KEY"
wandb.login(key=WANDB_API_KEY)


## Chuẩn bị dữ liệu:
"""
1. Load dataset gốc từ Hugging Face ("fcsn37/vietnamese-text-summarization-30k") và stopwords từ file txt.
2. Tiền xử lý: Tokenize input (text) và target (summary) với max_length 1024/128, sử dụng ViT5 tokenizer.
3. Augmentation: Sử dụng Gemini để tạo synthetic data (2 variants per sample, thay đổi diễn đạt nhưng giữ ý nghĩa), xử lý rate limit bằng retry và sleep. Augment 200-1000 samples từ train split.
4. Kết hợp data: Concatenate dataset gốc với augmented, lưu thành dataset mới và push lên Hugging Face ("fcsn37/vietnamese-text-summarization-augmented_dataset").
5. Chia subset: Train (5000 samples), validation/test (200 samples each), shuffle với seed 42.
"""
data = pd.read_csv("hf://datasets/fcsn37/vietnamese-text-summarization/data_summary.csv")
data = data[['Text', 'Summary']]
stopword = pd.read_csv("hf://datasets/fcsn37/vietnamese-stopwords/vietnamese-stopwords.txt", header=None, names=["stopword"])
print(stopword.head())
### Tiền xử lý dữ liệu:
data = data.sample(n=30000, random_state=42).reset_index(drop=True)
print(data.shape)
data.head(1)
###Loại bỏ phần tử rỗng:
data.drop_duplicates(subset=['Text'],inplace=True)
data.dropna(axis=0,inplace=True)


## Chuẩn hoá văn bản:
def text_cleaner(text, num=0):
    # 1. Chuyển toàn bộ văn bản thành chữ thường
    newString = text.lower()
    # 2. Loại bỏ nội dung nằm trong dấu ngoặc tròn ( )
    newString = re.sub(r'\([^)]*\)', '', newString)
    # 3. Xóa dấu ngoặc kép "
    newString = newString.replace('"', '')
    # 4. Loại bỏ dấu câu và ký tự đặc biệt
    # Chỉ giữ lại chữ cái tiếng Việt, chữ số và khoảng trắng
    newString = re.sub(
        r"[^0-9a-zA-Záàảãạăắằẵặẳâấầậẫẩéèẻẽẹêếềểễệ"
        r"íìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữự"
        r"ýỳỷỹỵđ\s]",
        " ",
        newString
    )
    # 5. Chuẩn hóa khoảng trắng thừa
    newString = re.sub(r"\s+", " ", newString).strip()
    # 6. Tách từ bằng underthesea
    tokens = word_tokenize(newString, format="text").split()
    # 7. Loại bỏ stopword
    if num == 0:
        tokens = [w for w in tokens if w not in stopword]
    # 8. Loại bỏ những từ chỉ gồm 1 ký tự
    longwords = [w for w in tokens if len(w) > 1]
    return " ".join(longwords)

cleaned_text = []
for t in data['Text']:
    cleaned_text.append(text_cleaner(t,0))
# cleaned_text[:1]

cleaned_summary = []
for t in data['Summary']:
    cleaned_summary.append(text_cleaner(t,1))
# cleaned_summary[:1]

# Thay đổi dữ liệu đã được làm sạch vào dataframe
data['cleaned_text']=cleaned_text
data['cleaned_summary']=cleaned_summary

### Lưu lại dữ liệu đã được xử lý
dataset = data[['cleaned_text', 'cleaned_summary']].copy()

dataset = dataset.rename(columns={
    'cleaned_text': 'text',
    'cleaned_summary': 'summary'
})
hf_dataset = Dataset.from_pandas(dataset, preserve_index=False)
dataset_dict = hf_dataset.train_test_split(test_size=0.2, seed=42)
dataset_dict = DatasetDict({
    'train': dataset_dict['train'],
    'validation': dataset_dict['test'].train_test_split(test_size=0.5, seed=42)['train'],
    'test': dataset_dict['test'].train_test_split(test_size=0.5, seed=42)['test']
})
dataset_dict
login(token="HF_TOKEN")
# dataset_dict.push_to_hub("fcsn37/vietnamese-text-summarization-30k")

### Dữ liệu nguyên mẫu:
#### Tải dữ liệu đã xử lý từ Hugging Face:
dataset = load_dataset("fcsn37/vietnamese-text-summarization-30k")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
model     = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")

MAX_INPUT_LENGTH  = 1024
MAX_TARGET_LENGTH = 128

def preprocess_function(examples):
    inputs  = examples["text"]
    targets = examples["summary"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

print(tokenized_datasets)

dataset["train"].select(range(1)).to_pandas()

###Dữ liệu tổng hợp:
#### Xử lý dữ liệu tổng hợp:
login(token="HF_TOKEN")
synthetic_dataset = load_dataset("fcsn37/vietnamese-text-summarization-augmented_dataset")
synthetic_dataset["train"].select(range(1)).to_pandas()
cleaned_synthetic_text = []
for t in synthetic_dataset["train"]['text']:
    cleaned_synthetic_text.append(text_cleaner(t,0))

cleaned_synthetic_summary = []
for t in synthetic_dataset["train"]['summary']:
    cleaned_synthetic_summary.append(text_cleaner(t,1))

synthetic_dataset["train"] = Dataset.from_dict({
    "text": cleaned_synthetic_text,
    "summary": cleaned_synthetic_summary
})

synthetic_dataset["train"].select(range(1)).to_pandas()

####Dữ liệu từ nguồn LLMs: Claude, ChatGPT, Grok
syn_data = pd.read_csv("synthetic_dataset.csv")
syn_dataset = Dataset.from_pandas(syn_data)

syn_dataset = syn_dataset.rename_column("Text", "text")
syn_dataset = syn_dataset.rename_column("Summary", "summary")

cleaned_synthetic_text = []
for t in syn_dataset['text']:
    cleaned_synthetic_text.append(text_cleaner(t,0))

cleaned_synthetic_summary = []
for t in syn_dataset['summary']:
    cleaned_synthetic_summary.append(text_cleaner(t,1))

syn_dataset.select(range(1)).to_pandas()

syn_dataset = Dataset.from_dict({
    "text": cleaned_synthetic_text,
    "summary": cleaned_synthetic_summary
})

syn_dataset.select(range(1)).to_pandas()

#### Kết hợp thành tập dữ liệu tạo sinh:

synthetic_datasets = concatenate_datasets([synthetic_dataset["train"], syn_dataset])

synthetic_datasets.select(range(1)).to_pandas()

login(token="HF_TOKEN")

dataset_to_push = DatasetDict({"train": synthetic_datasets})
dataset_to_push.push_to_hub(
    "fcsn37/vietnamese-text-summarization-synthetic-dataset",
    private=False
)

#### Dữ liệu tổng hợp:
"""
Mô hình LLMs được sử dụng để tạo sinh dữ liệu: Claude, ChatGPT, Grok.
Prompt sử dụng để tạo sinh:
    Bạn là chuyên gia tạo dữ liệu huấn luyện cho mô hình tóm tắt văn bản tiếng Việt.
    Nhiệm vụ của bạn: Tạo 20 cặp dữ liệu hoàn toàn mới (không sao chép từ bất kỳ nguồn nào trên Internet, không dùng dữ liệu thật đã tồn tại) theo định dạng sau:
    - Mỗi cặp gồm 2 trường: Text và Summary
    - Text: đoạn văn tiếng Việt tự nhiên, thuộc các chủ đề đa dạng (tin tức, kinh tế, công nghệ, sức khỏe, giáo dục, xã hội, thể thao, giải trí, môi trường, khoa học...), độ dài từ 150 đến 200 từ.
    - Summary: tóm tắt ngắn gọn, súc tích, bằng tiếng Việt, giữ đúng 100% nội dung chính, dưới 30 từ (tối đa 28 từ), viết tự nhiên như con người.
    Yêu cầu bắt buộc:
    1. Tất cả 20 cặp dữ liệu phải hoàn toàn do bạn sáng tạo, không trùng bất kỳ bài báo nào thật.
    2. Ngôn ngữ phải chuẩn tiếng Việt, tự nhiên, không có lỗi chính tả, lỗi dấu.
    3. Chủ đề phải đa dạng, không lặp lại quá nhiều cùng một lĩnh vực.
    4. Summary phải ngắn gọn, hấp dẫn, có thể dùng làm nhãn tốt cho mô hình học tóm tắt.
    Kết quả sau khi được tạo sinh sẽ được xử lý, tokenize và lưu tại hugging face để tiện trong quá trình huấn luyện dữ liệu
"""
synthetic_dataset = load_dataset("fcsn37/vietnamese-text-summarization-synthetic-dataset")

synthetic_dataset["train"].select(range(1)).to_pandas()

## **Huấn luyện mô hình:
"""
1. Define hyperparameters: Batch size 4, gradient accumulation 4, LR 3e-5, epochs 3, warmup 100, max_input 256, max_target 64.
2. Sử dụng DataCollatorForSeq2Seq và compute_metrics với ROUGE.
3. Huấn luyện baseline trên data gốc.
4. Huấn luyện augmented trên data kết hợp (gốc + 500 augmented samples).
5. Sử dụng Seq2SeqTrainer với callbacks để push checkpoint lên HF repo.
6. Lưu model final và evaluate trên validation.

Chỉ số theo dõi trong quá trình huấn luyện:
- [baseline](https://wandb.ai/fcsn_37-ton-duc-thang-university/huggingface/runs/fwyvfgre?nw=nwuserfcsn_37)
- [augmented-final](https://wandb.ai/fcsn_37-ton-duc-thang-university/huggingface/runs/lrjf4231?nw=nwuserfcsn_37)
"""

"""
Tổng quan mô hình: ViT5 là mô hình encoder–decoder dành cho tiếng Việt, được phát triển bởi VietAI, dựa trên kiến trúc T5 – Text-to-Text Transfer Transformer của Google. T5 là mô hình hợp nhất tất cả các tác vụ NLP vào một framework duy nhất: mọi bài toán đều được biểu diễn dưới dạng “text-to-text”.
ViT5 kế thừa hoàn toàn kiến trúc T5 gốc, bao gồm:
- Encoder để mã hóa câu đầu vào;
- Decoder để sinh văn bản đầu ra;
- Self-Attention và Cross-Attention trong các lớp Transformer;
- Cơ chế position-wise feed-forward và layer normalization.
"""
MODEL_NAME = "VietAI/vit5-base"
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 64
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 3e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 100
HF_TOKEN = "HF_TOKEN"
DATASET_NAME = "fcsn37/vietnamese-text-summarization-30k"
AUGMENTED_DATASET = "fcsn37/vietnamese-text-summarization-synthetic-dataset"
WANDB_PROJECT = "vit5-text-summarization"
WANDB_API_KEY = "WANDB_API_KEY"

dataset = load_dataset(DATASET_NAME)

train_size = 23435
dataset['train'] = dataset['train'].shuffle(seed=42).select(range(train_size))
dataset['validation'] = dataset['validation'].shuffle(seed=42).select(range(200))
dataset['test'] = dataset['test'].shuffle(seed=42).select(range(200))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    inputs = examples["text"]
    targets = examples["summary"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=None,
    padding=True
)

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    return {k: round(v * 100, 2) for k, v in result.items()}

class SafePushCallback(TrainerCallback):
    def __init__(self, tokenizer, repo_name, hf_token, experiment_name):
        self.tokenizer = tokenizer
        self.repo_name = repo_name
        self.api = HfApi(token=hf_token)
        self.experiment_name = experiment_name

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(state.epoch)
        checkpoint_name = f"{self.experiment_name}-epoch-{epoch}"
        checkpoint_dir = f"{args.output_dir}/{checkpoint_name}"

        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        try:
            self.api.upload_folder(
                folder_path=checkpoint_dir,
                repo_id=self.repo_name,
                path_in_repo=checkpoint_name,
                repo_type="model",
                commit_message=f"Checkpoint {checkpoint_name}"
            )
        except Exception as e:
            pass

        return control

def train_model(experiment_name, train_dataset, eval_dataset, repo_name):

    torch.cuda.empty_cache()

    output_dir = f"./results/{experiment_name}"

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.gradient_checkpointing_enable()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        warmup_steps=WARMUP_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        report_to="wandb",
        run_name=experiment_name,
        push_to_hub=False,
        label_smoothing_factor=0.1,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[SafePushCallback(tokenizer, repo_name, HF_TOKEN, experiment_name)]
    )

    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith(experiment_name)]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
            trainer.train(resume_from_checkpoint=os.path.join(output_dir, latest))
        else:
            trainer.train()
    else:
        trainer.train()

    final_dir = f"{output_dir}/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    try:
        api = HfApi(token=HF_TOKEN)
        api.upload_folder(
            folder_path=final_dir,
            repo_id=repo_name,
            path_in_repo=f"{experiment_name}-final",
            repo_type="model"
        )
    except Exception as e:
        pass

    eval_results = trainer.evaluate()

    return eval_results, trainer

"""
Kích thước input/output (256, 64)
Batch size hiệu dụng (16) nhờ gradient accumulation
Tốc độ học 3e-5
3 epoch và warm-up 100 bước
Evaluate và save mỗi epoch
Tính Rouge để chọn mô hình tốt nhất
Bật FP16 để giảm bộ nhớ và tăng tốc độ
"""

### Huấn luyện trên tập dữ liệu nguyên mẫu:
baseline_results, baseline_trainer = train_model(
    experiment_name="baseline",
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    repo_name="fcsn37/vit5-summarization-baseline"
)
wandb.finish()

### Huấn luyện trên tập dữ liệu kết hợp:
original = load_dataset("fcsn37/vietnamese-text-summarization-30k")

synthetic = load_dataset("fcsn37/vietnamese-text-summarization-synthetic-dataset")

combined = concatenate_datasets([
    original["train"],
    synthetic["train"]
])

combined_tokenized = combined.map(
    preprocess_function,
    batched=True,
    remove_columns=combined.column_names,
    num_proc=4
)

augmented_results, augmented_trainer = train_model(
    experiment_name="augmented-final",
    train_dataset=combined_tokenized,
    eval_dataset=tokenized_datasets["validation"],
    repo_name="fcsn37/vit5-vietnamese-summarization-augmented-final"
)
wandb.finish()

### Kiểm thử trên tập test:
login(token="HF_TOKEN")
# model = AutoModelForSeq2SeqLM.from_pretrained("./results/augmented-final/final")
# tokenizer = AutoTokenizer.from_pretrained("./results/augmented-final/final")
# model.push_to_hub("fcsn37/vit5-vietnamese-summarization-final")
# tokenizer.push_to_hub("fcsn37/vit5-vietnamese-summarization-final")
login(token="HF_TOKEN")

MODEL_REPO = "fcsn37/vit5-vietnamese-summarization-final"
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

test_dataset = load_dataset("fcsn37/vietnamese-text-summarization-30k", split="test")
test_samples = test_dataset.shuffle(seed=42).select(range(20))
predictions = []
references = []
original_texts = []

summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

for i, example in enumerate(test_samples):
    pred = summarizer(example["text"], max_length=128, min_length=30, truncation=True)[0]["summary_text"]
    predictions.append(pred)
    references.append(example["summary"])
    original_texts.append(example["text"])

    if (i + 1) % 10 == 0 or i < 5:
        print(f"  → Đã xử lý: {i+1}/20")

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
r1 = [scorer.score(ref, pred)["rouge1"].fmeasure for ref, pred in zip(references, predictions)]
r2 = [scorer.score(ref, pred)["rouge2"].fmeasure for ref, pred in zip(references, predictions)]
rl = [scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(references, predictions)]

print(f"ROUGE-1  : {np.mean(r1)*100:5.2f}%")
print(f"ROUGE-2  : {np.mean(r2)*100:5.2f}%")
print(f"ROUGE-L  : {np.mean(rl)*100:5.2f}%")

"""
Nhận xét:
- Mức ROUGE-1 > 50%: mô hình nhận diện được hầu hết từ quan trọng.
- ROUGE-2 thấp hơn nhiều: mô hình cần cải thiện khả năng kết nối từ thành câu hợp lý.
- ROUGE-L trung bình: mô hình tóm tắt chưa hoàn toàn giữ được cấu trúc và dòng chảy của tóm tắt gốc.
"""
N_DISPLAY = 3
for i in range(N_DISPLAY):
    print(f"MẪU {i+1}")
    print(f"VĂN BẢN GỐC (rút gọn):")
    text = original_texts[i]
    print(text[:500] + "..." if len(text) > 500 else text)

    print(f"TÓM TẮT THAM CHIẾU:")
    print(references[i])

    print(f"TÓM TẮT MÔ HÌNH SINH RA:")
    print(predictions[i]+"\n")

data = []
for i in range(20):
    orig = test_samples[i]["text"]
    short_orig = orig[:600] + " [...]" if len(orig) > 600 else orig

    data.append({
        "STT": i+1,
        "Văn bản gốc": short_orig,
        "Tóm tắt tham chiếu (Reference)": references[i],
        "Tóm tắt mô hình sinh ra (Prediction)": predictions[i],
        "ROUGE-1": round(r1[i], 4),
        "ROUGE-2": round(r2[i], 4),
        "ROUGE-L": round(rl[i], 4),
        "Gemini_Coherence": "",
        "Gemini_Relevance": "",
        "Gemini_Fluency": "",
        "Gemini_Factuality": "",
        "Gemini_Conciseness": "",
        "Gemini_Overall": "",
        "Gemini_Explanation": ""
    })
df = pd.DataFrame(data)
df.to_csv("20_ketqua.csv", index=False, encoding="utf-8-sig")
files.download("20_ketqua.csv")

### Kiểm thử trực tiếp: Nội dung được thể hiện cụ thể thông qua đường dẫn: https://colab.research.google.com/drive/1NQ3kWLSxm0j9xkvRDRFUgtjZTpGipHyW

import ipywidgets as widgets
from IPython.display import display, clear_output
text_input = widgets.Textarea(
    value='',
    placeholder='Nhập hoặc dán văn bản tiếng Việt cần tóm tắt...',
    layout=widgets.Layout(width='100%', height='200px')
)

# Nút bấm
button = widgets.Button(description="Tóm tắt")

# Ô kết quả
output = widgets.Output()

def on_button_click(b):
    with output:
        clear_output()
        text = text_input.value.strip()
        if not text:
            print("Văn bản trống.")
            return

        print("Đang xử lý...")
        with torch.no_grad():
            result = summarizer(
                text,
                max_new_tokens=128,
                min_length=30,
                truncation=True,
                do_sample=False
            )
        print("\nTóm tắt:\n")
        print(result[0]["summary_text"])

button.on_click(on_button_click)

# Hiển thị
display(text_input)
display(button)
display(output)

## Đánh giá và phân tích:
"""
[Kết quả đánh giá trực tiếp](https://gemini.google.com/share/1c1f7d13ae70)
"""
