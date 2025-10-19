import os
import torch
import re
from transformers import TrainingArguments, default_data_collator
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import Trainer
from datasets import Dataset
import pandas as pd
import math
import json
import time
from transformers import TrainerCallback
from tqdm.auto import tqdm
import inspect
from transformers import TrainingArguments

# 1. 모델과 토크나이저 로드
token = "your/token/info"  # Hugging Face에서 발급받은 토큰

model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-hf",
    token=token,
    torch_dtype=torch.float16,           # ✅ fp16
    device_map="auto"                    # ✅ 자동 GPU 할당
)

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf",use_fast=False, token=token)
tokenizer.pad_token = tokenizer.eos_token

# 2. LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    # ["q_proj", "v_proj"]
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)



# 4. 특수 토큰 정의
# special_tokens = {"additional_special_tokens": ["<bug>", "</bug>", "<context>", "</context>", "<fix>", "</fix>"]}
special_tokens = {"additional_special_tokens": ["[bug_function], [reference_function], [fix_code]"]}

# 5. 토크나이저에 추가 / 모델 임베딩 사이즈 조정
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))
embedding_weights = model.get_input_embeddings().weight
print("Embedding shape:", embedding_weights.shape)

# 3. LoRA 적용
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# def is_deletion_case(buggy_code: str, fixed_code: str) -> bool:
#     """
#     fixed code가 // fix_start // fix_end 인 경우 찾기
#     """
#     bug_block = extract_bug_block(buggy_code)
#     buggy_tokens = set(bug_block.strip().split())
#     fixed_tokens = set(fixed_code.strip().split())

#     deleted_tokens = buggy_tokens - fixed_tokens
#     inserted_or_modified = fixed_tokens - buggy_tokens

#     return len(deleted_tokens) > 0 and len(inserted_or_modified) == 0

def load_all_pf_pairs_from_dict(data, parent_key=""):
    pf_pairs = []

    if isinstance(data, list):
        top_level_items = enumerate(data)
    elif isinstance(data, dict):
        top_level_items = data.items()
    else:
        return pf_pairs

    for key, entry in top_level_items:
        if not isinstance(entry, dict):
            continue

        p_dict = {}
        f_dict = {}

        for file_key, value in entry.items():
            if file_key.endswith("_P"):
                base = file_key[:-2]
                p_dict[base] = value
            elif file_key.endswith("_F"):
                base = file_key[:-2]
                f_dict[base] = value

        for base in set(p_dict.keys()) & set(f_dict.keys()):
            pf_pairs.append({
                "source_file": str(parent_key),
                "group": str(key),
                "base": base,
                "buggy_code": p_dict[base],
                "fixed_code": f_dict[base]
            })

    return pf_pairs

def collect_all_pf_pairs_from_directory(directory_path):
    all_pairs = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pairs = load_all_pf_pairs_from_dict(data, parent_key=filename)
                all_pairs.extend(pairs)
            except Exception as e:
                print(f"❌ 파일 '{filename}' 처리 중 오류 발생: {e}")

    return all_pairs

# 정확한 파일 경로 (MyDrive 기준)
json_path = "/home/selab/Desktop/Dataset/Dataset_ver2"
pf_pairs = collect_all_pf_pairs_from_directory(json_path)

# 5. buggy/fixed 코드 추출 및 필터링
processed_data = []
before_count = 0
deletion_samples = []

# prefix_prompt = "Fix each <bug> block. Use <context> only for reference.\n"
    
for pair in pf_pairs:
    buggy_code = pair["buggy_code"]
    fixed_code = pair["fixed_code"]
    before_count += 1

    # 공통 데이터는 항상 포함
    sample = {
        "input": buggy_code,
        "target": fixed_code
    }
    processed_data.append(sample)
    
    # 비어 있는 <fix> </fix> 만 있는 경우 수집
    marker = "\n// fix_start\n \n// fix_end\n"
    if marker in fixed_code:
        deletion_samples.append(sample)
        
# ⏫ oversampling: 삭제 케이스 3배 추가
processed_data.extend(deletion_samples * 7)  # 총 8배 (1 + 7)

# 2. 통계 출력
print("총 샘플 수:", len(processed_data))
print("삭제 케이스 수:", len(deletion_samples))
print(f"최종 학습 샘플 수 (oversampled): {len(processed_data)}")

# 전체 카운트 출력
# print(f"\n전체 bug-fix 쌍 수 (원본): {before_count}")
# print(f"삭제 전용 쌍 수: {len(deletion_samples)}")
# print(f"최종 학습 샘플 수 (oversampled): {len(processed_data)}")

# #데이터 쌍 개수 확인
# print(f"Before filtering: {before_count} bug-fix pairs")
# print(f"After filtering: {after_count} bug-fix pairs")

# 6. Pandas DataFrame으로 변환
df = pd.DataFrame(processed_data)

# 7. Hugging Face Dataset으로 변환
dataset = Dataset.from_pandas(df)

# 8. 전처리 함수
def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["target"]

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for input_text, target_text in zip(inputs, targets):
        prefix = ("\n[bug_function]\n" + input_text + "\n[fix_code]\n")
        full_text = prefix +  target_text

        # 전체 시퀀스 토크나이즈
        tokenized_full = tokenizer(
            full_text,
            max_length=2048,
            truncation=True,
            padding="max_length",
            add_special_tokens=True
        )

        # input 길이 계산
        tokenized_input = tokenizer(
            prefix,
            truncation=True,
            max_length=1024,
            padding=False,
            add_special_tokens=True
        )
        input_len = len(tokenized_input["input_ids"])

        # labels 구성
        labels = tokenized_full["input_ids"].copy()
        labels[:input_len] = [-100] * input_len
        labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]

        # 결과 리스트에 추가
        input_ids_list.append(tokenized_full["input_ids"])
        attention_mask_list.append(tokenized_full["attention_mask"])
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }

# 9. 전처리 + 분할
split_dataset = dataset.train_test_split(test_size=0.2)
tokenized_dataset = split_dataset.map(preprocess_function, batched=True)
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]

# 학습설정
# print(TrainingArguments)
training_args = TrainingArguments(
    output_dir="./codellama-lora-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    warmup_steps=200,
    num_train_epochs=2,
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    fp16=True,  # GPU가 AMP 지원할 경우
    logging_steps=1000,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    disable_tqdm=False,
    remove_unused_columns=False
)

# Data Collator 구성
data_collator = default_data_collator
    
class TqdmProgressCallback(TrainerCallback):
    def __init__(self):
        self.pbar = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.pbar = tqdm(total=state.max_steps, desc="Training", position=0)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.pbar and logs is not None:
            desc = f"Step {state.global_step} | "

            if "loss" in logs:
                desc += f"Loss: {logs['loss']:.4f} | "

            if "eval_loss" in logs:
                desc += f"Eval Loss: {logs['eval_loss']:.4f} | "
                # Perplexity 계산
                try:
                    ppl = math.exp(logs["eval_loss"])
                    desc += f"PPL: {ppl:.2f}"
                except OverflowError:
                    desc += "PPL: inf"

            if "epoch" in logs:
                desc += f" | Epoch: {logs['epoch']:.2f}"

            self.pbar.set_description_str(desc)

    def on_step_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()
    
# Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[TqdmProgressCallback()]
)


# 현재 프로세스가 사용 중인 GPU 정보 출력
print(f"🖥️  Rank {torch.cuda.current_device()} | GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

trainer.train()

# 11. 마지막 모델 저장
timestamp = time.strftime("%m%d-%H%M")
save_path = f"/home/selab/Desktop/Lightweight-main/output/checkpoint-last-{timestamp}"
model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)
