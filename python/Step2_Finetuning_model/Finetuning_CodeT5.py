import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
import pandas as pd
import math
import json
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
import time

# 1. 모델 및 토크나이저 불러오기 (로컬 저장 불필요)
model_name = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. 추가할 special tokens 정의
special_tokens = {
    "additional_special_tokens": ["<bug1>", "<bug2>", "<context>", "</context>", "<bug3>", "<bug4>", "<bug5>", "<bug6>", "<bug7>","<bug8>","<bug9>","<bug10>",
                                  "<bug11>","<bug12>","<bug13>","<bug14>","<bug15>","<bug16>","<bug17>","<bug18>","<bug19>","<bug20>", "</bug>", "<bug>",
                                  "<bug21>","<bug22>","<bug23>","<bug24>","<bug25>","<bug26>","<bug27>","<bug28>","<bug29>","<bug30>","<bug31>","<bug32>"]
}

# 3. 토크나이저에 special tokens 추가
num_added_tokens = tokenizer.add_special_tokens(special_tokens)

# 4. 모델 임베딩 사이즈 재조정
model.resize_token_embeddings(len(tokenizer))
embedding_weights = model.get_input_embeddings().weight
print("Embedding shape:", embedding_weights.shape)

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
json_path = "to/your/path"
pf_pairs = collect_all_pf_pairs_from_directory(json_path)

# 5. buggy/fixed 코드 추출 및 필터링
processed_data = []
before_count = 0
after_count = 0

for pair in pf_pairs:
    buggy_code = pair["buggy_code"]
    fixed_code = pair["fixed_code"]

    before_count += 1

    # <bug1> 같은 특정 태그 조건 필터링 (필요 시 커스터마이즈)
    if isinstance(fixed_code, str) and re.match(r'^\s*<bug1>', fixed_code):
        processed_data.append({
            "input": buggy_code,
            "target": fixed_code
        })
        after_count += 1

    # # 그냥 일반 쌍 이코드 사용
    # processed_data.append({
    #     "input": buggy_code,
    #     "target": fixed_code
    # })

#데이터 쌍 개수 확인
print(f"Before filtering: {before_count} bug-fix pairs")
print(f"After filtering: {after_count} bug-fix pairs")

# 6. Pandas DataFrame으로 변환
df = pd.DataFrame(processed_data)

# 7. Hugging Face Dataset으로 변환
dataset = Dataset.from_pandas(df)

# 8. 전처리 함수
def preprocess_function(examples):
    prefix = "fix: "
    inputs = [prefix + ex for ex in examples["input"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target"], max_length=512, truncation=True, padding="max_length")["input_ids"]

    labels = [[l if l != tokenizer.pad_token_id else -100 for l in label] for label in labels]
    model_inputs["labels"] = labels
    return model_inputs

# 6. 전처리 + 분할
split_dataset = dataset.train_test_split(test_size=0.2)
tokenized_dataset = split_dataset.map(preprocess_function, batched=True)
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]


# 🔧 여기만 바꿔주세요
dataset_size = len(train_dataset)  # 예: 7000개든 50만 개든 자동 계산됨
batch_size = 16
accum_steps = 2
# epochs = 3

training_args = Seq2SeqTrainingArguments(
    output_dir="./Lightweight-main/python/output",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=accum_steps,
    # ✅ CodeT5 기본 하이퍼파라미터 적용
    learning_rate=3e-4,
    weight_decay=0.01,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,

    warmup_steps=int(10000 * 0.06),  # 직접 지정 or 계산
    max_steps=10000,  # ✅ 고정 step 수 20000/10000
    evaluation_strategy="steps",
    eval_steps=1000,  # ✅ 2000/1000 step마다 평가
    save_strategy="steps",
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=1,

    logging_steps=500,
    logging_dir="./Lightweight-main/python/output/logs",
    fp16=True,
    predict_with_generate=False,
    disable_tqdm=False,
    report_to="none"
)

# 10. Trainer 정의 및 학습 실행
class QuietEvalTrainer(Seq2SeqTrainer):
    def evaluation_loop(self, dataloader=None, description="Evaluation", prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        # 👇 평가 중에는 tqdm 비활성화
        original_tqdm_state = self.args.disable_tqdm
        self.args.disable_tqdm = True

        result = super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )

        # 👇 평가 끝나면 원래대로 복원
        self.args.disable_tqdm = original_tqdm_state
        return result
    
class ProgressPrinterCallback(TrainerCallback):
    def __init__(self, total_steps):
        self.start_time = None
        self.total_steps = total_steps

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"🚀 Training started. Total steps: {self.total_steps}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.start_time is None:
            return
        current_step = state.global_step
        elapsed = time.time() - self.start_time
        progress = current_step / self.total_steps
        eta = elapsed / progress - elapsed if progress > 0 else 0
        print(f"🔄 Step {current_step}/{self.total_steps} "
              f"({progress*100:.1f}%) | "
              f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

# ✅ 그 아래 trainer 정의 시에만 바꿔주면 끝!
trainer = QuietEvalTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    callbacks=[ProgressPrinterCallback(total_steps=training_args.max_steps)]
)

# 현재 프로세스가 사용 중인 GPU 정보 출력
print(f"🖥️  Rank {torch.cuda.current_device()} | GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

trainer.train()

# 11. 마지막 모델 저장
timestamp = time.strftime("%m%d-%H%M")
save_path = f"./Lightweight-main/python/output/checkpoint-last-{timestamp}"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

with open(f"./Lightweight-main/python/output/training_log_{timestamp}.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)