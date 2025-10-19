from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import json
import re

# 1. 모델 및 토크나이저 로딩
model_path = "./Lightweight-main/python/output/checkpoint-last-0421-2157"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# 2. 추가할 special tokens 정의
special_tokens = {
    "additional_special_tokens": ["<bug1>", "<bug2>", "<context>", "</context>", "<bug3>", "<bug4>", "<bug5>", "<bug6>", "<bug7>","<bug8>","<bug9>","<bug10>",
                                  "<bug11>","<bug12>","<bug13>","<bug14>","<bug15>","<bug16>","<bug17>","<bug18>","<bug19>","<bug20>", "</bug>", "<bug>"]
}

# 3. 토크나이저에 special tokens 추가
num_added_tokens = tokenizer.add_special_tokens(special_tokens)

# 4. 모델 임베딩 사이즈 재조정
model.resize_token_embeddings(len(tokenizer))


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# 2. 예측 함수 (unique 500개 추출하기)
# def predict_unique_variants(input_code: str, desired_unique: int = 500, batch_size: int = 100, max_length: int = 512):
#     prefix = "fix: "
#     input_text = prefix + input_code
#     inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)

#     unique_outputs = set()
#     tries = 0

#     while len(unique_outputs) < desired_unique and tries < 20:  # 최대 20번 반복
#         outputs = model.generate(
#             **inputs,
#             do_sample=True,
#             top_k=50,
#             top_p=0.95,
#             temperature=0.9,
#             repetition_penalty=1.2,
#             num_return_sequences=batch_size,
#             max_length=max_length,
#         )
#         #모델 출력 디코딩 skip_special_tokens=False로 해야 <bug1>태그 붙어서 결과 출력
#         decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
#         # 보고 싶은 토큰만 남기기
#         allowed_tags = {f"<bug{i}>" for i in range(1, 33)} | {"<bug>", "</bug>"} 
#         cleaned = []

#         for line in decoded:
#             tokens = line.split()
#             filtered = [tok for tok in tokens if tok in allowed_tags or not tok.startswith("<")]
#             cleaned.append(" ".join(filtered))


#         unique_outputs.update([o.strip() for o in cleaned])
#         tries += 1
#         print(f"🔁 {tries}회차 생성 | 고유 결과 수: {len(unique_outputs)}")

#     return list(unique_outputs)[:desired_unique]  # 딱 500개만 잘라서 반환


##########################################################################################
def predict_unique_variants(input_code: str, batch_size: int = 100, max_length: int = 512):
    prefix = "fix: "
    input_text = prefix + input_code
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)

    all_cleaned = []

    for i in range(1):  # 🔁 5번만 반복
        outputs = model.generate(
            **inputs,
            do_sample=True,
            top_k=100,
            top_p=0.95,
            temperature=1.0,
            repetition_penalty=1.2,
            num_return_sequences=batch_size,
            max_length=max_length,
        )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False) #True토큰 제거, False 토큰 추가 
        filter_tags = True  # ✅ 여기서 <bugX> 태그 포함할지 결정 False토큰제거, True토큰 포함
        allowed_tags = {f"<bug{i}>" for i in range(1, 33)} 
        unwanted_tokens = {"<pad>", "<s>", "</s>", "<unk>"}
        cleaned = []

        for line in decoded:
            # 🔧 특수 토큰 먼저 제거
            for tok in unwanted_tokens:
                line = line.replace(tok, "")

            tokens = line.split()

            if filter_tags:
                # 허용된 태그 or 일반 토큰만 남기기
                filtered = [tok for tok in tokens if tok in allowed_tags or not tok.startswith("<")]
            else:
                filtered = tokens

            cleaned.append(" ".join(filtered))

        all_cleaned.extend(cleaned)
        print(f"🔁 {i+1}회차 생성 완료 | 누적 총 후보: {len(all_cleaned)}")

    # ✅ 중복 제거
    unique_outputs = list(dict.fromkeys([o.strip() for o in all_cleaned]))
    print(f"✅ 고유한 결과 개수: {len(unique_outputs)}")

    return unique_outputs

# 4. 입력 파일 경로
# base_dir = "to/your/path"
# input_path = os.path.join(base_dir, "Lightweight_buggy_method_Context.txt")
# output_path = os.path.join(base_dir, "lWCP.txt")

# with open(input_path, "r", encoding="utf-8") as f:
#     buggy_code = f.read()

# # 5. 실행
# predictions = predict_unique_variants(buggy_code)

# # 6. 결과 저장
# with open(output_path, "w", encoding="utf-8") as f:
#     for pred in predictions:
#         f.write(pred + "\n")

# print(f"✅ 총 {len(predictions)}개의 중복 제거된 출력이 저장되었습니다.")

def process_json_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    folder_key = next(iter(data.keys()))
    items = data[folder_key]

    global_lwcp_index = 1  # 전역 lwcp 번호 시작

    for item in items:
        lwbm = item["lwbm"]
        predictions = predict_unique_variants(lwbm)

        # 전역 번호로 lwcp 저장
        item["lwcp"] = {}
        for pred in sorted(predictions):
            item["lwcp"][f"lwcp{global_lwcp_index}"] = pred
            global_lwcp_index += 1

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ {json_path} 처리 완료. {len(items)}개의 항목에 총 {global_lwcp_index - 1}개의 lwcp 저장됨.")

def main():
    base_dir = "to/your/path"

    # Chart_* 폴더 순회
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):  # Chart 여부 제한 제거
            # 내부 JSON 파일 찾기
            for file_name in os.listdir(folder_path):
                if re.match(r"Lightweight_buggy_method_Context.*\.json$", file_name):
                    json_path = os.path.join(folder_path, file_name)
                    try:
                        process_json_file(json_path)
                    except Exception as e:
                        print(f"❌ {json_path} 처리 중 오류 발생: {e}")

if __name__ == "__main__":
    main()