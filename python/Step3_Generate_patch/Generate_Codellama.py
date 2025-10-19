import os
import re
import fnmatch 
import json
import torch
import torch.nn as nn
from itertools import chain
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel


# ====== 모델 및 토크나이저 불러오기 ======
def load_codellama_lora_model(model_path):
    try:
        # ✅ tokenizer 설정
        tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLLaMA-7b-hf", use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"additional_special_tokens": ["[bug_function], [reference_function], [fix_code]"]})
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,  # 필요 시 설정
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            "codellama/CodeLLaMA-7b-hf",
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quant_config,
            trust_remote_code=True
        )
        base_model.resize_token_embeddings(len(tokenizer))

        # ✅ LoRA 어댑터 적용
        model = PeftModel.from_pretrained(base_model, model_path, torch_dtype=torch.float16)
        # model.resize_token_embeddings(len(tokenizer))
        model.eval()

        # ✅ 디바이스 정보 출력
        device = next(model.parameters()).device
        print(f"✅ 모델 주요 파라미터가 올라간 디바이스: {device}")
        if torch.cuda.is_available():
            print(f"🎯 GPU 이름: {torch.cuda.get_device_name(device.index)}")

        return model, tokenizer

    except Exception as e:
        print(f"❌ 모델 로딩 중 오류 발생: {e}")
        return None, None

# ====== 예측 함수 ======
# def trim_after_bug(pred: str, lwbm: str) -> str:
#     """
#     예측 결과에서 <bug>...</bug> 이후 <context> 이전까지만 남김.
#     """
#     # </bug> 와 <context> 사이 텍스트 추출
#     post_bug_match = re.search(r"</bug>(.*?)<context>", lwbm, re.DOTALL)
#     if not post_bug_match:
#         return pred.strip()

#     post_bug_code = post_bug_match.group(1).strip()
    
#     if not post_bug_code:
#         # 사이에 아무 코드 없으면 잘라낼 게 없음
#         return pred.strip()

#     if post_bug_code in pred:
#         pred = pred.split(post_bug_code, 1)[0] + post_bug_code

#     return pred.strip()

def generate_single_prediction(pair, model, tokenizer, device, max_new_tokens=512, total=10, chunk_size=10, diversity_penalty=1.3):
    # prefix_prompt = ""
    buggy_code = pair["lwbm"]
    
    bug_end_marker = "\n// bug_end\n"
    fix_end_marker = "\n// fix_end\n"
    bug_end_count = buggy_code.count(bug_end_marker)
    
    input_text = "\n[bug_function]\n" + buggy_code + "\n[fix_code]\n"# ✅ prefix 포함한 입력
    all_predictions = set()

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    inputs_len = input_ids.shape[1] 

    num_iterations = total // chunk_size

    with torch.no_grad():
        for _ in range(num_iterations):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # ↓ 여기부터 추가된 옵션
                num_beams=chunk_size,
                num_beam_groups=chunk_size,
                diversity_penalty=diversity_penalty,
                num_return_sequences=chunk_size,
                # ↑ 추가 끝
                # num_beams=chunk_size,                # 🔁 빔의 개수 설정 (예: 5개 빔)
                # num_return_sequences=chunk_size,     # 🔁 최종으로 반환할 시퀀스 개수 (ex. top 10)
                early_stopping=True,         # ✔️ 빔 모두 EOS 만나면 멈춤
                # early_stopping=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False              # 🔁 Beam search는 sampling 꺼야 안정적
                )
            
            # outputs = model.generate(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     do_sample=True,
            #     top_k=50,
            #     top_p=0.9,
            #     temperature=0.5,
            #     max_new_tokens=max_new_tokens,
            #     num_return_sequences=chunk_size,
            #     pad_token_id=tokenizer.pad_token_id,
            #     eos_token_id=tokenizer.eos_token_id
            # )
            
            output_ids = outputs[:, inputs_len:]
            decoded_outputs = tokenizer.batch_decode(
                output_ids,
                # outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # for pred in decoded_outputs:
            #     # clean_pred = trim_after_bug(pred, buggy_code)
            #     all_predictions.add(pred.strip())

            for pred in decoded_outputs:
                trimmed = pred.strip()
                fix_count = trimmed.count(fix_end_marker)
                
                if fix_count > bug_end_count > 0:
                    parts = trimmed.split(fix_end_marker)
                    allowed = parts[:bug_end_count]
                    trimmed = fix_end_marker.join(allowed) + fix_end_marker
                elif fix_count > 0 and bug_end_count == 0 :
                    trimmed = trimmed.split(fix_end_marker)[0]
                
                all_predictions.add(trimmed)
            
            torch.cuda.empty_cache()

    return sorted(all_predictions)[:total]

# ====== JSON 처리 및 예측 저장 ======
def process_json_file(json_path, model, tokenizer):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folder_key = next(iter(data.keys()))
        items = data[folder_key]
        lwcp_counter = 1  # ✅ 전체 JSON 기준으로 순차 증가할 번호
        
        # items = [i for i in items if i.get("id") == "1_50%"]
        for item in tqdm(items, desc=f"Generating lwcp for {folder_key}"):
            pred_list = generate_single_prediction(item, model, tokenizer, device)
            unique_predictions = sorted(set(pred_list))[:100]

            item["lwcp"] = {}

            if unique_predictions:
                for p in unique_predictions:
                    item["lwcp"][f"lwcp{lwcp_counter}"] = p
                    lwcp_counter += 1
            else:
                # 예측이 아무것도 없을 경우에도 lwcp1 넣기
                item["lwcp"][f"lwcp{lwcp_counter}"] = "empty"
                lwcp_counter += 1

        # ✅ 새 파일명 생성
        new_filename = os.path.basename(json_path).replace(".json", "_codellama.json")
        new_path = os.path.join(os.path.dirname(json_path), new_filename)

        with open(new_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ 저장 완료: {new_path}")
        return True

    except Exception as e:
        print(f"❌ {json_path} 처리 중 오류 발생: {e}")
        return False

# def process_all_txt_in_base_dir(base_dir, model, tokenizer):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     failed_files = []

#     for project_folder in sorted(os.listdir(base_dir)):
#         # ✅ Lightweight 폴더만 처리하고 싶다면:
#         if project_folder != "Token300":
#             print(f"🚫 {project_folder} 폴더는 스킵됨.")
#             continue

#         project_path = os.path.join(base_dir, project_folder)
#         if not os.path.isdir(project_path):
#             continue

#         for case_folder in sorted(os.listdir(project_path)):
#             case_path = os.path.join(project_path, case_folder)
#             if not os.path.isdir(case_path):
#                 continue

#             txt_candidates = [f for f in os.listdir(case_path)
#                               if ('Original_buggy_method' in f and 'Original_buggy_method_by_line' not in f and f.endswith('.txt'))]

#             if not txt_candidates:
#                 print(f"⚠️ {case_path} 내 처리할 TXT 없음.")
#                 continue

#             for txt_file in txt_candidates:
#                 txt_path = os.path.join(case_path, txt_file)
#                 try:
#                     generate_patch_for_txt_file(txt_path, model, tokenizer, device)
#                 except Exception as e:
#                     print(f"❌ 실패: {txt_path} | 에러: {e}")
#                     failed_files.append(txt_path)

#     if failed_files:
#         print("\n❗ 처리 실패한 파일 목록:")
#         for f in failed_files:
#             print("  -", f)
#     else:
#         print("\n🎉 모든 TXT 파일 처리 완료!")


# # ====== 설정 ======
# model_path = "./Lightweight-main/output/checkpoint-last-0316-1124/pytorch_model_0316-1124.bin"
# base_dir = "to/your/path"

# if __name__ == "__main__":
#     model, tokenizer = load_model_and_tokenizer()
#     process_all_txt_in_base_dir(base_dir, model, tokenizer)


#================== Original_buggy_method.txt 파일 500개 생성==================
# def generate_patch_for_txt_file(txt_path, model, tokenizer, device):
#     """txt 파일의 첫 줄을 읽고 모델 예측 결과 100개를 candidate_patch*.txt에 저장"""
#     with open(txt_path, 'r', encoding='utf-8') as f:
#         first_line = f.readline().strip()

#     if not first_line:
#         print(f"⚠️ {txt_path} 첫 줄이 비어 있음. 건너뜀.")
#         return

#     pred_list = generate_single_prediction(first_line, model, tokenizer, device)
#     unique_predictions = sorted(set(pred_list))[:500]

#     # 출력 파일명 구성
#     filename = os.path.basename(txt_path)
#     patch_filename = filename.replace("Original_buggy_method", "candidate_patch_codellama")
#     patch_path = os.path.join(os.path.dirname(txt_path), patch_filename)

#     with open(patch_path, 'w', encoding='utf-8') as f:
#         for pred in unique_predictions:
#             f.write(pred + '\n')

#     print(f"✅ {patch_path} 저장 완료 ({len(unique_predictions)}개 패치)")



# # ====== 전체 폴더 내 JSON 순회 처리 ======
# def process_all_json_in_base_dir(base_dir, model, tokenizer):
#     failed_folders = []

#     for mid_folder in sorted(os.listdir(base_dir)):
#         mid_folder_path = os.path.join(base_dir, mid_folder)
#         if not os.path.isdir(mid_folder_path):
#             continue

#         for chart_folder in sorted(os.listdir(mid_folder_path)):
#             chart_folder_path = os.path.join(mid_folder_path, chart_folder)
#             if not os.path.isdir(chart_folder_path):
#                 continue

#             json_files = [f for f in os.listdir(chart_folder_path) if f.endswith(".json")]
#             if not json_files:
#                 print(f"⚠️ {chart_folder_path} 내 JSON 파일 없음. 건너뜀.")
#                 continue

#             for json_file in json_files:
#                 json_path = os.path.join(chart_folder_path, json_file)
#                 success = process_json_file(json_path, model, tokenizer)
#                 if not success:
#                     failed_folders.append(chart_folder_path)

#     if failed_folders:
#         print("\n❗ 처리 실패한 폴더 목록:")
#         for folder in failed_folders:
#             print("  -", folder)
#     else:
#         print("\n🎉 모든 JSON 파일 처리 완료!")

# ====== 전체 폴더 내 특정 JSON파일만 순회 처리 ======
def process_all_json_in_base_dir(base_dir, model, tokenizer):
    failed_folders = []

    for project_folder in sorted(os.listdir(base_dir)):
        if project_folder != "Diffllama":  # ✅ Lightweight 폴더만 처리
        # if project_folder in ["Codellama", "multi"]:
        # if project_folder = "Lightweight":  # ✅ Lightweight 폴더만 제외
            print(f"🚫 {project_folder} 폴더는 스킵됨.")
            continue

        project_path = os.path.join(base_dir, project_folder)
        if not os.path.isdir(project_path):
            continue

        for case_folder in sorted(os.listdir(project_path)):
            case_path = os.path.join(project_path, case_folder)
            if not os.path.isdir(case_path):
                continue

            json_candidates = [f for f in os.listdir(case_path)
                            if fnmatch.fnmatch(f, "*Lightweight_method_llama_diff*.json")
                            and "codellama" not in f]

            # json_candidates = [f for f in os.listdir(case_path)
            #     if f == "Lightweight_buggy_method_Context.json"]

            if not json_candidates:
                print(f"⚠️ {case_path} 내 대상 JSON 없음. 건너뜀.")
                continue

            for json_filename in json_candidates:
                json_path = os.path.join(case_path, json_filename)
                success = process_json_file(json_path, model, tokenizer)
                if not success:
                    failed_folders.append(case_path)

    if failed_folders:
        print("\n❗ 처리 실패한 폴더 목록:")
        for folder in failed_folders:
            print("  -", folder)
    else:
        print("\n🎉 모든 JSON 파일 처리 완료!")


# ====== 설정 ======
model_path = "./Lightweight-main/output/checkpoint-last-0723-2121"
base_dir = "to/your/path"

if __name__ == "__main__":
    model, tokenizer = load_codellama_lora_model(model_path)
    process_all_json_in_base_dir(base_dir, model, tokenizer)


# ====== 단일 폴더 내 JSON 순회 처리 ======
# def process_all_json_in_target_folder(target_folder, model, tokenizer):
#     failed_files = []

#     if not os.path.isdir(target_folder):
#         print(f"❗ 유효하지 않은 폴더 경로입니다: {target_folder}")
#         return

#     json_files = [f for f in os.listdir(target_folder) if f.endswith(".json")]
#     if not json_files:
#         print(f"⚠️ {target_folder} 내 JSON 파일 없음. 건너뜀.")
#         return

#     for json_file in json_files:
#         json_path = os.path.join(target_folder, json_file)
#         success = process_json_file(json_path, model, tokenizer)
#         if not success:
#             failed_files.append(json_file)

#     if failed_files:
#         print("\n❗ 처리 실패한 JSON 파일 목록:")
#         for file in failed_files:
#             print("  -", file)
#     else:
#         print(f"\n🎉 모든 JSON 파일 처리 완료! (총 {len(json_files)}개)")

# # ====== 설정 ======
# model_path = "./Lightweight-main/python/output/checkpoint-last-0316-1124/pytorch_model_0316-1124.bin"
# target_folder = "to/your/path"  # 🔁 여기를 원하는 폴더로 수정

# if __name__ == "__main__":
#     model, tokenizer = load_model_and_tokenizer()
#     process_all_json_in_target_folder(target_folder, model, tokenizer)


# ====== 단일 폴더 내 특정JSON파일만 순회 처리 ======
# def process_specific_json_in_target_folder(target_folder, model, tokenizer, target_keyword):
#     failed_files = []

#     if not os.path.isdir(target_folder):
#         print(f"❗ 유효하지 않은 폴더 경로입니다: {target_folder}")
#         return

#     # 폴더 내에서 target_keyword를 포함하고 .json 확장자인 모든 파일을 찾음
#     json_files = [
#         f for f in os.listdir(target_folder)
#         if f.endswith(".json") and target_keyword in f
#     ]

#     if not json_files:
#         print(f"⚠️ {target_folder} 내 '{target_keyword}' 포함된 JSON 없음. 건너뜀.")
#         return

#     for json_file in json_files:
#         json_path = os.path.join(target_folder, json_file)
#         success = process_json_file(json_path, model, tokenizer)
#         if not success:
#             failed_files.append(json_file)

#     if failed_files:
#         print("\n❗ 처리 실패한 JSON 파일 목록:")
#         for file in failed_files:
#             print("  -", file)
#     else:
#         print(f"\n🎉 '{target_keyword}' 포함된 모든 JSON 처리 완료!")

# def process_multiple_folders(base_path, folder_list, model, tokenizer, target_keyword):
#     for folder_name in folder_list:
#         target_folder = os.path.join(base_path, folder_name)
#         process_specific_json_in_target_folder(target_folder, model, tokenizer, target_keyword)

# # ====== 설정 ======
# model_path = "./Lightweight-main/python/output/checkpoint-last-0316-1124/pytorch_model_0316-1124.bin"
# base_path = "./Lightweight-main/result_defects4j/version1/Lightweight"  # 여러 폴더들이 있는 기본 경로
# # target_folders = ["Closure_65", "Chart_4", "Chart_26","Closure_66","Closure_119", "Chart_13", "Closure_71", "Closure_123","Lang_16","Lang_58", "Math_24"]  # 🔥 처리하고 싶은 폴더들 리스트
# target_folders = ["Math_38","Closure_65"] 
# target_keyword = "Lightweight_buggy_method_Context_diff"

# if __name__ == "__main__":
#     model, tokenizer = load_model_and_tokenizer()
#     # process_multiple_folders(base_path, target_folders, model, tokenizer, target_filename)
#     process_multiple_folders(base_path, target_folders, model, tokenizer, target_keyword)
#     # process_specific_json_in_target_folder(target_folders, model, tokenizer, target_filename)