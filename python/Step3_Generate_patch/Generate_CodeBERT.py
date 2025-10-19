import os
import fnmatch 
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from itertools import chain
from model import Seq2Seq  # 너의 model.py에 정의된 클래스

# ====== 모델 및 토크나이저 불러오기 ======
def load_model_and_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<bug>", "</bug>", "<context>", "</context>"]})
    # tokenizer.add_tokens(["<bug>", "</bug>", "<context>", "</context>", "<fix>", "</fix>"])

    config = RobertaConfig.from_pretrained("microsoft/codebert-base")
    encoder = RobertaModel.from_pretrained("microsoft/codebert-base", config=config)
    encoder.resize_token_embeddings(len(tokenizer))

    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        config=config,
        beam_size=100,  # ✅ beam size 100으로 설정
        max_length=512,
        sos_id=tokenizer.cls_token_id,
        eos_id=tokenizer.sep_token_id,
    )

    model.load_state_dict(torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tokenizer

# ====== 예측 함수 ======
def generate_single_prediction(text, model, tokenizer, device):
    tokens = tokenizer.tokenize(text)[:510]
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(tokens)
    source_mask = [1] * len(source_ids)

    padding_length = 512 - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    source_mask += [0] * padding_length

    source_ids = torch.tensor([source_ids], dtype=torch.long).to(device)
    source_mask = torch.tensor([source_mask], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(source_ids=source_ids, source_mask=source_mask)

    # Beam 결과가 하나의 텐서로 묶여 있으면 outputs: [beam_size, max_length]
    # 리스트 안에 텐서들이 있는 구조면 그대로 사용
    if isinstance(outputs, list):
        outputs = torch.stack(outputs, dim=0)  # 리스트를 텐서로 변환

    predictions = []
    for i in range(outputs.size(0)):
        tokens = outputs[i].cpu().numpy().tolist()
        if isinstance(tokens[0], list):  # ✅ 리스트 안 리스트 처리
            tokens = list(chain.from_iterable(tokens))
        if 0 in tokens:
            tokens = tokens[:tokens.index(0)]
        decoded = tokenizer.decode(tokens, clean_up_tokenization_spaces=True).strip()
        predictions.append(decoded)

    return predictions[:]  # ✅ beam 후보 전체 리스트 반환

# ====== JSON 처리 및 예측 저장 ======
def process_json_file(json_path, model, tokenizer):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folder_key = next(iter(data.keys()))
        items = data[folder_key]
        global_id = 1

        for item in tqdm(items, desc=f"Generating lwcp for {folder_key}"):
            lwbm_input = item["lwbm"]

            pred_list = generate_single_prediction(lwbm_input, model, tokenizer, device)
            unique_predictions = sorted(set(pred_list))[:100]  # 중복 제거
            item["lwcp"] = {f"lwcp{global_id + i}": p for i, p in enumerate(unique_predictions)}
            global_id += 100

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ {json_path} 처리 완료. 모든 항목에 lwcp1~100 저장됨.")
        return True

    except Exception as e:
        print(f"❌ {json_path} 처리 중 오류 발생: {e}")
        return False

# # ====== 전체 폴더 내 JSON 순회 처리 ======
# def process_all_json_in_base_dir(base_dir, model, tokenizer):
#     failed_folders = []

#     for mid_folder in sorted(os.listdir(base_dir)):
#         mid_folder_path = os.path.join(base_dir, mid_folder)
        model_emb = encoder.get_input_embeddings()
        with torch.no_grad():
            # Xavier Uniform initialization ensures better weight distribution
            model_emb.weight[-num_added_tokens:] = torch.nn.init.xavier_uniform_(torch.empty(num_added_tokens, config.hidden_size))

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
    target_folders = ["DiffBERT"]

    for project_folder in sorted(os.listdir(base_dir)):
        if project_folder not in target_folders:
        # if project_folder != "Lightweight":  # ✅ Lightweight 폴더만 처리
        # if project_folder == "Lightweight":  # ✅ Lightweight 폴더만 제외
            print(f"🚫 {project_folder} 폴더는 스킵됨.")
            continue

        project_path = os.path.join(base_dir, project_folder)
        if not os.path.isdir(project_path):
            continue

        for case_folder in sorted(os.listdir(project_path)):
            case_path = os.path.join(project_path, case_folder)
            if not os.path.isdir(case_path):
                continue

            # ✅ 모든 접두어를 허용하는 JSON 패턴
            json_candidates = [f for f in os.listdir(case_path) 
                               if fnmatch.fnmatch(f, "*Lightweight_method_BERT_diff*.json")]
                            #    if fnmatch.fnmatch(f, "*buggy_and_context_method*.json") and "diff" not in f]

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
model_path = "./Lightweight-main/output/checkpoint-last-0316-1124/pytorch_model_0316-1124.bin"
base_dir = "to/your/path"

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
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
# base_path = "to/your/path"  # 여러 폴더들이 있는 기본 경로
# # target_folders = ["Closure_65", "Chart_4", "Chart_26","Closure_66","Closure_119", "Chart_13", "Closure_71", "Closure_123","Lang_16","Lang_58", "Math_24"]  # 🔥 처리하고 싶은 폴더들 리스트
# target_folders = ["Math_38","Closure_65"] 
# target_keyword = "Lightweight_buggy_method_Context_diff"

# if __name__ == "__main__":
#     model, tokenizer = load_model_and_tokenizer()
#     # process_multiple_folders(base_path, target_folders, model, tokenizer, target_filename)
#     process_multiple_folders(base_path, target_folders, model, tokenizer, target_keyword)
#     # process_specific_json_in_target_folder(target_folders, model, tokenizer, target_filename)