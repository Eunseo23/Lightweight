import re
import json
import time
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# 모델 및 토크나이저 불러오기
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# 디바이스 정보 출력
if device == "cuda":
    print(f"✅ Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("⚠️ Using CPU")

# 연산자 및 기호 목록
operators_v1 = [r'==', r'!=', r'>=', r'<=', r'\+\+', r'--', r'\+=', r'-=', r'\*=', r'/=', r'%=', r'&=', r'\|=', r'\^=', r'>>=', r'<<=',
    r'=', r'<', r'>', r'\+', r'-', r'\*', r'/', r'%', r'&', r'\|', r'\^', r'~', r'!', r'>>', r'<<', r'\?', r':', r'\.']
code_delimiters = [r'\{', r'\}', r';', r'\(', r'\)', r'\.', r',']

def protect_bug_tags(text):
    text = re.sub(r'\s*<bug>\s*', '##BUGSTART##', text)
    text = re.sub(r'\s*</bug>\s*', '##BUGEND##', text)
    return text

def restore_bug_tags(text):
    text = text.replace('##BUGSTART##', '<bug> ').replace('##BUGEND##', ' </bug>')
    text = re.sub(r'<bug>\s*<bug>', '<bug>', text)
    text = re.sub(r'</bug>\s*</bug>', '</bug>', text)
    text = re.sub(r'<bug>\s+', '<bug> ', text)
    text = re.sub(r'\s+</bug>', ' </bug>', text)
    return text

def add_spaces_around_symbols(text, operators_v1, delimiters):
    for operator in operators_v1:
        text = re.sub(f'(?<=\\w)({operator})', r' \1 ', text)
        text = re.sub(f'({operator})(?=\\w)', r' \1 ', text)
    for delimiter in delimiters:
        text = re.sub(f'(?<=\\w)({delimiter})', r' \1 ', text)
        text = re.sub(f'({delimiter})(?=\\w)', r' \1 ', text)
        text = re.sub(f' *({delimiter}) *', r' \1 ', text)
    return text

def remove_spaces_in_double_char_operators(text):
    text = re.sub(r'\= \=', '==', text)
    text = re.sub(r'\+ \+', '++', text)
    text = re.sub(r'\- \-', '--', text)
    text = re.sub(r'\! \=', '!=', text)
    text = re.sub(r'\> \=', '>=', text)
    text = re.sub(r'\< \=', '<=', text)
    return text

def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text)

def process_line(line):
    line = protect_bug_tags(line)
    line = add_spaces_around_symbols(line, operators_v1, code_delimiters)
    line = remove_spaces_in_double_char_operators(line)
    line = remove_extra_spaces(line)
    line = restore_bug_tags(line)
    return line

def get_embedding(text, model, tokenizer, max_length=512, overlap=100):
    inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=False).to(device)
    input_ids = inputs["input_ids"].squeeze(0)
    seq_length = input_ids.shape[0]

    if seq_length <= max_length:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    else:
        embeddings = []
        start = 0
        while start < seq_length:
            end = min(start + max_length, seq_length)
            chunk_text = tokenizer.decode(input_ids[start:end], skip_special_tokens=True)
            inputs_chunk = tokenizer(chunk_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            with torch.no_grad():
                outputs_chunk = model(**inputs_chunk)
            embeddings.append(outputs_chunk.last_hidden_state[:, 0, :].squeeze().cpu().numpy())
            start += max_length - overlap
        return np.mean(embeddings, axis=0)

def file_to_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [process_line(line.strip()) for line in file.readlines()]
        # lines = [process_line(line) for line in file.readlines()]
    return lines

def list_to_dataframe(lines):
    return pd.DataFrame({'java_code': lines})

def calculate_total_tokens(df, column_name):
    return sum(len(tokenizer.tokenize(str(text))) for text in df[column_name].astype(str))

def find_top_n_matches(original_methods, methods_list, n):
    split_methods = []
    current_method = []
    for line in methods_list:
        if line.strip():
            current_method.append(line.strip())
        else:
            if current_method:
                split_methods.append(current_method)
                current_method = []
    if current_method:
        split_methods.append(current_method)

    original_methods_list = [line.strip() for line in original_methods if line.strip()]
    original_text = " ".join(original_methods_list)

    original_embedding = get_embedding(original_text, model, tokenizer)

    method_embeddings = [
        (idx, method, get_embedding(" ".join(method), model, tokenizer))
        for idx, method in enumerate(split_methods, start=1)
    ]

    similarities = []
    for method_idx, method_text, method_embedding in method_embeddings:
        similarity = cosine_similarity([original_embedding], [method_embedding])[0][0]
        similarities.append((method_idx, method_text, similarity))

    sorted_matches = sorted(similarities, key=lambda x: x[2], reverse=True)
    return [match[1] for match in sorted_matches[1:n+1]]

def calculate_total_tokens(df, column_name):
    total_tokens = 0
    for text in df[column_name].astype(str):  # 해당 컬럼의 텍스트를 문자열로 변환
        tokens = tokenizer.tokenize(text)  # 텍스트를 토큰화
        total_tokens += len(tokens)  # 토큰 개수 추가
    return total_tokens

def count_tokens_at_indexes(df, different_indexes):
    total_tokens = 0

    # different_indexes에 해당하는 행만 선택하여 토큰 개수 계산
    for idx in different_indexes:
        java_code_text = df.loc[idx, 'java_code']  # 해당 인덱스의 java_code 텍스트
        tokens = tokenizer.tokenize(str(java_code_text))  # 텍스트를 토큰화
        total_tokens += len(tokens)  # 토큰 개수 추가

    return total_tokens

# 자바 기본 데이터 타입 목록
java_types = {
    'int', 'short', 'long', 'float', 'double', 'char', 'boolean', 'byte', 'String',
    'void', 'Integer', 'Boolean', 'Double', 'Character', 'Float', 'Long', 'Short', 'Byte'}

# 연산자 목록
operators = {'=', '+', '-', '*', '/', '%', '==', '!=', '>', '<', '>=', '<=', '&&', '||', '!', '&', '|', '^', '<<', '>>', '>>>', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^='}

# index별로 거리 점수 구하기
def calculate_dis_score(line_number, bug_line):
    if line_number == bug_line:
        return 1.0
    result = 1.0 / (abs(bug_line - line_number) + 1)
    return result

def noLightweight(buggy_method_lines, context_lines, a=0.5):
    buggy_full = ' '.join(str(x).strip() for x in buggy_method_lines if str(x).strip())
    context_full = ' '.join(str(x).strip() for x in context_lines if str(x).strip())

    if context_full:
        return f"{buggy_full} <context> {context_full} </context>".strip()
    else:
        return buggy_full.strip()
    
#################################################################################    
def process_inner_folder(inner_path, case_path):
    """CategoryPlot 같은 실제 파일이 있는 폴더 처리"""

    # methods 파일 찾기
    method_files = [f for f in os.listdir(inner_path) if f.endswith("_methods.txt")]
    if not method_files:
        print(f"⚠️ {inner_path} - methods 파일 없음. 건너뜀.")
        return
    methods_path = os.path.join(inner_path, method_files[0])
    methods_lines = file_to_lines(methods_path)

    # buggy 파일 찾기
    buggy_files = [f for f in os.listdir(inner_path) if "Original_buggy_method_by_line" in f and f.endswith(".txt")]
    if not buggy_files:
        print(f"⚠️ {inner_path} - buggy 파일 없음. 건너뜀.")
        return

    folder_name = os.path.basename(inner_path)
    a_value_map = {"50%": 0.5}

    for buggy_file in sorted(buggy_files):
        buggy_path = os.path.join(inner_path, buggy_file)
        buggy_lines = file_to_lines(buggy_path)

        # top 5 context
        top_contexts = find_top_n_matches(buggy_lines, methods_lines, 5)

        json_results = []
        for label, a in a_value_map.items():
            for i, context in enumerate(top_contexts, start=1):
                context_lines = context
                lwbm_result = noLightweight(buggy_lines, context_lines, a=a)
                json_results.append({
                    "id": f"{i}_{label}",
                    "lwbm": lwbm_result
                })

        final_output = {folder_name: json_results}

        # 파일명 변환
        # 예: CategoryPlot_Original_buggy_method_by_line1.txt →
        #     CategoryPlot_buggy_and_context_method1.json
        if "Original_buggy_method_by_line" in buggy_file:
            prefix, _, suffix = buggy_file.partition("Original_buggy_method_by_line")
            number_part = ''.join(ch for ch in suffix if ch.isdigit())
            save_name = f"{prefix}buggy_and_context_method{number_part}.json"
        else:
            save_name = "buggy_and_context_method.json"

        # 저장: inner_path (CategoryPlot) 와 case_path 두 곳에 저장
        save_path_inner = os.path.join(inner_path, save_name)
        save_path_outer = os.path.join(case_path, save_name)

        for save_path in [save_path_inner, save_path_outer]:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)
            print(f"✅ {save_path} 저장 완료")


def process_case_folder(case_path):
    # case_folder 바로 안에서 먼저 시도
    process_inner_folder(case_path, case_path)

    # 그리고 서브폴더들 순회
    subfolders = [os.path.join(case_path, d) for d in os.listdir(case_path) if os.path.isdir(os.path.join(case_path, d))]
    for inner in subfolders:
        process_inner_folder(inner, case_path)
        
#####################################################################################

# if __name__ == "__main__":
#     ##하나의 폴더에서 하나만 처리
#     methods_path = "to/your/path"
#     buggy_path = "to/your/path"

#     methods_lines = file_to_lines(methods_path)
#     buggy_lines = file_to_lines(buggy_path)

#     top_contexts = find_top_n_matches(buggy_lines, methods_lines, 5)

#     folder_name = "Chart-2"

#     a_value_map = {
#         "50%": 0.5,
#         # "0%": 0.0,
#         # "100%": 1.0
#     }

#     json_results = []

#     for label, a in a_value_map.items():
#         for i, context in enumerate(top_contexts, start=1):
#             context_lines = context
#             lwbm_result = noLightweight(buggy_lines, context_lines, a=a)
#             json_results.append({
#                 "id": f"{i}_{label}",   # <-- id를 1_30%, 2_30%, 1_0%, 2_0% 이런 식으로
#                 "lwbm": lwbm_result
#             })

#     final_output = {
#         folder_name: json_results
#     }

#     save_dir = "to/your/path"  # 원하는 저장 폴더
#     os.makedirs(save_dir, exist_ok=True)  # 폴더가 없으면 자동 생성

#     save_path = os.path.join(save_dir, "buggy_and_context_method2.json")

#     with open(save_path, "w", encoding="utf-8") as f:
#         json.dump(final_output, f, indent=2, ensure_ascii=False)

#     print("✅ Saved buggy_and_context_method.json")


# ====== 메인 실행 영역 ======   multi method
if __name__ == "__main__":
    base_dir = "to/your/path"
    project_folder = "Test"
    project_path = os.path.join(base_dir, project_folder)

    if not os.path.isdir(project_path):
        print(f"🚫 {project_folder} 폴더가 존재하지 않음.")
    else:
        for case_folder in sorted(os.listdir(project_path)):
            case_path = os.path.join(project_path, case_folder)
            if not os.path.isdir(case_path):
                continue
            process_case_folder(case_path)




# ====== 메인 실행 영역 ======  single method
# if __name__ == "__main__":

#     base_dir = "to/your/path"

#     project_folder = "single"  
#     project_path = os.path.join(base_dir, project_folder)

#     if not os.path.isdir(project_path):
#         print(f"🚫 {project_folder} 폴더가 존재하지 않음.")
#     else:
#         for case_folder in sorted(os.listdir(project_path)):
#             case_path = os.path.join(project_path, case_folder)
#             if not os.path.isdir(case_path):
#                 continue

#             buggy_file = os.path.join(case_path, "Original_buggy_method_by_line.txt")
#             method_files = [f for f in os.listdir(case_path) if f.endswith("_methods.txt")]

#             if not os.path.isfile(buggy_file) or not method_files:
#                 print(f" {case_path} - 필요한 파일 없음. 건너뜀.")
#                 continue

#             methods_path = os.path.join(case_path, method_files[0])
#             methods_lines = file_to_lines(methods_path)
            
#             buggy_lines = file_to_lines(buggy_file)
#             # print(buggy_lines)
#             top_contexts = find_top_n_matches(buggy_lines, methods_lines, 5)

#             folder_name = os.path.basename(case_path)

#             a_value_map = {
#                 "50%": 0.5,
#                 # "0%": 0.0,
#                 # "100%": 1.0
#             }

#             json_results = []
#             for label, a in a_value_map.items():
#                 for i, context in enumerate(top_contexts, start=1):
#                     context_lines = context
#                     lwbm_result = noLightweight(buggy_lines, context_lines, a=a)
#                     json_results.append({
#                         "id": f"{i}_{label}",
#                         "lwbm": lwbm_result
#                     })

#             final_output = {
#                 folder_name: json_results
#             }

#             save_path = os.path.join(case_path, "buggy_and_context_method.json")
#             with open(save_path, "w", encoding="utf-8") as f:
#                 json.dump(final_output, f, indent=2, ensure_ascii=False)

#             print(f" {case_path} - buggy_and_context_method.json 저장 완료")