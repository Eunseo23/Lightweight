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

def lightweight_context(dfc, indices, model, tokenizer, a, target_token_limit, max_length=512, overlap=100):
    """청크 단위를 포함하여 긴 텍스트에 대한 임베딩을 생성하고 문맥을 요약"""
    # 각 bug_line에 대해 거리 점수 계산
    dis_scores = {f'dis_score_{index}': [calculate_dis_score(line_number, index) for line_number in dfc.index] for index in indices}
    dis_scores_df = pd.DataFrame(dis_scores)

    # 각 라인별 임베딩 수행 (청크 포함)
    dfc['embedding'] = dfc['java_code'].apply(lambda x: get_embedding(x, model, tokenizer, max_length, overlap))

    # 코사인 유사도 계산
    sim_scores = {}
    for target_index in indices:
        if target_index < len(dfc):
            target_embedding = [dfc.loc[target_index, 'embedding']]  # 리스트로 감싸기
            similarities = [cosine_similarity(target_embedding, [emb])[0][0] for emb in dfc['embedding']]
            sim_scores[f'sim_scores_{target_index}'] = similarities

    # DataFrame 변환
    sim_scores_df = pd.DataFrame(sim_scores)
    df_combined = pd.concat([dfc, sim_scores_df, dis_scores_df], axis=1)

    # 총 점수 계산
    sum_scores = {}
    dis_keys = list(sorted(dis_scores.keys()))
    sim_keys = list(sorted(sim_scores.keys()))
    a = 0.5
    for dis_key, sim_key in zip(dis_keys, sim_keys):
        if dis_key in dis_scores and sim_key in sim_scores:  # 키가 존재하는지 확인
            sum_scores[dis_key] = [a * float(x) + (1 - a) * float(y) for x, y in zip(dis_scores[dis_key], sim_scores[sim_key])]

    list_length = len(next(iter(sum_scores.values()))) if sum_scores else 0
    total_scores1 = [0] * list_length
    for scores in sum_scores.values():
        total_scores1 = [total + score for total, score in zip(total_scores1, scores)]

    num_keys = len(sum_scores)
    total_scores = [score / num_keys for score in total_scores1]
    df_combined['total_scores'] = total_scores

    # **반복해서 제거하면서 토큰 개수를 200 이하로 줄이는 과정**
    max_iterations = 1000  # 최대 반복 횟수 설정
    iteration_count = 0  # 현재 반복 횟수
    time_limit = 30  # 30초 제한
    start_time = time.time()

    while True:
        iteration_count += 1
        current_time = time.time()
        if current_time - start_time > time_limit:
            print(f"Time limit reached ({time_limit} seconds), breaking the loop.")
            break
        if iteration_count > max_iterations:
            print(f"Max iterations reached ({max_iterations}), breaking the loop.")
            break

        # 가장 낮은 점수의 행 제거
        min_total_score_index = df_combined['total_scores'].idxmin()
        df_combined = df_combined.drop(min_total_score_index)

        # 현재 java_code들을 하나의 문자열로 결합
        lwm = ' '.join(df_combined['java_code'])

        # 토큰 개수 계산
        tokens = tokenizer.tokenize(str(lwm))

        if len(tokens) < target_token_limit:  # 목표 토큰 개수 이하라면 중지
            break

    # 최종적으로 결합된 `java_code` 반환
    lwct = "<context> {} </context>".format(' '.join(df_combined['java_code']).replace('\n', ''))

    return lwct


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

#데이터프레임내에서 가장 유사한 라인 찾기
def find_most_similar_line(bug_embeddings, dfc):
    if dfc["embedding"].isna().all():  # 모든 embedding이 None이면 빈 리스트 반환
        return []

    # 유사한 인덱스를 저장할 집합 (중복 방지)
    similar_indices = set()

    # 모든 bug_embedding에 대해 유사한 인덱스 찾기
    embeddings = np.stack(dfc["embedding"].dropna().values)  # None 값 제거 후 배열 변환

    for bug_embedding in bug_embeddings:
        similarities = cosine_similarity([bug_embedding], embeddings)[0]  # 1D 배열

        best_idx = np.argmax(similarities)  # 가장 높은 유사도를 가진 인덱스
        best_index = int(dfc.dropna(subset=["embedding"]).iloc[best_idx].name)  # 원래 인덱스 가져오기
        similar_indices.add(best_index)  # 중복되지 않게 추가

    return list(similar_indices)  # 리스트로 변환하여 반환

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

def lightweightdefects4j(buggy_method_lines, context_lines, a=0.5):
    df = list_to_dataframe(buggy_method_lines)
    dfc = list_to_dataframe(context_lines)
    bug_indices = df[df['java_code'].fillna('').str.strip().str.startswith('<bug>')].index.tolist()

    total_tokens_df = calculate_total_tokens(df, 'java_code')
    total_tokens_dfc = calculate_total_tokens(dfc, 'java_code')

    if total_tokens_df < 700:
        # lwbmo = ' '.join(df['java_code'])
        lwbmo = ' '.join(line.strip() for line in df['java_code'])
        bug_contents = re.findall(r"<bug>(.*?)</bug>", lwbmo) if re.search(r"<bug>(.*?)</bug>", lwbmo) else []
        bug_embeddings = [get_embedding(content.strip(), model, tokenizer) for content in bug_contents] if bug_contents else []

        if total_tokens_dfc < 300 + (700 - total_tokens_df):
            lwct = "<context> {} </context>".format(' '.join(dfc['java_code']).replace('\n', ' '))
        else:
            dfc["embedding"] = dfc["java_code"].apply(lambda x: get_embedding(x, model, tokenizer) if x.strip() else None)
            dfc_indices = find_most_similar_line(bug_embeddings, dfc)
            lwct = lightweight_context(dfc, dfc_indices, model, tokenizer, a, target_token_limit=300 + (700 - total_tokens_df))

        return f'{lwbmo} {lwct}'.strip()
    else:
        dis_scores = {
            f'dis_score_{bug_line}': [calculate_dis_score(line_number, bug_line) for line_number in df.index]
            for bug_line in bug_indices
        }
        dis_scores_df = pd.DataFrame(dis_scores)
        df['embedding'] = df['java_code'].apply(lambda x: get_embedding(x, model, tokenizer))

        sim_scores = {}
        for target_index in bug_indices:
            if target_index < len(df):
                target_embedding = [df.loc[target_index, 'embedding']]
                similarities = [cosine_similarity(target_embedding, [emb])[0][0] for emb in df['embedding']]
                sim_scores[f'sim_scores_{target_index}'] = similarities

        sim_scores_df = pd.DataFrame(sim_scores)
        df_combined = pd.concat([df, sim_scores_df, dis_scores_df], axis=1)

        sum_scores = {}
        # a = 0.3
        for dis_key, sim_key in zip(sorted(dis_scores.keys()), sorted(sim_scores.keys())):
            if dis_key in dis_scores and sim_key in sim_scores:
                sum_scores[dis_key] = [a * float(x) + (1 - a) * float(y) for x, y in zip(dis_scores[dis_key], sim_scores[sim_key])]

        total_scores = np.mean(list(sum_scores.values()), axis=0) if sum_scores else []
        df_combined['total_scores'] = total_scores
        # df_combined.drop(columns=['embedding'], errors='ignore').to_csv("df_with_scores.csv", index=False) ##################################

        max_iterations = 1000
        time_limit = 30
        start_time = time.time()
        iteration_count = 0

        while True:
            iteration_count += 1
            if time.time() - start_time > time_limit or iteration_count > max_iterations:
                break

            min_total_score_index = df_combined['total_scores'].idxmin()
            df_combined = df_combined.drop(min_total_score_index)

            lwbm = ''.join(df_combined['java_code'])
            tokens = tokenizer.tokenize(str(lwbm))
            if len(tokens) < 700:
                break

        # lwbmo = ' '.join(df_combined['java_code'])
        lwbmo = ' '.join(line.strip() for line in df_combined['java_code'])
        bug_contents = re.findall(r"<bug>(.*?)</bug>", lwbmo) if re.search(r"<bug>(.*?)</bug>", lwbmo) else []
        bug_embeddings = [get_embedding(content.strip(), model, tokenizer) for content in bug_contents] if bug_contents else []

        if total_tokens_dfc < 300:
            lwct = "<context> {} </context>".format(' '.join(dfc['java_code']).replace('\n', ' '))
        else:
            dfc["embedding"] = dfc["java_code"].apply(lambda x: get_embedding(x, model, tokenizer) if x.strip() else None)
            dfc_indices = find_most_similar_line(bug_embeddings, dfc)
            lwct = lightweight_context(dfc, dfc_indices, model, tokenizer, a, target_token_limit=300)

        return f'{lwbmo} {lwct}'.strip()

###################################################################################
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
    a_value_map = {
        "50%": 0.5,
        "0%": 0.0,
        "100%": 1.0
    }

    for buggy_file in sorted(buggy_files):
        buggy_path = os.path.join(inner_path, buggy_file)
        buggy_lines = file_to_lines(buggy_path)

        # top 5 context
        top_contexts = find_top_n_matches(buggy_lines, methods_lines, 5)

        json_results = []
        for label, a in a_value_map.items():
            for i, context in enumerate(top_contexts, start=1):
                context_lines = context
                lwbm_result = lightweightdefects4j(buggy_lines, context_lines, a=a)
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
            save_name = f"{prefix}Lightweight_method_llama_diff{number_part}.json"
        else:
            save_name = "Lightweight_method_llama_diff.json"

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
# #############################################################################
if __name__ == "__main__":

    base_dir = "to/your/path"
    project_folder = "multi"
    project_path = os.path.join(base_dir, project_folder)

    if not os.path.isdir(project_path):
        print(f"🚫 {project_folder} 폴더가 존재하지 않음.")
    else:
        for case_folder in sorted(os.listdir(project_path)):
            case_path = os.path.join(project_path, case_folder)
            if not os.path.isdir(case_path):
                continue
            process_case_folder(case_path)    
    
##하나의 폴더에서 하나만 처리
# if __name__ == "__main__":

#     methods_path = "to/your/path"
#     buggy_path = "to/your/path"

#     methods_lines = file_to_lines(methods_path)
#     buggy_lines = file_to_lines(buggy_path)
#     top_contexts = find_top_n_matches(buggy_lines, methods_lines, 5)
    
#     folder_name = "Cli-12"
#     a_value_map = {
#         "50%": 0.5,
#         # "0%": 0.0,
#         # "100%": 1.0
#     }
#     json_results = []
#     for label, a in a_value_map.items():
#         for i, context in enumerate(top_contexts, start=1):
#             context_lines = context
#             lwbm_result = lightweightdefects4j(buggy_lines, context_lines, a=a)
#             json_results.append({
#                 "id": f"{i}_{label}",   # <-- id를 1_30%, 2_30%, 1_0%, 2_0% 이런 식으로
#                 "lwbm": lwbm_result
#             })
#     final_output = {
#         folder_name: json_results
#     }
#     save_dir = "to/your/path"  # 원하는 저장 폴더
#     os.makedirs(save_dir, exist_ok=True)  # 폴더가 없으면 자동 생성
#     save_path = os.path.join(save_dir, "Lightweight_buggy_method_Context_llama.json")

#     with open(save_path, "w", encoding="utf-8") as f:
#         json.dump(final_output, f, indent=2, ensure_ascii=False)

#     print("✅ Saved Lightweight_buggy_method_Context_llama.json")







# ====== 메인 실행 영역 ======
# if __name__ == "__main__":

#     base_dir = "to/your/path"

#     project_folder = "Codellama"  # 특정 폴더만 처리
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
#                 print(f"⚠️ {case_path} - 필요한 파일 없음. 건너뜀.")
#                 continue

#             methods_path = os.path.join(case_path, method_files[0])
#             methods_lines = file_to_lines(methods_path)
            
#             buggy_lines = file_to_lines(buggy_file)
#             # print(buggy_lines)
#             top_contexts = find_top_n_matches(buggy_lines, methods_lines, 5)

#             folder_name = os.path.basename(case_path)

#             a_value_map = {
#                 "50%": 0.5,
#                 "0%": 0.0,
#                 "100%": 1.0
#             }

#             json_results = []
#             for label, a in a_value_map.items():
#                 for i, context in enumerate(top_contexts, start=1):
#                     context_lines = context
#                     lwbm_result = lightweightdefects4j(buggy_lines, context_lines, a=a)
#                     json_results.append({
#                         "id": f"{i}_{label}",
#                         "lwbm": lwbm_result
#                     })

#             final_output = {
#                 folder_name: json_results
#             }

#             save_path = os.path.join(case_path, "Lightweight_method_llama_diff.json")
#             with open(save_path, "w", encoding="utf-8") as f:
#                 json.dump(final_output, f, indent=2, ensure_ascii=False)

#             print(f"✅ {case_path} - Lightweight_method_llama_diff.json 저장 완료")