import os
# //인덴트 제거
# def extract_info_from_file(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()

#     extracted_info = []
#     for line in lines:
#         # 경로와 # 뒤에 있는 숫자를 추출
#         if 'java' in line and '#' in line:
#             parts = line.strip().split('#')
#             file_path = parts[0]
#             line_number = int(parts[1])
#             extracted_info.append((file_path, line_number))
    
#     return extracted_info


# def find_java_file_in_directory(base_dir, file_path):
#     full_path = os.path.join(base_dir, file_path)
#     return full_path if os.path.exists(full_path) else None

# # 테스트
# chart_number = 1  # 예시로 Chart-1.buggy.lines 처리
# file_name = f'Chart-{chart_number}.buggy.lines'
# extracted_info = extract_info_from_file(file_name)

# for file_path, line_number in extracted_info:
#     base_dir = 'C:\\Users\\UOS\\Desktop\\MCRepair-main\\APR_Resources\\localization\\defects4j_faults_20220718\\'
#     if base_dir:
#         java_file_path = find_java_file_in_directory(base_dir, file_path)
#         if java_file_path:
#             print(f"Found Java file: {java_file_path} at line {line_number}")

# import torch
# import os

# # 파일 경로 설정
# checkpoint_path = "E:/checkpoint-best-ppl-0926-0023/pytorch_model_0926-0023.bin"  # 실제 pytorch_model.bin 파일 경로로 수정하세요

# # 파일이 존재하는지 확인
# if os.path.exists(checkpoint_path):
#     # 체크포인트 로드
#     checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
#     # 체크포인트의 state_dict 키 확인
#     print("Checkpoint Keys:")
#     checkpoint_keys = checkpoint.keys()
#     for key in checkpoint_keys:
#         print(key)

#     # # 현재 모델의 state_dict 키 확인
#     # model_state_dict = model.state_dict()
#     # print("Model State Dict Keys:")
#     # for key in model_state_dict.keys():
#     #     print(key)

#     # 체크포인트와 모델의 키 비교
#     checkpoint_keys_set = set(checkpoint_keys)
#     # model_keys_set = set(model_state_dict.keys())

#     # # 체크포인트에는 있지만 모델에는 없는 키
#     # extra_keys_in_checkpoint = checkpoint_keys_set - model_keys_set
#     # # 모델에는 있지만 체크포인트에는 없는 키
#     # missing_keys_in_checkpoint = model_keys_set - checkpoint_keys_set

#     # print(f"Extra keys in checkpoint: {extra_keys_in_checkpoint}")
#     # print(f"Missing keys in checkpoint: {missing_keys_in_checkpoint}")
# else:
#     print(f"Checkpoint file not found at {checkpoint_path}")

#original buggy method 한줄로 만들기
# import os
# import pandas as pd
# import re

# # 자바 연산자 리스트 및 코드 여닫는 문자 리스트 추가
# operators = [
#     r'==', r'!=', r'>=', r'<=', r'\+\+', r'--', r'\+=', r'-=', r'\*=', r'/=', r'%=', r'&=', r'\|=', r'\^=', r'>>=', r'<<=',
#     r'=', r'<', r'>', r'\+', r'-', r'\*', r'/', r'%', r'&', r'\|', r'\^', r'~', r'!', r'>>', r'<<', r'\?', r':', r'\.'
# ]
# code_delimiters = [r'\{', r'\}', r';', r'\(', r'\)', r'\.', r',']  # ',' 추가

# # 보호 및 변환 함수들 정의
# def protect_bug_tags(text):
#     text = text.replace("<bug>", "SPECIAL_BUG_OPEN_TAG")
#     text = text.replace("</bug>", "SPECIAL_BUG_CLOSE_TAG")
#     return text

# def restore_bug_tags(text):
#     text = text.replace("SPECIAL_BUG_OPEN_TAG", "<bug>")
#     text = text.replace("SPECIAL_BUG_CLOSE_TAG", "</bug> ")
#     return text

# def add_spaces_around_bug_tags(text):
#     text = re.sub(r'(?<=\w)(SPECIAL_BUG_OPEN_TAG)', r' \1 ', text)
#     text = re.sub(r'(SPECIAL_BUG_OPEN_TAG)(?=\w)', r' \1 ', text)
#     text = re.sub(r'(?<=\w)(SPECIAL_BUG_CLOSE_TAG)', r' \1 ', text)
#     text = re.sub(r'(SPECIAL_BUG_CLOSE_TAG)(?=\w)', r' \1 ', text)
#     return text

# def add_spaces_around_symbols(text, operators, delimiters):
#     for operator in operators:
#         text = re.sub(f'(?<=\w)({operator})', r' \1 ', text)
#         text = re.sub(f'({operator})(?=\w)', r' \1 ', text)
#     for delimiter in delimiters:
#         text = re.sub(f'(?<=\w)({delimiter})', r' \1 ', text)
#         text = re.sub(f'({delimiter})(?=\w)', r' \1 ', text)
#         text = re.sub(f' *({delimiter}) *', r' \1 ', text)
#     return text

# def remove_spaces_in_double_char_operators(text):
#     text = re.sub(r'\= \=', '==', text)
#     text = re.sub(r'\+ \+', '++', text)
#     text = re.sub(r'\- \-', '--', text)
#     text = re.sub(r'\! \=', '!=', text)
#     text = re.sub(r'\> \=', '>=', text)
#     text = re.sub(r'\< \=', '<=', text)
#     return text

# def remove_extra_spaces(text):
#     text = re.sub(r'\s+', ' ', text)
#     return text

# def process_line(line):
#     line = protect_bug_tags(line)
#     line = add_spaces_around_symbols(line, operators, code_delimiters)
#     line = add_spaces_around_bug_tags(line)
#     line = remove_spaces_in_double_char_operators(line)
#     line = remove_extra_spaces(line)
#     line = restore_bug_tags(line)
#     return line

# # 파일 내용을 하나의 라인으로 만들고 변환 함수 적용
# def file_to_single_java_line_with_processing(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         lines = file.readlines()

#     # 각 줄에 대해 process_line 적용
#     processed_lines = [process_line(line.strip()) for line in lines]

#     # 모든 줄을 하나로 이어 붙이기
#     single_java_line = ''.join(processed_lines)
    
#     return single_java_line

# # 디렉토리 내 파일 처리 함수
# def process_files_in_directory(base_directory):
#     for chart_directory in os.listdir(base_directory):
#         chart_path = os.path.join(base_directory, chart_directory)

#         if os.path.isdir(chart_path):
#             for filename in os.listdir(chart_path):
#                 if filename.startswith("Original_buggy_method_by_line") and filename.endswith(".txt"):
#                     inputFilePath = os.path.join(chart_path, filename)

#                     # 파일을 처리하고 변환 함수 적용
#                     result = file_to_single_java_line_with_processing(inputFilePath)

#                     # 처리한 결과 파일명을 'Original_buggy_method'로 설정
#                     output_filename = filename.replace("Original_buggy_method_by_line", "Original_buggy_method")
#                     outputFilePath = os.path.join(chart_path, output_filename)

#                     # 동일한 이름의 파일이 있으면 뒤에 숫자를 붙여서 저장
#                     counter = 1
#                     while os.path.exists(outputFilePath):
#                         output_filename = f"Original_buggy_method{counter}.txt"
#                         outputFilePath = os.path.join(chart_path, output_filename)
#                         counter += 1

#                     # 결과를 새 파일로 저장
#                     with open(outputFilePath, 'w') as f:
#                         f.write(result)

#                     print(f"Processed {inputFilePath} and saved to {outputFilePath}")

# # 사용 예시
# base_directory = "C:/Users/UOS/Desktop/Closure_lightweight"
# process_files_in_directory(base_directory)

#파일 이름 바꾸기
import os

def rename_files_in_directory(base_directory):
    # 폴더 안으로 들어가 가장 안쪽 폴더의 파일을 처리
    for root, dirs, files in os.walk(base_directory):
        # 하위 디렉토리가 없는 가장 안쪽 폴더인지 확인
        if not dirs:  # 하위 폴더가 없으면 가장 안쪽 폴더임
            # 해당 폴더 내의 파일을 순회
            for filename in files:
                # Lightweight_buggy_method로 시작하고 .txt 확장자인 파일을 buggy_block.txt로 변경
                if filename.startswith("Original_candidate_patches") and filename.endswith(".txt"):
                    new_filename = "candidate_patches_buggy_block.txt"
                    old_file_path = os.path.join(root, filename)
                    new_file_path = os.path.join(root, new_filename)

                    # 파일이 이미 존재하는지 확인
                    if not os.path.exists(new_file_path):
                        # 파일 이름 변경
                        os.rename(old_file_path, new_file_path)
                        print(f"Renamed: {old_file_path} -> {new_file_path}")
                    else:
                        print(f"File already exists: {new_file_path}, skipping...")

                # # Original_buggy_method로 시작하고 .txt 확장자인 파일을 multi_chunk.txt로 변경
                # elif filename.startswith("Original_buggy_method") and filename.endswith(".txt"):
                #     new_filename = "multi_chunk.txt"
                #     old_file_path = os.path.join(root, filename)
                #     new_file_path = os.path.join(root, new_filename)

                #     # 파일 이름 변경
                #     os.rename(old_file_path, new_file_path)
                #     print(f"Renamed: {old_file_path} -> {new_file_path}")

# 사용 예시
base_directory = "C:/Users/UOS/Desktop/Chart_lightweight"
rename_files_in_directory(base_directory)