#-------------------------------------------------------------
# 특정 단어 있는 파일명 확인
import os

# # 검색할 디렉토리 경로 설정
# directory_path = "C:/Users/UOS/Desktop/LwD4j/Mockito/36"

# # word 단어가 포함된 파일명을 저장할 리스트
# files_with_word = []

# # 디렉토리 내의 모든 파일을 순회
# for file_name in os.listdir(directory_path):
#     file_path = os.path.join(directory_path, file_name)
    
#     # .txt 파일인지 확인
#     if os.path.isfile(file_path) and file_name.endswith(".txt"):
#         # 파일을 읽어서 adjustOffset 단어가 포함되어 있는지 확인
#         with open(file_path, 'r', encoding='utf-8') as file:
#             content = file.read()
#             if 'rawArguments' in content:
#                 files_with_word.append(file_name)

# # 결과 출력
# print(files_with_word)
#-----------------------------------------------------------------------
#중복 파일 제거 코드
# import os
# import hashlib

# # 비교할 디렉토리 경로 설정
# directory_path = "C:/Users/UOS/Desktop/LwD4j/Closure/"

# # 디렉토리 내 모든 폴더를 순회
# for folder_name in os.listdir(directory_path):
#     folder_path = os.path.join(directory_path, folder_name)
    
#     # 폴더인지 확인
#     if os.path.isdir(folder_path):
#         # 파일 해시값을 저장할 딕셔너리 초기화
#         file_hashes = {}
        
#         # 해당 폴더 내의 모든 파일을 순회
#         for file_name in os.listdir(folder_path):
#             file_path = os.path.join(folder_path, file_name)
            
#             # 파일인지 확인
#             if os.path.isfile(file_path) and file_name.endswith(".txt"):
#                 # 파일의 내용을 읽어서 해시값 계산
#                 with open(file_path, 'rb') as file:
#                     file_content = file.read()
#                     file_hash = hashlib.md5(file_content).hexdigest()
                
#                 # 동일한 해시값이 이미 존재하는지 확인
#                 if file_hash in file_hashes:
#                     # 중복 파일 삭제
#                     print(f"Deleting duplicate file: {file_path}")
#                     os.remove(file_path)
#                 else:
#                     file_hashes[file_hash] = file_name
#--------------------------------------------------------------------------------
# import os

# # 디렉토리 경로 설정
# directory_path = "C:/Users/UOS/Desktop/LwD4j/Closure/"

# # 파일이 한 개만 있는 폴더명을 저장할 리스트
# folders_with_one_file = []

# # baseDirPath 내의 모든 디렉토리를 순회
# for folder_name in os.listdir(directory_path):
#     folder_path = os.path.join(directory_path, folder_name)
    
#     # 폴더인지 확인
#     if os.path.isdir(folder_path):
#         # 해당 폴더 내의 파일 수를 계산
#         file_count = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
        
#         # 파일이 한 개만 있는 경우 폴더명 추가
#         if file_count == 1:
#             folders_with_one_file.append(folder_name)

# # 결과 출력
# print(folders_with_one_file)
#------------------------------------------------------------------
# 5개 이상 파일있는 폴더명 추출                    
# import os

# # 비교할 디렉토리 경로 설정
# directory_path = "C:/Users/UOS/Desktop/LwD4j/Closure/"

# # 5개 이상의 파일이 있는 폴더명을 저장할 리스트
# folders_with_more_than_5_files = []

# # 디렉토리 내 모든 폴더를 순회
# for folder_name in os.listdir(directory_path):
#     folder_path = os.path.join(directory_path, folder_name)
    
#     # 폴더인지 확인
#     if os.path.isdir(folder_path):
#         # 해당 폴더 내의 파일 수를 계산
#         file_count = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
        
#         # 파일이 5개 이상 있는 경우 폴더명 추가
#         if file_count >= 5:
#             folders_with_more_than_5_files.append(folder_name)

# # 결과 출력
# print("5개 이상의 파일이 있는 폴더들:")
# for folder in folders_with_more_than_5_files:
#     print(folder)
#--------------------------------------------------------------------
#<bug>토큰 붙이기
import os
import shutil

# 파일 경로 설정
base_dir = "C:/Users/UOS/Desktop/LwD4j/Time/1/"
output_dir = "C:/Users/UOS/Desktop/Lightweight/"
patch_file_path = "C:/Users/UOS/Desktop/MCRepair-main/APR_Resources/localization/defects4j_developers/Time/patches/1.src.patch"

# .txt 파일명을 찾는 함수
def find_txt_file(java_file_name, directory):
    java_base_name = java_file_name.split('.')[0]  # 확장자 제거
    for file_name in os.listdir(directory):
        if file_name.startswith(java_base_name) and file_name.endswith('.txt'):
            return os.path.join(directory, file_name)
    return None

# 패치 파일을 읽어와서 해당 Java 파일명을 추출
with open(patch_file_path, 'r') as patch_file:
    patch_file_content = patch_file.read()

# 파일별 diff 블록을 저장하기 위한 딕셔너리
diff_blocks = {}
current_file = None
txt_file_path = None  # 이 변수는 블록 내에서 사용되므로 초기화가 필요합니다.

# 패치 파일의 diff 내용 처리
for line in patch_file_content.splitlines():
    if line.startswith("diff --git"):
        # 파일명을 추출
        parts = line.split()
        current_file = parts[2].split('/')[-1]
        txt_file_path = find_txt_file(current_file, base_dir)
        if txt_file_path:
            diff_blocks[txt_file_path] = []
    elif current_file and txt_file_path:
        # 조건 수정: txt_file_path가 None이 아닐 때만 append
        diff_blocks[txt_file_path].append(line)

# 패치 내용을 적용하는 함수
def apply_patch(file_path, diff_block, output_dir):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i < len(diff_block):
            diff_line = diff_block[i]

            if diff_line.startswith("-") and diff_line[1:].strip() == line.strip():
                modified_lines.append(f"<bug>{line.strip()}</bug>\n")
                i += 1  # 삭제된 라인의 경우 diff 파일에서 라인 카운터를 증가
            elif diff_line.startswith("+"):
                modified_lines.append(f"<bug>{diff_line[1:].strip()}</bug>\n")
            else:
                modified_lines.append(line)
                i += 1  # 일반적인 라인의 경우 원본 파일에서 라인 카운터를 증가
        else:
            modified_lines.append(line)
            i += 1

    # 결과물을 output_dir에 저장
    output_file_path = os.path.join(output_dir, os.path.basename(file_path))
    with open(output_file_path, 'w') as output_file:
        output_file.writelines(modified_lines)

# 각 파일에 대해 패치 적용
os.makedirs(output_dir, exist_ok=True)  # output_dir이 없으면 생성
for file, diff_block in diff_blocks.items():
    if file:
        apply_patch(file, diff_block, output_dir)

print("패치가 성공적으로 적용되었고, 결과물이 지정된 폴더에 저장되었습니다.")