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
import os

# 비교할 디렉토리 경로 설정
directory_path = "C:/Users/UOS/Desktop/LwD4j/Mockito/"

# 5개 이상의 파일이 있는 폴더명을 저장할 리스트
folders_with_more_than_5_files = []

# 디렉토리 내 모든 폴더를 순회
for folder_name in os.listdir(directory_path):
    folder_path = os.path.join(directory_path, folder_name)
    
    # 폴더인지 확인
    if os.path.isdir(folder_path):
        # 해당 폴더 내의 파일 수를 계산
        file_count = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
        
        # 파일이 5개 이상 있는 경우 폴더명 추가
        if file_count == 1:
            folders_with_more_than_5_files.append(folder_name)

# 결과 출력
print("5개 이상의 파일이 있는 폴더들:")
for folder in folders_with_more_than_5_files:
    print(folder)
#--------------------------------------------------------------------
#<bug>토큰 붙이기
