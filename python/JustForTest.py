import difflib
import re

def compare_java_files(file1_path, file2_path):
    with open(file1_path, 'r', encoding='utf-8') as file1, open(file2_path, 'r', encoding='utf-8') as file2:
        file1_lines = file1.readlines()
        file2_lines = file2.readlines()

    max_len = max(len(file1_lines), len(file2_lines))

    differences = []
    
    for i in range(max_len):
        file1_line = file1_lines[i].strip() if i < len(file1_lines) else ""
        file2_line = file2_lines[i].strip() if i < len(file2_lines) else ""
        
        if file1_line != file2_line:
            differences.append(f"Line {i+1}:\nFile1: {file1_line}\nFile2: {file2_line}\n")

    if differences:
        return "\n".join(differences)
    else:
        return "The two files are identical."


# 파일 경로 설정
file1_path = "C:\\Users\\UOS\\Desktop\\sciclone\\data10\\mtufano\\deepLearningMutants\\out\\bugfixes\\code\\0a0aacca1c3c507ded2360cc737dd77c8819e458\\P_dir\\adapter\\src\\main\\java\\com\\addrone\\CommHandlerSimulator.java"
file2_path = "C:\\Users\\UOS\\Desktop\\sciclone\\data10\\mtufano\\deepLearningMutants\\out\\bugfixes\\code\\0a0aacca1c3c507ded2360cc737dd77c8819e458\\F_dir\\adapter\\src\\main\\java\\com\\addrone\\CommHandlerSimulator.java"

# 파일 비교
result = compare_java_files(file1_path, file2_path)
print(result)