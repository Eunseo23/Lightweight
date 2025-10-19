import subprocess
import os
import pandas as pd
import re
import json
import time
from tqdm import tqdm
import tokenize
from transformers import RobertaTokenizer

def run_java_program(input_file_path_p, input_file_path_f):
    # Java 클래스 이름 (패키지를 포함한 정규화된 이름)
    java_class = "com.lightweight.OverallProcess1"
    
    # 필요한 모든 JAR 파일들을 포함한 클래스패스 설정
    classpath = "./Lightweight-main/java/target/classes" + os.pathsep + "./Lightweight-main/java/target/dependency/*"
    
    # Java 프로그램 실행 명령어
    command = [
        "java", "-cp", classpath,
        java_class,
        input_file_path_p, input_file_path_f
    ]
    
    try:
        # Java 프로그램 실행
        result = subprocess.run(command, capture_output=True, text=True)
        
        # 실행 결과 출력
        # print("stdout:", result.stdout)
        # print("stderr:", result.stderr)
        
        # Java 프로그램 실행이 성공했는지 확인
        if result.returncode != 0:
            print("Java program failed with return code", result.returncode)
            return None, None
        else:
            print("Java program executed successfully")
            
            # stdout을 줄 단위로 분리
            lines = result.stdout.splitlines()
            original_lines = []
            revised_lines = []
            original_section = True

            for line in lines:
                if "Original Stripped Lines:" in line:
                    original_section = True
                    continue
                elif "Revised Stripped Lines:" in line:
                    original_section = False
                    continue

                if original_section:
                    original_lines.append(line)
                else:
                    revised_lines.append(line)

            return original_lines, revised_lines
    
    except Exception as e:
        print("An error occurred:", e)
        return None, None

def merge_adjacent_differences(file1_lines, file2_lines):
    merged_file1 = []
    merged_file2 = []

    i = 0
    while i < max(len(file1_lines), len(file2_lines)):
        file1_line = file1_lines[i].rstrip() if i < len(file1_lines) else ""
        file2_line = file2_lines[i].rstrip() if i < len(file2_lines) else ""

        # 만약 현재 줄이 다르다면
        if file1_line != file2_line:
            merged_file1_line = file1_line
            merged_file2_line = file2_line

            # 연속된 여러 줄이 차이가 있는지 확인
            while i + 1 < max(len(file1_lines), len(file2_lines)):
                next_file1_line = file1_lines[i + 1].rstrip() if i + 1 < len(file1_lines) else ""
                next_file2_line = file2_lines[i + 1].rstrip() if i + 1 < len(file2_lines) else ""

                if next_file1_line != next_file2_line:
                    # 차이가 있는 경우 줄을 병합
                    merged_file1_line += " " + next_file1_line
                    merged_file2_line += " " + next_file2_line
                    i += 1  # 인덱스를 증가시켜 연속된 줄들을 처리
                else:
                    break  # 차이가 없으면 루프 탈출

            # 병합된 줄 추가
            merged_file1.append(merged_file1_line)
            merged_file2.append(merged_file2_line)
        else:
            # 줄이 다르지 않으면 그대로 추가
            merged_file1.append(file1_line)
            merged_file2.append(file2_line)
        
        # 다음 줄로 이동
        i += 1

    return merged_file1, merged_file2

def list_to_dataframe(lines):
    # 데이터프레임으로 변환
    if not lines:
        raise ValueError("The input list is empty.")
    
    # 모든 요소가 문자열인지 확인
    if not all(isinstance(line, str) for line in lines):
        raise ValueError("All elements in the input list must be strings.")
    
    df = pd.DataFrame({
        'java_code': lines
    })
    return df

def calculate_total_tokens(df, column_name):
    total_tokens = 0
    for text in df[column_name].astype(str):  # 해당 컬럼의 텍스트를 문자열로 변환
        tokens = tokenizer.tokenize(text)  # 텍스트를 토큰화
        total_tokens += len(tokens)  # 토큰 개수 추가
    return total_tokens

def compare_df_find_index(df1, df2):
    # 두 데이터프레임의 행을 비교하여 차이점 찾기
    # 두 데이터프레임의 인덱스가 동일해야 함
    common_index = df1.index.intersection(df2.index)

    # 행 단위로 비교를 위한 데이터프레임 생성
    df1_common = df1.loc[common_index]
    df2_common = df2.loc[common_index]

    # 행 단위 비교
    different_rows = []
    for idx in common_index:
        if not df1_common.loc[idx].equals(df2_common.loc[idx]):
            different_rows.append(idx)

    # 차이점이 있는 인덱스를 반환
    return different_rows    

# 각 메서드 차이점이 담긴 토큰 개수
# 큰 값 기준으로 512에서 제외하고 lightweight한다.
# ex 512-207 = 305개까지 lightweight한다.
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

def extract_identifiers(line):
    identifiers = set()

    # 정규표현식을 사용하여 식별자 추출
    regex = r'\b[a-zA-Z_$][a-zA-Z_$0-9]*\b'
    matches = re.findall(regex, line)

    # 자바 기본 데이터 타입 제외
    for match in matches:
        if match not in java_types:
            identifiers.add(match)

    # 연산자 추출 및 추가
    for operator in operators:
        if operator in line:
            identifiers.add(operator)

    return identifiers

# index별로 거리 점수 구하기
def calculate_dis_score(line_number, bug_line):
    if line_number == bug_line:
        return 1.0
    return 1.0 / (abs(bug_line - line_number) + 1)

# 식별자 점수 구하기
def intersection_score(line_identifiers, target_identifiers):
    """Calculate the intersection score between two sets of identifiers."""
    intersection_count = len(line_identifiers.intersection(target_identifiers))
    target_identifier_count = len(target_identifiers)
    return intersection_count / target_identifier_count if target_identifier_count != 0 else 0

def LightweightProcess(inputFilePathp, inputFilePathf):

    # Java 프로그램 실행 및 출력 가져오기
    print(f"Processing P_dir: {inputFilePathp}")
    print(f"Processing F_dir: {inputFilePathf}")
    original_lines, revised_lines = run_java_program(inputFilePathp, inputFilePathf)

    if original_lines is not None and revised_lines is not None:
        # 파일 비교 및 병합
        merged_file1, merged_file2 = merge_adjacent_differences(original_lines, revised_lines)

    else:
        print("Failed to run the Java program or no output was captured.")
        return None, None
    
    if not merged_file1 or not merged_file2:
        print(f"Error: One of the merged files is empty for paths: {inputFilePathp}, {inputFilePathf}")
        return None, None

    df1 = list_to_dataframe(merged_file1)
    df2 = list_to_dataframe(merged_file2)
    
    # java_code 컬럼에 대한 총 토큰 개수 계산
    total_tokens_df1 = calculate_total_tokens(df1, 'java_code')
    total_tokens_df2 = calculate_total_tokens(df2, 'java_code')
    
    if total_tokens_df1 < 512 and total_tokens_df2 < 512:
        different_indexes = compare_df_find_index(df1, df2)
        # lightweight df에서 buggy line에 <bug>토큰 붙이기
        for index in different_indexes:
            df1.at[index, 'java_code'] = f"<bug>{df1.at[index, 'java_code']}</bug>"
            
        lwbm = ''.join(df1['java_code']).replace('\n', '')
        lwfm = ''.join(df2['java_code']).replace('\n', '')
        
    else:
        different_indexes = compare_df_find_index(df1, df2)
        total_tokens_diff_df1 = count_tokens_at_indexes(df1, different_indexes)
        total_tokens_diff_df2 = count_tokens_at_indexes(df2, different_indexes)
        total_bug_tokens = max(total_tokens_diff_df1, total_tokens_diff_df2)

        #식별자 추출
        line_identifiers = []
        for index, row in df1.iterrows():
            identifiers = extract_identifiers(row['java_code'])
            line_identifiers.append(identifiers)
        df1['identifiers'] = pd.Series(line_identifiers)

        # 각 bug_line에 대해 거리 점수 계산
        dis_scores = {}
        for bug_line in different_indexes:
            dis_scores[f'dis_score_{bug_line}'] = [calculate_dis_score(line_number, bug_line) for line_number in df1.index]

        dis_scores_df = pd.DataFrame(dis_scores)

        int_scores = {}
        for target_index in different_indexes:
            if target_index < len(line_identifiers):
                # 현재 라인 식별자와 대상 라인 식별자의 교집합 점수를 계산
                target_identifiers = line_identifiers[target_index]
                scores = []
                for idx, identifiers in enumerate(line_identifiers):
                    if idx == target_index:
                        scores.append(1)  # 동일한 인덱스의 식별자 세트는 점수 1
                    else:
                        score = intersection_score(identifiers, target_identifiers)
                        scores.append(score)
                int_scores[f'int_scores_{target_index}'] = scores

        int_scores_df = pd.DataFrame(int_scores)
        df_combined = pd.concat([df1, int_scores_df, dis_scores_df], axis=1)

        sum_scores = {}
        dis_keys = sorted(dis_scores.keys())
        int_keys = sorted(int_scores.keys())

        a = 0.5
        # 가정: 동일한 위치의 키가 서로 매핑되어야 함
        for dis_key, int_key in zip(dis_keys, int_keys):
            combined_scores = [a*x + (1-a)*y for x, y in zip(dis_scores[dis_key], int_scores[int_key])]
            sum_scores[dis_key] = combined_scores
        list_length = len(sum_scores[dis_key])  # 모든 리스트는 같은 길이라고 가정
        total_scores1 = [0] * len(sum_scores[dis_key])

        # 각 키의 리스트를 total_scores에 더함
        for scores in sum_scores.values():
            total_scores1 = [total + score for total, score in zip(total_scores1, scores)]

        # sum_scores의 키의 개수
        num_keys = len(sum_scores)

        # 각 요소를 키의 개수로 나누기
        total_scores = [score / num_keys for score in total_scores1]

        df_combined['total_scores'] = total_scores

        print("Starting the loop")
        max_iterations = 1000  # 최대 반복 횟수 설정
        iteration_count = 0  # 현재 반복 횟수
        
        time_limit = 30  # 예: 60초 제한
        start_time = time.time()  # 시작 시간 기록

        while True:
            iteration_count += 1
            current_time = time.time()
    
            if current_time - start_time > time_limit:
                print(f"Time limit reached ({time_limit} seconds), breaking the loop.")
                break
    
            if iteration_count > max_iterations:
                print(f"Max iterations reached ({max_iterations}), breaking the loop.")
                break
        #가장 낮은 점수 확인 후 그 행 제거
            min_total_score_index = df_combined['total_scores'].idxmin()
            df_combined = df_combined.drop(min_total_score_index)

            #java line을 하나로 만들기
            lwm = ''.join(df_combined['java_code'])

            #lwm에 있는 토큰 개수 확인하는 코드
            max_tokens = 0
            token_counts = []

            tokens = tokenizer.tokenize(str(lwm))
            if len(tokens) > max_tokens:
                max_tokens = len(tokens)
                token_counts.append(len(tokens))
            # print("max: ", max_tokens)

            if len(tokens) < (512-total_bug_tokens+total_tokens_diff_df1):
                # print("max: ", len(tokens))
                break

        # 현재 존재하는 인덱스 추출
        existing_indices = df_combined.index.tolist()

        # df2에서 existing_indices에 해당하는 인덱스만 추출하여 새로운 데이터프레임 생성
        filtered_df2 = df2.loc[df2.index.isin(existing_indices)]

        # lightweight df에서 buggy line에 <bug>토큰 붙이기
        for index in different_indexes:
            df_combined.at[index, 'java_code'] = f"<bug>{df_combined.at[index, 'java_code']}</bug>"

        lwbm = ''.join(df_combined['java_code']).replace('\n', '')
        lwfm = ''.join(filtered_df2['java_code']).replace('\n', '')
        
    print("successfully created lwbm and lwfm")  # 성공 메시지 출력
    return lwbm, lwfm


def findpath_and_save(root_directory, lwresult_directory, state):
    """ Traverse directories, process file paths, and save results to txt files. """
    
    # Ensure the lwresult_directory exists
    if not os.path.exists(lwresult_directory):
        os.makedirs(lwresult_directory)

    path_pairs = {}  # Dictionary to store paths of P_dir and F_dir files
    errors = []  # List to store error information

    print("Starting directory walk...")

    # Traverse the directory structure and process files in real-time
    files_to_process = [(root, file_name) for root, _, files in os.walk(root_directory) for file_name in files]
    print(f"Found {len(files_to_process)} files to process.")

    pair_count = state.get("pair_count", 0)  # Counter for pair folders, loaded from state

    try:
        for index, (root, file_name) in enumerate(tqdm(files_to_process, initial=state.get("current_index", 0), total=len(files_to_process), desc="Processing files")):
            # 이전에 처리된 파일들은 건너뛰기
            if index < state.get("current_index", 0):
                continue

            full_path = os.path.join(root, file_name)
            
            if 'P_dir' in full_path:
                if file_name not in path_pairs:
                    path_pairs[file_name] = {}
                path_pairs[file_name]['P_dir'] = full_path
            elif 'F_dir' in full_path:
                if file_name not in path_pairs:
                    path_pairs[file_name] = {}
                path_pairs[file_name]['F_dir'] = full_path

            # Once a pair is found, process it immediately
            paths = path_pairs.get(file_name, {})
            full_path_p = paths.get('P_dir')
            full_path_f = paths.get('F_dir')

            if full_path_p and full_path_f:  # Ensure both paths are available
                try:
                    result_p, result_f = LightweightProcess(full_path_p, full_path_f)

                    if result_p is not None and result_f is not None:
                        pair_count += 1
                        pair_folder = os.path.join(lwresult_directory, f"pair_{pair_count}")
                        os.makedirs(pair_folder, exist_ok=True)

                        p_file_name = f"{file_name}_P.txt"
                        p_file_path = os.path.join(pair_folder, p_file_name)
                        with open(p_file_path, 'w', encoding='utf-8') as p_file:
                            p_file.write(result_p)

                        f_file_name = f"{file_name}_F.txt"
                        f_file_path = os.path.join(pair_folder, f_file_name)
                        with open(f_file_path, 'w', encoding='utf-8') as f_file:
                            f_file.write(result_f)

                        # Remove the processed pair from path_pairs to free memory
                        del path_pairs[file_name]
                    else:
                        raise ValueError("Result is None")
                except Exception as e:
                    # Record the error and paths
                    print(f"Error processing pair {file_name}: {e}")
                    error_info = {
                        "error": str(e),
                        "P_dir_path": full_path_p,
                        "F_dir_path": full_path_f
                    }
                    errors.append(error_info)

            # 상태 업데이트 후 저장
            state["current_index"] = index + 1
            state["pair_count"] = pair_count
            save_state(state)

    except KeyboardInterrupt:
        print("\nExecution was interrupted. Saving state...")
        save_state(state)
        print(f"State saved. You can resume from file index {state['current_index']}.")

    # Save errors to a JSON file
    if errors:
        error_file_path = os.path.join(lwresult_directory, "errors.json")
        with open(error_file_path, 'w', encoding='utf-8') as error_file:
            json.dump(errors, error_file, indent=4)
    
    print("Processing complete.")
    
def save_state(state, filename='checkpoint.json'):
    """현재 상태를 파일에 저장합니다."""
    with open(filename, 'w') as f:
        json.dump(state, f)

def load_state(filename='checkpoint.json'):
    """이전 상태를 파일에서 불러옵니다."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        return None
                    
                    
if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # Set the root directory where the files are located
    root_directory = "./sciclone/data10/mtufano/deepLearningMutants/out/bugfixes/code/"
    
    # Set the output directory for the comparison results
    lwresult_directory = "/home/selab/lwresult"


    # 이전 상태 로드
    state = load_state()
    if state is None:
        state = {
            "current_index": 0,
            "pair_count": 0
        }

    # findpath_and_save 실행
    findpath_and_save(root_directory, lwresult_directory, state)