import pandas as pd
import re
import tokenize
from transformers import RobertaTokenizer
import time

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def file_to_lines(file_path):
    # 파일 경로를 받아서 파일 내용을 lines 리스트로 변환
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]  # 줄바꿈 문자 제거
        return lines
    except FileNotFoundError:
        raise FileNotFoundError(f"File at {file_path} not found.")
    except Exception as e:
        raise ValueError(f"An error occurred while reading the file: {e}")

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

    # <bug></bug> 태그를 기준으로 문자열을 분리
    parts = re.split(r'(<bug>|</bug>)', line)

    inside_bug_tag = False
    bug_tag_tokens_count = 0
    for part in parts:
        if part == '<bug>':
            inside_bug_tag = True
            bug_tag_tokens_count += 1  # '<bug>' 자체도 하나의 토큰으로 간주
        elif part == '</bug>':
            inside_bug_tag = False
            bug_tag_tokens_count += 1  # '</bug>' 자체도 하나의 토큰으로 간주
        elif inside_bug_tag:
            # <bug>...</bug> 안에 있는 부분은 제외하고 카운트 (태그 내부는 무시)
            bug_tag_tokens_count += len(re.findall(r'[a-zA-Z_$0-9]+|[><]', part))  # 토큰 개수 카운트
        else:
            # <bug></bug> 밖에 있는 부분에 대해서는 일반적인 식별자 추출
            matches = re.findall(r'\b[a-zA-Z_$][a-zA-Z_$0-9]*\b', part)
            
            # 자바 기본 데이터 타입 제외
            for match in matches:
                if match not in java_types:
                    identifiers.add(match)
            
            # 연산자 추출 및 추가
            for operator in operators:
                if operator in part:
                    identifiers.add(operator)
    
    # <bug> 태그 안에 있는 토큰이 2개 이상이면 개수에 포함
    if bug_tag_tokens_count > 4:
        identifiers.add(f'bug_tag_tokens_{bug_tag_tokens_count}')

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

def lightweightdefects4j(inputFilePath):
    
    lines = file_to_lines(inputFilePath)
    
    df = list_to_dataframe(lines)

    #버그 위치 인덱스 번호
    bug_indices = df[df['java_code'].str.startswith('<bug>')].index.tolist()

    total_tokens = calculate_total_tokens(df, 'java_code')

    if total_tokens < 512:
        lwbm = ''.join(df['java_code']).replace('\n', '')
    
    else:
        #식별자 추출
        line_identifiers = []
        for index, row in df.iterrows():
            identifiers = extract_identifiers(row['java_code'])
            line_identifiers.append(identifiers)
        df['identifiers'] = pd.Series(line_identifiers)

        # 각 bug_line에 대해 거리 점수 계산
        dis_scores = {}
        for bug_line in bug_indices:
            dis_scores[f'dis_score_{bug_line}'] = [calculate_dis_score(line_number, bug_line) for line_number in df.index]

        dis_scores_df = pd.DataFrame(dis_scores)

        int_scores = {}
        for target_index in bug_indices:
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
        df_combined = pd.concat([df, int_scores_df, dis_scores_df], axis=1)

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
            lwbm = ''.join(df_combined['java_code'])

            #lwm에 있는 토큰 개수 확인하는 코드
            max_tokens = 0
            token_counts = []

            tokens = tokenizer.tokenize(str(lwbm))
            if len(tokens) > max_tokens:
                max_tokens = len(tokens)
                token_counts.append(len(tokens))
            # print("max: ", max_tokens)

            if len(tokens) < 512:
                # print("max: ", len(tokens))
                break
    print("successfully created lwbm")  # 성공 메시지 출력
    return lwbm

if __name__ == "__main__":

    inputFilePath = "C:/Users/UOS/Desktop/data/chart2/Original_buggy_method2.txt"
    lightweightdefects4j(inputFilePath)
    