import pandas as pd
from transformers import RobertaTokenizer

# 파일 경로 설정
file_path1 = 'OriginalMethods1.txt'
file_path2 = 'RevisedMethods1.txt'

def txt_to_dataframe(file_path):
    # 파일 읽기
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 데이터프레임으로 변환
    df = pd.DataFrame({
        'java_code': lines
    })
    
    return df

# 데이터프레임 생성
df1 = txt_to_dataframe(file_path1)
df2 = txt_to_dataframe(file_path2)

# 데이터프레임을 텍스트 파일로 저장
df1.to_csv('OriginalDf.txt', index=False, sep='\t')
df2.to_csv('RevisedDf.txt', index=False, sep='\t')

# RobertaTokenizer 로드
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def calculate_total_tokens(df):
    """
    주어진 데이터프레임의 모든 텍스트 컬럼을 반복하여 총 토큰 개수를 계산합니다.
    """
    total_tokens = 0
    for column in df.columns:
        for text in df[column].astype(str):  # 텍스트를 문자열로 변환
            tokens = tokenizer.tokenize(text)  # 텍스트를 토큰화
            total_tokens += len(tokens)  # 토큰 개수 추가
    return total_tokens

# 데이터프레임 1의 총 토큰 개수 계산
total_tokens_df1 = calculate_total_tokens(df1)
print(f"Total number of tokens in df1: {total_tokens_df1}")

# 데이터프레임 2의 총 토큰 개수 계산
total_tokens_df2 = calculate_total_tokens(df2)
print(f"Total number of tokens in df2: {total_tokens_df2}")

# 비교하여 결과 출력 
# 둘 중 하나라도 토큰 개수가 160이 넘으면 수식을 적용한다.
if total_tokens_df1 > total_tokens_df2:
    print("df1 has more tokens than df2.")
elif total_tokens_df1 < total_tokens_df2:
    print("df2 has more tokens than df1.")
else:
    print("Both dataframes have the same number of tokens.")
    
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

different_indexes = compare_df_find_index(df1, df2)

print(f"Indexes with different rows: {different_indexes}")