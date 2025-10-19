#patch  reconstruction 적용 후 저장
import re
import difflib
import pandas as pd
import os
import json
import shutil
from transformers import RobertaTokenizer

def compare_files_to_dataframe(candidate_patch_line, buggy_method_code_str):
    # <bug>와 </bug>를 고유한 알파벳 조합으로 대체하는 함수
    def process_content(content):
        # <bug>와 </bug>를 각각 $~$와 $~~$로 대체
        content = re.sub(r'<bug>', '$~$', content)
        content = re.sub(r'</bug>', '$~~$', content)

        # <omit>과 </omit>을 각각 ~\$~ 와 ~\$\$~로 대체
        content = re.sub(r'<omit>', '~$~', content)
        content = re.sub(r'</omit>', '~$$~', content)

        return content.strip()

    # 문자열을 전처리
    candidate_patch_line = process_content(candidate_patch_line)
    buggy_method_content = process_content(buggy_method_code_str)

    # 두 파일의 내용 비교
    data = []
    matcher = difflib.SequenceMatcher(None, buggy_method_content, candidate_patch_line)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        buggy_method_part = buggy_method_content[i1:i2]  # 토큰들을 다시 결합
        candidate_patch_part = candidate_patch_line[j1:j2]

        # 데이터에 tag 정보도 추가
        data.append([candidate_patch_part, buggy_method_part, tag])

    # DataFrame 생성 (tag 정보 포함)
    df = pd.DataFrame(data, columns=['Candidate Patch Line', 'Buggy Method Content', 'Tag'])

    # 새로운 컬럼을 위한 리스트 생성
    lightweight = []
    bug_status = []

    # 각 행을 확인하고 값 비교
    for idx, row in df.iterrows():
        # 'Candidate Patch Line'과 'Buggy Method Content'가 동일한지 확인
        if row['Tag'] == 'equal':
            lightweight.append(row['Candidate Patch Line'])
        else:
            lightweight.append('')


        # $~$와 $~~$를 탐지하여 상태 기억
        has_bug_start = '$~$' in row['Candidate Patch Line'] or '$~$' in row['Buggy Method Content']
        has_bug_end = '$~~$' in row['Candidate Patch Line'] or '$~~$' in row['Buggy Method Content']

        total_bug_count = int(has_bug_start) + int(has_bug_end)

        # 첫 번째 조건: has_bug_start와 has_bug_end가 각각 하나씩만 있을 때
        if total_bug_count == 2:
            bug_status.append(0)  # 둘 다 하나씩만 있으면 0

        # 두 번째 조건: has_bug_start와 has_bug_end가 있고, 추가로 하나 이상 있는 경우
        elif total_bug_count >= 3:
            bug_status.append(3)  # 둘 중 하나 이상이 추가로 있으면 3

        # 세 번째 조건: 둘 중 하나만 있을 때
        elif total_bug_count == 1:
            bug_status.append(1)  # 하나만 있으면 1

        # 네 번째 조건: 둘 다 없으면 None
        else:
            bug_status.append(None)  # 없으면 None

        # Tag가 delete이고 Buggy Method Content가 $~~$ 혹은 $~$이며, Candidate Patch Line이 빈 경우
        if row['Tag'] == 'delete' and row['Buggy Method Content'].strip() in ['$~$', '$~~$'] and row['Candidate Patch Line'].strip() == '':
            lightweight[idx] = row['Buggy Method Content']  # 해당 인덱스의 lightweight에 Buggy Method Content 값 반영

    # replace이고 Candidate Patch Line이 비어있지 않으며, 특정 조건에 따라 처리
    for idx, row in df.iterrows():
        if row['Tag'] == 'replace' and row['Candidate Patch Line'].strip() != '':
            count_tilde_1 = row['Buggy Method Content'].count('$~$')
            count_tilde_2 = row['Buggy Method Content'].count('$~~$')
            total_count = count_tilde_1 + count_tilde_2

            # 총합이 4개 이상일 경우 처리
            if total_count >= 4:
                # 앞에 $~$를 붙이고 뒤에 $~~$를 붙인 뒤, 나머지 개수만큼 추가
                lightweight[idx] = '$~$ ' + row['Candidate Patch Line'] + ' $~~$ ' + \
                                  (' $~$ $~~$' * ((total_count - 2) // 2)).strip()

            # Buggy Method Content가 $~$로 시작하고 $~~$로 끝나고 그 사이에 다른 문자가 있을 때
            elif row['Buggy Method Content'].strip().startswith('$~$') and \
                row['Buggy Method Content'].strip().endswith('$~~$') and \
                len(row['Buggy Method Content'].strip()[3:-4].strip()) > 0:
                # Candidate Patch Line 앞에는 $~$, 뒤에는 $~~$를 붙여 lightweight에 저장
                lightweight[idx] = '$~$ ' + row['Candidate Patch Line'] + ' $~~$'

    # 이후 기존의 다른 조건들
    # bug_status가 NaN이고 tag가 replace인 경우, Candidate Patch Line을 lightweight에 넣음
    for idx, row in df.iterrows():
        if pd.isna(bug_status[idx]) and row['Tag'] == 'replace':
            lightweight[idx] = row['Candidate Patch Line']
        # tag가 insert이고 Buggy Method Content가 비어있으면 Candidate Patch Line을 lightweight에 넣음
        elif row['Tag'] == 'insert' and row['Buggy Method Content'].strip() == '':
            lightweight[idx] = row['Candidate Patch Line']

    # bug_status가 0이고 tag가 replace인 경우, Candidate Patch Line 내용을 lightweight에 넣고 Buggy Method Content를 확인
    for idx, status in enumerate(bug_status):
        if status == 0 and df.at[idx, 'Tag'] == 'replace' and df.at[idx, 'Candidate Patch Line'].strip() != '':
            buggy_content = df.at[idx, 'Buggy Method Content']

            # 새로운 조건 추가: $~$과 $~~$의 개수를 세고 순서대로 기억, 그리고 마지막이 $~$ 또는 $~~$로 끝나지 않은 경우에만 적용
            tokens = re.findall(r'\$~\$|\$~~\$', buggy_content)  # $~$와 $~~$의 순서를 기록
            if len(tokens) >= 3 and not (df.at[idx, 'Candidate Patch Line'].strip().endswith('$~$') or df.at[idx, 'Candidate Patch Line'].strip().endswith('$~~$')):
                # Candidate Patch Line에 기억한 순서대로 $~$와 $~~$ 추가
                lightweight[idx] = df.at[idx, 'Candidate Patch Line'] + ' ' + ' '.join(tokens)
            else:
                # 기존 조건: $~$ $~~$ 패턴이 있을 때
                if '$~$ $~~$' in buggy_content:
                    lightweight[idx] = '$~$ ' + df.at[idx, 'Candidate Patch Line'] + ' $~~$'
                elif '$~~$ $~$' in buggy_content:
                    lightweight[idx] = df.at[idx, 'Candidate Patch Line'] + ' $~~$ $~$'

        # bug_status가 0이고 tag가 replace인 경우, Candidate Patch Line이 비어있고 Buggy Method Content가 $~~$ $~$로 시작하면
        elif status == 0 and df.at[idx, 'Tag'] == 'replace' and df.at[idx, 'Candidate Patch Line'].strip() == '' and df.at[idx, 'Buggy Method Content'].strip().startswith('$~~$ $~$'):
            lightweight[idx] = df.at[idx, 'Buggy Method Content']

        # Tag가 'replace'이고, Buggy Method Content가 $~~$로 시작할 경우
        elif row['Tag'] == 'replace' and status ==  1 and row['Buggy Method Content'].strip().startswith('$~~$'):
            lightweight[idx] = row['Buggy Method Content']  # Buggy Method Content 값을 그대로 lightweight에 넣음

        # 기존 조건: Tag가 delete이고 Buggy Method Content가 $~~$ 혹은 $~$이며, Candidate Patch Line이 빈 경우
        if row['Tag'] == 'delete' and row['Buggy Method Content'].strip() in ['$~$', '$~~$'] and row['Candidate Patch Line'].strip() == '':
            lightweight[idx] = row['Buggy Method Content']

    # 보완된 조건 추가
    for idx, status in enumerate(bug_status):
        if status == 0 and df.at[idx, 'Tag'] == 'delete' and df.at[idx, 'Candidate Patch Line'].strip() == '':
            buggy_content = df.at[idx, 'Buggy Method Content']

            # $~$와 $~~$가 나타나는 순서 기록
            tokens = re.findall(r'\$~\$|\$~~\$', buggy_content)

            # 순서대로 한 칸 띄워서 lightweight에 넣기
            lightweight[idx] = ' '.join(tokens)

    ## 새로운 조건 추가: tag가 replace이고 bug_status가 1이고 Buggy Method Content가 $~$로 시작할 때
    for idx, status in enumerate(bug_status):
        if status == 1 and df.at[idx, 'Tag'] == 'replace':
            buggy_content = df.at[idx, 'Buggy Method Content'].strip()
            candidate_line = df.at[idx, 'Candidate Patch Line'].strip()

            # Buggy Method Content가 $~$로 시작할 때
            if buggy_content.startswith('$~$'):
                lightweight[idx] = '$~$ ' + candidate_line

            # 추가된 조건: Buggy Method Content가 $~~$로 끝나고, Candidate Patch Line이 비어 있지 않을 때
            if buggy_content.endswith('$~~$') and candidate_line != '':
                # 기존 lightweight 값에 $~~$를 추가
                lightweight[idx] = candidate_line + ' $~~$'

            # Buggy Method Content가 $~~$로 시작하고 Candidate Patch Line이 비어 있지 않을 때
            if buggy_content.startswith('$~~$') and candidate_line != '':
                # lightweight 인덱스에 '$~~$' + candidate_line을 저장
                lightweight[idx] = '$~~$ ' + candidate_line

    # 새로운 조건 추가: tag가 delete이고 bug_status가 1이고 Buggy Method Content가 $~~$로 시작하고 Candidate Patch Line이 비었을 때
    for idx, status in enumerate(bug_status):
        if status == 1 and df.at[idx, 'Tag'] == 'delete' and df.at[idx, 'Buggy Method Content'].strip().startswith('$~~$') and df.at[idx, 'Candidate Patch Line'].strip() == '':
            lightweight[idx] = '$~~$'

        # 새로운 조건 추가: tag가 delete이고 bug_status가 1이고 Buggy Method Content가 $~$로 시작하고 Candidate Patch Line이 비었을 때
        elif status == 1 and df.at[idx, 'Tag'] == 'delete' and df.at[idx, 'Buggy Method Content'].strip().startswith('$~$') and df.at[idx, 'Candidate Patch Line'].strip() == '':
            lightweight[idx] = '$~$'

    # **추가된 조건 1**: bug_status가 1이고 tag가 delete이며, Buggy Method Content에 $~$가 있고 Candidate Patch Line이 비어 있을 때
    for idx, status in enumerate(bug_status):
        if status == 1 and df.at[idx, 'Tag'] == 'delete' and '$~$' in df.at[idx, 'Buggy Method Content'] and df.at[idx, 'Candidate Patch Line'].strip() == '':
            lightweight[idx] = '$~$'
        elif status == 1 and df.at[idx, 'Tag'] == 'delete' and '$~~$' in df.at[idx, 'Buggy Method Content'] and df.at[idx, 'Candidate Patch Line'].strip() == '':
            lightweight[idx] = '$~~$'

    # **추가된 조건 2**: df의 맨 마지막 줄이고 bug_status가 0이고 tag가 delete이며, Buggy Method Content가 $~~$ $~$로 시작할 때
    if len(df) > 0:  # 데이터가 존재할 때
        last_idx = len(df) - 1  # 마지막 인덱스 확인
        if bug_status[last_idx] == 0 and df.at[last_idx, 'Tag'] == 'delete' and df.at[last_idx, 'Buggy Method Content'].strip().startswith('$~~$ $~$'):
            lightweight[last_idx] = '$~~$ $~$ $~~$'

    # 추가된 조건: bug_status가 0이고, tag가 replace이며, Buggy Method Content가 $~~$ $~$로 시작하고 Candidate Patch Line이 비어있지 않으면
    for idx, status in enumerate(bug_status):
        if status == 0 and df.at[idx, 'Tag'] == 'replace' and df.at[idx, 'Buggy Method Content'].strip().startswith('$~~$ $~$') and df.at[idx, 'Candidate Patch Line'].strip() != '':
            lightweight[idx] = '$~~$ $~$ ' + df.at[idx, 'Candidate Patch Line']

        # 추가된 조건: bug_status가 0이고, tag가 replace이며, Buggy Method Content가 $~~$ $~$로 끝나고 Candidate Patch Line이 비어있지 않으면
        elif status == 0 and df.at[idx, 'Tag'] == 'replace' and df.at[idx, 'Buggy Method Content'].strip().endswith('$~~$ $~$') and df.at[idx, 'Candidate Patch Line'].strip() != '':
            lightweight[idx] = df.at[idx, 'Candidate Patch Line'] + ' $~~$ $~$'

    # 최종적으로 lightweight에 있는 $~$와 $~~$를 <bug>와 </bug>로 변환
    lightweight = [re.sub(r'\$~\$', '<bug> ', item) for item in lightweight]
    lightweight = [re.sub(r'\$~~\$', ' </bug>', item) for item in lightweight]

    # lightweight 컬럼 추가
    df['lightweight'] = lightweight

    # bug_status 컬럼 추가
    df['bug_status'] = bug_status

    # DataFrame 반환
    return df


def apply_patch_to_original(lightweight_patch: str, original_buggy_method: str) -> str:
    def extract_tag_blocks(text, tag='bug'):
        return [(m.start(), m.end()) for m in re.finditer(fr'<{tag}>.*?</{tag}>', text)]

    def extract_tag_contents(text, tag='bug'):
        return [m.group(1) for m in re.finditer(fr'<{tag}>(.*?)</{tag}>', text, flags=re.DOTALL)]

    obm = original_buggy_method
    lcp = lightweight_patch

    obm_bug_blocks = extract_tag_blocks(obm, 'bug')
    lcp_bug_contents = extract_tag_contents(lcp, 'bug')

    obm_omit_blocks = extract_tag_blocks(obm, 'omit')
    lcp_omit_contents = extract_tag_contents(lcp, 'omit')

    if len(obm_bug_blocks) == len(lcp_bug_contents) and len(obm_omit_blocks) == len(lcp_omit_contents):
        updated = obm
        offset = 0

        # bug 블록 교체
        for (obm_start, obm_end), replacement in zip(obm_bug_blocks, lcp_bug_contents):
            replacement_with_tags = f"<bug>{replacement}</bug>"
            updated = updated[:obm_start + offset] + replacement_with_tags + updated[obm_end + offset:]
            offset += len(replacement_with_tags) - (obm_end - obm_start)

        # omit 블록 교체
        for (obm_start, obm_end), replacement in zip(obm_omit_blocks, lcp_omit_contents):
            replacement_with_tags = f"<omit>{replacement}</omit>"
            updated = updated[:obm_start + offset] + replacement_with_tags + updated[obm_end + offset:]
            offset += len(replacement_with_tags) - (obm_end - obm_start)

        # ✅ 태그 제거 시 공백으로 대체
        updated = re.sub(r'\s*</?(bug|omit)>\s*', ' ', updated)

        # ✅ 연속된 공백은 하나로 줄임
        updated = re.sub(r'\s{2,}', ' ', updated)

        return updated.strip()
    else:
        return obm

# ✅ RobertaTokenizer 초기화
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def get_token_count(text):
    return len(tokenizer.tokenize(text))

def process_json_folder(base_dir, target_folder):
    folder_path = os.path.join(base_dir, target_folder)

    json_file = None
    original_buggy_file = None

    # 파일 탐색
    for filename in os.listdir(folder_path):
        if filename.startswith("Lightweight_buggy_method_Context") and "diff" not in filename and "codellama" not in filename and filename.endswith(".json"):
        # if filename.startswith("Lightweight_buggy_method_Context") and "diff" not in filename and filename.endswith(".json"):
            json_file = os.path.join(folder_path, filename)
        elif filename.startswith("Original_buggy_method") and "by_line" not in filename and filename.endswith(".txt"):
            original_buggy_file = os.path.join(folder_path, filename) 

    if not json_file or not original_buggy_file:
        print(f"[ERROR] 필요한 파일이 {folder_path}에 없습니다.")
        return

    # ✅ 토큰 수 체크
    with open(original_buggy_file, 'r', encoding='utf-8') as f:
        original_buggy_method_text = f.read()

    token_count = get_token_count(original_buggy_method_text)
    if token_count < 200:                                          ###############################################
        print(f"[SKIP] {target_folder} → 토큰 수 {token_count} < 200")
        return    

    print(f"[INFO] 처리 시작: {json_file}")

    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    chart_key = list(json_data.keys())[0]
    items = json_data[chart_key]

    # 저장 파일명: Lightweight → Original로 교체
    output_json_filename = os.path.basename(json_file).replace(
        "Lightweight_buggy_method_Context", "Original_candidate_patch"
    )
    output_json_path = os.path.join(folder_path, output_json_filename)

    # original buggy method 텍스트 전체 로딩
    with open(original_buggy_file, 'r', encoding='utf-8') as f:
        original_buggy_method_text = f.read()

    for item in items:
        buggy_code = item.get("lwbm", "")
        buggy_code = re.sub(r"<context>.*?</context>", "", buggy_code, flags=re.DOTALL)

        lwcp_block = item.get("lwcp", None)
        if not lwcp_block:
            print(f"[WARNING] id={item.get('id')} 에 lwcp 없음. 건너뜀.")
            continue

        for key in sorted(lwcp_block.keys(), key=lambda k: int(k.replace('lwcp', ''))):
            candidate_patch = lwcp_block[key]

            # Step 1: lightweight 결과 생성
            df = compare_files_to_dataframe(candidate_patch, buggy_code)
            lightweight_patch = ''.join(df['lightweight'].apply(lambda x: x.replace('\n', '')))

            # Step 2: original 결과 생성
            final_patch = apply_patch_to_original(lightweight_patch, original_buggy_method_text)

            lwcp_block[key] = final_patch

        # ✅ 최종 필터링: <bug> 또는 </bug> 포함된 항목 제거
        filtered_lwcp = {}
        seen_values = set()

        for key in sorted(lwcp_block.keys(), key=lambda k: int(k.replace("lwcp", ""))):
            val = lwcp_block[key]

            # <bug> 태그가 포함되어 있으면 스킵
            if '<bug>' in val or '</bug>' in val:
                continue

            # 중복 값이면 스킵
            if val in seen_values:
                continue

            # 유일한 값만 추가
            filtered_lwcp[key] = val
            seen_values.add(val)

        item["lwcp"] = filtered_lwcp

    # 결과 저장
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)

    print(f"[DONE] 저장 완료: {output_json_path}")
    
def process_json_file(json_file_path, original_buggy_file_path):
    if not os.path.exists(json_file_path) or not os.path.exists(original_buggy_file_path):
        print(f"[ERROR] 파일 경로가 존재하지 않습니다.")
        return

    with open(original_buggy_file_path, 'r', encoding='utf-8') as f:
        original_buggy_method_text = f.read()

    token_count = get_token_count(original_buggy_method_text)
    if token_count < 200:                                        ###############################################
        print(f"[SKIP] {original_buggy_file_path} → 토큰 수 {token_count} < 200")
        return

    print(f"[INFO] 처리 시작: {json_file_path}")

    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    chart_key = list(json_data.keys())[0]
    items = json_data[chart_key]

    output_json_filename = os.path.basename(json_file_path).replace(
        "Lightweight_buggy_method_Context", "Original_candidate_patch"
    )
    output_json_path = os.path.join(os.path.dirname(json_file_path), output_json_filename)

    for item in items:
        buggy_code = item.get("lwbm", "")
        buggy_code = re.sub(r"<context>.*?</context>", "", buggy_code, flags=re.DOTALL)

        lwcp_block = item.get("lwcp", None)
        if not lwcp_block:
            print(f"[WARNING] id={item.get('id')} 에 lwcp 없음. 건너뜀.")
            continue

        for key in sorted(lwcp_block.keys(), key=lambda k: int(k.replace('lwcp', ''))):
            candidate_patch = lwcp_block[key]
            df = compare_files_to_dataframe(candidate_patch, buggy_code)
            lightweight_patch = ''.join(df['lightweight'].apply(lambda x: x.replace('\n', '')))
            final_patch = apply_patch_to_original(lightweight_patch, original_buggy_method_text)
            lwcp_block[key] = final_patch

        item["lwcp"] = {k: v for k, v in lwcp_block.items() if "<bug>" not in v and "</bug>" not in v}
    
    value_to_keys = defaultdict(list)

    # 값 → (item, key) 매핑 수집
    for item in items:
        for key, val in item.get("lwcp", {}).items():
            value_to_keys[val].append((item, key))

    # 중복된 값은 모두 제거
    for val, occurrences in value_to_keys.items():
        if len(occurrences) > 1:  # 2번 이상 등장하면 중복
            for item, key in occurrences:
                if key in item["lwcp"]:
                    del item["lwcp"][key]

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)

    print(f"[DONE] 저장 완료: {output_json_path}")

# 🟦 main 함수 추가
def main():
    base_dir = "to/your/path"
    target_folder = "target_folder"

    target_path = os.path.join(base_dir, target_folder)

    if not os.path.exists(target_path):
        print(f"[ERROR] 경로가 존재하지 않습니다: {target_path}")
        return

    # 하위 폴더 순회
    for subfolder in os.listdir(target_path):
        subfolder_path = os.path.join(target_folder, subfolder)
        full_path = os.path.join(base_dir, subfolder_path)

        if not os.path.isdir(full_path):
            continue

        # ✅ 파일 개수 검사
        files = os.listdir(full_path)
        context_files = [
            f for f in files
            if f.startswith("Lightweight_buggy_method_Context") and "diff" not in f and "codellama" not in f and f.endswith(".json")
            # if f.startswith("Lightweight_buggy_method_Context_codellama") and "diff" not in f and f.endswith(".json")
        ]
        original_files = [
            f for f in files
            if f.startswith("Original_buggy_method") and "by_line" not in f and f.endswith(".txt")
        ]

        if len(context_files) == 1 and len(original_files) == 1:
            print(f"\n[INFO] 하위 폴더 처리 시작: {subfolder_path}")
            process_json_folder(base_dir, subfolder_path)
        else:
            print(f"\n[SKIP] {subfolder_path} → 처리하지 않음 (context: {len(context_files)}, original: {len(original_files)})")

            
# 🟨 실행 조건
if __name__ == "__main__":
    #전체 폴더 처리할때
    main() 
    
    # #개별 폴더 직접 처리할때
    # json_path = "to/your/path"
    # txt_path  = "to/your/path"
    # process_json_file(json_path, txt_path)
