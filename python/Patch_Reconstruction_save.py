#patch  reconstruction 적용 후 저장
import re
import difflib
import pandas as pd
import os

def compare_files_to_dataframe(candidate_patch_line, buggy_method_file):
    # buggy_method_file의 내용을 읽기
    with open(buggy_method_file, 'r', encoding='utf-8') as bm_file:
        buggy_method_content = bm_file.read().strip()

    # <bug>와 </bug>를 고유한 알파벳 조합으로 대체하는 함수
    def process_content(content):
        # <bug>와 </bug>를 각각 $~$와 $~~$로 대체
        content = re.sub(r'<bug>', '$~$', content)
        content = re.sub(r'</bug>', '$~~$', content)

        # <omit>과 </omit>을 각각 $$~$$와 $$~~$$로 대체
        content = re.sub(r'<omit>', '~$~', content)
        content = re.sub(r'</omit>', '~$$~', content)

        return content

    # <bug>와 </bug>를 각각 고유한 토큰으로 대체한 후 비교
    candidate_patch_line = process_content(candidate_patch_line)
    buggy_method_content = process_content(buggy_method_content)

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
    lightweight = [re.sub(r'\$~\$', '<bug>', item) for item in lightweight]
    lightweight = [re.sub(r'\$~~\$', '</bug>', item) for item in lightweight]

    # lightweight 컬럼 추가
    df['lightweight'] = lightweight

    # bug_status 컬럼 추가
    df['bug_status'] = bug_status

    # DataFrame 반환
    return df

def count_and_extract_bug_blocks(line, tag='bug'):
    return [(m.start(), m.end()) for m in re.finditer(fr'<{tag}>.*?</{tag}>', line)]

def process_files_and_update(lightweight_candidate_patch_file, original_buggy_method_file, output_file):
    with open(original_buggy_method_file, 'r', encoding='utf-8') as obm_file:
        obm_line = obm_file.readline().strip()

    with open(output_file, 'w') as output:
        with open(lightweight_candidate_patch_file, 'r', encoding='utf-8') as lcp_file:
            lcp_lines = lcp_file.readlines()

        for i in range(min(500, len(lcp_lines))):
            lcp_line = lcp_lines[i].strip()

            obm_bug_blocks = count_and_extract_bug_blocks(obm_line, 'bug')
            lcp_bug_blocks = count_and_extract_bug_blocks(lcp_line, 'bug')

            obm_omit_blocks = count_and_extract_bug_blocks(obm_line, 'omit')
            lcp_omit_blocks = count_and_extract_bug_blocks(lcp_line, 'omit')

            if len(obm_bug_blocks) == len(lcp_bug_blocks) and len(obm_omit_blocks) == len(lcp_omit_blocks):
                updated_obm_line = obm_line
                offset = 0

                for (obm_start_idx, obm_end_idx), (lcp_start_idx, lcp_end_idx) in zip(obm_bug_blocks, lcp_bug_blocks):
                    replacement = lcp_line[lcp_start_idx:lcp_end_idx]
                    updated_obm_line = updated_obm_line[:obm_start_idx + offset] + replacement + updated_obm_line[obm_end_idx + offset:]
                    offset += len(replacement) - (obm_end_idx - obm_start_idx)

                for (obm_start_idx, obm_end_idx), (lcp_start_idx, lcp_end_idx) in zip(obm_omit_blocks, lcp_omit_blocks):
                    replacement = lcp_line[lcp_start_idx:lcp_end_idx]
                    updated_obm_line = updated_obm_line[:obm_start_idx + offset] + replacement + updated_obm_line[obm_end_idx + offset:]
                    offset += len(replacement) - (obm_end_idx - obm_start_idx)

                updated_obm_line = re.sub(r'<bug>|</bug>|<omit>|</omit>', '', updated_obm_line)
                output.write(updated_obm_line + '\n')
            else:
                output.write(obm_line + '\n')

def process_folder(folder_path):
    # 각 파일을 저장할 변수
    lightweight_buggy_methods = []
    original_buggy_methods = []
    candidate_patches_buggy_blocks = []

    # 숫자 추출 함수
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)  # 파일명에서 숫자 추출
        return match.group(1) if match else None

    # 폴더 내 파일들 확인
    for filename in os.listdir(folder_path):
        if filename.startswith('Lightweight_buggy_method'):
            lightweight_buggy_methods.append(os.path.join(folder_path, filename))
        elif filename.startswith('Original_buggy_method') and 'by_line' not in filename:
            original_buggy_methods.append(os.path.join(folder_path, filename))
        elif filename.startswith('candidate_patches_buggy_block'):
            candidate_patches_buggy_blocks.append(os.path.join(folder_path, filename))

    # 각 파일 쌍 확인 및 처리
    for lightweight_buggy_method in lightweight_buggy_methods:
        lw_number = extract_number(os.path.basename(lightweight_buggy_method))

        # 같은 숫자의 Original_buggy_method와 candidate_patches_buggy_block 파일 찾기
        original_buggy_method = next((obm for obm in original_buggy_methods if extract_number(os.path.basename(obm)) == lw_number), None)
        candidate_patches_buggy_block = next((cpb for cpb in candidate_patches_buggy_blocks if extract_number(os.path.basename(cpb)) == lw_number), None)

        # 세 파일 모두 존재하는지 확인
        if lightweight_buggy_method and original_buggy_method and candidate_patches_buggy_block:
            output_suffix = f"{lw_number}" if lw_number else ""
            lightweight_candidate_patch_file = os.path.join(folder_path, f"Lightweight_candidate_patches{output_suffix}.txt")
            original_candidate_patches_file = os.path.join(folder_path, f"Original_candidate_patches{output_suffix}.txt")

            # 경로 출력 및 처리 시작
            print(f"Processing files with number {lw_number} in folder: {folder_path}")
            compare_files_and_process(lightweight_buggy_method, candidate_patches_buggy_block, original_buggy_method, lightweight_candidate_patch_file, original_candidate_patches_file)
        else:
            print(f"Matching files for number {lw_number} are missing in {folder_path}")

def process_all_folders(base_directory):
    if not os.path.exists(base_directory):
        print(f"Error: {base_directory} does not exist.")
        return

    print(f"Processing base directory: {base_directory}")
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)
        if os.path.isdir(folder_path) and folder_name.startswith('Chart_'):
            process_folder(folder_path)

def compare_files_and_process(lightweight_buggy_method, candidate_patches_buggy_block, original_buggy_method, output_lightweight_file, output_original_file):
    with open(candidate_patches_buggy_block, 'r', encoding='utf-8') as cp_file:
        lines = cp_file.readlines()
        lightweight_candidate_patches = []

        for i in range(min(500, len(lines))):
            candidate_patch_line = lines[i].strip()
            df = compare_files_to_dataframe(candidate_patch_line, lightweight_buggy_method)

            lightweight_candidate_patch = ''.join(df['lightweight'].apply(lambda x: x.replace('\n', '')))
            lightweight_candidate_patches.append(lightweight_candidate_patch)

    with open(output_lightweight_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lightweight_candidate_patches))

    process_files_and_update(output_lightweight_file, original_buggy_method, output_original_file)

def main():
    # 전체 폴더 경로 설정 및 함수 실행
    base_directory = os.path.join("C:\\","Users","UOS","Desktop", "Chart_lightweight")
    process_all_folders(base_directory)

# main 함수 호출
if __name__ == "__main__":
    main()