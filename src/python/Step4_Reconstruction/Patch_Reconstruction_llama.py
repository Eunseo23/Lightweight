import os
import json
import re
from collections import defaultdict

# ✅ 경로 직접 지정
base_dir = "to/your/path"

BUG_BLOCK_PATTERN = re.compile(r"<bug>(.*?)</bug>", flags=re.DOTALL)
FIX_BLOCK_PATTERN = re.compile(r"// fix_start\n(.*?)\n// fix_end", flags=re.DOTALL)

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_single_topkey_items(json_data):
    if not isinstance(json_data, dict) or len(json_data) != 1:
        raise ValueError("JSON 최상위 구조가 단일 키가 아닙니다.")
    chart_key = next(iter(json_data))
    items = json_data[chart_key]
    if not isinstance(items, list):
        raise ValueError("최상위 키의 값이 리스트가 아닙니다.")
    return chart_key, items

def extract_bug_blocks(text):
    return BUG_BLOCK_PATTERN.findall(text)

def extract_fix_blocks(lwcp_text):
    return FIX_BLOCK_PATTERN.findall(lwcp_text)

def apply_fixes_to_original(original_text, fix_contents):
    fix_iter = iter(fix_contents)
    def repl(_match):
        return next(fix_iter)
    return re.sub(r"<bug>.*?</bug>", repl, original_text, flags=re.DOTALL)

def sorted_lwcp_keys(lwcp_dict):
    def key_fn(k):
        try:
            return (0, int(k.replace("lwcp", "")))
        except Exception:
            return (1, k)
    return sorted(lwcp_dict.keys(), key=key_fn)

def process_folder(folder_path):
    json_path = os.path.join(folder_path, "Lightweight_method_llama_diff_codellama.json")
    original_path = os.path.join(folder_path, "Original_buggy_method.txt")
    output_path = os.path.join(folder_path, "Original_candidate_patch.json")

    if not (os.path.isfile(json_path) and os.path.isfile(original_path)):
        print(f"[SKIP] {folder_path}: required files not found")
        return

    try:
        json_data = load_json(json_path)
        chart_key, items = get_single_topkey_items(json_data)
    except Exception as e:
        print(f"[SKIP] {folder_path}: {e}")
        return

    original_text = read_text(original_path)
    bug_blocks = extract_bug_blocks(original_text)
    if len(bug_blocks) == 0:
        print(f"[SKIP] {folder_path}: no <bug> blocks in original")
        return

    seen_raw_candidates = set()

    for item_index, item in enumerate(items):
        lwcp_dict = dict(item.get("lwcp") or {})
        if not lwcp_dict:
            item["lwcp"] = {}
            continue

        new_lwcp = {}
        for k in sorted_lwcp_keys(lwcp_dict):
            raw_candidate = lwcp_dict[k]

            if raw_candidate in seen_raw_candidates:
                print(f"[DEDUP] {os.path.basename(folder_path)} item#{item_index} {k}: duplicate removed")
                continue
            seen_raw_candidates.add(raw_candidate)

            fix_blocks = extract_fix_blocks(raw_candidate)
            if len(fix_blocks) == 0:
                print(f"[SKIP] {os.path.basename(folder_path)} item#{item_index} {k}: no fix blocks")
                continue
            if len(fix_blocks) != len(bug_blocks):
                print(f"[SKIP] {os.path.basename(folder_path)} item#{item_index} {k}: bug count ({len(bug_blocks)}) != fix count ({len(fix_blocks)})")
                continue

            try:
                patched_text = apply_fixes_to_original(original_text, fix_blocks)
                # ✅ 여기서 맨 끝 개행 제거
                patched_text = patched_text.rstrip("\n")
            except Exception as e:
                print(f"[SKIP] {os.path.basename(folder_path)} item#{item_index} {k}: patch apply error: {e}")
                continue

            new_lwcp[k] = patched_text   # ← 이제 개행 제거된 문자열 저장

        item["lwcp"] = new_lwcp

    out_data = {chart_key: items}
    write_json(output_path, out_data)
    print(f"[DONE] {folder_path}: saved -> {output_path}")

def process_root(root_path):
    if not os.path.isdir(root_path):
        print(f"[ERROR] root not found: {root_path}")
        return
    for name in os.listdir(root_path):
        subdir = os.path.join(root_path, name)
        if not os.path.isdir(subdir):
            continue
        process_folder(subdir)

if __name__ == "__main__":
    process_root(base_dir)