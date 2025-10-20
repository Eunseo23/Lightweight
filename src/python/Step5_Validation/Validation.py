import os
import json
import glob
import shutil
import subprocess

def run_cmd(cmd, cwd=None, timeout=300):
    try:
        result = subprocess.run(
            cmd, cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Command '{' '.join(cmd)}' timed out in {cwd}")
        return -1, "[TIMEOUT]"

priority_prefixes = ["Lang-60"]

def get_folder_name(path):
    return os.path.basename(path)

def folder_sort_key(path):
    name = get_folder_name(path)
    for idx, prefix in enumerate(priority_prefixes):
        if name.startswith(prefix):
            return (0, idx, name.lower())
    return (1, name.lower())

def validate_patch(pid, key_name, fixed_code, start, end, temp_loc):
    try:
        if not os.path.exists(temp_loc + ".bak"):
            shutil.copyfile(temp_loc, temp_loc + ".bak")
        shutil.copyfile(temp_loc + ".bak", temp_loc)

        with open(temp_loc, "r") as f:
            lines = f.readlines()
        lines[start - 1:end] = fixed_code.splitlines(keepends=True)
        with open(temp_loc, "w") as f:
            f.writelines(lines)
    except Exception as e:
        print(f"[ERROR] File modification failed: {e}")
        return None

    work_dir = f"./Lightweight-main/result_defects4j/checkout/{pid}"
    compile_code, _ = run_cmd(["defects4j", "compile"], cwd=work_dir, timeout=300)
    if compile_code != 0:
        return None
    print(f"[COMPILED] {pid} {key_name}")

    test_code, test_output = run_cmd(["defects4j", "test"], cwd=work_dir, timeout=600)

    shutil.copyfile(temp_loc + ".bak", temp_loc)
    os.remove(temp_loc + ".bak")

    if test_code == -1 and "[TIMEOUT]" in test_output:
        return "timeout"

    return "plausible" if test_code == 0 else "compiled"

def find_matching_json_key(bug_id, json_data_keys):
    candidates = [bug_id, bug_id.replace("-", "_"), bug_id.replace("_", "-")]
    for key in json_data_keys:
        if key in candidates:
            return key
    return None

def main():
    defects4j_path = "to/your/path"
    base_dir = "to/your/path"
    skipped_due_to_timeout = []
    plausible_limit_reached = []

    with open(defects4j_path) as f:
        defects_data = json.load(f)

    folders = sorted(glob.glob(os.path.join(base_dir, "*")), key=folder_sort_key)
    paths = []
    for folder in folders:
        # orig_path = os.path.join(folder, "Original_candidate_patch.json")
        orig_path = os.path.join(folder, "buggy_and_context_method.json")
        light_path = os.path.join(folder, "Lightweight_buggy_method_Context.json")
        if os.path.exists(orig_path):
            paths.append(orig_path)
        elif os.path.exists(light_path):
            paths.append(light_path)

    for json_path in paths:
        folder_name = os.path.basename(os.path.dirname(json_path))
        bug_id = folder_name

        if bug_id not in defects_data:
            print(f"[SKIP] {bug_id} not in defects4j data.")
            continue

        defect_entry = defects_data[bug_id]
        func_info = defect_entry["functions"][0] if "functions" in defect_entry else defect_entry

        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        json_key = find_matching_json_key(bug_id, json_data.keys())
        if not json_key:
            print(f"[SKIP] No matching key for {bug_id} in {json_path}")
            continue

        modified = False
        timeout_occurred = False
        plausible_count = 0  # 🔥 plausible 개수 카운트

        for entry in json_data[json_key]:
            if "lwcp" not in entry:
                continue

            lwcp_block = entry["lwcp"]
            original_keys = list(lwcp_block.keys())

            for key in sorted(original_keys, key=lambda k: int(k.replace("lwcp", ""))):
                patch_code = lwcp_block[key]
                if not patch_code.strip():
                    continue

                work_dir = f"./Lightweight-main/result_defects4j/checkout/{bug_id}"
                temp_loc = os.path.join(work_dir, func_info["loc"])

                result = validate_patch(
                    pid=bug_id,
                    key_name=key,
                    fixed_code=patch_code,
                    start=func_info.get("start_loc", func_info.get("start")),
                    end=func_info.get("end_loc", func_info.get("end")),
                    temp_loc=temp_loc
                )

                if result == "timeout":
                    print(f"[TIMEOUT] Skipping remaining patches for {bug_id}")
                    timeout_occurred = True
                    break
                elif result == "plausible":
                    lwcp_block[f"{key}_plausible"] = lwcp_block.pop(key)
                    print(f"[PLAUSIBLE] {bug_id} {key}")
                    modified = True
                    plausible_count += 1
                    if plausible_count >= 500:  # 🔥 limit 도달
                        print(f"[LIMIT REACHED] {bug_id} has {plausible_count} plausible patches. Skipping rest.")
                        plausible_limit_reached.append(bug_id)
                        break
                elif result == "compiled":
                    lwcp_block[f"{key}_compiled"] = lwcp_block.pop(key)
                    print(f"[COMPILED ONLY] {bug_id} {key}")
                    modified = True

            if timeout_occurred or plausible_count >= 500:
                break

        if timeout_occurred:
            skipped_due_to_timeout.append(bug_id)
            continue

        if modified:
            output_path = json_path.replace(".json", "_plausible.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2)
            print(f"[PLAUSIBLE] {bug_id} → {output_path}")
        else:
            print(f"[NO CHANGE] {bug_id}")

    if skipped_due_to_timeout:
        print("\n[SUMMARY] Skipped due to timeout:")
        for bug_id in skipped_due_to_timeout:
            print(f" - {bug_id}")

    if plausible_limit_reached:
        print("\n[SUMMARY] Skipped due to plausible limit (>=500):")
        for bug_id in plausible_limit_reached:
            print(f" - {bug_id}")

if __name__ == "__main__":
    main()

# def run_cmd(cmd, cwd=None):
#     result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#     return result.returncode, result.stdout + result.stderr

# def checkout_all_projects(defects4j_json_path, output_dir="to/your/path"):
#     with open(defects4j_json_path) as f:
#         defects_data = json.load(f)

#     os.makedirs(output_dir, exist_ok=True)

#     for pid in defects_data:
#         if "-" not in pid:
#             print(f"[SKIP] Invalid project-bug ID: {pid}")
#             continue

#         proj, bug_num = pid.split("-")
#         work_dir = os.path.join(output_dir, pid)
#         if os.path.exists(work_dir):
#             print(f"[SKIP] Already checked out: {work_dir}")
#             continue

#         print(f"[CHECKOUT] {pid} to {work_dir}")
#         code, output = run_cmd(["defects4j", "checkout", "-p", proj, "-v", f"{bug_num}b", "-w", work_dir])
#         if code == 0:
#             print(f"[SUCCESS] {pid}")
#         else:
#             print(f"[FAIL] {pid}:\n{output}")

# if __name__ == "__main__":
#     defects4j_path = "to/your/path"
#     checkout_all_projects(defects4j_path)