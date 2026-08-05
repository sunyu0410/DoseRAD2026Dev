import os
from huggingface_hub import list_repo_tree
from huggingface_hub.hf_api import RepoFile, RepoFolder
from tqdm import tqdm
import math
import pickle

REPO_ID = "LMUK-RADONC-PHYS-RES/DoseRAD2026"
BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"

# Fetch all file metadata recursively from the remote server
files = list_repo_tree(REPO_ID, repo_type="dataset", recursive=True)

folders = []
fl = []
others = []

for f in tqdm(files, total=122410):
    if isinstance(f, RepoFile):
        fl.append(f.path)
    elif isinstance(f, RepoFolder):
        folders.append(f.path)
    else:
        others.append(f.path)

info = dict(
    file_list = fl,
    folders = folders,
    others=others
)
pickle.dump(info, open('download/info.pickle', 'wb'))

# 3. GENERATE THE DIRECTORY LAYOUT ON RUNPOD 
print("Generating target directory tree maps...")
for folder_path in folders:
    os.makedirs(os.path.join(TARGET_DATA_DIR, folder_path), exist_ok=True)

# 4. CHUNK FILES INTO PARALLEL GROUPS AND WRITE WGET SCRIPTS
chunk_size = math.ceil(len(fl) / NUM_PARALLEL_SCRIPTS)

for i in range(NUM_PARALLEL_SCRIPTS):
    script_file_path = os.path.join(OUTPUT_DIR, f"download_part_{i+1}.sh")
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size
    script_files = fl[start_idx:end_idx]
    
    with open(script_file_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Parallel wget script generated to bypass SSL verification chains\n\n")
        
        for file_path in script_files:
            remote_url = f"{BASE_URL}/{file_path}"
            local_dest = os.path.join(TARGET_DATA_DIR, file_path)
            
            # --no-check-certificate: guarantees wget also bypasses the proxy error
            # -c: enables chunk resumption if a connection stream cuts out
            f.write(f'wget --no-check-certificate -c "{remote_url}" -O "{local_dest}"\n')
            
    os.chmod(script_file_path, 0o755)
    print(f"-> Created executable script: {script_file_path} ({len(script_files)} files)")

if others:
    with open(os.path.join(OUTPUT_DIR, "others.txt"), "w") as f:
        f.write("\n".join(others))

# Use wget to download a filelist
# -x will tell how many levels of folders to keep
# wget --no-check-certificate -c --show-progress -x --cut-dirs=4 -i /workspace/download_scripts/filelist.txt -P /workspace/DoseRAD2026/
