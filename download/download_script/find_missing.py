# !cd /workspace/

import os

original_list_path = '/workspace/fl.txt'
local_data_dir = '/workspace/data/huggingface.co/main'
missing_list_path = '/workspace/missing.txt'

with open(original_list_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

missing_urls = []

for url in urls:
    relative_path = url.split("/resolve/main/")[-1]
    local_file_path = os.path.join(local_data_dir, relative_path)
    if not os.path.exists(local_file_path):
        missing_urls.append(url)

with open(missing_list_path, "w") as f:
        for url in missing_urls:
                f.write(f"{url}\n")

