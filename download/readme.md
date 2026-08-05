```bash
# Parallel download
cat fl-0.txt | xargs -n 1 -P 80 wget --no-check-certificate -c -x --cut-dirs=4 -q -P data

# Count file
watch -n 10 "find /workspace/data -type f | wc -l"

# CPU / RAM usage
htop
```