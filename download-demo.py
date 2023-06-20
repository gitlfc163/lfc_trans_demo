

from huggingface_hub import snapshot_download
snapshot_download(repo_id="bert-base-chinese")

# snapshot_download(repo_id="bert-base-chinese", ignore_regex=["*.h5", "*.ot", "*.msgpack"])