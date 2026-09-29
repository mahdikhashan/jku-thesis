import sys
from pathlib import Path

from huggingface_hub import HfApi

import config as C

STAGES = {
    "stage1": [C.STAGE1_CKPT],
    "stage2": [C.STAGE2_CKPT],
    "code": ["config.py", "lizard_attention.py", "train.py", "evaluate.py"],
}


def push(stage, repo_id=C.HF_REPO):
    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    for path in map(Path, STAGES[stage]):
        if not path.exists():
            print(f"skipped {path}: file not found")
            continue
        api.upload_file(path_or_fileobj=path, path_in_repo=path.name,
                        repo_id=repo_id,
                        commit_message=f"{stage}: {path.name}")
        print(f"uploaded {path} to {repo_id}")


if __name__ == "__main__":
    for name in sys.argv[1:] or STAGES:
        push(name)
