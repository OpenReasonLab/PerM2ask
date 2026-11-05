import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from filelock import FileLock
from huggingface_hub import hf_hub_download

CACHE = Path(".cache/perm2ask")
BGE_REPO = "BAAI/bge-small-en-v1.5"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_FILES = {
    "onnx/model.onnx": "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "config.json": "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750",
    "README.md": "ddb964361a55c6e5dfca6361615854b260c9c960205d04c7520151aaa1d75837",
}
TOKENIZERS = {
    "Qwen/Qwen3.5-0.8B": (
        "2fc06364715b967f1860aea9cf38778875588b17",
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    ),
    "Qwen/Qwen3.5-2B": (
        "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    ),
    "Qwen/Qwen2.5-1.5B-Instruct": (
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
    "Qwen/Qwen2.5-3B-Instruct": (
        "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
    "meta-llama/Llama-3.2-1B-Instruct": (
        "9213176726f574b556790deb65791e0c5aa438b6",
        "5cc5f00a5b203e90a27a3bd60d1ec393b07971e8",
    ),
    "meta-llama/Llama-3.2-3B-Instruct": (
        "0cb88a4f764b7a12671c53f0838cd831a0843b95",
        "5cc5f00a5b203e90a27a3bd60d1ec393b07971e8",
    ),
}


def checksum(path, git_blob=False):
    hasher = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        hasher.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch(destination, repo, revision, files):
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(destination / ".download.lock"):
        actual = {}
        for name, expected in files.items():
            path = destination / name
            git_blob = len(expected) == 40
            valid = path.is_file() and checksum(path, git_blob) == expected
            if not valid:
                hf_hub_download(
                    repo_id=repo,
                    filename=name,
                    revision=revision,
                    local_dir=destination,
                    force_download=path.exists(),
                )
            if checksum(path, git_blob) != expected:
                raise ValueError(f"Checksum mismatch for {repo}/{name}")
            actual[name] = checksum(path)
        manifest = {"repo": repo, "revision": revision, "sha256": actual}
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination, delete=False
        ) as file:
            temporary = Path(file.name)
            json.dump(manifest, file, indent=2)
            file.write("\n")
        try:
            os.replace(temporary, destination / "source.json")
        finally:
            temporary.unlink(missing_ok=True)
    print(destination)


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify pinned model assets."
    )
    parser.add_argument(
        "--tokenizer",
        choices=sorted(TOKENIZERS),
        help="Download only the selected model's tokenizer.",
    )
    parser.add_argument("--out", type=Path, help="Local asset directory.")
    args = parser.parse_args()
    if args.tokenizer:
        repo = args.tokenizer
        revision, expected = TOKENIZERS[repo]
        path = args.out or CACHE / "tokenizers" / repo.split("/")[-1]
        fetch(path, repo, revision, {"tokenizer.json": expected})
    else:
        path = args.out or CACHE / "bge-small-en-v1.5"
        fetch(path, BGE_REPO, BGE_REVISION, BGE_FILES)


if __name__ == "__main__":
    main()
