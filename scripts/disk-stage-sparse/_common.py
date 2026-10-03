import json
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:1235"


def post(base, path, body, timeout):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def read_prompt(path):
    with open(path, encoding="utf-8", errors="ignore") as handle:
        return handle.read()


def tokenize(base, text):
    tokens = post(base, "/tokenize", {"content": text}, timeout=600)["tokens"]
    return [t["id"] if isinstance(t, dict) else t for t in tokens]
