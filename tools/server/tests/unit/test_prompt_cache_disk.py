import math
import struct

from utils import *


MIB = 1024 * 1024

LONG_PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom. "
) * 4


def make_server() -> ServerProcess:
    server = ServerPreset.tinygemma3()
    server.no_mmproj = True
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.kv_unified = True
    return server


def completion(server: ServerProcess, prompt: str, id_slot: int | None = None) -> dict:
    data = {
        "prompt": prompt,
        "cache_prompt": True,
    }
    if id_slot is not None:
        data["id_slot"] = id_slot
    res = server.make_request("POST", "/completion", data=data)
    assert res.status_code == 200
    return res.body["timings"]


def cache_files(cache_dir):
    return [path for path in cache_dir.iterdir() if path.is_file()]


def cache_data_files(cache_dir):
    return [path for path in cache_files(cache_dir) if path.name.endswith(".bin")]


def meta_checkpoint_count(path) -> int:
    # magic[8], version, flags, key_size, tokens_size, payload_size, main_size, drft_size
    header = path.read_bytes()[:64]
    assert header[:8] == b"LLPCACHE"
    return struct.unpack_from("<Q", header, 56)[0]


def test_disk_cache_restores_across_server_restart(tmp_path):
    server = make_server()
    server.cache_ram = 0
    server.cache_disk = str(tmp_path)
    server.cache_disk_max = 256
    server.start()

    timings_full = completion(server, LONG_PROMPT, 0)
    completion(server, "This prompt moves the first prompt into the disk cache.", 1)

    cached_files = {path.name for path in cache_files(tmp_path)}
    assert any(name.endswith(".bin") for name in cached_files)
    assert any(name.endswith(".bin.meta") for name in cached_files)

    # the test model uses SWA, so the cached state must carry context checkpoints
    metas = [path for path in cache_files(tmp_path) if path.name.endswith(".bin.meta")]
    assert all(meta_checkpoint_count(path) > 0 for path in metas)

    server.stop()
    assert cached_files.issubset({path.name for path in cache_files(tmp_path)})

    server.start()
    timings_restored = completion(server, LONG_PROMPT)
    assert timings_restored["cache_n"] + timings_restored["prompt_n"] == timings_full["prompt_n"]
    assert timings_restored["cache_n"] > timings_full["prompt_n"] * 0.8
    assert timings_restored["prompt_n"] < timings_full["prompt_n"] * 0.2
    assert cached_files.issubset({path.name for path in cache_files(tmp_path)})


def test_disk_cache_removes_incomplete_and_invalid_entries(tmp_path):
    incomplete = tmp_path / "llama-prompt-cache-incomplete.bin"
    invalid = tmp_path / "llama-prompt-cache-invalid.bin"
    invalid_metadata = tmp_path / "llama-prompt-cache-invalid.bin.meta"
    incomplete.write_bytes(b"incomplete")
    invalid.write_bytes(b"invalid")
    invalid_metadata.write_bytes(b"invalid metadata")

    server = make_server()
    server.cache_ram = 0
    server.cache_disk = str(tmp_path)
    server.cache_disk_max = 256
    server.start()

    assert not incomplete.exists()
    assert not invalid.exists()
    assert not invalid_metadata.exists()
    completion(server, "The server remains usable after cache cleanup.", 0)


def test_disk_cache_size_limit_keeps_recent_entries(tmp_path):
    server = make_server()
    server.cache_ram = 0
    server.cache_disk = str(tmp_path)
    server.cache_disk_max = -1
    server.start()

    completion(server, LONG_PROMPT, 0)
    completion(server, "Measure the disk state size.", 1)
    data_files = cache_data_files(tmp_path)
    assert len(data_files) == 1
    entry_size = data_files[0].stat().st_size

    server.stop()
    for path in cache_files(tmp_path):
        path.unlink()

    server.cache_disk_max = max(1, math.ceil(2.25 * entry_size / MIB))
    server.start()

    prompts = [f"Cache entry {index}. {LONG_PROMPT}" for index in range(4)]
    for index, prompt in enumerate(prompts):
        completion(server, prompt, index % 2)
    completion(server, "Move the last prompt into the disk cache.", 0)

    cache_limit = server.cache_disk_max * MIB
    assert sum(path.stat().st_size for path in cache_files(tmp_path)) <= cache_limit
    assert len(cache_data_files(tmp_path)) <= 2

    timings = completion(server, prompts[-1])
    assert timings["cache_n"] > 0
    assert timings["prompt_n"] < timings["cache_n"]
