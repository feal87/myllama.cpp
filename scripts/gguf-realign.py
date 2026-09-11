#!/usr/bin/env python3
# Copy a GGUF file with its tensor data aligned to a sector size.
#
# The disk decode cache (--load-mode dio) can share one expert tensor across all
# layers of a pool only when every layer's tensor starts at the same file-offset
# sector remainder. An unbuffered read writes directly into the shared slot, so
# the read destination must be sector-aligned, which needs that remainder to be
# zero. Plain GGUF files pack tensors at 32-byte alignment, so the remainders
# differ and the cache falls back to one tensor per layer.
#
# This script writes a copy with general.alignment set to the sector size (4096),
# leaving the original untouched. Run it on every shard of a split model; each
# output keeps the input shard's split.* metadata. Keep the shard naming
# (model-00001-of-00003.gguf -> model-aligned-00001-of-00003.gguf) or llama.cpp
# will not find the matching shards.
#
# Usage:
#   python scripts/gguf-realign.py input.gguf output.gguf [--alignment 4096] [--force]
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# use the local gguf package when run from a source checkout
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))

import gguf

logger = logging.getLogger("gguf-realign")


def realign(reader: gguf.GGUFReader, writer: gguf.GGUFWriter, alignment: int) -> None:
    # copy every metadata field except the two the writer owns: the architecture
    # (passed to the constructor) and the alignment (rewritten below)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if field.name == gguf.Keys.General.ALIGNMENT:
            continue

        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    # sets both the key and the padding used for the data section
    writer.add_custom_alignment(alignment)

    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.data.nbytes, tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    total = sum(t.n_bytes for t in reader.tensors)
    written = 0
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
        written += tensor.n_bytes
        logger.info("  %-60s %10.3f MiB  %5.1f%%", tensor.name,
                    tensor.n_bytes / (1024.0 * 1024.0), 100.0 * written / total)
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a GGUF with sector-aligned tensor data")
    parser.add_argument("input", type=Path, help="input GGUF file (one shard is fine)")
    parser.add_argument("output", type=Path, help="output GGUF file")
    parser.add_argument("--alignment", type=int, default=4096,
                        help="byte alignment of the data section and every tensor (default 4096)")
    parser.add_argument("--force", action="store_true", help="overwrite the output without asking")
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity")
    args = parser.parse_args(None if len(sys.argv) > 2 else ["--help"])

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    if args.alignment <= 0 or (args.alignment & (args.alignment - 1)) != 0:
        raise SystemExit(f"alignment must be a non-zero power of two, got {args.alignment}")

    if args.input.resolve() == args.output.resolve():
        raise SystemExit("input and output must differ; the original is never modified")

    if args.output.exists() and not args.force:
        answer = input(f'{args.output} exists, overwrite? enter exactly YES> ')
        if answer != "YES":
            raise SystemExit("aborted")

    logger.info("loading %s", args.input)
    reader = gguf.GGUFReader(args.input, "r")

    heads = sorted({int(t.data_offset) % args.alignment for t in reader.tensors})
    logger.info("%d tensors, %s", len(reader.tensors),
                "already aligned" if heads == [0] else f"offset remainders mod {args.alignment}: {heads[:8]}{'...' if len(heads) > 8 else ''}")

    # llama.cpp reconstructs the sibling shard paths from the first shard's name
    # by matching the trailing -NNNNN-of-MMMMM.gguf, so the output must keep it
    if gguf.Keys.Split.LLM_KV_SPLIT_COUNT in reader.fields:
        n_split = int(reader.fields[gguf.Keys.Split.LLM_KV_SPLIT_COUNT].contents())
        split_no = int(reader.fields[gguf.Keys.Split.LLM_KV_SPLIT_NO].contents()) \
            if gguf.Keys.Split.LLM_KV_SPLIT_NO in reader.fields else 0
        if n_split > 1:
            postfix = f"-{split_no + 1:05d}-of-{n_split:05d}.gguf"
            if not str(args.output).endswith(postfix):
                logger.warning("WARNING: output name should end with %r or llama.cpp will not find the "
                               "sibling shards; use a subdirectory with the same base name instead", postfix)

    # only the first shard of a split model carries the full header; the rest
    # carry split.* alone and have no architecture, so do not invent one
    has_arch = gguf.Keys.General.ARCHITECTURE in reader.fields
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents() if has_arch else "llama"
    logger.info("writing %s (alignment %d)", args.output, args.alignment)
    writer = gguf.GGUFWriter(str(args.output), arch=arch, endianess=reader.endianess)
    if not has_arch:
        # GGUFWriter adds the architecture in its constructor; drop it again so
        # the shard keeps exactly the metadata it had
        writer.remove_key(gguf.Keys.General.ARCHITECTURE)
    realign(reader, writer, args.alignment)

    # verify: reopen the copy and check the data section and every tensor start
    check = gguf.GGUFReader(args.output, "r")
    bad = [t.name for t in check.tensors if int(t.data_offset) % args.alignment != 0]
    if int(check.data_offset) % args.alignment != 0 or bad:
        raise SystemExit(f"alignment check failed: data_offset={check.data_offset}, {len(bad)} bad tensors")
    logger.info("done: data section and all %d tensor offsets are multiples of %d",
                len(check.tensors), args.alignment)


if __name__ == "__main__":
    main()
