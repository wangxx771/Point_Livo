#!/usr/bin/env python3
"""Salvage a truncated / unindexed ROS 1 bag into a valid one.

Handles the FAST-LIVO2-style dataset bags found half-copied on this machine:
the writer never got to the index (header index_pos points past EOF), so
rosbags' Reader rejects the file ("Bag index looks damaged"). The message
data lives in complete chunks at the front, so this tool:

  1. scans chunk records from the start of the data area,
  2. drops the truncated tail chunk (and refuses compressed chunks),
  3. rewrites a clean bag: header + verbatim chunks + per-chunk index data
     + connection records + chunk-info records + a patched index_pos.

Layout matches what rosbags' Reader expects (reader.py open()):
index_pos -> conn_count connection records, then chunk_count chunk-info
records; after each chunk record, its index-data records follow immediately.

Usage: salvage_rosbag1.py <in.bag> <out.bag> [--max-chunks N] [--stats-only]
"""

import argparse
import os
import struct
import sys

OP_MSG = 0x02
OP_BAGHEADER = 0x03
OP_INDEX_DATA = 0x04
OP_CHUNK = 0x05
OP_CHUNK_INFO = 0x06
OP_CONNECTION = 0x07


def parse_header_bytes(hdr):
    fields = {}
    p = 0
    while p < len(hdr):
        if p + 4 > len(hdr):
            return None
        flen = struct.unpack("<I", hdr[p : p + 4])[0]
        if p + 4 + flen > len(hdr):
            return None
        field = hdr[p + 4 : p + 4 + flen]
        name, _, val = field.partition(b"=")
        fields[name] = val
        p += 4 + flen
    return fields


def read_record(f, limit):
    """Read one record at f's position. Returns (fields, op, body, record_bytes)."""
    pos = f.tell()
    b = f.read(4)
    if len(b) < 4:
        return None
    hlen = struct.unpack("<I", b)[0]
    if hlen > 100_000_000 or pos + 4 + hlen + 4 > limit:
        return None
    hdr = f.read(hlen)
    if len(hdr) < hlen:
        return None
    h = parse_header_bytes(hdr)
    if h is None or b"op" not in h:
        return None
    op = h[b"op"][0]
    blen = struct.unpack("<I", f.read(4))[0]
    if pos + 4 + hlen + 4 + blen > limit:
        return None
    body = f.read(blen)
    if len(body) < blen:
        return None
    f.seek(pos)
    record_bytes = f.read(4 + hlen + 4 + blen)
    return h, op, body, record_bytes


def find_data_start(f, size):
    """Locate the first record after the (possibly space-padded) file header."""
    f.seek(13)
    hlen = struct.unpack("<I", f.read(4))[0]
    hdr = f.read(hlen)
    h = parse_header_bytes(hdr)
    if h is None or b"op" not in h or h[b"op"][0] != OP_BAGHEADER:
        return None
    after_header = 13 + 4 + hlen + 4  # + body length field (usually 0)
    for probe in (after_header, 4096, 4117, 8192):
        if probe + 12 >= size:
            continue
        f.seek(probe)
        b = f.read(4)
        if len(b) < 4:
            continue
        plen = struct.unpack("<I", b)[0]
        f.seek(probe + 4)
        phdr = f.read(min(plen, 1024))
        ph = parse_header_bytes(phdr)
        if ph is not None and b"op" in ph and ph[b"op"][0] in (OP_CHUNK, OP_CONNECTION):
            return probe
    # Fall back to a coarse scan; both known bags hit the probes above.
    step = 4
    for probe in range(after_header, min(size, 1 << 20), step):
        f.seek(probe)
        b = f.read(4)
        if len(b) < 4:
            break
        plen = struct.unpack("<I", b)[0]
        if plen < 4 or plen > 1024:
            continue
        ph = parse_header_bytes(f.read(plen))
        if ph is not None and b"op" in ph and ph[b"op"][0] in (OP_CHUNK, OP_CONNECTION):
            return probe
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_bag")
    ap.add_argument("out_bag", nargs="?")
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--stats-only", action="store_true")
    a = ap.parse_args()

    size = os.path.getsize(a.in_bag)
    f = open(a.in_bag, "rb")

    # ---- pass 1: walk chunks, collect index + connection records ----
    start = find_data_start(f, size)
    if start is None:
        sys.exit("could not locate the first chunk/connection record")

    conns = {}          # conn id -> raw connection record bytes (top-level or in-chunk)
    topics = {}         # conn id -> topic (for the report)
    chunks = []         # (pos, record_bytes, {conn_id: [(sec, nsec, offset), ...]})
    pos = start
    dropped_tail = 0
    while pos + 12 < size:
        f.seek(pos)
        r = read_record(f, size)
        if r is None:
            dropped_tail += 1
            break  # truncated tail or garbage; stop here
        h, op, body, record_bytes = r
        end = pos + len(record_bytes)
        if op == OP_CHUNK:
            comp = h.get(b"compression", b"none")
            if comp != b"none":
                sys.exit(f"compressed chunk ({comp!r}) at {pos} not supported")
            # Walk the records inside the chunk body to build the index.
            entries = {}
            p = 0
            while p + 4 <= len(body):
                ihlen = struct.unpack("<I", body[p : p + 4])[0]
                if p + 4 + ihlen + 4 > len(body):
                    break
                ih = parse_header_bytes(body[p + 4 : p + 4 + ihlen])
                if ih is None or b"op" not in ih:
                    break
                iop = ih[b"op"][0]
                iblen = struct.unpack("<I", body[p + 4 + ihlen : p + 8 + ihlen])[0]
                rec_start = p
                p += 4 + ihlen + 4 + iblen
                if p > len(body):
                    break
                if iop == OP_CONNECTION:
                    cid = struct.unpack("<I", ih[b"conn"])[0]
                    if cid not in conns:
                        conns[cid] = body[rec_start : p]
                        topics[cid] = ih.get(b"topic", b"?").decode(errors="replace")
                elif iop == OP_MSG:
                    cid = struct.unpack("<I", ih[b"conn"])[0]
                    sec, nsec = struct.unpack("<II", ih[b"time"])
                    entries.setdefault(cid, []).append((sec, nsec, rec_start))
            chunks.append((pos, record_bytes, entries))
            if a.max_chunks and len(chunks) >= a.max_chunks:
                break
        elif op == OP_CONNECTION:
            cid = struct.unpack("<I", h[b"conn"])[0]
            if cid not in conns:
                conns[cid] = record_bytes
                topics[cid] = h.get(b"topic", b"?").decode(errors="replace")
        # any other top-level record (message/index) is skipped
        pos = end

    counts = {cid: 0 for cid in conns}
    tmin = {cid: None for cid in conns}
    tmax = {cid: None for cid in conns}
    for _, _, entries in chunks:
        for cid, lst in entries.items():
            counts[cid] = counts.get(cid, 0) + len(lst)
            for sec, nsec, _ in lst:
                t = sec + nsec * 1e-9
                tmin[cid] = t if tmin[cid] is None else min(tmin[cid], t)
                tmax[cid] = t if tmax[cid] is None else max(tmax[cid], t)

    print(f"salvageable: {len(chunks)} complete chunks ({'tail dropped' if dropped_tail else 'EOF'})")
    for cid in sorted(conns):
        span = (tmax[cid] - tmin[cid]) if tmin[cid] is not None else 0.0
        print(f"  conn{cid} {topics[cid]:35s} {counts[cid]:8d} msgs  {span:7.1f}s")
    if not chunks or all(v == 0 for v in counts.values()):
        sys.exit("no messages found; refusing to write an empty bag")

    if a.stats_only or not a.out_bag:
        return

    # ---- pass 2: write the repaired bag ----
    out = open(a.out_bag, "wb")
    out.write(b"#ROSBAG V2.0\n")

    def field(name, val):
        return struct.pack("<I", len(name) + 1 + len(val)) + name + b"=" + val

    def record(header_fields, body=b""):
        hdr = b"".join(header_fields)
        return struct.pack("<I", len(hdr)) + hdr + struct.pack("<I", len(body)) + body

    # File header (index_pos patched after the layout is known).
    head_record = record(
        [
            field(b"index_pos", struct.pack("<Q", 0)),
            field(b"conn_count", struct.pack("<I", len(conns))),
            field(b"chunk_count", struct.pack("<I", len(chunks))),
            field(b"op", bytes([OP_BAGHEADER])),
        ]
    )
    head_pos = out.tell()
    out.write(head_record)

    # Chunks (verbatim) each followed by their index-data records. The new
    # positions differ from the source (no padding, index data between chunks),
    # so record them for the chunk-info records.
    new_chunk_pos = []
    for _, record_bytes, entries in chunks:
        new_chunk_pos.append(out.tell())
        out.write(record_bytes)
        for cid in sorted(entries):
            lst = entries[cid]
            hdr = b"".join(
                [
                    field(b"ver", struct.pack("<I", 1)),
                    field(b"conn", struct.pack("<I", cid)),
                    field(b"count", struct.pack("<I", len(lst))),
                    field(b"op", bytes([OP_INDEX_DATA])),
                ]
            )
            body = b"".join(
                struct.pack("<III", sec, nsec, off) for sec, nsec, off in lst
            )
            out.write(struct.pack("<I", len(hdr)) + hdr + struct.pack("<I", len(body)) + body)

    # Connection records, then chunk-info records; index_pos points here.
    index_pos = out.tell()
    for cid in sorted(conns):
        out.write(conns[cid])
    for new_pos, (_, _, entries) in zip(new_chunk_pos, chunks):
        times = [(s, n) for lst in entries.values() for (s, n, _) in lst]
        start_t = min(times) if times else (0, 0)
        end_t = max(times) if times else (0, 0)
        ci_hdr = b"".join(
            [
                field(b"ver", struct.pack("<I", 1)),
                field(b"chunk_pos", struct.pack("<Q", new_pos)),
                field(b"count", struct.pack("<I", len(entries))),
                field(b"start_time", struct.pack("<II", *start_t)),
                field(b"end_time", struct.pack("<II", *end_t)),
                field(b"op", bytes([OP_CHUNK_INFO])),
            ]
        )
        ci_body = b"".join(
            struct.pack("<II", cid, len(entries[cid])) for cid in sorted(entries)
        )
        out.write(struct.pack("<I", len(ci_hdr)) + ci_hdr + struct.pack("<I", len(ci_body)) + ci_body)

    # Patch index_pos.
    out.seek(head_pos)
    head_record = record(
        [
            field(b"index_pos", struct.pack("<Q", index_pos)),
            field(b"conn_count", struct.pack("<I", len(conns))),
            field(b"chunk_count", struct.pack("<I", len(chunks))),
            field(b"op", bytes([OP_BAGHEADER])),
        ]
    )
    out.write(head_record)
    out.close()
    f.close()
    print(f"wrote {a.out_bag}: {len(chunks)} chunks, {sum(counts.values())} messages")


if __name__ == "__main__":
    main()
