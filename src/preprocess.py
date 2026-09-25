"""Checkpointed preprocessing: process one source file per invocation.

Usage:
  python3 src/preprocess.py <tag> <src_label> <tsv_path>
  python3 src/preprocess.py <tag> assemble

Global token/phone vocab state lives in artifacts/<tag>_vocab.pkl; per-source
token arrays in artifacts/<tag>_<S>_*.npy.  `assemble` concatenates sources
(S1, S2, S3 order) into artifacts/<tag>_data.npz + <tag>_meta.pkl.
"""
import csv
import os
import pickle
import sys
import time
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from textnorm import build_record  # noqa: E402

ART = "artifacts"


def stream_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            yield r


def load_state(tag):
    path = f"{ART}/{tag}_vocab.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return dict(tok_vocab={}, phone_vocab={"": 0}, tok2phone=[],
                dom_vocab={"": 0}, country_ids={})


def process_source(tag, src_label, path):
    t0 = time.time()
    st = load_state(tag)
    tok_vocab = st["tok_vocab"]
    phone_vocab = st["phone_vocab"]
    tok2phone = st["tok2phone"]
    dom_vocab = st["dom_vocab"]
    country_ids = st["country_ids"]

    name_toks, addr_toks = [], []
    name_toks_alt, addr_toks_alt = [], []
    digits, pins, domains, countries = [], [], [], []
    states, eids = [], []
    eid2idx = {}
    raw_blob = bytearray()
    raw_off = [0]
    base = len(eids)

    def tok_id(t, ph):
        ti = tok_vocab.get(t)
        if ti is None:
            ti = len(tok_vocab)
            tok_vocab[t] = ti
            pi = phone_vocab.get(ph)
            if pi is None:
                pi = len(phone_vocab)
                phone_vocab[ph] = pi
            tok2phone.append(pi)
        return ti

    n = 0
    for r in stream_tsv(path):
        rec = build_record(r["entity_id"], src_label, r["country"],
                           r["business_name"], r["business_address"])
        ids = [tok_id(t, rec.name_ph[j] if j < len(rec.name_ph) else "")
               for j, t in enumerate(rec.name_toks)]
        name_toks.append(ids)
        ids = [tok_id(t, rec.addr_ph[j] if j < len(rec.addr_ph) else "")
               for j, t in enumerate(rec.addr_toks)]
        addr_toks.append(ids)

        if rec.name_t != rec.name or rec.addr_t != rec.addr:
            name_toks_alt.append([tok_id(t, "") for t in rec.name_rt_toks])
            addr_toks_alt.append([tok_id(t, "") for t in rec.addr_rt_toks])
        else:
            name_toks_alt.append(None)
            addr_toks_alt.append(None)

        digits.append([int(d) % (2 ** 31) for d in rec.addr_digits])
        pins.append(int(rec.addr_pin) if rec.addr_pin else -1)
        states.append(rec.addr_state)
        d = rec.domain
        di = dom_vocab.get(d)
        if di is None:
            di = len(dom_vocab)
            dom_vocab[d] = di
        domains.append(di)

        c = r["country"]
        ci = country_ids.get(c)
        if ci is None:
            ci = len(country_ids)
            country_ids[c] = ci
        countries.append(ci)

        eid = r["entity_id"]
        eids.append(eid)
        eid2idx[eid] = base + n
        raw_blob += rec.name_raw.encode("utf-8")
        raw_blob += b"\x00"
        raw_blob += rec.addr_raw.encode("utf-8")
        raw_blob += b"\x00"
        raw_off.append(len(raw_blob))
        n += 1
        if n % 1000000 == 0:
            print(f"  {src_label}: {n} rows ({time.time()-t0:.0f}s)", flush=True)

    print(f"  {src_label}: {n} rows, vocab={len(tok_vocab)} ({time.time()-t0:.0f}s)",
          flush=True)

    name_flat = np.array([t for lst in name_toks for t in lst], dtype=np.int32)
    name_off = np.zeros(n + 1, dtype=np.int64)
    name_off[1:] = np.cumsum([len(x) for x in name_toks])
    del name_toks
    addr_flat = np.array([t for lst in addr_toks for t in lst], dtype=np.int32)
    addr_off = np.zeros(n + 1, dtype=np.int64)
    addr_off[1:] = np.cumsum([len(x) for x in addr_toks])
    del addr_toks
    alt_rows = np.array([i for i, x in enumerate(name_toks_alt)
                         if x is not None or addr_toks_alt[i] is not None],
                        dtype=np.int64)
    alt_name_flat = np.array([t for i in alt_rows for t in (name_toks_alt[i] or [])],
                             dtype=np.int32)
    alt_name_off = np.zeros(len(alt_rows) + 1, dtype=np.int64)
    alt_name_off[1:] = np.cumsum([len(name_toks_alt[i] or []) for i in alt_rows])
    alt_addr_flat = np.array([t for i in alt_rows for t in (addr_toks_alt[i] or [])],
                             dtype=np.int32)
    alt_addr_off = np.zeros(len(alt_rows) + 1, dtype=np.int64)
    alt_addr_off[1:] = np.cumsum([len(addr_toks_alt[i] or []) for i in alt_rows])
    del name_toks_alt, addr_toks_alt
    dig_flat = np.array([d for lst in digits for d in lst], dtype=np.int64)
    dig_off = np.zeros(n + 1, dtype=np.int64)
    dig_off[1:] = np.cumsum([len(x) for x in digits])
    del digits

    os.makedirs(ART, exist_ok=True)
    p = f"{ART}/{tag}_{src_label}"
    np.savez(f"{p}_arrays.npz",
             name_flat=name_flat, name_off=name_off,
             addr_flat=addr_flat, addr_off=addr_off,
             alt_rows=alt_rows,
             alt_name_flat=alt_name_flat, alt_name_off=alt_name_off,
             alt_addr_flat=alt_addr_flat, alt_addr_off=alt_addr_off,
             dig_flat=dig_flat, dig_off=dig_off,
             pin=np.array(pins, dtype=np.int64),
             dom=np.array(domains, dtype=np.int32),
             country=np.array(countries, dtype=np.int8),
             eid=np.array(eids),
             raw_blob=np.frombuffer(bytes(raw_blob), dtype=np.uint8),
             raw_off=np.array(raw_off, dtype=np.int64))
    with open(f"{p}_state.pkl", "wb") as f:
        pickle.dump(dict(states=states, eid2idx=eid2idx, src=src_label), f,
                    protocol=4)
    st["tok_vocab"] = tok_vocab
    st["phone_vocab"] = phone_vocab
    st["tok2phone"] = tok2phone
    st["dom_vocab"] = dom_vocab
    st["country_ids"] = country_ids
    with open(f"{ART}/{tag}_vocab.pkl", "wb") as f:
        pickle.dump(st, f, protocol=4)
    print(f"saved {p}_arrays.npz ({time.time()-t0:.0f}s)", flush=True)


def assemble(tag):
    t0 = time.time()
    with open(f"{ART}/{tag}_vocab.pkl", "rb") as f:
        st = pickle.load(f)
    parts = []
    meta_states, eid2idx = [], {}
    for src_label in ("S1", "S2", "S3"):
        p = f"{ART}/{tag}_{src_label}_arrays.npz"
        if not os.path.exists(p):
            continue
        parts.append(np.load(p, allow_pickle=False))
        with open(f"{ART}/{tag}_{src_label}_state.pkl", "rb") as f:
            sp = pickle.load(f)
        meta_states.extend(sp["states"])
        base = len(eid2idx)
        for k, v in sp["eid2idx"].items():
            eid2idx[k] = base + v
    N = sum(int(p["name_off"][-1] if "name_off" in p else 0) for p in [])
    # concatenate
    out_arrays = {}
    n_rec = 0
    n_name = 0
    n_addr = 0
    n_alt = 0
    n_alt_name = 0
    n_alt_addr = 0
    n_dig = 0
    n_blob = 0
    for p in parts:
        for key in ("name_flat", "addr_flat", "alt_name_flat", "alt_addr_flat",
                    "dig_flat", "pin", "dom", "country", "eid"):
            out_arrays.setdefault(key, []).append(p[key])
        # offset arrays need shifting
        off = p["name_off"][:-1] + n_name
        out_arrays.setdefault("name_off", []).append(
            np.concatenate([off, [p["name_off"][-1] + n_name]]))
        off = p["addr_off"][:-1] + n_addr
        out_arrays.setdefault("addr_off", []).append(
            np.concatenate([off, [p["addr_off"][-1] + n_addr]]))
        ar = p["alt_rows"]
        out_arrays.setdefault("alt_rows", []).append(ar + n_rec)
        off = p["alt_name_off"][:-1] + n_alt_name
        out_arrays.setdefault("alt_name_off", []).append(
            np.concatenate([off, [p["alt_name_off"][-1] + n_alt_name]]))
        off = p["alt_addr_off"][:-1] + n_alt_addr
        out_arrays.setdefault("alt_addr_off", []).append(
            np.concatenate([off, [p["alt_addr_off"][-1] + n_alt_addr]]))
        off = p["dig_off"][:-1] + n_dig
        out_arrays.setdefault("dig_off", []).append(
            np.concatenate([off, [p["dig_off"][-1] + n_dig]]))
        blob = p["raw_blob"]
        out_arrays.setdefault("raw_blob", []).append(blob)
        off = p["raw_off"][:-1] + n_blob
        out_arrays.setdefault("raw_off", []).append(
            np.concatenate([off, [p["raw_off"][-1] + n_blob]]))
        n_rec += len(p["eid"])
        n_name += p["name_off"][-1]
        n_addr += p["addr_off"][-1]
        n_alt += len(ar)
        n_alt_name += p["alt_name_off"][-1]
        n_alt_addr += p["alt_addr_off"][-1]
        n_dig += p["dig_off"][-1]
        n_blob += len(blob)

    final = {k: (np.concatenate(v) if k not in ("name_off", "addr_off",
                                                "alt_name_off", "alt_addr_off",
                                                "dig_off", "raw_off")
                 else np.concatenate(v))
             for k, v in out_arrays.items()}
    os.makedirs(ART, exist_ok=True)
    np.savez(f"{ART}/{tag}_data.npz", **final)
    with open(f"{ART}/{tag}_meta.pkl", "wb") as f:
        pickle.dump(dict(
            tok_vocab=st["tok_vocab"],
            tok2phone=np.array(st["tok2phone"], dtype=np.int32),
            phone_vocab=st["phone_vocab"], dom_vocab=st["dom_vocab"],
            country_ids=st["country_ids"], states=meta_states,
            eid2idx=eid2idx,
        ), f, protocol=4)
    print(f"assembled N={n_rec} records -> {ART}/{tag}_data.npz "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    tag = sys.argv[1]
    if sys.argv[2] == "assemble":
        assemble(tag)
    else:
        process_source(tag, sys.argv[2], sys.argv[3])
