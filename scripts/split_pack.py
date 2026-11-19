r"""nvfp4 pack -> strata-split GGUF: experts NVFP4 blob -> Q4_0 tensors, dense BF16 matrices -> Q8_0,
indexer/hc stay BF16, other BF16/F16 -> F32, ple fp8 -> Q8_0, token_embd bf16 -> Q8_0.
shortcut: only NVFP4 blobs and Qwen3.8-Flash-Next names; upgrade if another pack/format shows up.
Run ON THE BOX: python split_pack.py <packdir> <outdir>
"""
import mmap, struct, sys, pathlib
import numpy as np

T_F32, T_F16, T_Q8_0, T_Q4_0, T_BF16 = 0, 1, 8, 2, 30  # ggml type codes (verified against GGMLQuantizationType)
BLOCKS = {T_F32: 4, T_F16: 2, T_BF16: 2}  # plain-type itemsize
QB = {T_Q8_0: (32, 34), T_Q4_0: (32, 18)}  # quant: elements per block, bytes per block

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32) * np.array([1, -1], np.float32)[:, None]  # [sign][mag]

def e4m3_tab():
    t = np.zeros(256, np.float32)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 15, b & 7
        if e == 0: v = m / 8 * 2 ** -6
        elif e == 15 and m == 7: v = np.nan
        else: v = (1 + m / 8) * 2 ** (e - 7)
        t[b] = -v if s else v
    return t
E4M3 = e4m3_tab()

def bf16_f32(raw, n): return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)[:n]
def f16_f32(raw, n): return np.frombuffer(raw, np.float16).astype(np.float32)[:n]

def _ue4m3_tab():
    t = np.zeros(256, np.float32)
    for b in range(256):
        if b == 0x7F:
            continue
        e, m = (b >> 3) & 15, b & 7
        t[b] = (m * 2.0 ** -9) if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return t
UE4M3 = _ue4m3_tab()

def dec_nvfp4(raw, rows, cols, s2):
    # block_nvfp4: d[4] UE4M3 scales FIRST then qs[32]; each 8-byte sub-block = low nibbles values 0..7, high
    # nibbles 8..15 (strata ggml-common.h); UE4M3 unsigned, 0x7F reads as 0
    sb = np.frombuffer(raw, np.uint8).reshape(rows, cols // 64, 36)
    sc = UE4M3[sb[:, :, :4]]
    qs = sb[:, :, 4:].reshape(rows, cols // 64, 4, 8)
    lo = E2M1[(qs & 15) >> 3, (qs & 15) & 7]
    hi = E2M1[(qs >> 4) >> 3, (qs >> 4) & 7]
    v = np.concatenate([lo, hi], axis=3).reshape(rows, cols // 16, 16)
    return (v * sc.reshape(rows, cols // 16, 1) * np.float32(s2)).reshape(rows, cols).astype(np.float32)

def enc_q4_0(x):
    r, c = x.shape
    b = x.reshape(r, c // 32, 32)
    amax = np.abs(b).max(2)
    sc = np.where(amax > 0, amax / 7.0, 1.0)
    q = np.clip(np.floor(b / sc[:, :, None] + 8.5), 0, 15).astype(np.uint8)
    out = np.empty((r, c // 32, 18), np.uint8)
    out[:, :, :2] = sc.astype(np.float16).view(np.uint8).reshape(r, c // 32, 2)
    out[:, :, 2:] = q[:, :, :16] | (q[:, :, 16:] << 4)   # ggml q4_0 packs byte j = q[j] | q[j+16]<<4
    return out.tobytes()

def enc_q8_0(x):
    r, c = x.shape
    b = x.reshape(r, c // 32, 32)
    amax = np.abs(b).max(2)
    sc = np.where(amax > 0, amax / 127.0, 1.0)
    q = np.clip(np.round(b / sc[:, :, None]), -127, 127).astype(np.int8)
    out = np.empty((r, c // 32, 34), np.uint8)
    out[:, :, :2] = sc.astype(np.float16).view(np.uint8).reshape(r, c // 32, 2)
    out[:, :, 2:] = q.view(np.uint8)
    return out.tobytes()

def selftest():
    x = np.random.default_rng(7).uniform(-3, 3, (4, 256)).astype(np.float32)
    d = np.reshape(np.frombuffer(enc_q8_0(x), np.uint8), (4, 8, 34))
    s = np.reshape(d[:, :, :2].copy().view(np.float16), (4, 8)).astype(np.float32)
    got = (np.reshape(d[:, :, 2:], (4, 8, 32)).astype(np.int8).astype(np.float32) * s[:, :, None]).reshape(4, 256)
    assert np.abs(got - x).max() < 0.3, "q8 self-check"
    d = np.reshape(np.frombuffer(enc_q4_0(x), np.uint8), (4, 8, 18))
    s = np.reshape(d[:, :, :2].copy().view(np.float16), (4, 8)).astype(np.float32)
    lo, hi = (d[:, :, 2:] & 15).astype(np.float32) - 8, (d[:, :, 2:] >> 4).astype(np.float32) - 8
    got = (np.concatenate([lo, hi], 2).reshape(4, 8, 32) * s[:, :, None]).reshape(4, 256)
    assert np.abs(got - x).max() < 1.0, "q4 self-check"
    r, c = 2, 640
    raw = np.random.default_rng(3).integers(0, 256, (r, c // 64, 36), dtype=np.uint8)
    raw[:, :, :4] &= 0x77   # UE4M3 scales are unsigned and never NaN like the real pack's
    mm = dec_nvfp4(raw.tobytes(), r, c, 2.0)
    assert mm.shape == (r, c) and np.isfinite(mm).all(), "nvfp4 self-check"
    print("SELFTEST-OK", flush=True)

# ---------------- gguf io ---------------- (via the fork's own gguf-py; loader reads the same layout)
from gguf import GGUFReader
from gguf.gguf_writer import GGUFWriter
from gguf.constants import GGUFValueType, GGMLQuantizationType

def read_gguf(p):
    r = GGUFReader(p)
    fields = [(k, fv) for k, fv in r.fields.items() if not k.startswith("GGUF.")]
    ts = [(t.name, [int(x) for x in t.shape], int(t.tensor_type), int(t.data_offset), int(t.n_bytes)) for t in r.tensors]  # np-order shape (outermost first)
    return r, fields, ts, int(r.data_offset)

ADD = {GGUFValueType.UINT8: "add_uint8", GGUFValueType.INT8: "add_int8", GGUFValueType.UINT16: "add_uint16",
       GGUFValueType.INT16: "add_int16", GGUFValueType.UINT32: "add_uint32", GGUFValueType.INT32: "add_int32",
       GGUFValueType.FLOAT32: "add_float32", GGUFValueType.BOOL: "add_bool", GGUFValueType.STRING: "add_string",
       GGUFValueType.ARRAY: "add_array", GGUFValueType.UINT64: "add_uint64", GGUFValueType.INT64: "add_int64",
       GGUFValueType.FLOAT64: "add_float64"}

def copy_fields(w, fields):
    for k, fv in fields:
        t = fv.types[0]
        v = fv.contents()
        if t is GGUFValueType.STRING:
            w.add_string(k, v if isinstance(v, str) else "".join(map(str, v)))
        elif t is GGUFValueType.ARRAY:
            # preserve the file's element type: reader values fit it exactly; writer derives ltype from Python ints, which can mis-derive uint32 > 2^31
            arr = [int(x.item()) if hasattr(x, "item") else x for x in v]
            w.add_key_value(k, arr, GGUFValueType.ARRAY, sub_type=fv.types[-1])
        else:
            d = v.item() if hasattr(v, "item") else v
            getattr(w, ADD[t])(k, d)

Q8_NAMES = ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_qkvz.weight",
            "ssm_out.weight", "ple_key.weight", "ple_value.weight", "output.weight",
            "hyper_connection_mixer.input_mix_weight_down.weight", "hyper_connection_mixer.input_mix_weight_up.weight")
Q8_NAMES = (".attn_gate.weight", ".attn_k.weight", ".attn_output.weight", ".attn_q.weight",
            ".attn_qkv.weight", ".attn_v.weight", ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight",
            ".ffn_down_shexp.weight", ".hc_attn_down.weight", ".hc_attn_up.weight",
            ".hc_ffn_down.weight", ".hc_ffn_up.weight", ".ple_key.weight", ".ple_value.weight",
            ".ssm_out.weight", "token_embd.weight", "output.weight", "output_hc_down.weight",
            "output_hc_up.weight")
F32_NAMES = (".ffn_gate_inp.weight", ".ffn_gate_inp_shexp.weight", ".hc_attn_inject.weight",
             ".hc_ffn_inject.weight", ".ssm_alpha.weight", ".ssm_beta.weight", ".ple_conv1d.weight")

def out_type(name, t):
    # authority = loader tensor_convert.hpp wanted_type(): name wins over source type
    if name.endswith(Q8_NAMES) or name in ("token_embd.weight", "output.weight",
                                           "output_hc_down.weight", "output_hc_up.weight"):
        return T_Q8_0
    if name.endswith(F32_NAMES):
        return T_F32
    if name.startswith("blk.") and "indexer.q_proj" in name or name.startswith("blk.") and "indexer.k_proj" in name:
        return T_BF16
    if t == T_F16: return T_F32
    return t

def main(pack, outd):
    pack = pathlib.Path(pack); outd = pathlib.Path(outd); outd.mkdir(parents=True, exist_ok=True)
    outp = outd / "qwen38fn-split-q4.gguf"
    for tmp in outd.glob("*.tmp"): tmp.unlink()
    dense_r, fields, dense_ts, dstart = read_gguf(str(pack.parent / "qwen-nvfp4-gptq-dense.gguf"))
    dense_f = open(str(pack.parent / "qwen-nvfp4-gptq-dense.gguf"), "rb")
    _er, emb_fields, emb_ts, emb_start = read_gguf(str(pack.parent / "token-embd-bf16.gguf"))
    emb_f = open(str(pack.parent / "token-embd-bf16.gguf"), "rb")
    _pr, ple_fields, ple_ts, ple_start = read_gguf(str(pack.parent / "ple-fp8.gguf"))
    ple_f = open(str(pack.parent / "ple-fp8.gguf"), "rb")
    ple_scale = 1.0
    for k, fv in ple_fields:
        if k == "strata.ple.scale": ple_scale = float(fv.contents())
    lay = {}
    for line in (pack / "native_experts.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip(): continue
        a = line.split()
        lay[int(a[0])] = (int(a[3]), int(a[1]))   # absolute offset in experts.bin, gu_type
    QT = {T_F32: np.float32, T_F16: np.float16, T_BF16: np.uint16, T_Q8_0: np.uint8, T_Q4_0: np.uint8}
    ef = open(pack / "experts.bin", "rb"); em = mmap.mmap(ef.fileno(), 0, access=mmap.ACCESS_READ)
    arch = dense_r.fields["general.architecture"].contents()
    if not isinstance(arch, str): arch = str(arch[0] if isinstance(arch, (list, np.ndarray)) else arch)
    w = GGUFWriter(str(outp), arch, use_temp_file=True)  # stream tensors to a temp file — otherwise 120 GB in RAM
    copy_fields(w, fields)
    copy_fields(w, [(k, fv) for k, fv in ple_fields if k.startswith("strata.")])  # strata.ple.* only — ple file's general.* would clobber
    seen = set()
    def emit(k, dims, t, arr):
        assert k not in seen, "dup tensor %s" % k
        seen.add(k)
        raw_dtype = {T_Q8_0: GGMLQuantizationType.Q8_0, T_Q4_0: GGMLQuantizationType.Q4_0, T_BF16: GGMLQuantizationType.BF16}.get(t)
        if t == T_BF16:
            w.add_tensor(k, np.frombuffer(arr, np.uint16), raw_shape=list(dims)[::-1], raw_dtype=GGMLQuantizationType.BF16)
        elif raw_dtype:  # quantized: np-order byte shape (dims reversed, last = file dim0 scaled to bytes)
            bs = list(dims)[::-1]; bs[-1] = bs[-1] * 34 // 32 if t == T_Q8_0 else bs[-1] * 18 // 32
            w.add_tensor(k, np.frombuffer(arr, np.uint8), raw_shape=bs, raw_dtype=raw_dtype)
        else:
            w.add_tensor(k, np.frombuffer(arr, QT[t]).reshape(list(dims)), raw_shape=list(dims)[::-1])
    print("start dense tensors", flush=True)
    for k, dims, t, off, nb in dense_ts:
        if "_exps" in k: continue  # experts live in the blob
        print("dense %s" % k, flush=True)
        ot = out_type(k, t); n = int(np.prod(dims))
        dense_f.seek(off)
        if ot == t:
            emit(k, dims, t, dense_f.read(nb))
        elif ot == T_F32:
            emit(k, dims, T_F32, (bf16_f32 if t == T_BF16 else f16_f32)(dense_f.read(n * 2), n).tobytes())
        else:
            raw = dense_f.read(n * 4) if t == T_F32 else dense_f.read(n * 2)
            conv = (lambda b, m: np.frombuffer(b, np.float32)) if t == T_F32 else (bf16_f32 if t == T_BF16 else f16_f32)
            x = conv(raw, n).reshape(list(dims))
            emit(k, dims, T_Q8_0, enc_q8_0(np.ascontiguousarray(x)))
    for l in range(48):
        off, gu = lay[l]
        assert gu == 40, "layer %d gu_type %d != nvfp4" % (l, gu)
        gsz = 640 * (1280 + 160); dsz = 2560 * (320 + 40); blob = gsz * 2 + dsz + 16
        for proj, rd, rc in (("gate", 640, 2560), ("up", 640, 2560), ("down", 2560, 640)):
            base = 0 if proj == "gate" else gsz if proj == "up" else gsz * 2
            so = gsz * 2 + dsz + (0 if proj == "gate" else 4 if proj == "up" else 8)
            buf = bytearray()
            for e in range(512):
                eb = off + e * blob
                s2 = struct.unpack_from('<f', em, eb + so)[0]
                buf += enc_q4_0(dec_nvfp4(em[eb + base: eb + base + rd * (rc // 2 + rc // 16)], rd, rc, s2))
            emit("blk.%d.ffn_%s_exps.weight" % (l, proj), [rc, rd, 512], T_Q4_0, bytes(buf))  # loader wants innermost-first dims
            print("layer %d %s done" % (l, proj), flush=True)
    # token_embd comes from dense (bf16 -> q8 via the dense loop); emb file is a duplicate of it
    pk, pdims, pt, poff, pnb = ple_ts[0]
    rows, cols = pdims[0], pdims[1]  # np-order (outermost first)
    ne = rows * cols
    buf = bytearray()
    for i in range(0, ne, 1 << 20):  # flat 1M-elem chunks — the tensor is one row of 320M, row-chunking would eat 200GB
        m = min(1 << 20, ne - i)
        ple_f.seek(poff + i)
        v = (E4M3[np.frombuffer(ple_f.read(m), np.uint8)] * np.float32(ple_scale)).reshape(-1, 32)
        buf += enc_q8_0(np.ascontiguousarray(v))
    emit("per_layer_token_embd.weight", pdims, T_Q8_0, bytes(buf))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file();    w.close(); em.close(); ef.close()
    print("SPLITPACK-DONE", str(outp), outp.stat().st_size, flush=True)

if __name__ == "__main__":
    selftest()
    main(sys.argv[1], sys.argv[2])
