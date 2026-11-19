"""In-place rewrite of the 144 expert tensors in the split GGUF with the corrected NVFP4 decode:
- per 8-byte sub-block: low nibbles = values 0..7, high nibbles = values 8..15 (NOT Q4_0 interleave)
- scales = UE4M3 unsigned: bit7 ignored, 0x7F -> 0 (kernel Ue4m3Table)
blob (per expert): gate rows | up rows | down rows | tail {s_gate, s_up, s_down, 0} fp32.
Row bytes: rc/64 blocks of 36 (4 d + 32 qs). Tensor bytes: np-order flat [512, rd, rc] -> Q4_0 blocks.
shortcut: expert-data-only rewrite; rerun the full converter if dense/PLE ever change.
"""
import mmap, os, struct, sys, pathlib
import numpy as np
sys.path.insert(0, ".")  # the forked `gguf` loader module directory — point this at your copy
from gguf import GGUFReader  # fork loader module (same one rawdims.py used)

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32) * np.array([1, -1], np.float32)[:, None]  # [sign][mag]

def ue4m3_tab():
    t = np.zeros(256, np.float32)
    for b in range(256):
        if b == 0x7F:
            continue
        e, m = (b >> 3) & 15, b & 7
        t[b] = (m * 2.0 ** -9) if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return t
UE4M3 = ue4m3_tab()

def dec_nvfp4(raw, rows, cols, s2):
    sb = np.frombuffer(raw, np.uint8).reshape(rows, cols // 64, 36)
    sc = UE4M3[sb[:, :, :4]]
    qs = sb[:, :, 4:].reshape(rows, cols // 64, 4, 8)          # 4 sub-blocks of 8 code bytes
    lo = E2M1[(qs & 15) >> 3, (qs & 15) & 7]                   # values 0..7 of each sub-block
    hi = E2M1[(qs >> 4) >> 3, (qs >> 4) & 7]                   # values 8..15
    v = np.concatenate([lo, hi], axis=3)                       # (rows, nblk, 4, 16)
    v = (v.reshape(rows, cols // 16, 16) * sc.reshape(rows, cols // 16, 1) * np.float32(s2)).reshape(rows, cols)
    return v.astype(np.float32)

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

def selftest():
    # one sub-block: byte j low nibble = value j, high = value j+8
    raw = np.zeros(36, np.uint8)
    raw[:4] = 0x38                                   # ue4m3 e=7,m=0 -> 1.0
    vals = np.arange(16) % 8                          # codes 0..7 magnitudes
    for j in range(8):
        raw[4 + j] = vals[j] | (vals[j + 8] << 4)
    v = dec_nvfp4(raw.tobytes(), 1, 64, 1.0)[0][:16]
    assert np.allclose(v[:8], E2M1[0][vals[:8]]) and np.allclose(v[8:], E2M1[0][vals[8:]]), "sub-block nibble order"
    assert UE4M3[0x7F] == 0.0 and UE4M3[0x80] == UE4M3[0] == 0.0, "ue4m3 unsigned/0x7F"
    print("selftest ok")

def main():
    selftest()
    root = pathlib.Path(os.environ.get("PACK_ROOT", "."))  # the model pack directory (pack/ + helper GGUFs inside)
    pack = root / "pack"
    lay = {}
    for line in open(pack / "native_experts.txt"):
        if line.startswith("#") or not line.strip():
            continue
        a = line.split()
        lay[int(a[0])] = (int(a[3]), int(a[1]))
    em = open(pack / "experts.bin", "rb")
    mm = mmap.mmap(em.fileno(), 0, access=mmap.ACCESS_READ)
    out_p = str(root / "split40" / "qwen38fn-split-q4.gguf")
    r = GGUFReader(out_p)
    ts = {t.name: t for t in r.tensors}
    f = open(out_p, "r+b")
    for l in range(48):
        off, gu = lay[l]
        assert gu == 40
        gsz = 640 * (1280 + 160); dsz = 2560 * (320 + 40); blob = gsz * 2 + dsz + 16
        for proj, rd, rc in (("gate", 640, 2560), ("up", 640, 2560), ("down", 2560, 640)):
            base = 0 if proj == "gate" else gsz if proj == "up" else gsz * 2
            so = gsz * 2 + dsz + (0 if proj == "gate" else 4 if proj == "up" else 8)
            t = ts["blk.%d.ffn_%s_exps.weight" % (l, proj)]
            nbytes = rd * rc * 512 // 32 * 18
            assert t.n_bytes == nbytes, (t.name, t.n_bytes, nbytes)
            buf = bytearray()
            for e in range(512):
                eb = off + e * blob
                s2 = struct.unpack_from('<f', mm, eb + so)[0]
                buf += enc_q4_0(dec_nvfp4(mm[eb + base: eb + base + rd * (rc // 2 + rc // 16)], rd, rc, s2))
            f.seek(int(t.data_offset)); f.write(bytes(buf))
            print("layer %d %s done" % (l, proj), flush=True)
    f.close()
    print("REWRITE-DONE")

main()
