# -*- coding: utf-8 -*-
# /root/autodl-tmp/DeepFRI/03_run_galaxy/002_prepare_npz_biopython_fixtmp_Calpha_seqres.py
"""
修正版 DeepFRI NPZ 构建脚本
- IN_DIR 和 OUT_DIR 已固定
- 避免 .tmp 文件重命名导致的报错
- 增加详细日志输出，方便排查问题
- 生成兼容 DeepFRI 的 C_alpha、seqres、S
"""

import os, sys, gzip, json, time, shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np

# ===================== 配置区 =====================
IN_DIR  = Path("/root/autodl-tmp/DeepFRI/cafa_pdb_source")
OUT_DIR = Path("/root/autodl-tmp/DeepFRI/cafa_npz_source")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORKERS        = min(48, max(1, cpu_count() - 1))
CONTACT_CUTOFF = 8.0
MODE           = "contact"
DIST_DTYPE     = "float32"
MIN_FREE_GB    = 10.0

# ===============================================================
AA3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G',
    'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
    'THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}
SUFFIXES = {".pdb", ".cif", ".pdb.gz", ".cif.gz"}

# 尝试选择解析后端
try:
    import gemmi
    BACKEND = "gemmi"
except Exception:
    try:
        from Bio.PDB import MMCIFParser, PDBParser, is_aa
        BACKEND = "biopython"
    except Exception:
        BACKEND = None

# ===================== 工具函数 =====================
def norm_acc_from_filename(p: Path) -> str:
    stem = p.name
    if stem.endswith(".gz"): stem = stem[:-3]
    for suf in (".pdb", ".cif"):
        if stem.endswith(suf):
            stem = stem[:-len(suf)]
            break
    if "__" in stem: stem = stem.split("__")[0]
    return stem

def read_text_auto(p: Path) -> str:
    if str(p).endswith(".gz"):
        with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        return p.read_text(encoding="utf-8", errors="ignore")

# ---------------- Gemmi 解析 ----------------
def _pseudo_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray):
    """Compute pseudo C_beta for GLY or missing CB using Dunbrack formula."""
    if n is None or ca is None or c is None:
        return None
    n_ca = n - ca
    c_ca = c - ca
    cross = np.cross(n_ca, c_ca)
    for vec in (n_ca, c_ca, cross):
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return None
    n_ca /= np.linalg.norm(n_ca)
    c_ca /= np.linalg.norm(c_ca)
    cross /= np.linalg.norm(cross)
    # coefficients from Park & Levitt backbone reconstruction
    return ca + (-0.58273431 * n_ca + 0.56802827 * c_ca - 0.54067466 * cross)


def parse_with_gemmi(p: Path):
    try:
        doc = gemmi.read_structure(str(p))
    except Exception as e:
        print(f"[ERROR] Gemmi failed to read {p}: {e}")
        return None

    best = None
    for model in doc:
        for chain in model:
            seq, ca_coords, cb_coords, res_ids = [], [], [], []
            for res in chain:
                # 仅考虑标准氨基酸
                name3 = res.name.upper().strip()
                if name3 not in AA3_TO_1:
                    continue
                ca = res.find_atom("CA", altloc='?')
                if ca is None:
                    continue
                seq.append(AA3_TO_1[name3])
                ca_pos = np.array([ca.pos.x, ca.pos.y, ca.pos.z], dtype=np.float32)
                ca_coords.append(ca_pos)

                cb = res.find_atom("CB", altloc='?')
                if cb is not None:
                    cb_pos = np.array([cb.pos.x, cb.pos.y, cb.pos.z], dtype=np.float32)
                else:
                    n = res.find_atom("N", altloc='?')
                    c = res.find_atom("C", altloc='?')
                    cb_pos = _pseudo_cb(
                        np.array([n.pos.x, n.pos.y, n.pos.z], dtype=np.float32) if n else None,
                        ca_pos,
                        np.array([c.pos.x, c.pos.y, c.pos.z], dtype=np.float32) if c else None,
                    )
                    if cb_pos is None:
                        cb_pos = ca_pos.copy()
                cb_coords.append(cb_pos.astype(np.float32))

                rid = f"{chain.name}:{res.seqid.num}:{res.seqid.icode or ''}"
                res_ids.append(rid)
            if seq:
                cand = (
                    chain.name,
                    "".join(seq),
                    np.asarray(ca_coords, dtype=np.float32),
                    np.asarray(cb_coords, dtype=np.float32),
                    res_ids,
                )
                if best is None or len(cand[1]) > len(best[1]):
                    best = cand
    return best

# ---------------- Biopython 解析 ----------------
def parse_with_biopython(p: Path):
    from Bio.PDB import MMCIFParser, PDBParser, is_aa
    if str(p).endswith(".cif") or str(p).endswith(".cif.gz"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    try:
        if str(p).endswith(".gz"):
            from io import StringIO
            txt = read_text_auto(p)
            handle = StringIO(txt)
            structure = parser.get_structure("S", handle)
        else:
            structure = parser.get_structure("S", str(p))
    except Exception as e:
        print(f"[ERROR] Biopython failed to read {p}: {e}")
        return None

    best = None
    for model in structure:
        for chain in model:
            seq, ca_coords, cb_coords, res_ids = [], [], [], []
            for res in chain:
                if not is_aa(res, standard=True):
                    continue
                ca = res["CA"] if "CA" in res else None
                if ca is None:
                    continue
                name3 = res.get_resname().upper().strip()
                if name3 not in AA3_TO_1:
                    continue
                seq.append(AA3_TO_1[name3])
                ca_pos = ca.get_coord().astype(np.float32)
                ca_coords.append(ca_pos)

                if "CB" in res and res["CB"] is not None:
                    cb_pos = res["CB"].get_coord().astype(np.float32)
                else:
                    n = res["N"] if "N" in res else None
                    c = res["C"] if "C" in res else None
                    cb_pos = _pseudo_cb(
                        n.get_coord().astype(np.float32) if n is not None else None,
                        ca_pos,
                        c.get_coord().astype(np.float32) if c is not None else None,
                    )
                    if cb_pos is None:
                        cb_pos = ca_pos.copy()
                cb_coords.append(cb_pos.astype(np.float32))

                icode = res.id[2] if len(res.id) > 2 else ""
                resnum = res.id[1]
                rid = f"{chain.id}:{resnum}:{icode or ''}"
                res_ids.append(rid)
            if seq:
                cand = (
                    chain.id,
                    "".join(seq),
                    np.asarray(ca_coords, dtype=np.float32),
                    np.asarray(cb_coords, dtype=np.float32),
                    res_ids,
                )
                if best is None or len(cand[1]) > len(best[1]):
                    best = cand
    return best

# ---------------- 计算距离与接触图 ----------------
def coords_to_dist_and_contacts(coords: np.ndarray, cutoff: float, dist_dtype: str):
    if coords.shape[0] == 0:
        return None, None
    x = coords.astype(np.float32)
    xx = np.sum(x*x, axis=1, keepdims=True)
    dist2 = xx + xx.T - 2.0 * (x @ x.T)
    np.maximum(dist2, 0.0, out=dist2)
    dist = np.sqrt(dist2, dtype=np.float32)
    np.fill_diagonal(dist, 0.0)
    contact_map = (dist <= cutoff).astype(np.uint8)
    np.fill_diagonal(contact_map, 0)
    if dist_dtype == "float16":
        dist = dist.astype(np.float16)
    return dist, contact_map

def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)

def find_all_inputs(root: Path):
    files = []
    for suf in SUFFIXES:
        files.extend(root.rglob(f"*{suf}"))
    return sorted([p for p in files if p.is_file() and not p.name.startswith(".")])

# ---------------- 多进程全局配置 ----------------
GLOBAL_CFG = {}
def init_worker(cfg):
    GLOBAL_CFG.update(cfg)

def _dist_matrix(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] == 0:
        return None
    x = coords.astype(np.float32)
    xx = np.sum(x * x, axis=1, keepdims=True)
    dist2 = xx + xx.T - 2.0 * (x @ x.T)
    np.maximum(dist2, 0.0, out=dist2)
    dist = np.sqrt(dist2, dtype=np.float32)
    np.fill_diagonal(dist, 0.0)
    return dist


def process_one(p: Path):
    out_dir   = GLOBAL_CFG["out_dir"]
    cutoff    = GLOBAL_CFG["cutoff"]
    mode      = GLOBAL_CFG["mode"]
    dist_dtype= GLOBAL_CFG["dist_dtype"]
    backend   = GLOBAL_CFG["backend"]
    min_free  = GLOBAL_CFG["min_free_gb"]

    acc = norm_acc_from_filename(p)
    out_npz = out_dir / f"{acc}.npz"

    try:
        if out_npz.exists():
            return {"acc": acc, "npz_path": str(out_npz.resolve()), "n_res": -1, "ok": True, "msg": "exists_skip"}

        if disk_free_gb(out_dir) < min_free:
            return {"acc": acc, "npz_path": None, "n_res": 0, "ok": False, "msg": "low_disk_stop"}

        if backend == "gemmi":
            parsed = parse_with_gemmi(p)
        else:
            parsed = parse_with_biopython(p)

        if parsed is None:
            print(f"[ERROR] Failed to parse {p} with {backend}")
            return {"acc": acc, "npz_path": None, "n_res": 0, "ok": False, "msg": "no_valid_chain"}

        chain_id, seq, ca_coords, cb_coords, res_ids = parsed
        if len(seq) < 2:
            print(f"[WARN] {p} chain {chain_id} too short: {len(seq)} residues")
            return {"acc": acc, "npz_path": None, "n_res": len(seq), "ok": False, "msg": "too_short"}

        # ---------------- 生成距离矩阵 ----------------
        C_alpha = _dist_matrix(ca_coords)
        C_beta = _dist_matrix(cb_coords) if cb_coords is not None else None
        if C_beta is None:
            C_beta = C_alpha.copy()

        # ---------------- 生成 contact_map ----------------
        contact_map = (C_alpha <= cutoff).astype(np.uint8)
        np.fill_diagonal(contact_map, 0)

        # ---------------- 生成占位 S ----------------
        S = np.zeros((len(seq), 1024), dtype=np.float32)

        # ---------------- 构建数组 ----------------
        arrays = {
            "seq": np.array(seq),
            "seqres": np.array(list(seq)),
            "res_ids": np.array(res_ids, dtype=object),
            "chain_id": np.array(chain_id),
            "C_alpha": C_alpha,
            "C_beta": C_beta,
            "S": S,
            "contact_map": contact_map,
        }

        if mode in ("dist","both"):
            arrays["dist"] = C_alpha

        np.savez_compressed(out_npz, **arrays)

        return {"acc": acc, "npz_path": str(out_npz.resolve()), "n_res": len(seq), "ok": True, "msg": "ok"}

    except Exception as e:
        print(f"[ERROR] Unexpected error for {p}: {e}")
        return {"acc": acc, "npz_path": None, "n_res": 0, "ok": False, "msg": f"error:{e}"}

# ===================== 主运行 =====================
if BACKEND is None:
    print("[ERROR] gemmi/biopython 都不可用，请先安装其一")
else:
    files = find_all_inputs(IN_DIR)
    print(f"[Config] BACKEND={BACKEND}")
    print(f"[Config] IN_DIR={IN_DIR}")
    print(f"[Config] OUT_DIR={OUT_DIR}")
    print(f"[Scan] found candidate structures: {len(files)}")
    print(f"[Parallel] workers={WORKERS}, mode={MODE}, dist_dtype={DIST_DTYPE}, min_free_gb={MIN_FREE_GB}")

    cfg = {
        "out_dir": OUT_DIR,
        "cutoff": CONTACT_CUTOFF,
        "mode": MODE,
        "dist_dtype": DIST_DTYPE,
        "backend": BACKEND,
        "min_free_gb": MIN_FREE_GB,
    }

    t0 = time.time()
    records = []
    stop = False

    with Pool(processes=WORKERS, initializer=init_worker, initargs=(cfg,)) as pool:
        for i, rec in enumerate(pool.imap_unordered(process_one, files, chunksize=16), 1):
            records.append(rec)
            if rec.get("msg") in ("low_disk_stop",):
                stop = True
                break
            if i % 200 == 0:
                ok_new = sum(1 for r in records if r.get("ok") and r.get("msg") == "ok")
                exist  = sum(1 for r in records if r.get("ok") and r.get("msg") == "exists_skip")
                fail   = sum(1 for r in records if not r.get("ok"))
                print(f"[Progress] done={i} | ok_new={ok_new} | exists={exist} | fail={fail}")
        if stop:
            pool.terminate()
        pool.close()
        pool.join()

    ok_new   = [r for r in records if r.get("ok") and r.get("msg") == "ok"]
    ok_exist = [r for r in records if r.get("ok") and r.get("msg") == "exists_skip"]
    fail     = [r for r in records if not r.get("ok")]
    dt = time.time() - t0

    # 写结构映射
    map_tsv = OUT_DIR / "structure_map_all.tsv"
    with open(map_tsv, "w", encoding="utf-8") as fo:
        fo.write("acc\tnpz_path\n")
        for r in ok_new + ok_exist:
            if r.get("acc") and r.get("npz_path"):
                fo.write(f"{r['acc']}\t{r['npz_path']}\n")

    # 写报告
    rep_json = OUT_DIR / "npz_build_report.json"
    report = {
        "backend": BACKEND,
        "in_dir": str(IN_DIR.resolve()),
        "out_dir": str(OUT_DIR.resolve()),
        "mode": MODE,
        "dist_dtype": DIST_DTYPE,
        "contact_cutoff_A": CONTACT_CUTOFF,
        "n_files": len(files),
        "n_ok_new": len(ok_new),
        "n_ok_exists": len(ok_exist),
        "n_fail": len(fail),
        "time_sec": round(dt, 2),
        "stop_due_to_low_disk": stop,
        "fail_examples": fail[:10],
    }
    with open(rep_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[Done] new={len(ok_new)} | exists={len(ok_exist)} | fail={len(fail)} | elapsed={round(dt,1)}s")
    if stop:
        print("[HINT] 触发低磁盘保护：清理空间后重新运行此脚本即可，已完成的npz会自动跳过。")
    print(f"[Saved] map -> {map_tsv}")
    print(f"[Saved] report -> {rep_json}")
