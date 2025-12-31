import os, glob, shutil
import numpy as np
import zarr

# 你要用过滤后的就改成 filtered；不改也行
NPZ_DIR  = "data/pickplace_scripted_npz_filtered"
OUT_ZARR = "data/pickplace/pickplace_replay.zarr"

RECREATE = True  # True=覆盖重建 OUT_ZARR


def _infer_action_key(d: np.lib.npyio.NpzFile) -> str:
    if "action" in d.files:
        return "action"
    if "act" in d.files:
        return "act"
    raise KeyError(f"Cannot find action key. Available keys: {d.files}")


def _infer_obs_arrays(d: np.lib.npyio.NpzFile) -> np.ndarray:
    """
    Return low-dim obs (T, Do) as float32.
    Priority:
      A) obs/observation + obs/achieved_goal + obs/desired_goal
      B) observation + achieved_goal + desired_goal
      C) obs (already flattened)
    """
    # A) obs/... style
    if "obs/observation" in d.files:
        o = d["obs/observation"].astype(np.float32)
        # goal keys optional but recommended
        if "obs/achieved_goal" in d.files and "obs/desired_goal" in d.files:
            ag = d["obs/achieved_goal"].astype(np.float32)
            dg = d["obs/desired_goal"].astype(np.float32)
            return np.concatenate([o, ag, dg], axis=-1)
        return o

    # B) flat keys
    if "observation" in d.files:
        o = d["observation"].astype(np.float32)
        if "achieved_goal" in d.files and "desired_goal" in d.files:
            ag = d["achieved_goal"].astype(np.float32)
            dg = d["desired_goal"].astype(np.float32)
            return np.concatenate([o, ag, dg], axis=-1)
        return o

    # C) already flattened
    if "obs" in d.files:
        return d["obs"].astype(np.float32)

    raise KeyError(f"Cannot infer obs. Available keys: {d.files}")


def load_one(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    act_key = _infer_action_key(d)
    obs = _infer_obs_arrays(d)            # (T, Do)
    act = d[act_key].astype(np.float32)   # (T, Da)
    if obs.shape[0] != act.shape[0]:
        raise ValueError(f"Length mismatch in {npz_path}: obs T={obs.shape[0]} vs act T={act.shape[0]}")
    return obs, act


def main():
    os.makedirs(os.path.dirname(OUT_ZARR), exist_ok=True)
    if RECREATE and os.path.exists(OUT_ZARR):
        shutil.rmtree(OUT_ZARR)

    paths = sorted(glob.glob(os.path.join(NPZ_DIR, "episode_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No episode_*.npz found in {NPZ_DIR}")

    obs_list, act_list, ends = [], [], []
    total_T = 0

    # quick sanity: print keys of first file
    with np.load(paths[0], allow_pickle=True) as d0:
        print("[INFO] first file:", paths[0])
        print("[INFO] keys:", d0.files)

    for p in paths:
        obs, act = load_one(p)
        T = obs.shape[0]
        obs_list.append(obs)
        act_list.append(act)
        total_T += T
        ends.append(total_T)

    obs_all = np.concatenate(obs_list, axis=0)
    act_all = np.concatenate(act_list, axis=0)
    ends = np.asarray(ends, dtype=np.int64)

    root = zarr.open(OUT_ZARR, mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")

    # diffusion_policy 常用的结构：data/* + meta/episode_ends 
    data.create_dataset(
        "obs", data=obs_all,
        chunks=(2048, obs_all.shape[1]), dtype=np.float32
    )
    data.create_dataset(
        "action", data=act_all,
        chunks=(2048, act_all.shape[1]), dtype=np.float32
    )
    meta.create_dataset(
        "episode_ends", data=ends,
        chunks=(1024,), dtype=np.int64
    )

    print("[DONE] Saved:", OUT_ZARR)
    print("[DONE] obs:", obs_all.shape, "action:", act_all.shape, "episodes:", len(ends))


if __name__ == "__main__":
    main()
