#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from biotite.structure.io import load_structure
from peppr_dockq import dockq

# Filter out the specific warning from biotite
warnings.filterwarnings("ignore", module="biotite")

# --- Globals (top of file) ---
G_ROWS = None  # list[dict]-style rows (indexable)
G_CHAIN = None  # chain_translate_dict
G_GT_DIR = None  # gt dir
G_GT_CACHE = {}  # per-process cache: pdb_id -> loaded AtomArray


def _init_pool(rows, chain_translate_dict, gt_dir):
    global G_ROWS, G_CHAIN, G_GT_DIR, G_GT_CACHE
    G_ROWS = rows
    G_CHAIN = chain_translate_dict
    G_GT_DIR = gt_dir
    G_GT_CACHE = {}


def _load_gt(pdb_id):
    """Lazy-load and cache GT structure per process."""
    if pdb_id not in G_GT_CACHE:
        path = os.path.join(G_GT_DIR, f"{pdb_id.lower()}/{pdb_id.lower()}.cif")
        if not os.path.exists(path):
            G_GT_CACHE[pdb_id] = None
        else:
            try:
                G_GT_CACHE[pdb_id] = load_structure(path)
            except Exception:
                G_GT_CACHE[pdb_id] = None
    return G_GT_CACHE[pdb_id]


def _worker(job):
    # job: JobSpec(row_index, seed_dir)
    row = G_ROWS[job.row_index]
    try:
        return compute_dockq_for_interfaces_idx(job.row_index, job.seed_dir)
    except Exception as e:
        print(f"Error in job {job}: {e}")
        return []


# ----------------------------
# Utilities & core computation
# ----------------------------


def get_homomeric_chains(structure, target_chain_id) -> list[str]:
    """Return chain IDs in `structure` that are homomeric with `target_chain_id`."""
    target_chain = structure[structure.chain_id == target_chain_id]
    target_seq = target_chain.res_name[~np.isnan(target_chain.coord).any(axis=-1)]

    homomeric_chain_ids = []
    for chain_id in np.unique(structure.chain_id):
        chain = structure[structure.chain_id == chain_id]
        seq = chain.res_name[~np.isnan(chain.coord).any(axis=-1)]
        if (
            len(seq) == len(target_seq)
            and np.array_equal(seq, target_seq)
            and target_chain.coord.shape == chain.coord.shape
        ):
            homomeric_chain_ids.append(chain_id)
    return homomeric_chain_ids


@dataclass(frozen=True)
class JobSpec:
    """A single (row, seed_dir) job."""

    row_index: int  # index into the dataframe
    seed_dir: Path  # path to .../<problem>/seed_<X>


# For OpenFold3 predictions
def find_valid_problem_seed_map_of3(
    pred_dir: str, pdb_ids: list[str], required_seeds: int = 115, lower: bool = False
) -> dict[str, list[Path]]:
    valid: dict[str, list[Path]] = {}
    for d in pdb_ids:
        problem_dir = Path(pred_dir) / (d.lower() if lower else d)
        if not problem_dir.is_dir():
            continue
        seed_dirs = list(problem_dir.glob("seed_*"))
        if len(seed_dirs) >= required_seeds:
            valid[d] = seed_dirs
    return valid


def load_chain_translate_dict(chain_metadata_path: str) -> dict[str, dict[str, str]]:
    """
    Build chain translation: chain_translate_dict[pdb_id][label_asym_id] -> mmCIF asym_id in preprocessed set.
    """
    with open(chain_metadata_path) as f:
        data = json.load(f)["structure_data"]
    chain_translate_dict: dict[str, dict[str, str]] = {}
    for pdb_id, entry in data.items():
        pdb_id = pdb_id.upper()
        chain_translate_dict[pdb_id] = {}
        for chain_to, info in entry["chains"].items():
            chain_from = info["label_asym_id"]
            chain_translate_dict[pdb_id][chain_from] = chain_to
    return chain_translate_dict


def compute_dockq_for_interfaces_idx(row_index: int, seed_dir: Path):
    """
    Same outputs as before, but:
      - pulls row by index from global G_ROWS
      - uses per-process GT cache
      - uses len(AtomArray) instead of .size
    """
    row = G_ROWS[row_index]
    pdb_id = row["pdb_id"]
    interface_id = row["interface_cluster_key"]

    # Chain translate
    try:
        chain1 = G_CHAIN[pdb_id][row["chain_id_1"]]
        chain2 = G_CHAIN[pdb_id][row["chain_id_2"]]
        pdb_chain1 = row["chain_id_1"]
        pdb_chain2 = row["chain_id_2"]
        # VS: OF3 direct inference outputs(non validation style cache) have same chain IDs as PDB
        pred_chain1 = row["chain_id_1"]
        pred_chain2 = row["chain_id_2"]
    except KeyError:
        return []

    # Predictions for this pdb in this seed
    pred_matches = list(seed_dir.glob("*.cif"))

    if len(pred_matches) == 0:
        return []

    # Load GT from per-process cache
    gt = _load_gt(pdb_id)
    if gt is None:
        return []

    gt_receptor = gt[gt.chain_id == chain1]
    gt_ligand = gt[gt.chain_id == chain2]
    if len(gt_receptor) == 0 or len(gt_ligand) == 0:
        return []

    # Masks & shapes
    receptor_mask = ~np.isnan(gt_receptor.coord).any(axis=-1)
    ligand_mask = ~np.isnan(gt_ligand.coord).any(axis=-1)
    gt_rec_masked = gt_receptor[receptor_mask]
    gt_lig_masked = gt_ligand[ligand_mask]

    gt_rec_shape = gt_receptor.coord.shape
    gt_lig_shape = gt_ligand.coord.shape

    # Homomeric candidates from a representative pred
    try:
        pred_rep = load_structure(pred_matches[0])
        homo_rec_ids = get_homomeric_chains(pred_rep, pred_chain1)
        homo_lig_ids = get_homomeric_chains(pred_rep, pred_chain2)
        if not homo_rec_ids or not homo_lig_ids:
            print("No rec/lig ids found")
            return []
    except Exception:
        return []

    seed = seed_dir.stem.split("_")[-1]
    records = []

    for pred_path in pred_matches:
        confidence_path = pred_path.parent / str(pred_path.name).replace(
            "_model.cif", "_confidences_aggregated.json"
        )

        if not confidence_path.exists():
            print(f"Confidence path missing: {confidence_path}")
            continue
        with open(confidence_path) as file:
            confidence_scores = json.load(file)

        try:
            pred = load_structure(pred_path)
        except Exception:
            continue

        best = None
        best_score = -1.0
        best_model_score = None
        iptm, bespoke_iptm, sample_ranking_score = 0, 0, 0

        for rec_id in homo_rec_ids:
            for lig_id in homo_lig_ids:
                if rec_id == lig_id:
                    continue

                chain_pair = sorted((rec_id.item(), lig_id.item()))
                chain_pair = f"({chain_pair[0]}, {chain_pair[1]})"
                if "chain_pair_iptm" in confidence_scores:
                    iptm = confidence_scores["chain_pair_iptm"][chain_pair]
                    bespoke_iptm = confidence_scores["bespoke_iptm"][chain_pair]
                    sample_ranking_score = confidence_scores["sample_ranking_score"]

                pred_receptor = pred[pred.chain_id == rec_id]
                pred_ligand = pred[pred.chain_id == lig_id]

                if len(pred_receptor) == 0 or len(pred_ligand) == 0:
                    print(f"Pred shape 0 {pred_path}")
                    continue

                # Require raw shapes to match GT before masking
                if (
                    pred_receptor.coord.shape != gt_rec_shape
                    or pred_ligand.coord.shape != gt_lig_shape
                ):
                    print(f"Shape mismatch {pred_path}")
                    if pred_receptor.coord.shape != gt_rec_shape:
                        print(f"Receptor {pred_receptor.coord.shape}!={gt_rec_shape}")
                    else:
                        print(f"Ligand {pred_ligand.coord.shape}!={gt_lig_shape}")
                    continue

                pred_rec_masked = pred_receptor[receptor_mask]
                pred_lig_masked = pred_ligand[ligand_mask]

                try:
                    res = dockq(
                        gt_rec_masked, gt_lig_masked, pred_rec_masked, pred_lig_masked
                    )
                    if res.score > best_score:
                        best_score = res.score
                        best = res
                        best_model_score = [iptm, bespoke_iptm, sample_ranking_score]
                except Exception:
                    print("Dockq failed")
                    continue

        if best is None:
            continue

        diffusion_id = pred_path.stem.split("_")[-2]

        records.append(
            (
                pdb_id,
                pdb_chain1,
                pdb_chain2,
                interface_id,
                seed,
                diffusion_id,
                best_model_score[0],
                best_model_score[1],
                best_model_score[2],
                float(best.score),
                float(best.fnat),
                float(best.irmsd),
                float(best.lrmsd),
            )
        )

    return records


# ----------------------------
# Orchestration
# ----------------------------


def build_jobs(
    df: pd.DataFrame, pdb_to_seed_dirs: dict[str, list[Path]]
) -> list[JobSpec]:
    """Create one JobSpec per (row, seed_dir) only for rows whose pdb_id exists in predictions."""
    jobs: list[JobSpec] = []
    for idx, row in df.iterrows():
        pid = row["pdb_id"]
        seed_dirs = pdb_to_seed_dirs.get(pid, [])
        for sdir in seed_dirs:
            jobs.append(JobSpec(row_index=idx, seed_dir=sdir))
    return jobs


def run_jobs_parallel(df, jobs, chain_translate_dict, gt_dir, n_procs=None):
    # Convert rows to plain dicts (pickle-friendly & light)
    rows = [df.loc[i].to_dict() for i in df.index]
    n_procs = n_procs or max(1, mp.cpu_count() - 1)

    with mp.Pool(
        processes=n_procs,
        initializer=_init_pool,
        initargs=(rows, chain_translate_dict, gt_dir),
    ) as pool:
        total = max(1, len(jobs))
        chunksize = max(1, total // (n_procs * 8) or 1)
        results = [
            res
            for res in tqdm.tqdm(
                pool.imap_unordered(_worker, jobs, chunksize=chunksize), total=total
            )
        ]

    return [rec for sub in results for rec in (sub or [])]


# ----------------------------
# Main
# ----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute DockQ over valid problems/seeds.")
    p.add_argument(
        "-p",
        "--pred_dir",
        required=True,
        type=str,
        help="Directory containing predictions",
    )
    p.add_argument(
        "-o", "--output_path", required=True, type=str, help="Path to save CSV output"
    )
    p.add_argument(
        "--ref_dir",
        required=True,
        type=str,
        help="Base dir containing metadata_interface.csv and gt structures",
    )
    p.add_argument(
        "-j", "--jobs", default=128, type=int, help="Number of parallel processes"
    )
    p.add_argument(
        "--required_seeds",
        default=50,
        type=int,
        help="Required entries per problem to consider valid",
    )
    return p.parse_args()


def main():
    args = parse_args()

    ref_dir = args.ref_dir
    meta_csv = os.path.join(ref_dir, "metadata_interface.csv")

    chain_metadata_path = os.path.join(
        ref_dir, "pdb_preprocessing/preprocessed_pdbs/metadata.json"
    )
    gt_dir = os.path.join(
        ref_dir, "pdb_preprocessing/preprocessed_pdbs/structure_files"
    )

    # Load metadata & chain mapping
    df = pd.read_csv(meta_csv).assign(pdb_id=lambda x: x["pdb_id"].str.upper())
    pdb_ids = df["pdb_id"].unique().tolist()

    chain_translate = load_chain_translate_dict(chain_metadata_path)

    # Identify valid problems & their seeds, then invert to pdb_id -> seed_dirs
    valid_map = find_valid_problem_seed_map_of3(
        args.pred_dir, pdb_ids, required_seeds=args.required_seeds
    )

    # Filter df to rows that have predictions (valid problems only)
    valid_pdbs = set(valid_map.keys())
    df_valid = df[df["pdb_id"].isin(valid_pdbs)].reset_index(drop=True)
    print(f"Valid problems (directories): {len(valid_map)}")
    print(f"Rows to evaluate: {len(df_valid)}")

    # Build jobs across (row, seed)
    jobs = build_jobs(df_valid, valid_map)
    print(f"Total (row, seed) jobs: {len(jobs)}")

    # Run in parallel
    records = run_jobs_parallel(
        df=df_valid,
        jobs=jobs,
        chain_translate_dict=chain_translate,
        gt_dir=gt_dir,
        n_procs=args.jobs,
    )

    # Save
    if records:
        out_df = pd.DataFrame(
            records,
            columns=[
                "pdbid",
                "chain_id_1",
                "chain_id_2",
                "interface_cluster",
                "seed_number",
                "sample_number",
                "iptm",
                "bespoke_iptm",
                "sample_ranking_score",
                "dockq",
                "fnat",
                "irmsd",
                "lrmsd",
            ],
        ).assign(
            ref_interface=lambda x: (
                x["pdbid"].str.lower() + ":" + x["chain_id_1"] + x["chain_id_2"]
            ),
            pdbid=lambda x: x["pdbid"].str.lower(),
        )

        out_df = out_df[
            [
                "pdbid",
                "ref_interface",
                "interface_cluster",
                "seed_number",
                "sample_number",
                "iptm",
                "bespoke_iptm",
                "sample_ranking_score",
                "dockq",
                "fnat",
                "irmsd",
                "lrmsd",
            ]
        ]

        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        out_df.to_csv(args.output_path, index=False)
        print(f"Wrote {len(out_df)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
