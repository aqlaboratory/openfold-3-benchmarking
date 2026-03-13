# %%
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument(
    "--ost_dir",
    type=str,
    required=True,
    help="Directory containing OST result JSON files",
)
parser.add_argument(
    "--pred_dir",
    type=str,
    required=True,
    help="Directory containing prediction confidence JSON files",
)
parser.add_argument(
    "--output_csv", type=str, required=True, help="Output CSV file path"
)
parser.add_argument(
    "--no_confidence", action="store_true", help="If set, do not parse confidence data "
)
parser.add_argument(
    "--method_name",
    type=str,
    required=True,
    default="of3",
    help="Method name to tag in the output CSV",
)

args = parser.parse_args()
# %%
all_dfs = []

entries_failed = []
entries_empty = []

ost_result_dir = Path(args.ost_dir)
print(f"Processing directory: {ost_result_dir}")
result_files = list(Path(ost_result_dir).glob("*.json"))
if not args.no_confidence:
    print("Will parse confidence data as well.")
for ost_result_json in tqdm(result_files, total=len(result_files)):
    query_name, _, seed, _, sample, _, _, _ = ost_result_json.name.split("_")

    pred_dir = Path(args.pred_dir)
    confidence_json = (
        pred_dir / f"{query_name}/seed_{seed}/{query_name}_seed_"
        f"{seed}_sample_{sample}_confidences_aggregated.json"
    )

    with open(ost_result_json) as f:
        ost_data = json.load(f)

    if ost_data["status"] == "FAILURE":
        entries_failed.append((query_name, seed, sample, "ost_failure"))
        continue

    if not confidence_json.exists():
        entries_failed.append((query_name, seed, sample, "missing_confidence"))
        continue

    with open(confidence_json) as f:
        confidence_data = json.load(f)

    lddt_data = []
    rmsd_data = []

    for e in ost_data["lddt_pli"]["assigned_scores"]:
        lddt_data.append((e["model_ligand"], e["reference_ligand"], e["score"]))
    for e in ost_data["rmsd"]["assigned_scores"]:
        rmsd_data.append(
            (
                e["model_ligand"],
                e["reference_ligand"],
                e["score"],
                e["lddt_lp"],
                e["bb_rmsd"],
            )
        )

    df_lddt = pd.DataFrame(
        lddt_data, columns=["model_ligand", "ligand_instance_chain", "lddt_pli"]
    )
    df_rmsd = pd.DataFrame(
        rmsd_data,
        columns=[
            "model_ligand",
            "ligand_instance_chain",
            "rmsd",
            "lddt_lp",
            "bb_rmsd",
        ],
    )

    # Merge the dataframes to add LDDT scores to RMSD dataframe
    df_combined = df_rmsd.merge(
        df_lddt,
        on=["ligand_instance_chain"],
        how="left",
        suffixes=("_chain_rmsd", "_chain_lddt_pli"),
    )
    if (
        df_combined[
            [
                "model_ligand_chain_rmsd",
                "ligand_instance_chain",
                "model_ligand_chain_lddt_pli",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        print(f"chain mapping error in {query_name}:{seed}:{sample}")
        entries_failed.append((query_name, seed, sample, "rmsd-lddt chain mapping"))
        continue

    # Rename the score columns for clarity
    df_combined = df_combined.rename(
        columns={"score_rmsd": "rmsd_score", "score_lddt": "lddt_score"}
    )

    df_combined["method"] = [args.method_name] * len(df_combined)
    df_combined["target"] = [f"{query_name}"] * len(df_combined)
    df_combined["seed"] = [seed] * len(df_combined)
    df_combined["sample"] = [sample] * len(df_combined)
    if not args.no_confidence:
        ## oracle results would basically all be empty here:
        df_combined["ranking_score"] = confidence_data["sample_ranking_score"]

        iptm_map = {
            tuple(k.strip("()").split(", ")): v
            for k, v in confidence_data["chain_pair_iptm"].items()
        }
        model_ligand_chains = sorted(
            set(df_combined["model_ligand_chain_rmsd"].str[:-3])
            | set(df_combined["model_ligand_chain_lddt_pli"].str[:-3])
        )
        iptm_data = {
            i: []
            for i in [
                "prot_lig_chain_iptm_average_rmsd",
                "prot_lig_chain_iptm_min_rmsd",
                "prot_lig_chain_iptm_max_rmsd",
                "prot_lig_chain_iptm_average_lddt_pli",
                "prot_lig_chain_iptm_min_lddt_pli",
                "prot_lig_chain_iptm_max_lddt_pli",
                "lig_prot_chain_iptm_average_rmsd",
                "lig_prot_chain_iptm_min_rmsd",
                "lig_prot_chain_iptm_max_rmsd",
                "lig_prot_chain_iptm_average_lddt_pli",
                "lig_prot_chain_iptm_min_lddt_pli",
                "lig_prot_chain_iptm_max_lddt_pli",
            ]
        }
        for l in df_combined["model_ligand_chain_rmsd"].str[:-3]:
            pl_iptm = np.array(
                [
                    v
                    for k, v in iptm_map.items()
                    if (
                        (k[0] == l and k[1] not in model_ligand_chains)
                        | (k[1] == l and k[0] not in model_ligand_chains)
                    )
                ]
            )
            iptm_mean = np.mean(pl_iptm) if len(pl_iptm) > 0 else np.nan
            iptm_min = np.min(pl_iptm) if len(pl_iptm) > 0 else np.nan
            iptm_max = np.max(pl_iptm) if len(pl_iptm) > 0 else np.nan
            iptm_data["prot_lig_chain_iptm_average_rmsd"].append(iptm_mean)
            iptm_data["lig_prot_chain_iptm_average_rmsd"].append(iptm_mean)
            iptm_data["prot_lig_chain_iptm_min_rmsd"].append(iptm_min)
            iptm_data["lig_prot_chain_iptm_min_rmsd"].append(iptm_min)
            iptm_data["prot_lig_chain_iptm_max_rmsd"].append(iptm_max)
            iptm_data["lig_prot_chain_iptm_max_rmsd"].append(iptm_max)

        for l in df_combined["model_ligand_chain_lddt_pli"].str[:-3]:
            pl_iptm = np.array(
                [
                    v
                    for k, v in iptm_map.items()
                    if (
                        (k[0] == l and k[1] not in model_ligand_chains)
                        | (k[1] == l and k[0] not in model_ligand_chains)
                    )
                ]
            )
            iptm_mean = np.mean(pl_iptm) if len(pl_iptm) > 0 else np.nan
            iptm_min = np.min(pl_iptm) if len(pl_iptm) > 0 else np.nan
            iptm_max = np.max(pl_iptm) if len(pl_iptm) > 0 else np.nan
            iptm_data["prot_lig_chain_iptm_average_lddt_pli"].append(iptm_mean)
            iptm_data["lig_prot_chain_iptm_average_lddt_pli"].append(iptm_mean)
            iptm_data["prot_lig_chain_iptm_min_lddt_pli"].append(iptm_min)
            iptm_data["lig_prot_chain_iptm_min_lddt_pli"].append(iptm_min)
            iptm_data["prot_lig_chain_iptm_max_lddt_pli"].append(iptm_max)
            iptm_data["lig_prot_chain_iptm_max_lddt_pli"].append(iptm_max)

        df_combined = pd.concat([df_combined, pd.DataFrame(iptm_data)], axis=1)

    if df_combined.empty:
        entries_empty.append((query_name, seed, sample))
        continue

    all_dfs.append(df_combined)

df_combined = pd.concat(all_dfs, ignore_index=True)


base_cols = [
    "target",
    "method",
    "seed",
    "sample",
    "lddt_pli",
    "model_ligand_chain_lddt_pli",
    "rmsd",
    "lddt_lp",
    "bb_rmsd",
    "model_ligand_chain_rmsd",
    "ligand_instance_chain",
    # "ligand_is_proper",
    # "sucos_shape",
    # "sucos_shape_pocket_qcov",
    # "protein_fident_weighted_sum",
    # "topological_tanimoto",
    # "morgan_tanimoto",
]

confidence_cols = [
    "ranking_score",
    "prot_lig_chain_iptm_average_rmsd",
    "prot_lig_chain_iptm_min_rmsd",
    "prot_lig_chain_iptm_max_rmsd",
    "prot_lig_chain_iptm_average_lddt_pli",
    "prot_lig_chain_iptm_min_lddt_pli",
    "prot_lig_chain_iptm_max_lddt_pli",
    "lig_prot_chain_iptm_average_rmsd",
    "lig_prot_chain_iptm_min_rmsd",
    "lig_prot_chain_iptm_max_rmsd",
    "lig_prot_chain_iptm_average_lddt_pli",
    "lig_prot_chain_iptm_min_lddt_pli",
    "lig_prot_chain_iptm_max_lddt_pli",
]

select_cols = base_cols + (confidence_cols if not args.no_confidence else [])
df_combined = df_combined[select_cols]

df_combined = df_combined.assign(
    model_ligand_chain_lddt_pli=lambda x: x.model_ligand_chain_lddt_pli.str.replace(
        ".1.", ""
    ),
    model_ligand_chain_rmsd=lambda x: x.model_ligand_chain_rmsd.str.replace(".1.", ""),
    ligand_instance_chain=lambda x: x.ligand_instance_chain.str.replace(".1.", ""),
)[
    [
        "target",
        "ligand_instance_chain",
        "method",
        "seed",
        "sample",
    ]  ## meta
    + ["lddt_pli", "rmsd"]
    + ["lig_prot_chain_iptm_average_rmsd", "ranking_score"]
    ## this is mean iptm, based on the `collate_rnp.py`,
    # but still applied at the interface level
].rename(
    columns={
        "target": "pdbid",
        "ligand_instance_chain": "ref_interface",
        "seed": "seed_number",
        "sample": "sample_number",
        "method": "model",
        "lig_prot_chain_iptm_average_rmsd": "iptm",
    }
)[
    [
        "model",
        "pdbid",
        "seed_number",
        "sample_number",
        "ref_interface",
    ]
    + ["lddt_pli", "rmsd", "iptm", "ranking_score"]
]
# %%
df_combined.to_csv(
    args.output_csv,
    header=True,
    index=False,
)
# %%
print(
    f"""Num entries failed: {len(entries_failed)}/{
        len(all_dfs) + len(entries_failed) + len(entries_empty)
    }"""
)
print(
    f"""Num entries empty: {len(entries_empty)}/{
        len(all_dfs) + len(entries_failed) + len(entries_empty)
    }"""
)
print("Failed Entries:")
print(entries_failed)

print("Empty Entries")
print(entries_empty)
# %%
