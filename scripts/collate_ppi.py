# %%
import json
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path

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
    query_name, seed, sample = itemgetter(0, 2, 4)(ost_result_json.stem.split("_"))

    pred_dir = Path(args.pred_dir)
    confidence_json = (
        pred_dir / f"{query_name}/seed_{seed}/{query_name}_"
        f"seed_{seed}_sample_{sample}_confidences_aggregated.json"
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

    """
    As per the OST docs: 
    "dockq_interfaces" is a subset of
    "dockq_reference_interfaces" that contains interfaces
    that can be mapped to the model. They are stored as
    lists in format [ref_ch1, ref_ch2, mdl_ch1, mdl_ch2].
    """
    ost_data["ref_interface"] = [
        "".join(sorted(ig[:2])) for ig in ost_data["dockq_interfaces"]
    ]
    ost_data["model_interface"] = [
        "".join(sorted(ig[2:])) for ig in ost_data["dockq_interfaces"]
    ]
    interface_keys = [
        "dockq_interfaces",
        "ref_interface",
        "model_interface",
        "dockq",
        "fnat",
        "fnonnat",
        "irmsd",
        "lrmsd",
        "nnat",
        "nmdl",
    ]
    single_keys = [
        "dockq_ave",
        "dockq_wave",
        "dockq_ave_full",
        "dockq_wave_full",
        "tm_score",
        "lddt",
        "ilddt",
    ]

    metric_df = pd.DataFrame({key: ost_data[key] for key in interface_keys})
    is_monomer = metric_df.shape[0] == 0
    if is_monomer:
        single_keys = ["tm_score", "lddt"]
        interface_keys = []
        metric_df = pd.DataFrame({key: [ost_data[key]] for key in single_keys})
    else:
        for key in single_keys:
            metric_df[key] = ost_data[key]

    metric_df["query"] = query_name
    metric_df["seed"] = seed
    metric_df["sample"] = sample
    metric_df["method"] = args.method_name
    metric_df["ost_output"] = str(ost_result_json)
    metric_df = metric_df[
        ["query", "seed", "sample", "method", "ost_output"]
        + interface_keys
        + single_keys
    ]

    if not args.no_confidence:
        ## oracle results would basically all be empty here:
        metric_df["ranking_score"] = confidence_data["sample_ranking_score"]
        metric_df["avg_plddt"] = confidence_data["avg_plddt"]
        metric_df["ptm"] = confidence_data["ptm"]
        if not is_monomer:
            iptm_df = pd.DataFrame(
                [
                    ("".join(sorted(k.strip("()").split(", "))), v)
                    for k, v in confidence_data["chain_pair_iptm"].items()
                ],
                columns=["model_interface", "iptm"],
            )
            old_shape = metric_df.shape[0]
            metric_df = metric_df.merge(iptm_df, how="left")
            assert old_shape == metric_df.shape[0]
            assert metric_df["iptm"].notnull().all()

    all_dfs.append(metric_df)

# %%
df_combined = pd.concat(all_dfs).rename(
    columns={
        "query": "pdbid",
        "seed": "seed_number",
        "sample": "sample_number",
        "lddt": "lddt_score",
        "ranking_score": "aggregate_score",
    }
)
if "ref_interface" in df_combined.columns:
    df_combined = df_combined.assign(
        ref_interface=lambda x: x["ref_interface"].str.replace("-", "")
    )


# %%
df_combined.to_csv(
    args.output_csv,
    header=True,
    index=False,
)
# %%
print(
    f"Num entries failed: "
    f"{len(entries_failed)}/{len(all_dfs) + len(entries_failed) + len(entries_empty)}"
)
print(
    f"Num entries empty: "
    f"{len(entries_empty)}/{len(all_dfs) + len(entries_failed) + len(entries_empty)}"
)
# %%
