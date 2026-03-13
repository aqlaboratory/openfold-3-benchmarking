# %%
import json
from pathlib import Path

import click
import pandas as pd
import tqdm

"""
NOTE:
DockQ output looks something like this :
'BF': {
   'DockQ': 0.05604053796645724,
   'F1': 0.0,
   'iRMSD': 7.577608617698875,
   'LRMSD': 21.94888563752535,
   'fnat': 0,
   'nat_correct': 0,
   'nat_total': 1,
   'fnonnat': 1.0,
   'nonnat_count': 26,
   'model_total': 26,
   'clashes': 2,
   'len1': 121,
   'len2': 7,
   'class1': 'receptor',
   'class2': 'ligand',
   'is_het': False,
   'chain1': 'A',
   'chain2': 'D',
   'chain_map': {'A': 'B', 'B': 'A', 'C': 'C', 'D': 'F', 'E': 'E', 'F': 'D'}}

I am taking this to mean that the topmost key `BF` is the native/reference interface
and that chain1 and chain2 are model interface chains 
I can't find any concrete documentation to support this in the repo, but 
Gemini supports this 

The topmost keys `BF` also appears to be stable across runs 
"""


@click.command()
@click.option(
    "--dockq_outdir",
    type=Path,
    help="Path to directory containing DockQ output json files",
)
@click.option(
    "--prediction_outdir",
    type=Path,
    help="Path to directory containing prediction outputs including confidence files",
)
@click.option("--output_file", type=Path, help="Path to write collated results csv")
def main(dockq_outdir: Path, prediction_outdir: Path, output_file: Path):
    dockq_files = list(dockq_outdir.rglob("*.json"))
    clean_results = []
    for df in tqdm.tqdm(dockq_files):
        with open(df) as f:
            dockq_results = json.load(f)

        pdb_id, _, seed, _, sample = df.stem.split("_")[:5]
        confidence_file = (
            prediction_outdir
            / f"{pdb_id}/seed_{seed}/{pdb_id}_seed_{seed}_sample_{sample}_confidences_aggregated.json"
        )
        with open(confidence_file) as f:
            conf = json.load(f)

        for intr_id, intr_results in dockq_results["best_result"].items():
            model_chain_pair = sorted([intr_results["chain1"], intr_results["chain2"]])
            chain_pair_key = f"({model_chain_pair[0]}, {model_chain_pair[1]})"
            iptm = conf["chain_pair_iptm"][chain_pair_key]
            bespoke_iptm = conf["bespoke_iptm"][chain_pair_key]
            clean_results.append(
                (
                    pdb_id,
                    seed,
                    sample,
                    intr_id,
                    "".join(model_chain_pair),
                    intr_results["DockQ"],
                    intr_results["F1"],
                    intr_results["iRMSD"],
                    intr_results["LRMSD"],
                    conf["sample_ranking_score"],
                    iptm,
                    bespoke_iptm,
                )
            )

    clean_results = pd.DataFrame(
        clean_results,
        columns=[
            "pdbid",
            "seed_number",
            "sample_number",
            "ref_interface",
            "model_interface",
            "dockq",
            "f1",
            "iRMSD",
            "LRMSD",
            "sample_ranking_score",
            "iptm",
            "bespoke_iptm",
        ],
    )
    clean_results.to_csv(output_file, index=False)
    return


if __name__ == "__main__":
    main()
