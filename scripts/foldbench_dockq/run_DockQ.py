import json
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import biotite.structure.io.pdbx as pdbx
import click
import numpy as np
import tqdm
from biotite.structure import get_residues
from eval_by_dockqv2 import dockq

from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.core.data.primitives.structure.metadata import get_cif_block
from openfold3.core.data.resources.residues import MoleculeType

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.WARNING)
# %%


def add_entity_poly_seq_to_cif(model_path, fixed_model_path):
    # 1. Load the model

    parsed_structure = parse_mmcif(model_path)
    atom_array = parsed_structure.atom_array
    # 2. load CIF block to so we cna write out
    cif_file = pdbx.CIFFile.read(model_path)
    cif_block = get_cif_block(cif_file)
    # 4. Construct _entity_poly_seq category
    # We need to map entity_id -> sequence

    entity_poly_seq_data = {"entity_id": [], "num": [], "mon_id": [], "hetero": []}

    # Iterate over entities
    unique_entities = np.unique(atom_array.entity_id)
    for entity_id in unique_entities:
        # Get atoms for this entity
        entity_mask = atom_array.entity_id == entity_id
        entity_atoms = atom_array[entity_mask]

        # Check molecule type
        mol_type_id = entity_atoms.molecule_type_id[0]  # Assuming consistent per entity
        try:
            mol_type = MoleculeType(mol_type_id)
        except ValueError:
            # Skip if unknown or not a polymer type we care about
            continue

        if mol_type not in [MoleculeType.PROTEIN, MoleculeType.RNA, MoleculeType.DNA]:
            continue

        # Get residues
        # We need the sequence of residues for the entity.
        # Since entity implies identical chains, we can just take the first chain for this entity.
        chain_ids = np.unique(entity_atoms.chain_id)
        first_chain_mask = entity_atoms.chain_id == chain_ids[0]
        first_chain_atoms = entity_atoms[first_chain_mask]

        # Get residue names in order
        _, res_names = get_residues(first_chain_atoms)
        for i, res_name in enumerate(res_names):
            entity_poly_seq_data["entity_id"].append(entity_id)
            entity_poly_seq_data["num"].append(str(i + 1))
            entity_poly_seq_data["mon_id"].append(res_name)
            entity_poly_seq_data["hetero"].append(
                "n"
            )  # Assuming standard residues for now

    # Add category to block
    cif_block["entity_poly_seq"] = pdbx.CIFCategory(entity_poly_seq_data)

    # 5. Write the fixed file
    cif_file.write(fixed_model_path)
    return


def run_DockQ(
    args,
    tmp_dir,
):
    try:
        (
            native_cif,
            model_cif,
            out_json,
        ) = args
        stem = model_cif.stem
        rstem = native_cif.stem
        tmp_model_cif = f"{tmp_dir}/MODEL_{stem}_ref_{rstem}.cif"
        add_entity_poly_seq_to_cif(model_cif, tmp_model_cif)
        result = dockq(
            model_path=str(tmp_model_cif),
            native_path=str(native_cif),
            model_chains=None,
            native_chains=None,
            small_molecule=False,
            allowed_mismatches=10,
        )
        with open(out_json, "w") as f:
            json.dump(
                result,
                f,
                indent=4,
                default=lambda x: x.item() if hasattr(x, "item") else x,
            )
        return True, model_cif
    except Exception as e:
        print(f"Error processing {model_cif}: {e}")
        return False, model_cif


@click.command()
@click.option(
    "--model_cif_dir",
    required=True,
    help=("Path to a dir of OF3 outputs."),
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--reference_cif_dir",
    required=True,
    help=("Path to reference cifs "),
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--output_dir",
    required=True,
    help=("Path to output directory"),
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option("--num_workers", required=True, help=("Number of workers"), type=int)
@click.option(
    "--skip_existing", is_flag=True, help="Whether to skip existing output files"
)
@click.option(
    "--tmp_dir",
    required=False,
    help="Path to temporary directory for intermediate files",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
    default=Path("/tmp/"),
)
def main(
    model_cif_dir: Path,
    reference_cif_dir: Path,
    output_dir: Path,
    num_workers: int,
    tmp_dir: Path,
    skip_existing: bool = False,
):
    test_pids = {
        "8SV3",  # div by 0 error
        "8Q43",  # operand broadcast error
    }
    output_dir.mkdir(exist_ok=True, parents=True)
    model_cif_files = model_cif_dir.rglob("*.cif")
    args = []
    for cf in model_cif_files:
        pdb_id = cf.stem.split("_")[0]
        stem = cf.stem
        ref_cifs = list(reference_cif_dir.glob(f"{pdb_id.lower()}*.cif"))
        if ref_cifs:
            for rc in ref_cifs:
                ref_name = rc.stem
                outpath = Path(f"{output_dir}/{stem}_ref_{ref_name}.json")
                if skip_existing and outpath.exists():
                    continue
                args.append((rc, cf, Path(f"{output_dir}/{stem}_ref_{ref_name}.json")))
        else:
            print(f"Warning: No reference CIF found for {pdb_id}, skipping.")

    success = []
    failed = []
    pfunc = partial(
        run_DockQ,
        tmp_dir=tmp_dir,
    )

    with Pool(num_workers) as p:
        for result in tqdm.tqdm(
            p.imap_unordered(pfunc, args, chunksize=3), total=len(args)
        ):
            succeeded, cif_file = result
            if succeeded:
                success.append(cif_file)
            else:
                failed.append(cif_file)
    print(f"Successfully processed {len(success)} files.")

    for f in failed:
        print(f"Failed to process {f}")
    return


if __name__ == "__main__":
    main()
