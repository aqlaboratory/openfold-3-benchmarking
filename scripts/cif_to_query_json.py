from __future__ import annotations

import json
import logging
import os
import string
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import biotite.structure.io.pdbx as pdbx
import click
import numpy as np
import pandas as pd
import requests
import tqdm
from Bio import SeqIO
from biotite.structure import AtomArray
from biotite.structure.bonds import BondList, BondType
from biotite.structure.io.pdbx import CIFFile
from pdbeccdutils.core.ccd_reader import Component

from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.core.data.primitives.structure.cleanup import (
    prefilter_bonds,
    remove_crystallization_aids,
    remove_hydrogens,
    remove_waters,
)
from openfold3.core.data.primitives.structure.component import (
    find_cross_chain_bonds,
    pdbeccdutils_component_from_ccd,
)
from openfold3.core.data.primitives.structure.labels import (
    MoleculeType,
    assign_molecule_type_ids,
)
from openfold3.core.data.primitives.structure.metadata import (
    get_chain_to_canonical_seq_dict,
    get_chain_to_three_letter_codes_dict,
    get_cif_block,
)

logger = logging.getLogger("generate_inputs")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# Constants
STANDARD_PROTEIN_3 = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
}
STANDARD_RNA = {"A", "G", "C", "U", "N"}
STANDARD_DNA = {"DA", "DG", "DC", "DT", "DN"}


@dataclass(frozen=True)
class ChainInfo:
    chain_id: str
    molecule_type: str  # 'protein'|'rna'|'dna'|'other'
    sequence_1letter: str
    sequence_3letter: list[str]
    # list of (1-based index, 3-letter-code)
    modified_residues: list[tuple[int, str]]


@dataclass(frozen=True)
class LigandInfo:
    chain_id: str
    ccd_code: str
    smiles: str | None


@dataclass(frozen=True)
class BondInfo:
    # (chain, res_id, atom_name) for each atom in a cross-chain bond
    a: tuple[str, int, str]
    b: tuple[str, int, str]
    # optional: raw indices for debugging
    atom_index_a: int
    atom_index_b: int


@dataclass(frozen=True)
class GenerateInputsResult:
    pdb_id: str
    chains: dict[str, ChainInfo]
    ligands: dict[str, LigandInfo]
    cross_chain_bonds: list[BondInfo]

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(
            {
                "pdb_id": self.pdb_id,
                "chains": {k: asdict(v) for k, v in self.chains.items()},
                "ligands": {k: asdict(v) for k, v in self.ligands.items()},
                "cross_chain_bonds": [asdict(b) for b in self.cross_chain_bonds],
            },
            indent=indent,
        )


# Helper functions
def download_cif(pdbid: str, dst: Path) -> None:
    url = f"https://files.rcsb.org/download/{pdbid.upper()}.cif"
    logger.info(f"Downloading {url} -> {dst}")
    try:
        headers = {"User-Agent": "openfold3-generate-inputs/1.0"}
        rsp = requests.get(url, headers=headers, timeout=30)
        rsp.raise_for_status()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(rsp.content)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download {pdbid}.cif: {e}") from e


def get_smiles_value(descriptors: Sequence[Component.Descriptor]) -> str | None:
    """
    Prefer 'SMILES_CANONICAL', then 'SMILES', else None.
    """
    if not descriptors:
        return None
    for d in descriptors:
        # pdbeccdutils uses an enum-like .type; accept str as fallback
        t = d.type.name if hasattr(d.type, "name") else str(d.type)
        if t == "SMILES_CANONICAL":
            return d.value
    for d in descriptors:
        t = d.type.name if hasattr(d.type, "name") else str(d.type)
        if t == "SMILES":
            return d.value
    return None


def molecule_type_to_str(mt_id: int) -> str:
    try:
        mt = MoleculeType(mt_id)
    except Exception:
        return "other"
    if mt == MoleculeType.PROTEIN:
        return "protein"
    if mt == MoleculeType.RNA:
        return "rna"
    if mt == MoleculeType.DNA:
        return "dna"
    if mt == MoleculeType.LIGAND:
        return "ligand"
    return "other"


def modified_residues(
    three_letter_seq: Sequence[str], molecule_type_str: str
) -> list[tuple[int, str]]:
    if molecule_type_str == "protein":
        standard = STANDARD_PROTEIN_3
    elif molecule_type_str == "rna":
        standard = STANDARD_RNA
    elif molecule_type_str == "dna":
        standard = STANDARD_DNA
    else:
        return []

    out = []
    for i, code in enumerate(three_letter_seq):
        if code not in standard:
            out.append((i + 1, code))
    return out


def generate_chain_ids(exclude: Iterable[str], count: int) -> list[str]:
    """
    Generate up to `count` unique chain IDs avoiding those in `exclude`.
    Single letters (A..Z) first, then 2-letter A..Z namespace.
    """
    exclude_set = set(exclude)
    alphabet = string.ascii_uppercase
    ids: list[str] = []

    # 1-letter
    for ch in alphabet:
        if len(ids) >= count:
            return ids
        if ch not in exclude_set:
            ids.append(ch)
            exclude_set.add(ch)

    # 2-letter
    first, second = 0, 0
    while len(ids) < count:
        cid = alphabet[first] + alphabet[second]
        if cid not in exclude_set:
            ids.append(cid)
            exclude_set.add(cid)
        second += 1
        if second == len(alphabet):
            second = 0
            first += 1
    return ids


def additional_bond_filtering(
    atom_array: AtomArray,
    keep_polymer_ligand: bool = True,
    keep_ligand_ligand: bool = True,
    remove_all_metal_coordination: bool = True,
) -> AtomArray:
    """
    Post-filter biotite BondList with more explicit rules.
    """
    out = atom_array.copy()
    if out.bonds is None or len(out.bonds.as_array()) == 0:
        return out

    bonds = out.bonds.as_array()  # shape (N, 3) [i, j, BondType]
    partners = bonds[:, :2]
    valid = np.ones(len(partners), dtype=bool)

    if (not keep_polymer_ligand) or (not keep_ligand_ligand):
        moltypes = out.molecule_type_id[partners]  # shape (N, 2)
        is_lig = np.isin(moltypes, [MoleculeType.LIGAND.value])

        if not keep_polymer_ligand:
            is_poly = np.isin(
                moltypes,
                [
                    MoleculeType.PROTEIN.value,
                    MoleculeType.RNA.value,
                    MoleculeType.DNA.value,
                ],
            )
            is_poly_lig = is_poly.any(axis=1) & is_lig.any(axis=1)
            valid[is_poly_lig] = False

        if not keep_ligand_ligand:
            is_lig_lig = is_lig.all(axis=1)
            valid[is_lig_lig] = False

    if remove_all_metal_coordination:
        btypes = bonds[:, 2]
        is_coord = btypes == BondType.COORDINATION
        valid[is_coord] = False

    out.bonds = BondList(atom_count=len(out), bonds=bonds[valid])
    return out


def expand_operation_expression(expression):
    """
    Parses a PDB assembly operator expression (e.g., "1,2,5-7")
    and returns the count of operators.
    """
    expression = expression.strip("()")
    if not expression:
        return 0

    count = 0
    parts = expression.split(",")

    for part in parts:
        if "-" in part:
            try:
                start, end = part.split("-")
                count += int(end) - int(start) + 1
            except ValueError:
                continue
        else:
            count += 1
    return count


def get_complete_stoichiometry_map(file_path, assembly_id="1"):
    """
    Returns a dictionary mapping Entity ID to its count in the biological assembly.
    Correctly handles Ligands and Water by using 'label_asym_id'.
    """
    try:
        cif_file = pdbx.CIFFile.read(file_path)
    except FileNotFoundError:
        return "File not found"

    if hasattr(cif_file, "block"):
        block = cif_file.block
    else:
        block = next(iter(cif_file.values()))

    # 1. Map Asym IDs (System Chain IDs) to Entity IDs
    asym_to_entity = {}
    if "struct_asym" in block:
        struct_asym = block["struct_asym"]

        # --- FIX: STRICTLY USE SYSTEM IDs (label_asym_id) ---
        # The assembly instructions (pdbx_struct_assembly_gen) always refer to
        # label_asym_id, NOT auth_asym_id.
        if "label_asym_id" in struct_asym:
            chain_ids = struct_asym["label_asym_id"].as_array()
        else:
            # Fallback for minimal files that only have the primary key
            chain_ids = struct_asym["id"].as_array()

        ent_ids = struct_asym["entity_id"].as_array()

        for a_id, e_id in zip(chain_ids, ent_ids):
            asym_to_entity[a_id] = e_id

    # 2. Calculate Counts from Assembly Generation
    entity_counts = {}

    if "pdbx_struct_assembly_gen" in block:
        gen_cat = block["pdbx_struct_assembly_gen"]

        gen_assembly_ids = gen_cat["assembly_id"].as_array()
        gen_opers = gen_cat["oper_expression"].as_array()
        gen_asym_lists = gen_cat["asym_id_list"].as_array()

        for i in range(len(gen_assembly_ids)):
            if gen_assembly_ids[i] != assembly_id:
                continue

            # Get multiplier
            multiplier = expand_operation_expression(gen_opers[i])

            # Get chains involved
            chain_list = gen_asym_lists[i].split(",")

            for chain_id in chain_list:
                chain_id = chain_id.strip()

                # Now looking up the correct system ID
                entity_id = asym_to_entity.get(chain_id)

                if entity_id:
                    entity_counts[entity_id] = (
                        entity_counts.get(entity_id, 0) + multiplier
                    )
    return {k.item(): v for k, v in entity_counts.items()}


def generate_inputs(
    pdb_id: str,
    cif_path: Path,
    cif_dir: Path,
    ccd_cif: CIFFile,
    apply_additional_bond_filter: bool = False,
) -> GenerateInputsResult:
    """
    Args:
        pdb_id: 4-letter PDB ID (case-insensitive).
        cif_dir: Directory to read/write the PDB .cif file.
        ccd_cif: Path to components.cif (CCD dictionary).
        apply_additional_bond_filter (bool): If True, apply extra filtering after prefilter_bonds().

    Returns:
        GenerateInputsResult
    """

    cif_file = CIFFile.read(str(cif_path))
    cif_block = get_cif_block(cif_file)
    parsed = parse_mmcif(
        str(cif_path), expand_bioassembly=True, renumber_chain_ids=True
    )
    atom_array = parsed.atom_array

    # Clean up
    atom_array = remove_waters(atom_array)
    atom_array = remove_hydrogens(atom_array)
    atom_array = remove_crystallization_aids(atom_array)

    ## re-numbering chains will expand homomeric chains - check here
    cid_map = np.sort(
        np.unique(
            np.column_stack((atom_array.chain_id, atom_array.label_asym_id)), axis=0
        )
    )

    unique_label_asym_ids = np.unique(atom_array.label_asym_id)
    for chain_id in unique_label_asym_ids:
        cid_map_asym = cid_map[cid_map[:, 1] == chain_id]
        chain_numbering = [""] + [str(i) for i in range(1, cid_map_asym.shape[0] + 1)][
            1:
        ]
        for i in range(cid_map_asym.shape[0]):
            # 0 is numeric chain id, 1 is the original chain ID
            mask = atom_array.chain_id == cid_map_asym[i, 0]
            atom_array.chain_id[mask] = str(cid_map_asym[i, 1]) + chain_numbering[i]
    ## verify that the stoichiometry for each entity matches
    entity2stoich = get_complete_stoichiometry_map(str(cif_path), assembly_id="1")
    chain2entity_unique = np.sort(
        np.unique(np.column_stack((atom_array.entity_id, atom_array.chain_id)), axis=0)
    )
    entities, counts = np.unique(chain2entity_unique[:, 0], return_counts=True)
    for i in range(len(entities)):
        ent = entities[i]
        count = counts[i]
        expected_count = entity2stoich.get(ent, None)
        if expected_count is not None and expected_count != count:
            raise ValueError(
                f"Entity {ent} in {pdb_id} has {count} copies after expansion, but expected stoichiometry {expected_count}"
            )

    assign_molecule_type_ids(atom_array, cif_file)
    chain_to_seq = get_chain_to_canonical_seq_dict(atom_array, cif_block, ccd=ccd_cif)
    chain_to_3 = get_chain_to_three_letter_codes_dict(atom_array, cif_block)

    chains: dict[str, ChainInfo] = {}
    for chain_id, seq1 in chain_to_seq.items():
        chain_mask = atom_array.chain_id == chain_id
        if not np.any(chain_mask):
            continue
        moltype_id = int(atom_array[chain_mask].molecule_type_id[0])
        mtype = molecule_type_to_str(moltype_id)
        seq3 = chain_to_3.get(chain_id, [])
        modres = modified_residues(seq3, mtype)
        chains[chain_id] = ChainInfo(
            chain_id=str(chain_id),
            molecule_type=mtype,
            sequence_1letter=str(seq1),
            sequence_3letter=list(seq3),
            modified_residues=modres,
        )

    # Handle ligands:
    ligand_mask = atom_array.molecule_type_id == MoleculeType.LIGAND.value
    ligands = atom_array[ligand_mask]
    ligand_chain_ids = np.unique(ligands.chain_id)
    all_chain_ids = np.unique(atom_array.chain_id)

    # 2) Break glycan-like ligand chains with multiple res_ids into separate chains
    for cid in ligand_chain_ids:
        lig = ligands[ligands.chain_id == cid]
        uniq_res = np.unique(lig.res_id)
        if len(uniq_res) > 1:
            new_ids = generate_chain_ids(all_chain_ids, len(uniq_res) - 1)
            all_chain_ids = np.append(all_chain_ids, new_ids)
            assign_idx = 0
            for resid in uniq_res:
                if resid == 1:
                    continue
                mask = (atom_array.chain_id == cid) & (atom_array.res_id == resid)
                atom_array.chain_id[mask] = new_ids[assign_idx]
                assign_idx += 1

    # 3) Normalize all ligand res_id to 1 (after splitting)
    ligand_mask = atom_array.molecule_type_id == MoleculeType.LIGAND.value
    atom_array.res_id[ligand_mask] = 1

    # 4) Extract ligand info: chain -> (ccd_code, smiles)
    ligands_out: dict[str, LigandInfo] = {}
    ligand_chains_after = np.unique(atom_array[ligand_mask].chain_id)
    for chain_id in ligand_chains_after:
        lig = atom_array[(atom_array.chain_id == chain_id) & ligand_mask]
        if len(lig) == 0:
            continue
        ccd_codes = np.unique(lig.res_name)
        ccd_code = str(ccd_codes[0]) if len(ccd_codes) else "UNK"
        try:
            comp = pdbeccdutils_component_from_ccd(ccd_code, ccd_cif)
            smiles = get_smiles_value(comp.descriptors)
        except Exception:
            smiles = None
        ligands_out[str(chain_id)] = LigandInfo(
            chain_id=str(chain_id), ccd_code=ccd_code, smiles=smiles
        )

    # 5) Bond filtering
    atom_array = prefilter_bonds(
        atom_array=atom_array,
        remove_inter_chain_dative=True,
        remove_inter_chain_poly_links=True,
        remove_intra_chain_poly_links=True,
        remove_longer_than=2.4,
    )

    if apply_additional_bond_filter:
        atom_array = additional_bond_filtering(
            atom_array=atom_array,
            keep_polymer_ligand=True,
            keep_ligand_ligand=True,
            remove_all_metal_coordination=True,
        )

    # 6) Cross-chain bonds
    cross = find_cross_chain_bonds(atom_array)
    bonds_out: list[BondInfo] = []
    for i, j, *_ in cross:
        a = atom_array[i]
        b = atom_array[j]
        bonds_out.append(
            BondInfo(
                a=(str(a.chain_id), int(a.res_id), str(a.atom_name).strip()),
                b=(str(b.chain_id), int(b.res_id), str(b.atom_name).strip()),
                atom_index_a=int(i),
                atom_index_b=int(j),
            )
        )

    return GenerateInputsResult(
        pdb_id=pdb_id, chains=chains, ligands=ligands_out, cross_chain_bonds=bonds_out
    )


def create_of3_json_inputs(
    input_dir: str,
    pdb_id: str,
    result: GenerateInputsResult,
    refined=True,
    repr_mapping: dict = None,
    msa_dir: str = None,
):
    """
    Creates an OF3 input JSON file for protein structure prediction.

    Args:
        input_dir (str): Directory where the JSON file will be saved
        pdb_id (str): Identifier for the PDB structure
        result: GenerateInputsResult object from generate_inputs()
        refined (bool): Whether to include refined features like modified residues and bonds

    Returns:
        str: Path to the created JSON file
    """
    os.makedirs(input_dir, exist_ok=True)
    chain_groups = defaultdict(list)

    # process biopolymers
    for chain_id, chain_info in result.chains.items():
        sequence = chain_info.sequence_1letter
        modified_residues = chain_info.modified_residues
        molecule_type = chain_info.molecule_type

        if refined and modified_residues:
            key = (molecule_type, sequence, tuple(modified_residues))
        else:
            key = (molecule_type, sequence)
        chain_groups[key].append(chain_id)

    # process ligands
    for chain_id, ligand_info in result.ligands.items():
        sequence = ligand_info.smiles or "UNK"
        molecule_type = "ligand"

        key = (molecule_type, sequence)
        chain_groups[key].append(chain_id)

    chains = []
    needs_msa_mapping = []
    for (molecule_type, sequence, *extras), chain_ids in chain_groups.items():
        chain_entry = {
            "molecule_type": molecule_type,
            "chain_ids": chain_ids,
        }

        if molecule_type in ["protein", "rna", "dna"]:
            chain_entry.update({"sequence": sequence})
        else:
            chain_entry.update({"smiles": sequence})

        ## set MSA/template to blank or based on repr
        if molecule_type in ["protein", "rna"]:
            ## first set to blanks
            msa_fp = "/absolute/path/to/main_msa.sto/a3m"
            template_fp = "/absolute/path/to/main_msa/hmm_output.sto"
            repr_id = None
            if repr_mapping:
                repr_id = repr_mapping.get(chain_entry["sequence"])
                if repr_id:
                    msa_fp = f"{msa_dir}/{repr_id}"
                    template_fp = f"{msa_dir}/{repr_id}/hmm_output.sto"

            chain_entry.update(
                {
                    "main_msa_file_paths": msa_fp,
                }
            )
            if molecule_type == "protein":
                chain_entry.update(
                    {
                        "template_alignment_file_path": template_fp,
                    }
                )
            ## if we don't have an MSA for this sequence, generate input so we can
            ## create an output fasta
            if msa_fp == "/absolute/path/to/main_msa.sto/a3m":
                needs_msa_mapping.append(
                    (pdb_id, chain_ids[0], molecule_type, chain_entry["sequence"])
                )

        if (
            refined and extras and extras[0]
        ):  # extras[0] would be modified_residues tuple
            modified_residues = extras[0]
            if modified_residues:
                chain_entry["non_canonical_residues"] = {
                    str(pos): residue_name for pos, residue_name in modified_residues
                }
        chains.append(chain_entry)

    # are covalently bonded ligands supported?
    # if refined and False:
    #     raise NotImplementedError("Bond handling is not yet implemented. We don't even have that feature available!")

    of3_input = {"queries": {pdb_id: {"chains": chains}}}

    return of3_input, needs_msa_mapping


def cif_to_json(
    args: tuple,
    outdir: Path,
    ccd_cif: CIFFile,
    repr_mapping: dict = None,
    msa_dir: str = None,
):
    pdb_id, cif_file = args
    outdir.mkdir(parents=True, exist_ok=True)
    if cif_file is not None:
        pdb_id = cif_file.stem.upper()
    elif pdb_id is not None:
        pdb_id = pdb_id.upper()
        cif_file = outdir / f"{pdb_id}.cif"
        if not cif_file.exists():
            download_cif(pdb_id, cif_file)
    else:
        raise ValueError("Either --cif-file or --pdb-id must be provided.")
    try:
        result = generate_inputs(
            pdb_id=pdb_id,
            cif_path=cif_file,
            cif_dir=outdir,
            ccd_cif=ccd_cif,
            apply_additional_bond_filter=True,
        )

        of3_input, needs_msa_mapping = create_of3_json_inputs(
            input_dir=str(outdir),
            pdb_id=pdb_id,
            result=result,
            refined=True,
            repr_mapping=repr_mapping,
            msa_dir=msa_dir,
        )
    except Exception as e:
        logger.error(f"Error processing {pdb_id}: {e}")
        return None
    return of3_input, needs_msa_mapping


def df_to_fasta(df, outfile):
    if df.shape[0] > 0:
        with open(outfile, "w+") as f:
            for _, row in df.iterrows():
                f.write(
                    f">{row.pdb_id}_{row.chain_id}\t{row.moltype}\n{row.sequence}\n"
                )


def parallel_local_cif_to_json(
    pdb_id_file: Path,
    input_cif_dir: Path,
    outdir: Path,
    ccd_cif: CIFFile,
    repr_mapping: dict = None,
    msa_dir: Path = None,
    num_workers: int = 4,
):
    with open(pdb_id_file) as f:
        pdb_ids = [line.strip().lower() for line in f if line.strip()]
    input_cif_files = []
    for pdb_id in pdb_ids:
        cif_path = input_cif_dir / f"{pdb_id}.cif"
        if not cif_path.exists():
            print(f"CIF file not found for PDB ID {pdb_id} at {cif_path}")
        else:
            input_cif_files.append(cif_path)

    args = list(zip(pdb_ids, input_cif_files))

    pfunc = partial(
        cif_to_json,
        outdir=outdir,
        ccd_cif=ccd_cif,
        repr_mapping=repr_mapping,
        msa_dir=msa_dir,
    )
    complete_config = {"queries": {}}
    all_missing_msas = []
    with Pool(num_workers) as pool:
        for output in tqdm.tqdm(pool.imap_unordered(pfunc, args), total=len(args)):
            if output:
                of3_input, needs_msa_mappping = output
                complete_config["queries"].update(of3_input["queries"])
                all_missing_msas.extend(needs_msa_mappping)

    # write out the combined JSON
    out_json_file = outdir / "of3_inputs.json"
    with open(out_json_file, "w") as f:
        json.dump(complete_config, f, indent=4)

    # write out missing msa fasta
    missing_msa_df = pd.DataFrame(
        all_missing_msas, columns=["pdb_id", "chain_id", "moltype", "sequence"]
    )
    missing_msa_fasta = outdir / "missing_msas.fasta"
    df_to_fasta(missing_msa_df, missing_msa_fasta)


descr = """
Generate OF3 JSON inputs from CIF files. Three input options:
- input a pdb id, and download CIF from the pdb (--pdb-id)
- a single CIF file (--cif-file)
- a list of PDB IDs + directory with pre-downloaded CIF files (--pdb-id-file)+\
(--input-cif-dir)
"""


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to save the output JSON file",
)
@click.option(
    "--pdb-id-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="File with list of PDB IDs to process",
)
@click.option(
    "--input-cif-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory containing input CIF files for batch processing",
)
@click.option(
    "--num-workers",
    type=int,
    default=4,
    help="Number of parallel workers for batch processing",
)
@click.option("--pdb-id", type=str, default=None, help="PDB ID to download and process")
@click.option(
    "--cif-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to an input CIF file",
)
@click.option(
    "--ccd-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the CCD CIF file",
)
@click.option(
    "--repr-fasta",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to FASTA containing protein/RNA",
)
@click.option(
    "--msa-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Path to directory with MSAs",
)
def main(
    output_dir,
    pdb_id_file,
    input_cif_dir,
    num_workers,
    pdb_id,
    cif_file,
    ccd_file,
    repr_fasta,
    msa_dir,
):
    ccd_cif = CIFFile.read(ccd_file)
    if repr_fasta:
        repr_mapping = {}
        for record in SeqIO.parse(repr_fasta, "fasta"):
            repr_mapping[str(record.seq)] = record.id
    else:
        repr_mapping = None
    if pdb_id_file:
        parallel_local_cif_to_json(
            pdb_id_file=pdb_id_file,
            input_cif_dir=input_cif_dir,
            outdir=output_dir,
            ccd_cif=ccd_cif,
            repr_mapping=repr_mapping,
            msa_dir=msa_dir,
            num_workers=num_workers,
        )
    else:
        result = cif_to_json(
            args=(pdb_id, cif_file),
            outdir=Path(output_dir),
            ccd_cif=ccd_cif,
            repr_mapping=repr_mapping,
            msa_dir=msa_dir,
        )
        if result:
            of3_input, needs_msa = result
        else:
            return

        if not pdb_id:
            pdb_id = cif_file.stem.upper()

        config_outfile = output_dir / f"{pdb_id}.json"
        with open(config_outfile, "w+") as ofl:
            json.dump(of3_input, ofl, indent=4)

        missing_msa_df = pd.DataFrame(
            needs_msa, columns=["pdb_id", "chain_id", "moltype", "sequence"]
        )
        missing_msa_fasta = output_dir / f"{pdb_id}_missing_msas.fasta"
        df_to_fasta(missing_msa_df, missing_msa_fasta)


if __name__ == "__main__":
    main()
