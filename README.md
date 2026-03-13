# OpenFold3 Benchmarking pipeline 

This directory contains scripts and instructions for benchmarking OpenFold3 and comparing it to other models. We use Openstructure as our main evaluation tool, and have several wrapper scripts to run and collate these results. 
The general workflow is 
1. predict structures
2. run ost on the predicted structures
3. collate results into a csv file 


## Setup 

- Install openfold3 and openstructure (OST). It is highly recommended to install these into SEPARATE environments, as the two can cause conflicts. Installing OST is very annoying and system dependent. These instructions are what we use for the cluster we are on but YMMV:

```
mamba create -n ost28 -c conda-forge -y python=3.9 "cmake>=3.23,<3.26" boost=1.76 zlib eigen=3.4.* sqlite=3.35.* fftw=3.3.* libtiff=4.2.* libpng=1.6.* "qt=5.15.*" "pyqt=5.*" mesa-libgl-devel-cos7-x86_64 mesa-libglu-devel-cos7-x86_64 freeglut pydantic tqdm pyyaml networkx scipy pandas numpy
conda activate ost28
mamba install -c conda-forge 

git clone https://git.scicore.unibas.ch/schwede/openstructure.git
cd openstructure
git checkout 2.8.0

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DPython_FIND_VIRTUALENV=FIRST -DPython_FIND_STRATEGY=LOCATION -DPython_ROOT_DIR="$CONDA_PREFIX" -DPython_EXECUTABLE="$CONDA_PREFIX/bin/python"
cmake --build . -j"$(nproc)"
cmake --install .
test with ost --help to make sure that the preliminary installation succeeded
create compound library following these instructions: https://openstructure.org/docs/2.11/conop/compoundlib/#mmcif-convert

## if chemdict-tools doesn’t work, you may need to add stuff to LD_LIBRARY_PATH
LD_LIBRARY_PATH="<path/to/your/ost/env>/lib64:<path/to/your/ost/env>/lib:$LD_LIBRARY_PATH" chemdict_tool --help
redo steps 9.-11. but at step 9. add -DCOMPOUND_LIB=<path/to/your/compounds.chemlib> to your cmake command

ost --help
```
- download data from s3: `aws s3 sync s3://openfold3-data/benchmarking/ . ` . This will download the reference structures, MSAs, existing results, and template files.(currents only has RnP, FoldBench, and GDM's AbAg set) 

- edit the query jsons and runner yaml so input paths match the paths to the relevant downloaded data. Additionally set the number of nodes/gpus in the runner yaml. 

## Running Predictions 

Ensure you have Openfold3 set up correctly and you have the weights downloaded. Run predictions like this:

```
run_openfold.py predict \
    --query_json benchmark_query_jsons/rnp.json \
    --runner_yaml runner.yaml \
    --output_dir predictions/rnp \
    --use_msa_server False  \
    --use_templates True 
```


## Evaluating structures 

We have three evaluations protocols:
- OST for protein-ligand, protein-protein, protein-peptide and monomers. Run this with `scripts/run_ost.py`, selecting an ost configuration in `ost_configs/` accordingly 
- A custom evaluation script(`script/evaluate_abag.py`) for AbAg structures
- DockQv2 as modified by the FoldBench authors for protein-nucleic acids



Here are some example evaluation calls 

```
## RnP
python scripts/run_ost.py  \
    --pred-dir predictions/rnp \
    --ref-dir reference_structures/rnp \
    --output-dir evaluation/rnp \
    --runner-yml ost_configs/ost_config_rnp.yml \
    --log-dir ost_logs/rnp

## FoldBench PPI
python scripts/run_ost.py  \
    --pred-dir predictions/interface_protein_protein \
    --ref-dir reference_structures/foldbench \
    --output-dir evaluation/interface_protein_protein \
    --runner-yml ost_configs/ost_config_fb.yml \
    --log-dir ost_logs/interface_protein_protein

## Antibody Antigen
python scripts/evaluate_abag.py \ 
    --pred_dir predictions/abag_gdm \
    --output_path collated_abag_gdm.csv \
    --ref_dir reference_structures/abag_gdm

## FoldBench Protein-RNA
python scripts/foldbench_dockq/run_DockQ.py \
    --model_cif_dir predictions/interface_protein_rna \
    --reference_cif_dir reference_structures/foldbench \
    --output_dir evaluation/interface_protein_rna \
    --num_workers 16
```


## Collate ost results

Finally, run collate scripts to condense evaluation output + confidence metrics into tables

example calls:

```
## rnp
python collate_pli.py  \
    --ost_dir evaluation/rnp/with_templates/  \
    --pred_dir predictions/rnp/with_templates/  \
    --output_csv rnp_with_templates_collated.csv \
    --annot_df rnp_annotations.csv \
    --method_name "<name of run>"

## foldbench protein

python collate_ppi.py \
    --ost_dir evaluation/foldbench_protein/with_templates/ \
    --pred_dir predictions/foldbench_protein/with_templates/ \
    --output_csv foldbench_protein_with_templates_collated.csv \
    --method_name "<name of run>"

## AbAg evaluation script also collates

python scripts/collate_DockQ.py \
    --dockq_outdir evaluation/interface_protein_rna \
    --prediction_outdir predictions/interface_protein_rna \
    --output_file collated/interface_protein_rna.csv

```

