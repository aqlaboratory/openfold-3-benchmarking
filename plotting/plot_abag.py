from abag_scaling import load_config_from_dict, load_all_models, apply_blacklist, subset_common, model_summary
from abag_scaling import score_predictions, add_success_rates
from abag_scaling import bootstrap_scaling, clip_scaling_to_seeds
from abag_scaling import plot_dockq_bars, plot_success_bars, plot_scaling, plot_difficulty, save_plot
import pandas as pd 
import plotnine as pn 

## define a model, source and color here for it be included in analysis 
## Non OF3 models are in `../existing_results_other_models/abag` 
## and OF3 models are in `../pipeline_output/collated_results`
ALL_MODELS={
    "AlphaFold3":{
        "source":"existing_results_other_models/abag/AF3_ABAG_matrix_1000s_scored.csv",
        "color": "#EA157A"
    },
    "OF3p2":{
        "source":"collated_results/ne1ce3lb_235-155000/AMDrocmBLAS/gdm_abag/inference/w_templates/collated.csv",
        "color":"#007EEA"
    },
    "Boltz-1":{
        "source": "existing_results_other_models/abag/Boltz-1.csv",        
        "color": "#754AB2"
    },
    "Chai-1":{
        "source" :"existing_results_other_models/abag/Chai-1.csv",
        "color" :  "#FEB80A"
    },
    "Protenix-v1":{
        "source":"existing_results_other_models/abag/protenix-v1_gdm_abag_1000s_assembly_expand.csv",
        "color":"#7FD13B"
    }
}
## structures to NOT consider - these will be removed from the analysis 
BLACKLIST=["7XQ8", "7RU6", "7T6X"]
config = load_config_from_dict(ALL_MODELS, BLACKLIST)
df = load_all_models(config)
model_summary(df)

# restrict to 1K seeds and apply blacklist 
seed_lim = 1000
df = df.query("seed_number < @seed_lim")
df = apply_blacklist(df, config.blacklist)
model_summary(df)

## scaling analysis:
## this will run the seed subsampling scaling analysis.
## By default, it will run on all available seeds for a model.
## If you want to run on a subset, first subset the input, then 
## run the scaling analysis
means = bootstrap_scaling(df, n_jobs=128, cache_dir="cache")

# plot mean dockq
(
    plot_scaling(means, config) + 
    #pn.geom_ribbon(pn.aes(x="n_seeds", ymin="ci_low", ymax="ci_high", fill="model"), alpha=0.2, color=None) +
    pn.scale_x_log10()  + 
    pn.theme(legend_text=pn.element_text(ha='left', ma = "left")) 
)

## plot success rate scaling
ranked = plot_difficulty(means, config, assessment="ranked") + pn.scale_x_log10()
oracle = plot_difficulty(means, config, assessment="oracle") + pn.scale_x_log10()
plot = ranked/oracle 
plot + pn.theme(figure_size = (15,10))
