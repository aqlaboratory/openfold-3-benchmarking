"""Core module for antibody-antigen benchmark analysis."""

import pickle
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as pn
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    name: str
    source: str
    color: str
    order: int


@dataclass
class BenchConfig:
    blacklist: list
    models: list  # List[ModelConfig]

    @property
    def color_map(self) -> dict:
        return {m.name: m.color for m in self.models}

    @property
    def model_order(self) -> list:
        return [m.name for m in self.models]


def load_config(path: str) -> BenchConfig:
    """Read YAML model registry and return a BenchConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    models = []
    for i, (name, info) in enumerate(raw["models"].items()):
        models.append(ModelConfig(
            name=name,
            source=info["source"],
            color=info["color"],
            order=i,
        ))
    return BenchConfig(
        blacklist=raw.get("blacklist_structures", []),
        models=models,
    )

def load_config_from_dict(raw: dict, black_list_structures: list = []) -> BenchConfig:
    """Read YAML model registry and return a BenchConfig."""
    models = []
    for i, (name, info) in enumerate(raw.items()):
        models.append(ModelConfig(
            name=name,
            source=info["source"],
            color=info["color"],
            order=i,
        ))
    return BenchConfig(
        blacklist=black_list_structures,
        models=models,
    )

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

SCHEMA = [
    "model", "pdbid", "interface_cluster", "ref_interface", "seed_number",
    "sample_number", "iptm", "bespoke_iptm", "dockq",
]


def load_all_models(config: BenchConfig) -> pd.DataFrame:
    """Load all models defined in config, return single DataFrame."""
    frames = []
    for mc in config.models:
        df = pd.read_csv(mc.source).assign(model=mc.name)
        frames.append(df[[c for c in SCHEMA if c in df.columns]])

    all_df = pd.concat(frames, ignore_index=True)

    # Set model as ordered categorical
    all_df["model"] = pd.Categorical(
        all_df["model"], categories=config.model_order, ordered=True,
    )

    return all_df


def apply_blacklist(df: pd.DataFrame, structures: list) -> pd.DataFrame:
    """Remove blacklisted structures (case-insensitive)."""
    bl_lower = {s.lower() for s in structures}
    return df[~df["pdbid"].str.lower().isin(bl_lower)].reset_index(drop=True)


def subset_common(
    df: pd.DataFrame,
    level: str = "ref_interface",
    equalize_seeds: bool = True,
    equalize_samples: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Keep only targets present in ALL models, optionally equalizing seeds and samples.

    Steps:
        1. Filter to targets (at 'level') present in every model.
        2. (if equalize_seeds) For each target, subsample each model's seeds down to the
           minimum seed count across models.
        3. (if equalize_samples) For each target, match the per-seed sample count
           distribution across models by ranking seeds by sample count and taking
           element-wise minimums.

    Args:
        df:
            DataFrame with columns: model, pdbid, ref_interface, seed_number,
            sample_number
        level:
            Grouping level for common-target filtering ("ref_interface" or "pdbid")
        equalize_seeds:
            Whether to equalize the number of seeds across models per target
        equalize_samples:
            Whether to equalize per-seed sample counts across models per target
        random_state:
            Seed for reproducible random subsampling
    """
    rng = np.random.RandomState(random_state)

    # Step 1: Keep only targets present in ALL models
    n_models = df["model"].nunique()
    counts = df.groupby(level, observed=True)["model"].nunique()
    common = counts[counts == n_models].index
    df = df[df[level].isin(common)].reset_index(drop=True)

    if not equalize_seeds:
        return df

    # Step 2: Equalize number of seeds per target across models
    group_cols = ["pdbid", "ref_interface"]

    seed_counts = (
        df.groupby(group_cols + ["model"])["seed_number"]
        .nunique()
        .reset_index(name="n_seeds")
    )
    min_seeds = (
        seed_counts
        .groupby(group_cols)["n_seeds"]
        .min()
        .reset_index(name="target_n_seeds")
    )

    def _subsample_seeds(group, target_n_seeds_map):
        key = tuple(group[c].iloc[0] for c in group_cols)
        target_n = target_n_seeds_map[key]
        available_seeds = group["seed_number"].unique()
        if len(available_seeds) <= target_n:
            return group
        selected = rng.choice(available_seeds, size=target_n, replace=False)
        return group[group["seed_number"].isin(selected)]

    target_n_seeds_map = {
        tuple(row[c] for c in group_cols): row["target_n_seeds"]
        for _, row in min_seeds.iterrows()
    }

    df = (
        df.groupby(group_cols + ["model"], group_keys=False)
        .apply(lambda g: _subsample_seeds(g, target_n_seeds_map))
        .reset_index(drop=True)
    )

    if not equalize_samples:
        return df

    # Step 3: Equalize per-seed sample count distributions across models
    def _equalize_per_seed_samples(target_group):
        counts = (
            target_group
            .groupby(["model", "seed_number"])
            .size()
            .reset_index(name="n_samples")
        )

        # Rank seeds within each model by sample count (descending)
        counts["seed_rank"] = (
            counts
            .groupby("model")["n_samples"]
            .rank(method="first", ascending=False)
            .astype(int)
        )

        # Element-wise min across models at each seed rank
        target_profile = (
            counts
            .groupby("seed_rank")["n_samples"]
            .min()
            .reset_index()
            .rename(columns={"n_samples": "target_n"})
        )

        counts = counts.merge(target_profile, on="seed_rank")

        # Subsample each (model, seed) to its target count
        results = []
        for _, row in counts.iterrows():
            seed_data = target_group[
                (target_group["model"] == row["model"])
                & (target_group["seed_number"] == row["seed_number"])
            ]
            target_n = int(row["target_n"])
            if target_n == 0:
                continue
            if len(seed_data) <= target_n:
                results.append(seed_data)
            else:
                results.append(seed_data.sample(n=target_n, random_state=rng))

        if results:
            return pd.concat(results)
        return target_group.iloc[:0]

    df = (
        df.groupby(group_cols, group_keys=False)
        .apply(_equalize_per_seed_samples)
        .reset_index(drop=True)
    )

    return df


def model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model summary stats, including missing structures."""
    all_structures = set(df["pdbid"].unique())

    n_interfaces = (
        df.groupby("model", observed=True)
        .apply(lambda g: g[["pdbid", "ref_interface"]].drop_duplicates().shape[0],
               include_groups=False)
        .rename("n_interfaces")
    )
    missing = (
        df.groupby("model", observed=True)["pdbid"]
        .apply(lambda s: sorted(all_structures - set(s.unique())))
        .rename("missing_structures")
    )
    summary = df.groupby("model", observed=True).agg(
        n_seeds=("seed_number", "nunique"),
        n_structures=("pdbid", "nunique"),
        n_rows=("dockq", "size"),
    ).join(n_interfaces).join(missing).reset_index()
    return summary[["model", "n_seeds", "n_structures", "n_interfaces", "n_rows", "missing_structures"]]


def clip_scaling_to_seeds(df_raw, means, frac=0.8):
    """Filter scaling results to n_seeds <= frac * each model's max seed count."""
    max_seeds = df_raw.groupby("model", observed=True)["seed_number"].nunique()
    limits = (max_seeds * frac).astype(int).reset_index(name="max_n")
    return (means.merge(limits, on="model")
            .query("n_seeds <= max_n")
            .drop(columns="max_n")
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_predictions(
    df: pd.DataFrame,
    rank_by: str = "bespoke_iptm",
    metric: str = "dockq",
    seeds=None,
    include_avg: bool = False,
) -> pd.DataFrame:
    """Compute oracle and ranked assessments (optionally avg).

    If *seeds* is given, filter to those seed numbers first.
    Set *include_avg=True* to add a mean-over-seeds "avg" assessment
    (used for bar plots, not for bootstrap scaling).
    """
    if seeds is not None:
        df = df[df["seed_number"].isin(seeds)].reset_index(drop=True)

    gcols = ["model", "pdbid", "ref_interface"]
    outcols = ["model", "pdbid", "interface_cluster", "ref_interface"]

    # Ranked: pick sample with highest rank_by, report its metric
    ranked_idx = df.groupby(gcols, observed=True)[rank_by].idxmax().tolist()
    ranked = df.iloc[ranked_idx][outcols + [metric]].assign(assessment="ranked")

    # Oracle: best metric per group (merge interface_cluster back)
    oracle = (df.groupby(gcols, observed=True)[metric]
              .max().reset_index().assign(assessment="oracle"))
    ref_map = df[["pdbid", "ref_interface", "interface_cluster"]].drop_duplicates()
    oracle = oracle.merge(ref_map, on=["pdbid", "ref_interface"])

    parts = [oracle, ranked]

    if include_avg:
        avg = (df.groupby(gcols, observed=True)[metric]
               .mean().reset_index().assign(assessment="avg"))
        avg = avg.merge(ref_map, on=["pdbid", "ref_interface"])
        parts.append(avg)

    return pd.concat(parts, ignore_index=True)


def add_success_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean DockQ success columns."""
    return df.assign(
        is_acceptable=lambda x: x["dockq"] >= 0.23,
        is_medium=lambda x: x["dockq"] >= 0.49,
        is_high=lambda x: x["dockq"] >= 0.80,
    )


# ---------------------------------------------------------------------------
# Bootstrap + caching
# ---------------------------------------------------------------------------

_worker_data: dict = {}


def _load_cache(cache_dir: str, key: str):
    path = Path(cache_dir) / f"{key}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(cache_dir: str, key: str, data):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = Path(cache_dir) / f"{key}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _init_worker(df_bytes):
    df = pickle.loads(df_bytes)
    _worker_data["sampler"] = PerInterfaceSampler(df)

    # Pre-compute numpy arrays so the hot loop never touches pandas
    gids = df.groupby(["pdbid", "ref_interface"], observed=True).ngroup().values
    _worker_data["group_ids"] = gids
    _worker_data["n_groups"] = int(gids.max()) + 1
    for col in ("bespoke_iptm", "iptm", "dockq"):
        if col in df.columns:
            _worker_data[f"col_{col}"] = df[col].values

    # Map each (pdbid, ref_interface) group → interface_cluster for hierarchical averaging
    group_ref = (
        df.assign(_gid=gids)
        .drop_duplicates("_gid")
        .sort_values("_gid")
        .reset_index(drop=True)
    )
    ref_cat = pd.Categorical(group_ref["interface_cluster"])
    g2r = ref_cat.codes.copy()
    n_refs = len(ref_cat.categories)
    _worker_data["group_to_ref"] = g2r
    _worker_data["ref_names"] = ref_cat.categories.tolist()
    _worker_data["n_refs"] = n_refs
    _worker_data["groups_per_ref"] = np.bincount(g2r, minlength=n_refs).astype(float)
    _worker_data["model_name"] = str(df["model"].iloc[0])


class PerInterfaceSampler:
    """Fast per-interface seed sampler with pre-computed index arrays.

    For each (model, pdbid, ref_interface), pre-computes the row indices
    belonging to each seed_number.  Sampling draws n_seeds (with replacement)
    from each interface's own seed pool, so every interface always contributes
    regardless of how many seeds it has.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        # (model, pdbid, iface, seed) -> row indices into df
        full_groups = df.groupby(
            ["model", "pdbid", "ref_interface", "seed_number"], observed=True,
        ).indices

        # Re-organise: interface -> list of index arrays (one per seed)
        iface_map: dict[tuple, list[np.ndarray]] = {}
        for (model, pdbid, iface, _seed), idx in full_groups.items():
            iface_map.setdefault((model, pdbid, iface), []).append(idx)

        self.iface_seeds = list(iface_map.values())
        self.rng = np.random.default_rng()

    def sample(self, n_seeds: int) -> pd.DataFrame:
        parts = []
        for seed_arrays in self.iface_seeds:
            chosen = self.rng.integers(0, len(seed_arrays), size=n_seeds)
            parts.append(np.concatenate([seed_arrays[i] for i in chosen]))
        return self.df.iloc[np.concatenate(parts)].reset_index(drop=True)

    def sample_indices(self, n_seeds: int) -> np.ndarray:
        """Return sampled row indices without building a DataFrame."""
        parts = []
        for seed_arrays in self.iface_seeds:
            chosen = self.rng.integers(0, len(seed_arrays), size=n_seeds)
            parts.append(np.concatenate([seed_arrays[i] for i in chosen]))
        return np.concatenate(parts)


def _worker_fn(n_seeds, rank_by, metric, n_seed_samples):
    sampler = _worker_data["sampler"]
    group_ids = _worker_data["group_ids"]
    n_groups = _worker_data["n_groups"]
    rank_all = _worker_data[f"col_{rank_by}"]
    metric_all = _worker_data[f"col_{metric}"]
    g2r = _worker_data["group_to_ref"]
    n_refs = _worker_data["n_refs"]
    gpr = _worker_data["groups_per_ref"]

    # Accumulators per (pdbid, ref_interface) group
    o_dq = np.zeros(n_groups)
    o_acc = np.zeros(n_groups)
    o_med = np.zeros(n_groups)
    o_hi = np.zeros(n_groups)
    r_dq = np.zeros(n_groups)
    r_acc = np.zeros(n_groups)
    r_med = np.zeros(n_groups)
    r_hi = np.zeros(n_groups)

    for _ in range(n_seed_samples):
        idx = sampler.sample_indices(n_seeds)
        gids = group_ids[idx]
        ranks = rank_all[idx]
        mvals = metric_all[idx]

        # Oracle: max metric per group
        oracle = np.full(n_groups, -np.inf)
        np.maximum.at(oracle, gids, mvals)
        o_dq += oracle
        o_acc += (oracle >= 0.23)
        o_med += (oracle >= 0.49)
        o_hi += (oracle >= 0.80)

        # Ranked: metric at the row with max rank_by, per group
        order = np.lexsort((-ranks, gids))
        sg = gids[order]
        sm = mvals[order]
        first = np.empty(len(sg), dtype=bool)
        first[0] = True
        first[1:] = sg[1:] != sg[:-1]
        ranked = np.full(n_groups, -np.inf)
        ranked[sg[first]] = sm[first]
        r_dq += ranked
        r_acc += (ranked >= 0.23)
        r_med += (ranked >= 0.49)
        r_hi += (ranked >= 0.80)

    # Mean across bootstrap samples
    inv_n = 1.0 / n_seed_samples
    o_dq *= inv_n; o_acc *= inv_n; o_med *= inv_n; o_hi *= inv_n
    r_dq *= inv_n; r_acc *= inv_n; r_med *= inv_n; r_hi *= inv_n

    # Aggregate (pdbid, ref_interface) groups into interface_cluster means
    def _ref_mean(vals):
        out = np.zeros(n_refs)
        np.add.at(out, g2r, vals)
        return out / gpr

    # Build output DataFrame (only pandas usage — once per task, not per sample)
    model_name = _worker_data["model_name"]
    ref_names = _worker_data["ref_names"]
    ref_arr = np.array(ref_names)
    parts = []
    for assess, dq, acc, med, hi in [
        ("oracle", o_dq, o_acc, o_med, o_hi),
        ("ranked", r_dq, r_acc, r_med, r_hi),
    ]:
        vals = np.column_stack([_ref_mean(dq), _ref_mean(acc),
                                _ref_mean(med), _ref_mean(hi)])
        for j, mn in enumerate(["dockq", "is_acceptable", "is_medium", "is_high"]):
            parts.append(pd.DataFrame({
                "model": model_name,
                "interface_cluster": ref_arr,
                "assessment": assess,
                "metric": mn,
                "value": vals[:, j],
                "n_seeds": n_seeds,
            }))
    return pd.concat(parts, ignore_index=True)


import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

# 1. Define the worker function at the top level of your script
def _bootstrap_chunk(args):
    """Worker function to process a chunk of bootstrap iterations."""
    df, n_iterations = args
    chunk_results = []
    
    for _ in range(n_iterations):
        # Sample
        bs = df.groupby(["model", "assessment", "metric", "n_seeds"], observed=True).sample(
            frac=1.0, replace=True, random_state=None
        )
        # Get mean
        bs_mean = bs.groupby(["model", "assessment", "metric", "n_seeds"], observed=True)["value"].mean().reset_index(name="mean_value")
        chunk_results.append(bs_mean)
        
    # Concatenate at the worker level to minimize data sent back to the main process
    return pd.concat(chunk_results, ignore_index=True)

# 2. Main execution logic
def compute_confidence_intervals(means_df, total_iterations=10000, num_cores = 128):
    # Calculate how many iterations each core should handle
    iters_per_core = total_iterations // num_cores
    remainder = total_iterations % num_cores
    
    # Create arguments for each core (passing the dataframe and the number of iterations)
    tasks = [
        (means_df, iters_per_core + (1 if i < remainder else 0)) 
        for i in range(num_cores)
    ]
    
    print(f"Computing 95% confidence intervals across {num_cores} cores...")
    bs_df_all = []
    
    # Initialize the Pool
    with mp.Pool(processes=num_cores) as pool:
        # We use imap_unordered so tqdm updates as soon as a chunk finishes
        for chunk_result in tqdm(pool.imap_unordered(_bootstrap_chunk, tasks), total=num_cores, desc="Processing chunks"):
            bs_df_all.append(chunk_result)
            
    # Combine all chunks from all cores
    bs_df_all = pd.concat(bs_df_all, ignore_index=True)
    
    # Calculate final quantiles
    ci_95 = bs_df_all.groupby(["model", "assessment", "metric", "n_seeds"], observed=True)["mean_value"].quantile([0.025, 0.975]).unstack().reset_index()
    ci_95.columns = ["model", "assessment", "metric", "n_seeds", "ci_low", "ci_high"]
    
    return ci_95

def bootstrap_scaling(
    df: pd.DataFrame,
    n_seed_samples: int = 1000,
    rank_by: str = "bespoke_iptm",
    metric: str = "dockq",
    n_jobs: int = 128,
    cache_dir: str = "cache",
):
    """Run bootstrap scaling analysis with caching.

    Returns means_df with columns: model, assessment, metric, n_seeds, mean_value
    """
    all_means = []

    for model_name in df["model"].cat.categories:
        model_df = df[df["model"] == model_name].reset_index(drop=True)
        if model_df.empty:
            continue

        key = f"{model_name}_n{len(model_df)}_{rank_by}_s{n_seed_samples}"
        print(key)
        cached = _load_cache(cache_dir, key)

        if cached is not None:
            print(f"  [{model_name}] loaded from cache")
            all_means.append(cached)
            continue

        print(f"  [{model_name}] cache miss, running bootstrap...")
        max_seeds = model_df["seed_number"].max()
        seeds = list(range(1, max_seeds))

        df_bytes = pickle.dumps(model_df)
        means_list = []

        with Pool(min(n_jobs, len(seeds)),
                  initializer=_init_worker, initargs=(df_bytes,)) as pool:
            pfunc = partial(
                _worker_fn,
                rank_by=rank_by,
                metric=metric,
                n_seed_samples=n_seed_samples,
            )
            for avg in tqdm(
                pool.imap_unordered(pfunc, seeds),
                total=len(seeds),
                desc=model_name,
            ):
                means_list.append(avg)

        means_df = pd.concat(means_list, ignore_index=True)

        
    
        ci_95 = compute_confidence_intervals(means_df, total_iterations=10000)
        # Aggregate means across clusters -> one value per model/assessment/metric/n_seeds
        means_agg = (
            means_df
            .groupby(["model", "assessment", "metric", "n_seeds"], observed=True)
            ["value"].mean().reset_index(name="mean_value")
        ).merge(ci_95)

        _save_cache(cache_dir, key, means_agg)
        print(f"  [{model_name}] cached in {cache_dir}/")

        all_means.append(means_agg)

    return pd.concat(all_means, ignore_index=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "is_acceptable": "Acceptable \n(DockQ >= 0.23)",
    "is_medium": "Medium \n(DockQ >= 0.49)",
    "is_high": "High \n(DockQ >= 0.80)",
    "dockq": "DockQ",
}



def plot_dockq_bars(scored, config, assessments=("oracle", "ranked", "avg")):
    """Grouped bar chart of mean DockQ scores with SEM error bars."""
    # Aggregate ref_interfaces within each interface_cluster, then flat mean
    iface = (
        scored[scored["assessment"].isin(assessments)]
        .groupby(["model", "interface_cluster", "assessment"], observed=True)["dockq"]
        .mean().reset_index()
    )
    pdf = (
        iface
        .groupby(["model", "assessment"], observed=True)["dockq"]
        .agg(["mean", "sem"]).reset_index()
        .assign(model=lambda x: pd.Categorical(
            x["model"], categories=config.model_order, ordered=True))
    )
    return (
        pn.ggplot(pdf)
        + pn.geom_bar(
            pn.aes(x="model", y="mean", fill="assessment"),
            stat="identity", position=pn.position_dodge())
        + pn.geom_errorbar(
            pn.aes(x="model", y="mean", ymin="mean-sem", ymax="mean+sem",
                    group="assessment"),
            position=pn.position_dodge(width=0.9), width=0.25)
        + pn.theme_minimal()
        + pn.theme(axis_text_x=pn.element_text(angle=45, hjust=1))
        + pn.ylab("Mean Dockq")
    )


def plot_success_bars(scored, config, assessments=("oracle", "ranked", "avg")):
    """Faceted bar chart of success rates by threshold."""
    df = scored[scored["assessment"].isin(assessments)]

    melted = df.melt(
        id_vars=["model", "pdbid", "interface_cluster", "ref_interface", "dockq", "assessment"],
        value_vars=["is_acceptable", "is_medium", "is_high"],
        var_name="threshold",
        value_name="frac",
    )
    # Aggregate ref_interfaces within each interface_cluster, then flat mean
    iface = (
        melted
        .groupby(["model", "interface_cluster", "assessment", "threshold"],
                 observed=True)["frac"]
        .mean().reset_index()
    )
    pdf = (
        iface
        .groupby(["model", "assessment", "threshold"], observed=True)["frac"]
        .agg(["mean", "sem"]).reset_index()
        .assign(
            model=lambda x: pd.Categorical(
                x["model"], categories=config.model_order, ordered=True),
            threshold=lambda x: pd.Categorical(
                x["threshold"],
                categories=["is_acceptable", "is_medium", "is_high"],
                ordered=True,
            ),
        )
        .dropna()
    )
    return (
        pn.ggplot(pdf)
        + pn.geom_bar(
            pn.aes(x="model", y="mean", fill="assessment"),
            stat="identity", position=pn.position_dodge())
        + pn.geom_errorbar(
            pn.aes(x="model", y="mean", ymin="mean-sem", ymax="mean+sem",
                    group="assessment"),
            position=pn.position_dodge(width=0.9), width=0.25)
        + pn.facet_wrap("~threshold")
        + pn.theme_bw()
        + pn.theme(
            axis_text_x=pn.element_text(angle=45, hjust=1),
            figure_size=(12, 4))
        + pn.ylab("Success Rate")
    )


def plot_scaling(means, config, metric="dockq"):
    """Line+point scaling plot, faceted by assessment."""
    pdf = means.query("metric == @metric").assign(
        model=lambda x: pd.Categorical(
            x["model"], categories=config.model_order, ordered=True),
    )

    return (
        pn.ggplot(pdf)
        + pn.geom_line(pn.aes(x="n_seeds", y="mean_value", color="model"))
        + pn.geom_point(pn.aes(x="n_seeds", y="mean_value", color="model"), size=1)
        + pn.scale_color_manual(values=config.color_map)
        + pn.facet_wrap("assessment", nrow=1, scales="free_y")
        + pn.theme_bw()
        + pn.ylab("Mean DockQ")
        + pn.xlab("# of seeds")
        + pn.ggtitle("Antibody-Antigen Scaling Analysis")
        + pn.theme(figure_size=(12, 6))
    )


def plot_difficulty(means, config, assessment="oracle"):
    """Line+point plot faceted by success threshold."""
    pdf = (
        means.query("metric != 'dockq' and assessment == @assessment")
        .assign(
            model=lambda x: pd.Categorical(
                x["model"], categories=config.model_order, ordered=True),
            metric=lambda x: pd.Categorical(
                x["metric"].map(METRIC_LABELS),
                categories=[
                    METRIC_LABELS["is_acceptable"],
                    METRIC_LABELS["is_medium"],
                    METRIC_LABELS["is_high"],
                ],
                ordered=True,
            ),
        )
    )

    return (
        pn.ggplot(pdf)
        + pn.geom_line(pn.aes(x="n_seeds", y="mean_value", color="model"))
        + pn.geom_point(pn.aes(x="n_seeds", y="mean_value", color="model"), size=1)
        + pn.scale_color_manual(values=config.color_map)
        + pn.facet_wrap("metric", nrow=1, scales="free_y")
        + pn.theme_bw()
        + pn.ylab("Success Rate")
        + pn.xlab("# of seeds")
        + pn.ggtitle(
            f"Antibody-Antigen Scaling by difficulty levels ({assessment})")
        + pn.theme(figure_size=(12, 6))
    )


def save_plot(plot, path, width=10, height=6, dpi=300):
    """Save plot to file, creating directory if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plot.save(path, width=width, height=height, dpi=dpi, verbose=False)
