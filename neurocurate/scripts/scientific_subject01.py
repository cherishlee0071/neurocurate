"""Phase 1.3 scientific analysis; exclusively pickle-free numeric exports."""
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import yaml
from PIL import Image
from neurocurate.eeg import EEGTrials, EEGConfig, extract_eeg_features
from neurocurate.metadata import align_features, validate_manifest
from neurocurate.realdata import validate_test_order
from neurocurate.rdm import compute_rdm, validate_rdm, nearest_neighbors
from neurocurate.scoring import jaccard_disagreement
from neurocurate.vision import extract_vision_features, save_feature_cache, load_feature_cache
from neurocurate.lowdisk import sha256
from neurocurate.visualization import plot_rdms


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def interval(values):
    values = np.asarray(values)
    return dict(median=float(np.median(values)), mean=float(np.mean(values)),
                random_split_95_percentile_interval=np.quantile(values, [.025, .975]).tolist())


def rank_vector(matrix, triangle):
    ranked = rankdata(matrix[triangle])
    centered = ranked - ranked.mean()
    return centered / np.linalg.norm(centered), ranked


def main():
    root = Path("results/real_phase0/sub01")
    exports = root / "pickle_free"
    assert (exports / "VERIFIED").is_file()
    export_manifest = json.loads((exports / "export_manifest.json").read_text())
    for name, digest in export_manifest["exports"].items():
        assert sha256(exports/name) == digest
    eeg = np.load(exports/"eeg.npy", allow_pickle=False, mmap_mode="r")
    times = np.load(exports/"times.npy", allow_pickle=False)
    channels = np.load(exports/"channels.npy", allow_pickle=False)
    metadata = json.loads((exports/"metadata.json").read_text())
    image_root = Path("data/local/real_phase0/stimuli")
    alignment = validate_test_order(metadata, image_root)
    rows = [dict(image_id=f"test_images/{c}/{f}", image_path=str((image_root/"test_images"/c/f).resolve()),
                 split="sanity", eeg_index=i) for i,(c,f) in enumerate(zip(metadata["test_img_concepts"], metadata["test_img_files"]))]
    table = validate_manifest(pd.DataFrame(rows))
    ids = table.image_id.to_numpy(dtype=str)
    paths = table.image_path.tolist()
    hashes = [sha256(p) for p in paths]
    assert len(set(hashes)) == 200
    alignment.update(content_alias_count=0, distinct_stimulus_hashes=200,
        source_code_provenance=json.loads(Path("outputs/real-phase0-source/source_provenance.json").read_text()))
    for item in alignment["source_code_provenance"]:
        assert sha256(item["path"]) == item["sha256"]
    dt = np.diff(times)
    assert np.allclose(dt, .01) and np.isfinite(eeg).all() and eeg.shape == (200,80,17,100)
    shape = dict(shape=list(eeg.shape), dtype=str(eeg.dtype), axes=["image","repetition","channel","time"],
        image_count=200, repetition_count=80, trial_count=16000, channel_count=17,
        channel_names=channels.tolist(), sampling_rate_hz=float(1/np.median(dt)),
        times_seconds=times.tolist(), missing_values=int(np.isnan(eeg).sum()),
        nonfinite_values=int((~np.isfinite(eeg)).sum()),
        image_axis_semantics="numeric test condition i+1 at image-axis index i, verified metadata/protocol convention",
        session_labels_available=False, pickle_free_export_verified=True)
    for name,obj in [("eeg_shape_report.json",shape),("image_alignment_report.json",alignment)]:
        if (root/name).exists() and not (root/(name+".phase12_backup")).exists():
            shutil.copyfile(root/name,root/(name+".phase12_backup"))
        write(root/name,obj)
    config = EEGConfig(time_window=(.1,.5), baseline=None, min_repetitions=40)
    experiment = dict(subject="sub-01", original_split="test", repository_split="sanity",
        preprocessing=asdict(config), brain_metric="correlation", vision_metric="cosine",
        split_seed=20261011, permutation_seed=1337, repetition_splits=30,
        permutations=1000, k_values=[5,10,20], primary_visualization_k=10,
        uncertainty="empirical 2.5–97.5% random-split variability interval, not a population confidence interval",
        conditional_on_shared_preprocessing=True, session_independent=False)
    (root/"scientific_config.yaml").write_text(yaml.safe_dump(experiment))
    def features(indices):
        selected = np.asarray(eeg[:,indices,:,:]).reshape(-1,17,100)
        trials = EEGTrials(selected, np.repeat(ids,len(indices)), times, channels, "sanity")
        feature_ids, values, _ = extract_eeg_features(trials,config)
        return align_features(ids,feature_ids,values)
    full_features = features(np.arange(80))
    brain = validate_rdm(compute_rdm(full_features))
    np.save(root/"eeg_features.npy",full_features,allow_pickle=False)
    np.save(root/"brain_rdm.npy",brain,allow_pickle=False)
    np.save(root/"image_ids.npy",ids,allow_pickle=False)
    table.to_csv(root/"image_mapping.csv",index=False)
    models = {}
    for kind, revision in [("dinov2","f9e44c814b77203eaa57a6bdbbd535f21ede1415"),
                           ("clip","3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")]:
        cache = root/f"{kind}_features.npz"
        if not cache.exists():
            emb, provenance = extract_vision_features(ids,paths,kind=kind,revision=revision,
                cache_dir="/workspace/setup/neurocurate/model_cache",local_files_only=True,batch_size=8)
            save_feature_cache(cache,ids,emb,split="sanity",model_name=kind,provenance=provenance)
        cached_ids, emb, provenance = load_feature_cache(cache,split="sanity",model_name=kind)
        assert np.array_equal(cached_ids,ids) and provenance["resolved_revision"] == revision
        assert [provenance["image_sha256"][i] for i in ids] == hashes
        models[kind] = validate_rdm(compute_rdm(emb,"cosine"))
        np.save(root/f"{kind}_rdm.npy",models[kind],allow_pickle=False)
        print(kind,emb.shape,"real features/RDM complete",flush=True)
        gc.collect()
    ks = experiment["k_values"]
    model_neighbors = {(kind,k):nearest_neighbors(rdm,k,ids) for kind,rdm in models.items() for k in ks}
    def disagreements(rdm):
        result = {}
        for k in ks:
            neighbors = nearest_neighbors(rdm,k,ids)
            for kind in models:
                result[(kind,k)] = jaccard_disagreement(neighbors,model_neighbors[(kind,k)])
            result[("mean_models",k)] = np.mean([result[(kind,k)] for kind in models],axis=0)
        return result
    full_scores = disagreements(brain)
    frame = pd.DataFrame({"image_id":ids, "eeg_index":np.arange(200)})
    for (kind,k),score in full_scores.items():
        frame[f"disagreement_{kind}_k{k}"] = score
        frame[f"rank_{kind}_k{k}"] = rankdata(-score,method="average")
    frame.to_csv(root/"disagreement_scores.csv",index=False)
    rng = np.random.default_rng(experiment["split_seed"])
    permutation_rng = np.random.default_rng(experiment["permutation_seed"])
    permutations = [permutation_rng.permutation(200) for _ in range(experiment["permutations"])]
    triangle = np.triu_indices(200,1)
    split_records, reliability, nulls = [], [], []
    stability = {key:dict(spearman=[],top20_overlap_fraction=[],top20_jaccard=[],permutation_null=[],top20_permutation_null=[]) for key in full_scores}
    for split in range(experiment["repetition_splits"]):
        order = rng.permutation(80)
        halves = [np.sort(order[:40]), np.sort(order[40:])]
        assert not set(halves[0]) & set(halves[1])
        rdms = [validate_rdm(compute_rdm(features(indices))) for indices in halves]
        first_rank, _ = rank_vector(rdms[0],triangle)
        second_rank, original_ranks = rank_vector(rdms[1],triangle)
        reliability.append(float(first_rank @ second_rank))
        rank_matrix = np.zeros((200,200));rank_matrix[triangle]=original_ranks;rank_matrix+=rank_matrix.T
        center = original_ranks.mean();norm=np.linalg.norm(original_ranks-center)
        null = [float(first_rank @ ((rank_matrix[np.ix_(p,p)][triangle]-center)/norm)) for p in permutations]
        nulls.append(null)
        scores = [disagreements(rdm) for rdm in rdms]
        for key,record in stability.items():
            a,b = scores[0][key],scores[1][key]
            record["spearman"].append(float(spearmanr(a,b).statistic))
            top_a=set(np.lexsort((ids,-a))[:20]);top_b=set(np.lexsort((ids,-b))[:20])
            record["top20_overlap_fraction"].append(len(top_a & top_b)/20)
            record["top20_jaccard"].append(len(top_a & top_b)/len(top_a | top_b))
            # Relabel scores, THEN apply the same deterministic tie rule. This
            # captures shared image-ID tie-breaking bias absent from 20/200.
            record["top20_permutation_null"].append([
                len(top_a & set(np.lexsort((ids,-b[p]))[:20]))/20 for p in permutations])
            ra=rankdata(a);rb=rankdata(b);ra-=ra.mean();rb-=rb.mean()
            ra/=np.linalg.norm(ra);rb/=np.linalg.norm(rb)
            record["permutation_null"].append([float(ra @ rb[p]) for p in permutations])
        split_records.append(dict(split=split,half_a=halves[0].tolist(),half_b=halves[1].tolist(),spearman=reliability[-1]))
        print("split",split+1,"RDM Spearman",round(reliability[-1],4),flush=True)
    median_null=np.median(np.asarray(nulls),axis=0)
    observed=float(np.median(reliability))
    reliability_report = dict(conditional_on_shared_preprocessing=True,session_independent=False,
        repetition_splits=30,repetitions_per_half=40,statistics=interval(reliability),splits=split_records,
        permutation=dict(unit="image identity: simultaneously permute rows and columns of half B",
            count=1000,statistic="median Spearman across 30 splits; same image permutation across splits",
            observed_median=observed,null_median=float(np.median(median_null)),
            null_95_percentile_interval=np.quantile(median_null,[.025,.975]).tolist(),
            one_sided_monte_carlo_p=float((1+np.sum(median_null>=observed))/1001)),
        uncertainty=experiment["uncertainty"],
        limitation="Disjoint repetition halves share supplied MVNN and possibly sessions; random splits overlap across reruns.")
    write(root/"brain_reliability.json",reliability_report)
    stability_report = dict(conditional_on_shared_preprocessing=True,session_independent=False,
        top_n=20,tie_policy="average ranks for Spearman; stable image-ID ordering for tied top20",
        uniform_random_top20_overlap_reference=.1,
        uniform_reference_is_not_tie_aware=True,results={})
    for (kind,k),record in stability.items():
        rank_null=np.median(np.asarray(record.pop("permutation_null")),axis=0)
        top_null=np.median(np.asarray(record.pop("top20_permutation_null")),axis=0)
        observed_rank=float(np.median(record["spearman"]))
        scores=full_scores[(kind,k)];boundary=np.sort(scores)[-20]
        stability_report["results"][f"{kind}_k{k}"]=dict(
            spearman=interval(record["spearman"]),top20_overlap_fraction=interval(record["top20_overlap_fraction"]),
            top20_jaccard=interval(record["top20_jaccard"]),raw_splits=record,
            full_score_unique_values=len(np.unique(scores)),full_top20_boundary_tie_size=int(np.sum(scores==boundary)),
            permutation_null_median=float(np.median(rank_null)),
            permutation_null_95_interval=np.quantile(rank_null,[.025,.975]).tolist(),
            tie_aware_top20_null_median=float(np.median(top_null)),
            tie_aware_top20_null_95_interval=np.quantile(top_null,[.025,.975]).tolist(),
            tie_aware_top20_one_sided_p=float((1+np.sum(top_null>=np.median(record["top20_overlap_fraction"])))/1001),
            exploratory_one_sided_p=float((1+np.sum(rank_null>=observed_rank))/1001))
    write(root/"disagreement_stability.json",stability_report)
    np.savez(root/"permutation_nulls.npz",rdm_median_null=median_null,allow_pickle=False)
    plot_rdms({"Subject-01 Brain (100–500 ms)":brain},ids,root/"brain_rdm.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top=np.lexsort((ids,-full_scores[("mean_models",10)]))[:20]
    neighbors={"Brain":nearest_neighbors(brain,2,ids),**{kind:nearest_neighbors(rdm,2,ids) for kind,rdm in models.items()}}
    fig,axes=plt.subplots(20,7,figsize=(15,38))
    for row,index in enumerate(top):
        shown=[index]+[j for name in ["Brain","dinov2","clip"] for j in neighbors[name][index]]
        labels=["Anchor","Brain NN1","Brain NN2","DINO NN1","DINO NN2","CLIP NN1","CLIP NN2"]
        for col,(j,label) in enumerate(zip(shown,labels)):
            with Image.open(paths[j]) as im:axes[row,col].imshow(im.convert("RGB"))
            axes[row,col].axis("off")
            title=f"{label}: condition {j+1}" if col else f"#{row+1} cond {j+1}\nD={full_scores[('mean_models',10)][index]:.3f}"
            axes[row,col].set_title(title,fontsize=8)
    boundary_ties=stability_report["results"]["mean_models_k10"]["full_top20_boundary_tie_size"]
    fig.suptitle(f"Subject-01: Top-20 mean DINOv2/CLIP Jaccard disagreement, K=10\n{boundary_ties}-way cutoff tie; image-ID tie-break. Exploratory, shared preprocessing.",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.985));fig.savefig(root/"top20_disagreement.png",dpi=150);plt.close(fig)
    # Independent, pickle-free saved-output verification.
    assert np.array_equal(np.load(root/"image_ids.npy",allow_pickle=False),ids)
    assert np.allclose(np.load(root/"brain_rdm.npy",allow_pickle=False),compute_rdm(np.load(root/"eeg_features.npy",allow_pickle=False)))
    for kind in models:
        _,saved,_=load_feature_cache(root/f"{kind}_features.npz",split="sanity",model_name=kind)
        assert np.allclose(np.load(root/f"{kind}_rdm.npy",allow_pickle=False),compute_rdm(saved,"cosine"))
    primary=stability_report["results"]["mean_models_k10"]
    summary=f'''# Subject-01 first real EEG scientific result

1. **Valid Brain RDM: yes.** Real float64 EEG (200,80,17,100), 100 Hz, 17 posterior channels, 80 repetitions/image. Features (200,680), correlation RDM (200,200). Finite, symmetric, zero-diagonal and saved-order/recomputation checks pass. All 200 numeric condition mappings and distinct stimulus files validate.
2. **Repetition-split reliability:** median Spearman **{observed:.4f}**; random-split 95% variability interval **{reliability_report['statistics']['random_split_95_percentile_interval']}**, 30 disjoint 40/40 splits. This interval is not a population confidence interval.
3. **Permutation control:** median null **{reliability_report['permutation']['null_median']:.4f}**, null 95% interval **{reliability_report['permutation']['null_95_percentile_interval']}**; one-sided Monte Carlo **p={reliability_report['permutation']['one_sided_monte_carlo_p']:.6f}**, 1,000 image-label permutations preserving matrix symmetry and pair dependencies.
4. **Disagreement stability: moderate score-ranking repeatability, NOT a definitive stable shortlist.** Primary mean-model K=10 ranking Spearman median **{primary['spearman']['median']:.4f}**, variability interval **{primary['spearman']['random_split_95_percentile_interval']}**. Top-20 overlap fraction median **{primary['top20_overlap_fraction']['median']:.3f}**; tie-aware permutation null median **{primary['tie_aware_top20_null_median']:.3f}**, **p={primary['tie_aware_top20_one_sided_p']:.4f}**. Full top-20 boundary tie size **{primary['full_top20_boundary_tie_size']}**; the visualized 20 are one deterministic selection from this tied group. Uniform-random overlap 0.10 is NOT the appropriate tie-aware control. K=5/10/20 and individual-model results are in disagreement_stability.json. Rankings require cross-subject replication.
5. **Exploratory scope:** ALL reliability, model comparison and ranking results are conditional on supplied whitening estimated using both training/test partitions and all repetitions. Halves have disjoint repetitions, not independently fitted preprocessing or independent sessions. Repeated random splits overlap. No cross-subject consensus, training improvement, label quality or causal benefit is established.
6. **Proceed to subject-02: GO for a narrowly scoped replication/reliability experiment**, not training or OOD. Replicate identical settings and image identities; assess cross-subject agreement before trusting selection rankings. No subject-02 data was downloaded here.

For this Phase 1.3 run, the two source NPY files were SHA256-verified and deserialized ONLY in an unprivileged, network-disabled, descriptor-only seccomp process with 4 GiB address-space, 120-second CPU and 512 MiB/file limits. All downstream arrays load with allow_pickle=False. Source archives and verified inputs are preserved. Frozen pinned DINOv2/CLIP cover all 200 real images; no weights or new subjects downloaded.
'''
    (root/"scientific_summary.md").write_text(summary)
    names=["eeg_shape_report.json","image_alignment_report.json","eeg_features.npy","brain_rdm.npy","brain_reliability.json","disagreement_scores.csv","disagreement_stability.json","brain_rdm.png","top20_disagreement.png","scientific_summary.md","scientific_config.yaml","image_ids.npy","image_mapping.csv","dinov2_rdm.npy","clip_rdm.npy"]
    write(root/"scientific_provenance.json",dict(experiment=experiment,isolated_export_manifest=export_manifest,
        input_image_hashes=dict(zip(ids,hashes)),output_sha256={name:sha256(root/name) for name in names},
        independently_verified_saved_numeric_outputs=True))
    print(summary,flush=True)


if __name__ == "__main__":
    main()
