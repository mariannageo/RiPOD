# Latent subtypes of longitudinal postpartum trajectories for identifying healthy women at risk for affective conditions

Analysis code accompanying the manuscript submitted to *Translational Psychiatry*.

**Contact:** mariannag43@gmail.com
**Publication / DOI:** [TBA]

## Project overview

This project investigates heterogeneity in postpartum symptom trajectories and associated clinical and psychosocial characteristics. It uses a discovery cohort (N=420) and a validation cohort (N=304), followed from childbirth to 12 weeks postpartum.

The workflow prepares longitudinal data, identifies two clusters in the discovery cohort, assigns validation participants to the discovery centroids, and compares trajectories, maternal attachment, risk factors, and participant characteristics. 

**Abbreviations:** HC, healthy controls; AD, adjustment disorder; PPD, postpartum depression; EPDS, Edinburgh Postnatal Depression Scale; MPAS, Maternal Postnatal Attachment Scale; EMA, ecological momentary assessment.

## Repository layout

Keep the notebooks and helper modules together in the repository root:

```text
.
├── README.md
├── preparation_disc_coh.ipynb
├── preparation_val_coh_and_clust.ipynb
├── plotting_kmeans_trajectories.ipynb
├── mean_std_trajectories.ipynb
├── mixed_linear_model_ad_vs_hc.ipynb
├── risk_factor_analysis.ipynb
├── clusters_effect_size_rr_forest_plots.ipynb
├── raw_effect_size.ipynb
├── compare_included_excluded_participants.ipynb
├── demograohics_tables.ipynb
├── repeated_measures_tests.ipynb
├── trajectory_clustering_utils.py
├── cohort_data_utils.py
└── files/
    ├── inputs/          # Source workbooks
    ├── interm_files/    # Prepared features, scaling parameters, cluster labels
    ├── tables/          # Statistical and descriptive tables
    └── plots/           # Exported figures
```


## Environment

The specified notebook environment uses **Python 3.11.5**. Install the packages in from the requirements.txt in that environment.

The `.py` files are imported helpers, not standalone analysis entry points:

| Module | Purpose |
| --- | --- |
| `trajectory_clustering_utils.py` | Availability filtering, interval preparation, normalization, shuffling, and clustering utilities. |
| `cohort_data_utils.py` | Cohort harmonization, long-format conversion, interval averaging, and statistical helpers. Formerly `operators.py`. |

## Data inputs

Place source workbooks in `files/inputs/` with the names expected by the notebooks:

| File | Used for |
| --- | --- |
| `combined_all_cohorts.xlsx` | Master longitudinal cohort data for preparation. |
| `Subskalen Hauptstudie.xlsx` | Discovery EPDS totals and subscales. |
| `Subscales_sample3.xlsx` | Validation EPDS totals and subscales. |
| `Subskalen_allcohorts.xlsx` | Alternative combined-cohort preparation branch. |
| `combined_cohorts - Copy.xlsx` | Risk-factor source data; distinct from the preparation workbook. |
| `variables_across_cohorts.xlsx` | Variable availability and type definitions for risk-factor analyses. |


**Data availability:** [Data is not publiclz available, as it contains personal participants' information] 

## Recommended execution order

Run each notebook from the first cell in a fresh kernel. Where indicated, run separately for discovery (`coh = [1, 2]`) and validation (`coh = [3]`). 

| Order | Notebook | Purpose and dependencies |
| --- | --- | --- |
| 1 | [preparation_disc_coh.ipynb](preparation_disc_coh.ipynb) | Prepare discovery trajectories, export raw and normalized features, save normalization means/SDs, and evaluate k-means silhouette scores with shuffled data. |
| 2 | [preparation_val_coh_and_clust.ipynb](preparation_val_coh_and_clust.ipynb) | Prepare validation trajectories using discovery scaling parameters. Fit discovery k-means with k=2, assign validation participants to those centroids, export both cohorts’ labels, and assess bootstrap stability and reduced-feature sensitivity. Requires step 1. |
| 3 | [plotting_kmeans_trajectories.ipynb](plotting_kmeans_trajectories.ipynb) | Plot raw and clustered trajectories and cluster composition; run time-specific group comparisons; export raw scores joined to cluster labels. Run for both cohorts after steps 1–2, subject to the schema issue below. |
| 4 | [mean_std_trajectories.ipynb](mean_std_trajectories.ipynb) | Export mean ± SD tables by diagnosis and by diagnosis/cluster for both cohorts. Requires raw trajectories and both joined tables from step 3. |
| 5 | [mixed_linear_model_ad_vs_hc.ipynb](mixed_linear_model_ad_vs_hc.ipynb) | Fit group-by-time random-intercept models comparing AD and HC in Cluster 2, separately by cohort. Requires cluster files from step 2 and raw MPAS from step 3. |
| 6 | [repeated_measures_tests.ipynb](repeated_measures_tests.ipynb) | Run within-group Friedman and paired Wilcoxon tests for HC C2 and AD C2. Run separately for both cohorts using step 3 outputs; see the diagnostic-cell issue below. |
| 7 | [raw_effect_size.ipynb](raw_effect_size.ipynb) | Calculate bootstrap Cohen’s d between diagnoses from participant-averaged raw trajectories and create forest/violin plots. Run for both cohorts; needs preparation outputs, not cluster labels. |
| 8 | [risk_factor_analysis.ipynb](risk_factor_analysis.ipynb) | Compare diagnosis groups and HC across clusters; export categorical relative risks and continuous effect sizes. Run for both cohorts using source risk-factor workbooks and step 2 labels. |
| 9 | [clusters_effect_size_rr_forest_plots.ipynb](clusters_effect_size_rr_forest_plots.ipynb) | Calculate clustered trajectory effect sizes and plot both cohorts’ risk-factor summaries. Requires step 3 joined data and step 8 exports for both cohorts. |
| 10 | [compare_included_excluded_participants.ipynb](compare_included_excluded_participants.ipynb) | Compare characteristics of included and excluded participants. Run for each cohort after providing the full-risk-factor workbooks described above. |
| 11 | [demograohics_tables.ipynb](demograohics_tables.ipynb) | Produce descriptive tables for included/excluded participants and HC clusters in both cohorts. Uses the same full-risk-factor workbooks; does not depend on step 10’s results. |



## Analysis conventions

- **Cohorts and labels:** cohorts 1 and 2 form discovery; cohort 3 is validation. Source diagnosis codes ND/AS are relabeled HC/AD for presentation. Cluster exports use IDX=0/1; several plotting notebooks convert these to displayed clusters 1/2. Use the expected convention for each input.
- **Trajectories:** mood/stress baseline is the first available numeric observation. Subsequent observations are averaged within study days 1–21, 22–42, 43–63, and 64–84; the trailing partial interval is discarded. Baseline is excluded from follow-up means. MPAS is available at T1–T4.
- **Scaling and clustering:** normalization uses a pooled mean and sample SD per feature block. Validation uses discovery parameters. Discovery silhouette evaluation includes six blocks, including MPAS; the final clustering notebook uses 25 features from five blocks and excludes MPAS. The supplied selection/evaluation distinction is preserved.


