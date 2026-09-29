"""Trajectory preparation, normalization, and clustering helpers."""

import pandas as pd
import os
from pathlib import Path
import cohort_data_utils as ops
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import mne.stats
import scipy
from scipy.cluster.hierarchy import cophenet
from scipy import stats
import random
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

DATA_DIR = Path(__file__).resolve().parent / "files"
INPUT_DATA_PATH = Path(os.environ.get("INPUT_DATA_PATH", DATA_DIR / "inputs"))
SAVE_INTERM_DATA_PATH = Path(os.environ.get("SAVE_INTERM_DATA_PATH", DATA_DIR / "interm_files"))
SAVE_DATA_PATH = Path(os.environ.get("SAVE_DATA_PATH", DATA_DIR / "plots"))

def import_data(which_cohort: str):

    """Load all, discovery (one_two), or validation (three) cohort rows from Excel."""
    common_variables = pd.read_excel(INPUT_DATA_PATH / 'variables_across_cohorts.xlsx')
    combined_cohorts = pd.read_excel(SAVE_INTERM_DATA_PATH / 'combined_all_cohorts.xlsx')

    if which_cohort == 'all':
        cohort = combined_cohorts
    elif which_cohort == 'one_two':
        cohort = combined_cohorts[combined_cohorts['Cohort'].isin([1,2])]
    elif which_cohort == 'three':
        cohort = combined_cohorts[combined_cohorts['Cohort'].isin([3])]

    return cohort

def assert_variables(dataframe):

    """Retain rows with all availability flags true and nonmissing ID/CODE."""
    c1 = dataframe['Mood Bool'] == True
    c2 = dataframe['Stress Bool'] == True
    c3 = dataframe['Dem Bool'] == True
    c4 = dataframe['ID'].notna()
    c5 = dataframe['CODE'].notna()

    dataframe_complete = dataframe[c1 & c2 & c3 & c4 & c5]

    return dataframe_complete

def mpas_epds_data_for_clustering(dataframe, list_of_parameters):

    """Select the requested columns and remove rows with missing values."""
    dataframe  = dataframe[list_of_parameters]
    wide = dataframe.dropna()

    return  wide

def select_mood_stress_values(choose_attribute: str, dataframe, interval_size: int, start_day:int, finish_day:int):
    """Select daily scores within inclusive day bounds and nonmissing ID/CODE; interval_size is unused."""
    columns = [f'{choose_attribute} Day {day}' for day in range(start_day, finish_day+1)]

    wide = dataframe[columns]

    wide['ID'] = dataframe['ID']
    wide['CODE'] = dataframe['CODE']

    df_cleaned = wide.dropna(subset=['ID', 'CODE'])

    return df_cleaned

def clean_mood_stress(choose_attribute: str, dataframe, interval_size: int, start_day:int, finish_day:int):                                             

    """Average daily scores into intervals and return complete long and wide trajectories."""
    columns = [f'{choose_attribute} Day {day}' for day in range(start_day, finish_day+1)]

    wide = dataframe[columns]

    wide['ID'] = dataframe['ID']
    wide['CODE'] = dataframe['CODE']

    df_cleaned = wide.dropna(subset=['ID', 'CODE'])
    df = ops.make_long(df_cleaned, f'{choose_attribute}')

    new = ops.convert_to_weekly(df, interval_size, start_day, finish_day, choose_attribute)   
                                                 
    new_unique = new.drop(['Day', f'{choose_attribute} Level'], axis=1).drop_duplicates(subset=['ID', 'T'])

    df_wide = new_unique.pivot(index=['ID', 'CODE'], columns='T', values=f'Mean_{choose_attribute}')

    df_wide.reset_index(inplace=True)
    df_wide.columns = [f'{choose_attribute} T' + str(col) if any(char.isdigit() for char in str(col)) else col for col in df_wide.columns]
    df_wide_cleaned = df_wide.dropna()

    mask = df_wide_cleaned['ID'].tolist()
    mean_filtered = new_unique[new_unique['ID'].isin(mask)]
   
    assert not mean_filtered.isnull().values.any()
    assert not df_wide_cleaned.isnull().values.any()

    return mean_filtered, df_wide_cleaned

def first_non_nan(col):
    """Return the first nonmissing value in the first eight entries, or None."""
    non_nan_values = [val for val in col[:8] if pd.notna(val)]
    return non_nan_values[0] if non_nan_values else None

def normalization(df):

    """Standardize numeric entries using one pooled mean and sample SD (ddof=1).
    Numeric identifiers contribute to the moments before their original values are restored."""
    numeric_columns = df.select_dtypes(include=['number'])
    assert not numeric_columns.empty, "DataFrame does not contain any numeric columns."

    mean_all = numeric_columns.stack().mean()
    std_all = numeric_columns.stack().std()

    normalized_df = (numeric_columns - mean_all) / std_all
    normalized_df['ID'] = df['ID']
    if "CODE" in df.columns:
        normalized_df['CODE'] = df['CODE']

    return normalized_df, mean_all, std_all

def normalization_validation(df, mean, std):

    """Apply the supplied mean and SD to numeric columns, then restore ID/CODE."""
    numeric_columns = df.select_dtypes(include=['number'])
    assert not numeric_columns.empty, "DataFrame does not contain any numeric columns."

    normalized_df = (numeric_columns - mean) / std
    normalized_df['ID'] = df['ID']
    normalized_df['CODE'] = df['CODE']

    return normalized_df

def extract_cluster_data(linkage_matrix, data):
    """Run MNE permutation tests at each linkage merge.
    Return final-merge statistics and null distribution, plus p-values from all merges."""
    n_samples = len(linkage_matrix) + 1                               

    tree_clusters = {i: {i} for i in range(n_samples)}
    p = []
                                                             
    for i in range(len(linkage_matrix)):
        print(i)
                                                                   
        cluster_idx1, cluster_idx2 = int(linkage_matrix[i, 0]), int(linkage_matrix[i, 1])

        new_cluster = tree_clusters[cluster_idx1].union(tree_clusters[cluster_idx2])

        first_cluster = tree_clusters[cluster_idx1]            
        selected_data_one = data[list(first_cluster)]
        cluster_one_data = np.vstack(selected_data_one)
    
        second_cluster = tree_clusters[cluster_idx2]            
        selected_data_two = data[list(second_cluster)]
        cluster_two_data = np.vstack(selected_data_two)

        data_for_permut = [cluster_one_data, cluster_two_data]
        F_obs, clusters, cluster_pv, h0 = mne.stats.permutation_cluster_test(data_for_permut, 
                                                                n_permutations=1000,
                                                                threshold=0.05, seed = 42)

        p.append(cluster_pv)

        tree_clusters[n_samples + i] = new_cluster

    return F_obs, clusters, p, h0                                                 

from collections import namedtuple
def welch_anova_np(*args):

    """Compute the legacy Welch-style statistic.
    The p-value argument order requires review; this helper is unused by the notebook."""
    F_onewayResult = namedtuple('F_onewayResult', ('statistic', 'pvalue'))

    data = list(map(np.asarray, args))
                      
    k = len(data)
                             
    ni =np.array([len(arg) for arg in data])
    mi =np.array([np.mean(arg) for arg in data])
    vi =np.array([np.var(arg,ddof=1) for arg in data])
    wi = ni/vi
    den = ni - 1 
    tmp =sum((1-wi/sum(wi))**2 / (ni-1))

    if (k**2 -1) != 0:
        tmp /= (k**2 -1)
    else:
        print('k**2 -1 = 0')

    dfbn = k - 1
    if (3 * tmp) != 0:
        dfwn = 1 / (3 * tmp)
    else:
        print('(3 * tmp) = 0')

    m = sum(mi*wi) / sum(wi)
    if ((dfbn) * (1 + 2 * (dfbn - 1) * tmp)) != 0:
        f = sum(wi * (mi - m)**2) /((dfbn) * (1 + 2 * (dfbn - 1) * tmp))
    else:
        print('final = 0')
    prob = scipy.stats.f.sf(dfbn, dfwn, f)                             
    return F_onewayResult(f, prob)

def top_down_traversal(Z, node, num_data_points, cluster_members):
    """Return current and child leaf lists, updating cluster_members for internal nodes."""
    if node < num_data_points:
                                                   
        current_cluster = [node]
        left_cluster = []
        right_cluster = []
    else:
                                               
        left_node = int(Z[node - num_data_points, 0])
        right_node = int(Z[node - num_data_points, 1])

        left_cluster, _, _ = top_down_traversal(Z, left_node, num_data_points, cluster_members)
        right_cluster, _, _ = top_down_traversal(Z, right_node, num_data_points, cluster_members)

        current_cluster = left_cluster + right_cluster
        cluster_members[node] = current_cluster

    return current_cluster, left_cluster, right_cluster

def calculate_shilouette(X, labels, metric):
    
    """Return the mean silhouette score for observations, labels, and distance metric."""
    avg_shilouette = silhouette_score(X, labels, metric=metric)

    return avg_shilouette

def permutation_shuffling_within_groups(left_array, right_array, num_permutations):
    """Return silhouettes after shuffling flattened values independently within each group."""
    permutation_wstat = []

    flattened_left = left_array.flatten()
    flattened_right = right_array.flatten()

    for _ in range(num_permutations):
        np.random.shuffle(flattened_left)
        np.random.shuffle(flattened_right)

        left = flattened_left.reshape(left_array.shape)
        right = flattened_right.reshape(right_array.shape)

        left_length = left.shape[0]
        
        right_length = right.shape[0]
        labels = [1] * left_length + [2] * right_length

        concat_data = np.concatenate((left, right), axis=0)

        wstat = calculate_shilouette(concat_data, labels, 'euclidean')
        permutation_wstat.append(wstat)

    return permutation_wstat

from scipy.spatial.distance import cdist

def elbow_method_perc(array, linkage_matrix):
                                  
    """Return candidate cluster counts, explained-variance percentages, and successive increments."""
    if array.ndim == 1:
        array = array.reshape(-1, 1)

    variance_explained = []
    total_variance = np.sum(cdist(array, np.mean(array, axis=0).reshape(1, -1)) ** 2)
                                                                     
    for k in range(1, len(array)):
        labels = fcluster(linkage_matrix, k, criterion='maxclust')

        within_cluster_variance = np.sum([np.sum(cdist(array[labels == j], np.mean(array[labels == j], axis=0).reshape(1, -1)) ** 2) for j in range(1, k + 1)])

        variance_expl = ((total_variance - within_cluster_variance) / total_variance) * 100
        variance_explained.append(variance_expl)

    vel = np.diff(variance_explained)

    vel = np.insert(vel, 0, 0)

    return np.arange(1, len(array)), variance_explained, vel

def hierarchical_clustering_results(datasets, datasets_names, metric, method, correlation_coefficient):

    """Plot cophenetic correlation tables for distance/linkage combinations; return table artists."""
    all_tables = []

    for dataset_index, array in enumerate(datasets):

        if array.ndim == 1:
                                                                        
            array = array[:, np.newaxis]    

        print(f"Processing dataset {dataset_index + 1}:")
        results = []

        for meth in method:
            avg_cc_for_method = 0 
            for metr in metric:
                distances = pdist(array, metric=metr)
                linkage_matrix = linkage(distances, method=meth)
                                                
                c, coph_dists = cophenet(linkage_matrix, distances)

                orig_dists = distances

                if correlation_coefficient == 'Pearsons':
                    cc = np.corrcoef(orig_dists, coph_dists)[0, 1]
                    avg_cc_for_method += cc 
                    results.append((meth, metr, cc))

                elif correlation_coefficient == 'Spearmans':
                    cc =  stats.spearmanr(orig_dists, coph_dists)
                    avg_cc_for_method += cc.statistic
                    results.append((meth, metr, cc.statistic))
            avg_cc_for_method /= len(metric)                                                                
            print(f'Average {correlation_coefficient} cc for {meth} method: {avg_cc_for_method}')
                                        
        methods, metrics, scores = zip(*results)

        results_df = pd.DataFrame({'Method': methods, 'Metric': metrics, f'{correlation_coefficient} Correlation Coefficient': scores})

        max_index = results_df[f'{correlation_coefficient} Correlation Coefficient'].idxmax()

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=results_df.values,
                         colLabels=results_df.columns,
                         loc='center')

        table[max_index + 1, 2].set_facecolor('yellow')                                       

        ax.set_title(f'{correlation_coefficient} Correlation Coef. for Different Method and Metric Combinations - {datasets_names[dataset_index]}')
        plt.show()

        all_tables.append(table)

    return all_tables

def shuffle(data):
    """Shuffle all flattened array values and restore the original shape without mutating the input."""
    stacked_data = data.flatten()

    np.random.shuffle(stacked_data)

    reshaped_data = stacked_data.reshape(data.shape)

    return reshaped_data

def permutation_compare_random_baseline(left_array, right_array, num_permutations):
    
    """Return silhouettes for uniform random matrices with fixed group sizes."""
    permutation_stat = []
    concat_data = np.concatenate((left_array, right_array), axis=0)
    left_length = left_array.shape[0]
    right_length = right_array.shape[0]
    
    labels = [1] * left_length + [2] * right_length

    for _ in range(num_permutations):

        data_shuffled = np.random.rand(concat_data.shape[0], concat_data.shape[1])
        wstat =  calculate_shilouette(data_shuffled, labels, 'euclidean')
        permutation_stat.append(wstat)

    return permutation_stat
