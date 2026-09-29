"""Cohort harmonization, interval aggregation, and statistical helpers."""

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from tabulate import tabulate
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scikit_posthocs import posthoc_dunn
import sys

def restructure_cohort_three(stress_df, mood_df):

    """Merge cohort-three stress records with diagnosis codes derived from mood indicators."""
    assert "CODE" not in stress_df.columns

    stress_df = stress_df.rename(columns={"Subj. ID":"ID"})
    mood_df["CODE"] = mood_df[["ND", "AS", "PPD"]].idxmax(1)

    diagnosis_one = mood_df[['ID', 'CODE']] 

    wide_df = pd.merge(stress_df, diagnosis_one, on='ID', how='left')

    wide_df = wide_df[ wide_df["CODE"].notnull() ]

    assert "ID" in wide_df.columns
    assert "CODE" in wide_df.columns
    assert wide_df["CODE"].isnull().sum() == 0

    return wide_df

def make_long(wide_df, score:str):

    """Reshape daily scores into ID/CODE/day rows, excluding missing and nonnumeric scores."""
    assert "ID" in wide_df.columns
    assert "CODE" in wide_df.columns

    before = wide_df.shape[0]
    wide_df = wide_df[wide_df['ID'].notnull()]
    after = wide_df.shape[0]

    day_wide_df = wide_df.filter(like=f'{score} Day', axis="columns")

    very_wide_df = pd.merge(wide_df[["ID","CODE"]], day_wide_df, left_index=True, right_index=True)
    inter_stress_one = pd.melt(very_wide_df, id_vars=['ID', 'CODE'], var_name='Day', value_name=f'{score} Level')
    inter_stress_one = inter_stress_one.dropna(subset=[f'{score} Level'])
    
    inter_stress_one['Day'] = inter_stress_one['Day'].str.extract('(\d+)')             
                                                                        
    inter_stress_one.sort_values(by=['ID',  'CODE'], inplace=True)        
    inter_stress_one.reset_index(drop=True, inplace=True)

    inter_stress_one = inter_stress_one[ inter_stress_one['Day'].notnull() ] 

    inter_stress_one['Day'] = inter_stress_one['Day'].astype(int)
    inter_stress_one = inter_stress_one[ pd.to_numeric(inter_stress_one[f'{score} Level'], errors="coerce").notnull() ]

    assert set(["ID","CODE","Day",f'{score} Level']) <= set(inter_stress_one.columns)                  
    assert not inter_stress_one['CODE'].isnull().any()
    assert not inter_stress_one['ID'].isnull().any()
    assert not inter_stress_one['Day'].isnull().any()

    return inter_stress_one

def concat_dfs_vertically (df_1, df_2):

    """Concatenate two DataFrames row-wise, preserving their indices."""
    union_df = pd.concat([df_1, df_2])

    return union_df

def convert_to_weekly(df, interval_size:int, start_day:int, finish_day:int, score:str):
    
    """Attach participant interval means and zero-based T labels to observed rows.
    Discard incomplete trailing intervals; interval_size need not equal seven."""
    assert "ID" in df.columns
    assert 'CODE' in df.columns
    assert 'Day' in df.columns

    intervals = []

    for i, current_start in enumerate(range(start_day, finish_day + 1, interval_size)):
        current_end = min(current_start + interval_size - 1, finish_day)
                                                              
        if current_end - current_start + 1 == interval_size:
            intervals.append((current_start, current_end))

    selected_rows = df[(df['Day'] >= intervals[0][0]) | (df['Day'] <= intervals[-1][1])]
    selected_rows['T'] = None                               
    selected_rows[f'Mean_{score}'] = None                                        

    for i, interval in enumerate(intervals):
        for id_value in selected_rows['ID'].unique():
            start, end = interval

            id_interval_selected_rows = selected_rows[(selected_rows['ID'] == id_value) & (selected_rows['Day'] >= start) & (selected_rows['Day'] <= end)]
            score_values = id_interval_selected_rows[f'{score} Level']

            if not score_values.isnull().all():
                mean_score = score_values.mean()
            else:
                mean_score = None         

            selected_rows.loc[(selected_rows['ID'] == id_value) & (selected_rows['Day'] >= start) & (selected_rows['Day'] <= end), 'T'] = i
            selected_rows.loc[(selected_rows['ID'] == id_value) & (selected_rows['Day'] >= start) & (selected_rows['Day'] <= end), f'Mean_{score}'] = mean_score
    selected_rows = selected_rows.dropna()

    assert 'CODE' in selected_rows.columns
                                            
    return selected_rows

def mean_scores_plot(mean_df: pd.DataFrame, variable: str, cohorts:list):

    """Plot existing Week and mean-score columns separately by CODE."""
    assert 'CODE' in mean_df.columns
    assert 'Week' in mean_df.columns
    assert 'Mean_Stress' or 'Mean_Mood' in mean_df.columns

    plt.figure(figsize=(12, 8))
    for code in mean_df["CODE"].unique():
        code_mean_df = mean_df[ mean_df["CODE"] == code ]
        plt.plot(code_mean_df["Week"], code_mean_df[f'Mean_{variable}'], label=f'Diagnosis {code}', marker='o', linestyle='-')

    plt.xlabel('Weeks')
    plt.ylabel(f'Mean {variable} Levels')
    plt.title(f'Mean {variable} Levels - Cohorts {cohorts}')
    plt.legend()
    plt.grid(True)

def standardize_df(stress_df, mood_df, columns):

    """Harmonize daily column names in place and outer-merge mood/stress on ID/CODE/Cohort."""
    assert isinstance(columns, list)
    assert not stress_df['ID'].isnull().any()
    assert not mood_df['ID'].isnull().any()
    assert not mood_df['CODE'].isnull().any()

    assert 'ID' in stress_df.columns
    assert 'ID' in mood_df.columns
    assert 'Cohort' in mood_df.columns

    stress_prefix = 'Stress Day '
    mood_prefix = 'Mood Day '

    stress_df.columns = [f'{stress_prefix}{col.split(" ")[-1]}' if 'Day' in col else col for col in stress_df.columns]
                                     
    mood_df.columns = [f'{mood_prefix}{col.split(" ")[-1]}'  if 'Day' in col else col for col in mood_df.columns]

    mood_df_sel = pd.DataFrame()
    mood_df_sel['Cohort'] = mood_df['Cohort']
    mood_df_sel['ID'] = mood_df['ID']
    mood_df_sel['Mood Bool'] = mood_df['Mood Bool']
    mood_df_sel['Dem Bool'] = mood_df['Dem Bool']

    filtered_columns = [col for col in columns if col != 'Cohort']
    mood_df_sel = pd.concat([mood_df_sel, mood_df[filtered_columns]], axis=1)

    day_columns = mood_df.filter(like='Day ', axis=1)

    mood_df_sel = pd.concat([mood_df_sel, mood_df[day_columns.columns.tolist()]], axis=1)

    stress_df_sel = pd.DataFrame()
    stress_df_sel['Cohort'] = stress_df['Cohort']
    stress_df_sel['ID'] = stress_df['ID']
    stress_df_sel['CODE'] = stress_df['CODE']

    stress_df_sel['Stress Bool'] = stress_df['Stress Bool']
    day_columns = stress_df.filter(like='Day ', axis=1)
    stress_df_sel = pd.concat([stress_df_sel, stress_df[day_columns.columns.tolist()]], axis=1)
    print(mood_df_sel[mood_df_sel["ID"]=="BH_CK1305"][['ID', 'CODE', 'Cohort']])
    print(stress_df_sel[stress_df_sel["ID"]=="BH_CK1305"][['ID', 'CODE', 'Cohort']])

    merged_df = pd.merge(mood_df_sel, stress_df_sel, on=['ID', 'CODE', 'Cohort'], how='outer')

    nan_percentage = merged_df.isna().sum(axis=1) / merged_df.shape[1] * 100
    assert not (nan_percentage >= 90).all()

    return merged_df

def select_cohort_columns(common_variables: pd.DataFrame, list_of_coh_number: list):
    """Return metadata columns marked True for every selected cohort."""
    select_cohorts = common_variables[common_variables['Cohort'].isin(list_of_coh_number)]                                                              
    selected_columns = select_cohorts.loc[:, (select_cohorts == True).all()]
    selected_columns_list = selected_columns.columns.tolist()
    
    return selected_columns_list                                       

def replace_rows_from_df(original_df, replacement_rows):

    """Replace original rows sharing replacement IDs, append replacements, and reset the index."""
    assert "ID" in original_df.columns
    assert "ID" in replacement_rows.columns

    original_filt = original_df[~original_df['ID'].isin(replacement_rows['ID'])]
    cor_df = concat_dfs_vertically (original_filt, replacement_rows)
    cor_df.reset_index(drop=True, inplace=True)

    print(f'Original: {original_df.shape[0]}, Original filter {original_filt.shape[0]}, Add {replacement_rows.shape[0]}, corrected {cor_df.shape[0]}')

    return cor_df

def data_selection(combined_cohorts, cohorts_number:list, common_variables, columns_wanted, condition2=None, 
                   condition3=None, condition4=None, condition5=None, condition6=None):
    
    """Select cohort rows using optional masks and shared metadata columns.
    columns_wanted is passed as cohort numbers to select_cohort_columns."""
    all_conditions = True

    if condition2 is not None:
        all_conditions &= condition2
    if condition3 is not None:
        all_conditions &= condition3
    if condition4 is not None:
        all_conditions &= condition4
    if condition5 is not None:
        all_conditions &= condition5
    if condition6 is not None:
        all_conditions &= condition6

    selected_cohorts = combined_cohorts['Cohort'].isin(cohorts_number)

    cohort = combined_cohorts[selected_cohorts & all_conditions]                                                      

    columns_wanted = select_cohort_columns(common_variables, columns_wanted)
    cohort_final = cohort[columns_wanted]
    cohort_final['ID'] = cohort['ID']

    return cohort_final

import numpy as np
from scipy import stats

def check_contingency_table(observe_table):

    """Return chi-square results and a flag allowing at most one expected cell below five."""
    chi2, p, dof, expected_table = stats.chi2_contingency(observe_table)

    num_cells_below_5 = np.sum(expected_table < 5)

    assumptions_met = num_cells_below_5 <= 1

    return assumptions_met, chi2, p, dof, expected_table

from statsmodels.stats.multitest import multipletests

def chi_square_test(combined_categorical, list_of_categorical_data_underscore):
    """Print screened chi-square tests and BH-adjusted p-values; omit the final variable-list entry."""
    significant_variables = []
    p_values = []

    for var in list_of_categorical_data_underscore[:-1]:
        contingency_table = pd.crosstab(combined_categorical['CODE'], combined_categorical[var])
        is_valid, chi2, p, dof, expected = check_contingency_table(contingency_table)

        if is_valid == False:
            print("var:", var, "One of the expected value is <5, cannot conduct the test")
            print(contingency_table)
            print(expected)

        elif is_valid == True:
            significant_variables.append(var)
            p_values.append(p)                                                 
            print("var:", var, "p:", p, "chi2", chi2, dof)
            print(contingency_table)

    if p_values:
        _, corrected_p_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        print("Adjusted p-values with FDR correction:")
        for var, p_corr in zip(significant_variables, corrected_p_values):
            print(f"{var}: {p_corr}")
    else:
        print("No significant variables found to correct.")

from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
import pandas as pd
from itertools import combinations

def pairwise_posthoc_chi2(df, variable, group_col='CODE'):
    """Print pairwise Fisher (2x2) or chi-square tests with within-variable BH correction."""
    pairs = list(combinations(df[group_col].unique(), 2))
    p_values = []
    comparisons = []

    for g1, g2 in pairs:
        sub_df = df[df[group_col].isin([g1, g2])]
        contingency = pd.crosstab(sub_df[group_col], sub_df[variable])

        if contingency.shape == (2, 2):
            _, p = fisher_exact(contingency)
        else:
            _, p, _, _ = chi2_contingency(contingency)

        comparisons.append(f"{g1} vs {g2}")
        p_values.append(p)

    _, corrected_p, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

    print(f"\nPost hoc test for variable: {variable}")
    for comp, raw_p, adj_p in zip(comparisons, p_values, corrected_p):
        print(f"{comp}: raw p = {raw_p:.4f}, FDR-adjusted p = {adj_p:.4f}")

def chi_square_test_new(combined_categorical, list_of_categorical_data_underscore):
    """Print BH-adjusted omnibus tests and run pairwise tests for adjusted p-values below 0.05."""
    significant_variables = []
    p_values = []

    for var in list_of_categorical_data_underscore[:-1]:
        contingency_table = pd.crosstab(combined_categorical['CODE'], combined_categorical[var])
        is_valid, chi2, p, dof, expected = check_contingency_table(contingency_table)

        if not is_valid:
            print("var:", var, "One of the expected values is <5, cannot conduct the test")
            print(contingency_table)
            print(expected)
        else:
            significant_variables.append(var)
            p_values.append(p)
            print("var:", var, "p:", p, "chi2:", chi2, "dof:", dof)
            print(contingency_table)

    if p_values:
        _, corrected_p_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        print("\nAdjusted p-values with FDR correction:")
        for var, p_corr in zip(significant_variables, corrected_p_values):
            print(f"{var}: {p_corr:.4f}")
            if p_corr < 0.05:
                pairwise_posthoc_chi2(combined_categorical, var, group_col='CODE')
    else:
        print("No significant variables found to correct.")

from scipy import stats
from scikit_posthocs import posthoc_dunn
import pandas as pd

def kruskal_wallis_test(df_cont, list_of_continuous_data):
    """Compare ND/PPD/AS scores and run Bonferroni Dunn tests after raw p<0.05.
    Return significant variable names; omnibus tests are not corrected across variables."""
    significant_var = []           
    
    for var in list_of_continuous_data:
        if var == 'CODE':
            continue                                  

        print(f"\nVariable: {var}")

        df_clean = df_cont.copy()
        df_clean[var] = pd.to_numeric(df_clean[var], errors='coerce')
        df_clean = df_clean.dropna(subset=[var, 'CODE'])

        nd_vals = df_clean[df_clean['CODE'] == 'ND'][var]
        ppd_vals = df_clean[df_clean['CODE'] == 'PPD'][var]
        as_vals = df_clean[df_clean['CODE'] == 'AS'][var]

        print("Summary per group:")
        for group_name, values in [('ND', nd_vals), ('PPD', ppd_vals), ('AS', as_vals)]:
            count = len(values)
            mean = values.mean()
            std = values.std()
            print(f"  {group_name}: count = {count}, mean = {mean:.2f}, std = {std:.2f}")

        Kruskal_result = stats.kruskal(nd_vals, ppd_vals, as_vals)

        if Kruskal_result.pvalue < 0.05:
            significant_var.append(var)
            print("Kruskal-Wallis test is significant (p < 0.05).")

            posthoc_result = posthoc_dunn(df_clean, val_col=var, group_col='CODE', p_adjust='bonferroni')
            print("Dunn's post hoc test results with Bonferroni correction:")
            print(posthoc_result)
        else:
            print("Kruskal-Wallis test is not significant.")

    return significant_var
