import matplotlib.pyplot as plt
import numpy as np
from scipy import interp, stats
import pandas as pd


from scipy.stats import chi2_contingency
from scipy.stats import chi2, kstest, ranksums



def print_statistics(df_cohort):
    df_pos = df_cohort[df_cohort["tni_class"] == 1]
    df_neg = df_cohort[df_cohort["tni_class"] == 0]
    print(f"Shape {df_pos.shape} for tni_class = 1")
    print(f"Shape {df_neg.shape} for tni_class = 0")
    print(f"---")

    def print_numeric(col):
        overall_stat = f"{df_cohort[col].describe()['50']} [{df_cohort[col].describe()['25%']}-{df_cohort[col].describe()['75%']}]"
        pos_stat = f"{df_pos[col].describe()['50%']} [{df_pos[col].describe()['25%']}-{df_pos[col].describe()['75%']}]"
        neg_stat = f"{df_neg[col].describe()['50%']} [{df_neg[col].describe()['25%']}-{df_neg[col].describe()['75%']}]"
        overall_missing = df_cohort[col].isna().sum()
        pos_missing = df_pos[col].isna().sum()
        neg_missing = df_pos[col].isna().sum()
        w, p_val = ranksums(df_pos[col], df_neg[col])

        print(f"{col},{overall_stat},{overall_missing}, {pos_stat}, {pos_missing}, {neg_stat}, {neg_missing}, {p_val}")

    def print_binary_col(col, val_1, val_2):
        overall_f = df_cohort[df_cohort[col] == val_1].shape[0]
        overall_m = df_cohort[df_cohort[col] == val_2].shape[0]
        contingency_table = [
            [df_pos[df_pos[col] == val_2].shape[0], df_neg[df_neg[col] == val_2].shape[0]],
            [df_pos[df_pos[col] == val_1].shape[0], df_neg[df_neg[col] == val_1].shape[0]]
        ]

        stat, p, dof, expected = chi2_contingency(contingency_table, correction=False)

        pos_f = df_pos[df_pos[col] == val_1].shape[0]
        pos_m = df_pos[df_pos[col] == val_2].shape[0]
        neg_f = df_neg[df_neg[col] == val_1].shape[0]
        neg_m = df_neg[df_neg[col] == val_2].shape[0]
        pos_f_missing = df_pos[df_pos[col] == val_1][col].isna().sum()
        neg_f_missing = df_neg[df_neg[col] == val_1][col].isna().sum()
        pos_m_missing = df_pos[df_pos[col] == val_2][col].isna().sum()
        neg_m_missing = df_neg[df_neg[col] == val_2][col].isna().sum()
        print(
            f"{col} - {val_1}, {pos_f} ({round(100 * overall_f / (overall_f + overall_m), 2)})({round(100 * pos_f / (pos_f + pos_m), 2)}), {pos_f_missing}, {neg_f} ({round(100 * neg_f / (neg_f + neg_m), 2)}), {neg_f_missing}, {p}")
        print(
            f"{col} - {val_2}, {pos_m} ({round(100 * overall_m / (overall_f + overall_m), 2)})({round(100 * pos_m / (pos_f + pos_m), 2)}), {pos_m_missing}, {neg_m} ({round(100 * neg_m / (neg_f + neg_m), 2)}), {neg_m_missing},")

    def print_risk_factor(col):
        pos_missing = df_pos[col].isna().sum()
        neg_missing = df_neg[col].isna().sum()
        pval = stats.ttest_ind(df_pos[col], df_neg[col], nan_policy="omit", equal_var=False).pvalue
        print(
            f"{col}, {int(df_pos[col].sum())} ({round(100 * df_pos[col].sum() / (len(df_pos[col]) - int(df_pos[col].isna().sum())), 2)}), {pos_missing}, {int(df_neg[col].sum())} ({round(100 * df_neg[col].sum() / (len(df_neg[col]) - int(df_neg[col].isna().sum())), 2)}), {neg_missing}, {pval}")

        # ---

    print("Characteristic, Pos Statistic, Pos Missing, Neg Statistic, Neg Missing, p-value")
    print_numeric("Age")
    print_binary_col("Gender", 0, 1)
    print_binary_col("st_related", 0, 1)
    print_numeric("tni_value")
    print_numeric("PR")
    print_numeric("HR")
    print_numeric("QT")
    print_numeric("SBP")
    print_numeric("QTc")
    print_numeric("QRS")
    print_numeric("QRSE")



df_tni = pd.read_csv(".")
print(df_tni.shape)
print(print_statistics(df_tni))