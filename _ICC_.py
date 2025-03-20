# -*- coding: utf-8 -*-
"""
Copyright (c) 2024 Paris Brain Institute. All rights reserved.

Created on March 2025

@author: Cassandra Dumas

"""

# ========================
# MODULES
# ========================
import numpy as np
import pingouin as pg
import mne
import mne_rsa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tools import discrete_cmap, generate_html_table

# ========================
# PARAMÈTRES
# ========================
data_path = "C:/Users/alexs/OneDrive/Bureau/Cours_inge_3/Projet EEG/_WEIBO_/"
subject_list = range(9)  # Liste des sujets

# Définition des paramètres de traitement
freqs = np.arange(0.5, 80, 0.5)  # Fréquences utilisées pour la TFR
n_cycles = freqs / 2  # Nombre de cycles pour l'ondelette de Morlet
low_freq, high_freq = 8, 30  # Filtrage en bande passante (rythme mu)
crop_params = dict(tmin=0, tmax=4, fmin=low_freq, fmax=high_freq)  # Paramètres de restriction des TFR

# Définition d'une palette de couleurs pour la visualisation des topographies
my_cmap = discrete_cmap(13, 'RdBu_r')

results = {}


# ========================
# ANALYSE PAR SUJET
# ========================
# Stockage des ICC sous forme de DataFrame
icc_df_list = []

for corr in ["_NO_CORRECTION_", "_CORRECTION_"]:
    report = mne.Report(title=f"Rapport Global RSA - {corr}")
    results[corr] = []
    
    for subject in subject_list:
        print(f"Traitement du sujet {subject + 1}...")

        # Charger les données EEG prétraitées
        epochs = mne.read_epochs(f"{data_path}{corr}/_EPOCHS_/MI_epochs_S{subject+1}-epo.fif", preload=True)
        epochs = epochs['right_hand']

        # Déterminer les indices temporels correspondant à 0 - 4s
        tmin, tmax = 0, 4  # Fenêtre en secondes
        tmin_idx, tmax_idx = epochs.time_as_index([tmin, tmax])

        # Extraire les données de la fenêtre temporelle pour tous les canaux
        data = epochs.get_data()[:, :, tmin_idx:tmax_idx]  # Shape: (n_epochs, n_channels, n_times)

        # Vérification du nombre d'epochs
        n_epochs, n_channels, _ = data.shape
        group_labels = np.repeat(np.arange(1, 11), n_epochs/10)

        icc_results = {}

        for ch_idx, ch_name in enumerate(epochs.ch_names):
            # Moyenne temporelle pour chaque epoch et canal
            data_mean = np.mean(data[:, ch_idx, :], axis=1)

            # Construire le DataFrame ICC
            df_icc = pd.DataFrame({
                "Group": group_labels,  # 10 groupes de 8 epochs
                "Epoch": np.tile(np.arange(1, int(n_epochs/10)+1), len(group_labels) // int(n_epochs/10)),
                "Measure": data_mean
            })

            # Calcul ICC(3,1)
            icc_result = pg.intraclass_corr(data=df_icc, targets="Group", raters="Epoch", ratings="Measure", nan_policy="omit")
            icc_3_1 = icc_result[icc_result["Type"] == "ICC3"]["ICC"].values[0]

            icc_results[ch_name] = icc_3_1

        # Stocker dans un DataFrame
        icc_df = pd.DataFrame({
            "Channel": list(icc_results.keys()),
            "ICC": list(icc_results.values()),
            "Subject": subject + 1,
            "Correction": corr
        })
        icc_df_list.append(icc_df)

# Concaténer tous les ICCs des sujets
final_icc_df = pd.concat(icc_df_list, ignore_index=True)

# Génération des figures individuelles par sujet
figs = []

for subject in final_icc_df["Subject"].unique():
    df_sub = final_icc_df[final_icc_df["Subject"] == subject]

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.violinplot(x="Correction", y="ICC", data=df_sub, inner="box", ax=ax, alpha=0.8, linewidth=1.2, width=0.6, cut=0)

    for ch in df_sub["Channel"].unique():
        icc_no_corr = df_sub[(df_sub["Channel"] == ch) & (df_sub["Correction"] == "_NO_CORRECTION_")]["ICC"].values[0]
        icc_corr = df_sub[(df_sub["Channel"] == ch) & (df_sub["Correction"] == "_CORRECTION_")]["ICC"].values[0]
        ax.plot([0, 1], [icc_no_corr, icc_corr], color='gray', alpha=0.3, linewidth=0.8)

    ax.set_xlim(-0.5, 1.5)

    icc_min, icc_max = df_sub["ICC"].min(), df_sub["ICC"].max()
    margin = (icc_max - icc_min) * 0.15
    ax.set_ylim(icc_min - margin, icc_max + margin)

    ax.set_title(f"Sujet {subject} - ICC(3,1)", fontsize=14, pad=20)
    ax.set_ylabel("ICC", fontsize=12, labelpad=10)
    ax.set_xlabel("Condition", fontsize=12, labelpad=10)

    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    plt.subplots_adjust(left=0.2, right=0.8, top=0.85, bottom=0.3)

    figs.append((fig, f"icc_subject_{subject}"))

# Création du rapport MNE
report = mne.Report(title="Rapport ICC")
for fig, fname in figs:
    report.add_figure(fig, title=fname, caption=f"ICC par canal - {fname}")

# Calcul de la moyenne des ICCs par sujet et condition
summary_df = final_icc_df.groupby(["Subject", "Correction"])["ICC"].mean().reset_index()

# Génération de la figure récapitulative
fig, ax = plt.subplots(figsize=(7, 7))

sns.scatterplot(data=summary_df, x="Correction", y="ICC", hue="Subject", palette="tab10", ax=ax, s=100)

for subject in summary_df["Subject"].unique():
    icc_no_corr = summary_df[(summary_df["Subject"] == subject) & (summary_df["Correction"] == "_NO_CORRECTION_")]["ICC"].values[0]
    icc_corr = summary_df[(summary_df["Subject"] == subject) & (summary_df["Correction"] == "_CORRECTION_")]["ICC"].values[0]
    ax.plot(["_NO_CORRECTION_", "_CORRECTION_"], [icc_no_corr, icc_corr], color='gray', alpha=0.5, linewidth=1)

ax.set_xlim(-0.5, 1.5)

ax.set_title("Résumé des ICC par sujet", fontsize=14, pad=20)
ax.set_ylabel("ICC moyen", fontsize=12, labelpad=10)
ax.set_xlabel("Condition", fontsize=12, labelpad=10)

ax.legend(title="Sujet", bbox_to_anchor=(0.85, 1), loc='upper left')

report.add_figure(fig, title="Résumé des ICC par sujet", caption="Comparaison des ICC moyens entre _NO_CORRECTION_ et _CORRECTION_")

# Sauvegarde du rapport
report.save(f"{data_path}Report/_ICC_/Global_ICC.html", overwrite=True)