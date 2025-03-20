# ========================
# MODULES
# ========================
import numpy as np
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
for corr in ["_NO_CORRECTION_", "_CORRECTION_"]:
    report = mne.Report(title=f"Rapport Global RSA - {corr}")
    results[corr] = []
    
    for subject in subject_list:
        print(f"Traitement du sujet {subject + 1}...")
        
        # Charger les données EEG prétraitées
        epochs = mne.read_epochs(f"{data_path}{corr}/_EPOCHS_/MI_epochs_S{subject+1}-epo.fif", preload=True)
        
        # Sélection des epochs de la main droite
        epochs_right_hand = epochs["right_hand"]
        
        # Sélection des 20 premières et 20 dernières epochs
        first_quarter = epochs_right_hand[:20]
        
        tfr_first = first_quarter.compute_tfr("morlet", freqs=freqs, n_cycles=n_cycles, n_jobs=15, average=True)
        tfr_first.apply_baseline(baseline=(-3, -1), mode="logratio")
        tfr_first.crop(**crop_params)
        
        # Calcul des topographies moyennes
        topographie_first = np.mean(tfr_first.data, axis=(1, 2))
        
        # Sélection des 20 dernières epochs
        last_quarter = epochs_right_hand[-20:]
        
        tfr_last = last_quarter.compute_tfr("morlet", freqs=freqs, n_cycles=n_cycles, n_jobs=15, average=True)
        tfr_last.apply_baseline(baseline=(-3, -1), mode="logratio")
        tfr_last.crop(**crop_params)
        
        # Calcul des topographies moyennes
        topographie_last = np.mean(tfr_last.data, axis=(1, 2))
        
        # vmin et vmax
        vmax = np.max(np.abs([topographie_first, topographie_last]))
        vmin = -vmax
        
        # Affichage des topographies
        fig_topo, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        mne.viz.plot_topomap(topographie_last, epochs_right_hand.info, vlim=(vmin, vmax), cmap=my_cmap, axes=axs[0], show=False)
        axs[0].set_title("First 20 Epochs")
        mne.viz.plot_topomap(topographie_first, epochs_right_hand.info, vlim=(vmin, vmax), cmap=my_cmap, axes=axs[1], show=False)
        axs[1].set_title("Last 20 Epochs")
    
        # Matrices de dissimilarités
        rdm_first = mne_rsa.compute_rdm(topographie_first, metric="euclidean")
        rdm_last = mne_rsa.compute_rdm(topographie_last, metric="euclidean")
        
        fig_rsa = mne_rsa.plot_rdms([rdm_first, rdm_last], names=["First 20 Epochs", "Last 20 Epochs"])
        fig_rsa.set_size_inches(4, 4)
    
        rsa_result = np.abs(mne_rsa.rsa(rdm_first, rdm_last, metric="spearman"))
        results[corr].append(rsa_result)
        
        table_data = {
            "Score de similarité (%) : ": rsa_result*100
        }
        
        html_code = generate_html_table(table_data)
        report.add_html(html_code, section=f"{subject}", title="Score de similartité")
        report.add_figure(fig_topo, title="Topographies", section=f"{subject}")
        report.add_figure(fig_rsa, title="RDM", section=f"{subject}")

    report.save(f"{data_path}Report/_RSA_/Global_RSA_{corr}.html", open_browser=False, overwrite=True)
    
    plt.close('all')


report = mne.Report(title="Rapport Global RSA")

# Conversion en DataFrame pour Seaborn
df = pd.DataFrame(results)
df = df.melt(var_name='Condition', value_name='Valeur')

# Création du plot combiné Boxplot + Violinplot
fig_box, ax_box = plt.subplots(figsize=(12, 7))
sns.violinplot(x='Condition', y='Valeur', data=df, inner=None, alpha=0.5)
sns.boxplot(x='Condition', y='Valeur', data=df, width=0.3, boxprops=dict(alpha=0.7))

# Ajout des lignes reliant les mêmes sujets entre les deux conditions
no_corr_values = results['_NO_CORRECTION_']
corr_values = results['_CORRECTION_']
num_points = len(no_corr_values)

for i in range(num_points):
    plt.plot([0, 1], [no_corr_values[i], corr_values[i]], marker='o', color='gray', alpha=0.6)

# Ajout de titres et labels
ax_box.set_xlabel("Condition")
ax_box.set_ylabel("RSA (%)")

report.add_figure(fig_box, title="Global RSA", section="GLOBAL")

report.save(f"{data_path}Report/_RSA_/Global_RSA.html", open_browser=False, overwrite=True)