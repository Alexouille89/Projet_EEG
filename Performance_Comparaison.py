# ========================
# MODULES
# ========================
import numpy as np
import mne
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
# PARAMÈTRES
# ========================

data_path = "C:/Users/alexs/OneDrive/Bureau/Cours_inge_3/Projet EEG/_WEIBO_/"
subject_list = range(9)  # Liste des sujets
csp_index_corrected = [5, 6, 1, 5, 6, 1, 3, 2, 6]
csp_index_uncorrected = [4, 6, 1, 5, 6, 1, 3, 3, 6]
freq_range = (8, 30)  # Bande de fréquences d'intérêt
scaling = 1e6  # Conversion V² -> µV²

# Fenêtres temporelles
baseline_window = (0, 2)  # Baseline avant la tâche
mi_window = (3, 7)  # Fenêtre d'imagination motrice

# Stockage des résultats
results = []
report = mne.Report(title="Rapport Global ERD")

# ========================
# ANALYSE PAR SUJET
# ========================

for subject in subject_list:
    print(f"Traitement du sujet {subject + 1}...")
    
    # Charger les données EEG prétraitées
    epochs = mne.read_epochs(f"{data_path}_NO_CORRECTION_/_EPOCHS_/MI_epochs_S{subject+1}-epo.fif", preload=True)
    
    # Charger les modèles CSP
    csp_paths = {
        "Corrected": f"{data_path}_CORRECTION_/_CSP_/CSP_{subject+1}.pkl",
        "Uncorrected": f"{data_path}_NO_CORRECTION_/_CSP_/CSP_{subject+1}.pkl"
    }
    csp_models = {key: joblib.load(path) for key, path in csp_paths.items()}

    # Index CSP sélectionnés
    csp_subject_indices = {
        "Corrected": [csp_index_corrected[subject]],
        "Uncorrected": [csp_index_uncorrected[subject]]
    }

    # Dictionnaire pour stocker les ERD
    erd_values = {"Sujet": subject + 1}

    # Figures pour le rapport
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_ylabel("ERD (%)")

    # Calcul pour chaque méthode (CSP et Laplacien)
    for label, csp_model in csp_models.items():
        csp_idx = csp_subject_indices[label][0]
        selected_filter = csp_model.filters_[csp_idx] / np.sum(csp_model.filters_[csp_idx])  # Normalisation

        # Application du filtre CSP
        X_csp = np.einsum('ij,kjl->kil', selected_filter[np.newaxis, :], epochs['right_hand'].get_data())
        epochs_csp = mne.EpochsArray(X_csp, mne.create_info(['CSP'], epochs.info['sfreq'], ch_types='eeg'))
        
        epochs_baseline = epochs_csp.copy().crop(tmin=baseline_window[0], tmax=baseline_window[1])
        epochs_mi = epochs_csp.copy().crop(tmin=mi_window[0], tmax=mi_window[1])
        
        # Filtrage 8-30 Hz
        X_csp_baseline = mne.filter.filter_data(epochs_baseline.get_data(), sfreq=epochs.info['sfreq'],
                                                l_freq=freq_range[0], h_freq=freq_range[1], method='fir', fir_design='firwin')
        X_csp_mi = mne.filter.filter_data(epochs_mi.get_data(), sfreq=epochs.info['sfreq'],
                                  l_freq=freq_range[0], h_freq=freq_range[1], method='fir', fir_design='firwin')
        
        # Extraction de la puissance instantanée
        P_baseline = np.mean(X_csp_baseline ** 2, axis=(1, 2)) * scaling ** 2
        P_mi = np.mean(X_csp_mi ** 2, axis=(1, 2)) * scaling ** 2


        # Calcul de l'ERD
        ERD_csp = ((P_mi - P_baseline) / P_baseline) * 100
        erd_values[f"ERD {label}"] = np.median(ERD_csp)

        # Ajout de la figure pour chaque CSP
        ax.plot(range(len(ERD_csp)), ERD_csp, label=f"{label} CSP")

    # Calcul de l'ERD pour le Laplacien
    X = epochs['right_hand'].get_data()
    X_laplacien = 4 * X[:, 24, :] - X[:, 17, :] - X[:, 15, :] - X[:, 34, :] - X[:, 32, :]
    X_laplacien = X_laplacien[:, np.newaxis, :]

    epochs_laplacien = mne.EpochsArray(X_laplacien, mne.create_info(['Laplacien'], epochs.info['sfreq'], ch_types='eeg'))
    epochs_baseline_lap = epochs_laplacien.copy().crop(tmin=baseline_window[0], tmax=baseline_window[1])
    epochs_mi_lap = epochs_laplacien.copy().crop(tmin=mi_window[0], tmax=mi_window[1])

    # Filtrage 8-30 Hz
    X_baseline_lap = mne.filter.filter_data(epochs_baseline_lap.get_data(), sfreq=epochs.info['sfreq'],
                                            l_freq=freq_range[0], h_freq=freq_range[1], method='fir', fir_design='firwin')
    X_mi_lap = mne.filter.filter_data(epochs_mi_lap.get_data(), sfreq=epochs.info['sfreq'],
                                      l_freq=freq_range[0], h_freq=freq_range[1], method='fir', fir_design='firwin')

    # Extraction de la puissance instantanée
    P_baseline_lap = np.mean(X_baseline_lap ** 2, axis=(1, 2)) * scaling ** 2
    P_mi_lap = np.mean(X_mi_lap ** 2, axis=(1, 2)) * scaling ** 2

    # Calcul de l'ERD Laplacien
    ERD_laplacien = ((P_mi_lap - P_baseline_lap) / P_baseline_lap) * 100
    erd_values["ERD Laplacien"] = np.median(ERD_laplacien)
    ax.plot(range(len(ERD_laplacien)), ERD_laplacien, label="Laplacien")
    ax.legend()
    ax.set_title(f"Évolution de l'ERD - Sujet {subject+1}")
    report.add_figure(fig, title=f"ERD Evolution - Sujet {subject+1}", section=f"Sujet {subject+1}")
    
    # Stockage des résultats
    results.append(erd_values)

# Création d'un DataFrame pour affichage
results_df = pd.DataFrame(results)
report.add_html(results_df.to_html(), title="Tableau ERD", section="Résumé des résultats")


# Ajout d'un boxplot et d'un violin plot avec Seaborn
fig_box, ax_box = plt.subplots(figsize=(12, 7))
sns.boxplot(data=results_df.drop(columns=["Sujet"]), ax=ax_box, palette="Set2", width=0.5)
sns.violinplot(data=results_df.drop(columns=["Sujet"]), ax=ax_box, palette="Set2", alpha=0.6, inner="quartile")

# Relier les points par sujet
for i in range(len(results_df)):
    plt.plot([0, 1, 2], results_df.iloc[i, 1:].values, marker='o', linestyle='-', color='black', alpha=0.5)

ax_box.set_title("Distribution des ERD par méthode")
ax_box.set_ylabel("ERD (%)")
ax_box.set_xticklabels(results_df.columns[1:])
report.add_figure(fig_box, title="Boxplot et Violin Plot des ERD", section="Résumé des résultats")

report.save(f"{data_path}Report/_PERFORMANCE_/ERD_Global_Report.html", overwrite=True)
print(results_df)

plt.close('all')
