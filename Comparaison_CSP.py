# ========================
# MODULES
# ========================
import os
import numpy as np
import mne
import joblib
import matplotlib.pyplot as plt


# ========================
# PARAMÈTRES GLOBAUX
# ========================

# Paramètres
data_path = "C:/Users/alexs/OneDrive/Bureau/Cours_inge_3/Projet EEG/_WEIBO_/"
subject_list = range(9)  # Liste des sujets optimisée
csp_index_corrected = [5, 6, 1, 5, 6, 1, 3, 2, 6]
csp_index_uncorrected = [4, 6, 1, 5, 6, 1, 3, 3, 6]
freq_range = (8, 30)  # Plage de fréquences à analyser
scaling = 1e6  # Conversion V² -> µV²

# Répertoire de sauvegarde des rapports
save_path = f"{data_path}Report/_COMPARAISON_"
os.makedirs(save_path, exist_ok=True)

# Boucle sur les sujets
for subject in subject_list:
    report = mne.Report(title=f"Comparaison CSPs - Sujet {subject}")

    # Charger les données EEG prétraitées
    epochs = mne.read_epochs(f"{data_path}_NO_CORRECTION_/_EPOCHS_/MI_epochs_S{subject+1}-epo.fif", preload=True)

    # Charger les modèles CSP
    csp_paths = {
        "Corrected": f"{data_path}_CORRECTION_/_CSP_/CSP_{subject+1}.pkl",
        "Uncorrected": f"{data_path}_NO_CORRECTION_/_CSP_/CSP_{subject+1}.pkl"
    }
    csp_models = {key: joblib.load(path) for key, path in csp_paths.items()}

    # Création des topoplots pour les CSPs (Corrected & Uncorrected)
    for label, csp_model in csp_models.items():
        fig_csp, ax_csp = plt.subplots(nrows=2, ncols=6, figsize=(16, 8))
        for j in range(6):
            mne.viz.plot_topomap(csp_model.patterns_[j], epochs.info, axes=ax_csp[0, j])
            mne.viz.plot_topomap(csp_model.filters_[j], epochs.info, axes=ax_csp[1, j])
        report.add_figure(fig_csp, title=f"CSP Pattern & Filters - {label}", section=f"CSP {label}")
        plt.close(fig_csp)  # Éviter surcharge mémoire

    # Index CSP sélectionnés pour le sujet
    csp_subject_indices = {
        "Corrected": [csp_index_corrected[subject]],
        "Uncorrected": [csp_index_uncorrected[subject]]
    }

    # Création de l'objet info MNE pour un seul canal CSP
    info_mne = mne.create_info(ch_names=['CSP'], sfreq=epochs.info['sfreq'], ch_types=['eeg'])

    # Figure pour la comparaison des PSDs
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("PSD (dB µV²/Hz)")
    ax.set_title("Comparaison des PSD - CSP vs Laplacien (en dB µV²/Hz)")

    # Calcul des PSDs pour chaque CSP sélectionné (Corrected & Uncorrected)
    for label, csp_model in csp_models.items():
        for i, csp_idx in enumerate(csp_subject_indices[label]):
            selected_filter = csp_model.filters_[csp_idx] / np.sum(csp_model.filters_[csp_idx])  # Normalisation
            
            # Application du filtre CSP
            X_csp = np.einsum('ij,kjl->kil', selected_filter[np.newaxis, :], epochs['right_hand'].get_data())
            epochs_csp = mne.EpochsArray(X_csp, info_mne)

            # Calcul du PSD
            psd_csp, freqs_csp = epochs_csp.compute_psd(fmin=8, fmax=30).get_data(return_freqs=True)
            psd_csp_mean_db = 10 * np.log10(np.mean(psd_csp, axis=(0, 1)) * scaling ** 2)

            # Ajout au plot
            ax.plot(freqs_csp, psd_csp_mean_db, label=f"CSP {label} - {csp_idx}", linestyle="-", linewidth=2)

    # Calcul du PSD pour le Laplacien
    X = epochs['right_hand'].get_data()
    X_laplacien = 4 * X[:, 24, :] - X[:, 17, :] - X[:, 15, :] - X[:, 34, :] - X[:, 32, :]
    X_laplacien = X_laplacien[:, np.newaxis, :]
    
    epochs_laplacien = mne.EpochsArray(X_laplacien, info_mne)
    psd_laplacien, freqs_laplacien = epochs_laplacien.compute_psd(fmin=8, fmax=30).get_data(return_freqs=True)
    psd_laplacien_mean_db = 10 * np.log10(np.mean(psd_laplacien, axis=(0, 1)) * scaling ** 2)

    # Ajout du PSD du Laplacien au plot
    ax.plot(freqs_laplacien, psd_laplacien_mean_db, label="Laplacien", linestyle="-", linewidth=2)

    # Finalisation du plot
    ax.legend()
    ax.grid()
    report.add_figure(fig, title="CSP Comparaison", section="Comparaison")
    plt.close('all')  # Fermeture pour éviter surcharge mémoire

    # Sauvegarde du rapport
    report.save(os.path.join(save_path, f"CSP_Comparaison_Report_S{subject+1}.html"), open_browser=False, overwrite=True)
