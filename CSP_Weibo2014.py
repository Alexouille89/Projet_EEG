# ========================
# MODULES
# ========================

import os
import joblib
import numpy as np
import mne
from mne.decoding import CSP
from moabb.datasets import Weibo2014
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from tools import bad_channels, generate_html_table, discrete_cmap

# Définition du répertoire de travail comme étant celui où se trouve ce script
os.chdir(os.path.dirname(os.path.realpath(__file__)))


# ========================
# PARAMÈTRES GLOBAUX
# ========================

# Définition du chemin de base pour sauvegarder les résultats
base_path = "C:/Users/cassandra.dumas/OneDrive - ICM/Documents/PHD/Data/PROJET_ESME/_WEIBO_/"

# Chargement du dataset EEG "Weibo2014"
dataset = Weibo2014()
subject_list = dataset.subject_list  # Liste des sujets disponibles

# Chargement des canaux à exclure (artefacts, mauvaise qualité)
channels_to_drop = bad_channels()

# Définition d'une palette de couleurs pour la visualisation des topographies
my_cmap = discrete_cmap(13, 'RdBu_r')

# Définition des paramètres de traitement
freqs = np.arange(0.5, 80, 0.5)  # Fréquences utilisées pour la TFR
n_cycles = freqs / 2  # Nombre de cycles pour l'ondelette de Morlet
low_freq, high_freq = 8, 30  # Filtrage en bande passante (rythme mu)
n_components = 6  # Nombre de composantes CSP à extraire
plot = True  # Activation des visualisations interactives
crop_params = dict(tmin=0, tmax=4, fmin=low_freq, fmax=high_freq)  # Paramètres de restriction des TFR

# Affichage du nombre de sujets disponibles
print(f"Nombre de sujets : {len(subject_list)}")
print(f"Identifiants des sujets : {subject_list}")


# ========================
# BOUCLE PRINCIPALE : TRAITEMENT DES SUJETS
# ========================

for subject in subject_list:  # Limité au premier sujet pour l'instant
    print(f"Traitement du sujet {subject}")

    # ------------------------
    # Initialisation du rapport HTML avec MNE
    # ------------------------
    report = mne.Report(title=f"CSP - Imagination Main droite VS Rest - Sujet {subject}")

    # Ajout des informations générales au rapport sous forme de tableau HTML
    report.add_html(
        generate_html_table({
            "Sujet": f"S{subject}",
            "Passe-Bande": f"{low_freq}-{high_freq} Hz",
            "Nombre de composantes CSP": n_components
        }),
        title="Processing Informations",
        section="Informations"
    )
    
    # ------------------------
    # Chargement et prétraitement des données EEG
    # ------------------------

    # Récupération des données EEG brutes pour le sujet
    data = dataset.get_data([subject])
    raw = data[subject]['0']['0']

    # Suppression des canaux à exclure
    raw.drop_channels(channels_to_drop, on_missing='ignore')

    # Application du montage standard "easycap-M1" pour localiser les électrodes
    raw.set_montage(mne.channels.make_standard_montage('easycap-M1'))

    # Filtrage passe-haut à 1 Hz pour supprimer les tendances lentes (sanity check)
    raw.filter(l_freq=1, h_freq=None)

    # Extraction des événements à partir des annotations
    events, _ = mne.events_from_annotations(raw)

    # Définition des classes d'intérêt
    event_id = {'right_hand': 6, 'rest': 5}

    # Segmentation des données en epochs
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-3, tmax=4, baseline=None, preload=True)

    # ------------------------
    # Sauvegarde des epochs non corrigés
    # ------------------------
    
    save_path = f"{base_path}_NO_CORRECTION_/_EPOCHS_/"
    os.makedirs(save_path, exist_ok=True)
    epochs.save(os.path.join(save_path, f"MI_epochs_S{subject}-epo.fif"), overwrite=True)

    
    # ------------------------
    # Correction ICA 
    # ------------------------

    # Création et ajustement du modèle ICA
    ica = mne.preprocessing.ICA(n_components=0.999, random_state=97)
    ica.fit(epochs)

    # Affichage des composantes indépendantes
    ica.plot_components()
    source = ica.get_sources(epochs)
    source.plot(block=True, picks='misc')

    # Sélection manuelle des artefacts ICA
    ica.exclude = [int(c) for c in input('Bad components : ').split() if c.isdigit()]

    # Sauvegarde du modèle ICA
    save_path = f"{base_path}_CORRECTION_/_ICA_/"
    os.makedirs(save_path, exist_ok=True)
    ica.save(os.path.join(save_path, f"S{subject}-ica.fif"), overwrite=True)

    # Application de la correction ICA aux epochs
    epochs_corrected = ica.apply(epochs.copy())

    # Sauvegarde des epochs corrigés
    save_path = f"{base_path}_CORRECTION_/_EPOCHS_/"
    os.makedirs(save_path, exist_ok=True)
    epochs_corrected.save(os.path.join(save_path, f"MI_epochs_S{subject}-epo.fif"), overwrite=True)
    
    report_ica = mne.Report(title="ICA example")
    report_ica.add_ica(
        ica=ica,
        title="ICA cleaning",
        picks=ica.exclude,  # Affichage des EOGs composantes
        inst=raw,
        n_jobs=None,
    )
    save_path = f"{base_path}_REPORT_/_ICA_"
    os.makedirs(save_path, exist_ok=True)
    report_ica.save(os.path.join(save_path, f"ICA_Report_S{subject}.html"), open_browser=False, overwrite=True)
    
    # ------------------------
    # Filtrage en bande passante (8-30 Hz)
    # ------------------------
    
    epochs.filter(l_freq=low_freq, h_freq=high_freq)
    epochs_corrected.filter(l_freq=low_freq, h_freq=high_freq)

    # ------------------------
    # Analyse TFR et visualisation des topographies
    # ------------------------

    for correction, epoch_data, folder in zip(
        ["NO_CORRECTION", "CORRECTION"],
        [epochs, epochs_corrected],
        ["_NO_CORRECTION_", "_CORRECTION_"]
    ):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

        for i, condition in enumerate(["right_hand", "rest"]):
            # Calcul de la TFR avec ondelettes de Morlet
            tfr = epoch_data[condition].compute_tfr("morlet", freqs=freqs, n_cycles=n_cycles, n_jobs=15, average=True)
            tfr.apply_baseline(baseline=(-3, -1), mode="logratio")
            tfr.crop(**crop_params)

            # Calcul des topographies moyennes
            topographie = np.mean(tfr.data, axis=(1, 2))

            # Normalisation des couleurs pour garantir une échelle homogène
            if condition == "right_hand":
                vmax = np.max(abs(topographie))
                vmin = -vmax

            # Sauvegarde et affichage
            save_path = f"{base_path}{folder}/_TOPOGRAPHIES_/S{subject}"
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, f"Topographies_{condition}.npy"), topographie)
            mne.viz.plot_topomap(topographie, raw.info, vlim=(vmin, vmax), cmap=my_cmap, axes=axs[i], show=False)
            axs[i].set_title(f"Topographie {condition}")

        report.add_figure(fig, title="Topographie - CSP Data", section=correction)
        
        
    # ------------------------
    # APPLICATION DU CSP
    # ------------------------

    for correction, epoch_data, folder in zip(
        ["NO_CORRECTION", "CORRECTION"],
        [epochs, epochs_corrected],
        ["_NO_CORRECTION_", "_CORRECTION_"]
    ):
        # Suppression des périodes avant le début de la tâche
        epoch_data.crop(tmin=0)

        # Extraction des données et étiquettes
        X, y = epoch_data.get_data(), epoch_data.events[:, -1]

        # Application du CSP
        csp = CSP(n_components=n_components)
        X_csp = csp.fit_transform(X, y)

        # Sauvegarde du modèle CSP
        save_path = f"{base_path}{folder}/_CSP_/"
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(csp, os.path.join(save_path, f"CSP_{subject}.pkl"), protocol=3)

        # Visualisation des patterns et filtres CSP
        fig, ax = plt.subplots(nrows=2, ncols=n_components, figsize=(16, 8))
        for j in range(n_components):
            mne.viz.plot_topomap(csp.patterns_[j], raw.info, axes=ax[0, j])
            mne.viz.plot_topomap(csp.filters_[j], raw.info, axes=ax[1, j])

        report.add_figure(fig, title="CSP Pattern & Filters", section=correction)
        

    # ------------------------
    # Sauvegarde du rapport HTML
    # ------------------------
    
    save_path = f"{base_path}_REPORT_/_CSP_"
    os.makedirs(save_path, exist_ok=True)
    report.save(os.path.join(save_path, f"CSP_Report_S{subject}.html"), open_browser=False, overwrite=True)

    plt.close("all")  # Fermeture des figures pour éviter une surcharge mémoire

print("Traitement terminé pour tous les sujets.")


