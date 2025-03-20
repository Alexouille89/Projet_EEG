# Modules
# -------
import os
import numpy as np

import matplotlib.pyplot as plt


# Functions
# ---------
def get_channels():
    """
    Retourne une liste des noms de canaux EEG utilisés dans une configuration standard.
    
    Cette liste inclut les noms des électrodes selon la nomenclature 10-10 
    ou 10-20 utilisée en électroencéphalographie.
    """

    channels = [
    'Fp1', 'Fpz', 'Fp2',
    'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2',
    'Iz'
]
    
    return channels


def generate_html_table(data):
    """
    Génère un tableau HTML stylisé à partir d'un dictionnaire de données.
    
    Parameters:
        data (dict): Dictionnaire contenant des paires clé-valeur à afficher dans le tableau.
        
    Returns:
        str: Code HTML représentant le tableau.
    """
    
    rows = []
    for i, (header, value) in enumerate(data.items()):
        background_color = "#f2f2f2" if i % 2 == 0 else "#ffffff"
        rows.append(f"""
        <tr style="background-color: {background_color};">
            <td style="border: 1px solid black; text-align: center; padding: 10px;"><b>{header}</b></td>
            <td style="border: 1px solid black; text-align: center; padding: 10px;">{value}</td>
        </tr>
        """)
    return f"""
    <table style="margin-left: auto; margin-right: auto; width: 60%; border-collapse: collapse; border: 1px solid black;">
        {''.join(rows)}
    </table>
    """
    
    
def discrete_cmap(N, base_cmap=None):
    """
    Crée une colormap discrète (divisée en N intervalles) à partir d'une colormap existante.
    
    Parameters:
        N (int): Nombre de couleurs distinctes souhaitées.
        base_cmap (str or Colormap, optional): Colormap de base à utiliser (nom ou instance).
    
    Returns:
        Colormap: Une nouvelle colormap discrète.
    """

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def bad_channels():
    """
    Retourne une liste des canaux à exclure lors du traitement EEG.
    
    Returns:
        list: Liste des noms de canaux à exclure.
    """
    channels_to_drop = ['HEO', 'VEO', 'CB1', 'CB2', 'STIM014', 'P5', 'PO7', 'PO5', 'PO6', 'FCz']
    
    return channels_to_drop
