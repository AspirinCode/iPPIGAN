#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import Chem
from rdkit.Chem import Descriptors
REMOVER = SaltRemover()
ORGANIC_ATOM_SET = set([5, 6, 7, 8, 9, 15, 16, 17, 35, 53])

df = pd.read_csv('chembl_29.csv')

def keep_largest_fragment(sml):
    """Function that returns the SMILES sequence of the largest fragment for a input
    SMILES sequnce.

    Args:
        sml: SMILES sequence.
    Returns:
        canonical SMILES sequnce of the largest fragment.
    """
    mol_frags = Chem.GetMolFrags(Chem.MolFromSmiles(sml), asMols=True)
    largest_mol = None
    largest_mol_size = 0
    for mol in mol_frags:
        size = mol.GetNumAtoms()
        if size > largest_mol_size:
            largest_mol = mol
            largest_mol_size = size
    return Chem.MolToSmiles(largest_mol)

def remove_salt_stereo(sml, remover):
    """Function that strips salts and removes stereochemistry information from a SMILES.
    Args:
        sml: SMILES sequence.
        remover: RDKit's SaltRemover object.
    Returns:
        canonical SMILES sequnce without salts and stereochemistry information.
    """
    try:
        sml = Chem.MolToSmiles(remover.StripMol(Chem.MolFromSmiles(sml),
                                                dontRemoveEverything=True),
                               isomericSmiles=False)
        if "." in sml:
            sml = keep_largest_fragment(sml)
    except:
        sml = np.float("nan")
    return(sml)

def organic_filter(sml):
    """Function that filters for organic molecules.
    Args:
        sml: SMILES sequence.
    Returns:
        True if sml can be interpreted by RDKit and is organic.
        False if sml cannot interpreted by RDKIT or is inorganic.
    """
    try:
        m = Chem.MolFromSmiles(sml)
        atom_num_list = [atom.GetAtomicNum() for atom in m.GetAtoms()]
        is_organic = (set(atom_num_list) <= ORGANIC_ATOM_SET)
        if is_organic:
            return True
        else:
            return False
    except:
        return False

def filter_smiles(sml):
    try:
        m = Chem.MolFromSmiles(sml)
        logp = Descriptors.MolLogP(m)
        mol_weight = Descriptors.MolWt(m)
        num_heavy_atoms = Descriptors.HeavyAtomCount(m)
        atom_num_list = [atom.GetAtomicNum() for atom in m.GetAtoms()]
        is_organic = set(atom_num_list) <= ORGANIC_ATOM_SET
        if ((logp > -5) & (logp < 7) & 
            (mol_weight > 300) & (mol_weight < 1100) &
            (num_heavy_atoms > 3) & (num_heavy_atoms < 50) &
            is_organic ):
            return Chem.MolToSmiles(m)
        else:
            return float('nan')
    except:
        return float('nan')

def preprocess_smiles(sml):
    """Function that preprocesses a SMILES string such that it is in the same format as
    the translation model was trained on. It removes salts and stereochemistry from the
    SMILES sequnce. If the sequnce correspond to an inorganic molecule or cannot be
    interpreted by RDKit nan is returned.

    Args:
        sml: SMILES sequence.
    Returns:
        preprocessd SMILES sequnces or nan.
    """
    new_sml = remove_salt_stereo(sml, REMOVER)
    new_sml = filter_smiles(new_sml)
    return new_sml

df["canonical_smiles"] = df["SMILES"].map(preprocess_smiles)
df = df.drop(["SMILES"], axis=1)
df = df.dropna(subset=["canonical_smiles"])
df = df.reset_index(drop=True)
df.head()

data = df[["chembl_id", "canonical_smiles"]]
data.to_csv("chembl29", index= False)