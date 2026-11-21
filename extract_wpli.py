#!/usr/bin/env python3
"""
Resting-State EEG wPLI Feature Extraction Pipeline
Dataset: Atieh Schizophrenia EEG (ASEEG) Resting-State Dataset (Zenodo: 18029536)
Condition: Eyes-Open (EO)
"""

import os
import io
import re
import gc
import warnings
import h5py
import rarfile
import numpy as np
import pandas as pd
import mne
from mne_connectivity import spectral_connectivity_epochs

# Disable noisy MNE logging and runtime warnings during batch iterations
mne.set_log_level('ERROR')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------
# 1. Pipeline Constants & Specifications
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "eeg_data")
RAR_PATH = os.path.join(DATA_DIR, "EO.rar")
EXCEL_PATH = os.path.join(DATA_DIR, "participants_info.xlsx")
OUTPUT_CSV = os.path.join(BASE_DIR, "wpli_features.csv")

SAMPLING_RATE = 500  # Hz
EPOCH_DURATION = 2.0  # seconds
EPOCH_OVERLAP = 0.0  # non-overlapping

ELECTRODES_1020 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4', 'T5',
    'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'
]

# Frequency bands (Hz)
BANDS = {
    'Delta': (1.0, 4.0),
    'Theta': (4.0, 8.0),
    'Alpha': (8.0, 13.0),
    'Beta': (13.0, 30.0),
    'Gamma': (30.0, 45.0)
}
BAND_NAMES = list(BANDS.keys())
FMIN = [BANDS[b][0] for b in BAND_NAMES]
FMAX = [BANDS[b][1] for b in BAND_NAMES]

# Upper triangular electrode pairs (19 * 18 / 2 = 171 pairs)
ELECTRODE_PAIRS = []
for i in range(len(ELECTRODES_1020)):
    for j in range(i + 1, len(ELECTRODES_1020)):
        ELECTRODE_PAIRS.append((i, j, f"{ELECTRODES_1020[i]}-{ELECTRODES_1020[j]}"))

FEATURE_COLUMNS = []
for band in BAND_NAMES:
    for _, _, pair_name in ELECTRODE_PAIRS:
        FEATURE_COLUMNS.append(f"wPLI_{band}_{pair_name}")

ALL_COLUMNS = ['Patient_ID', 'Schizophrenia_Label'] + FEATURE_COLUMNS
assert len(ALL_COLUMNS) == 857, f"Expected 857 columns, got {len(ALL_COLUMNS)}"


def parse_patient_info(archive_filename):
    fname = os.path.basename(archive_filename)
    lower = fname.lower()
    match_sz = re.search(r'sz(\d+)', lower)
    match_norm = re.search(r'n(\d+)', lower)

    if 'sz' in archive_filename.lower() or match_sz:
        num = match_sz.group(1) if match_sz else "".join(filter(str.isdigit, fname))
        return f"SZ_{num}", 1
    elif 'normal' in archive_filename.lower() or match_norm:
        num = match_norm.group(1) if match_norm else "".join(filter(str.isdigit, fname))
        return f"HC_{num}", 0
    else:
        num = "".join(filter(str.isdigit, fname))
        return f"SUBJ_{num}", 0


def process_single_mat(data_bytes):
    with h5py.File(io.BytesIO(data_bytes), 'r') as f:
        key = list(f.keys())[0]
        data = np.array(f[key], dtype=np.float64)

    if data.shape[0] > data.shape[1] and data.shape[1] == len(ELECTRODES_1020):
        data = data.T  # Shape -> (19, n_samples)
    elif data.shape[0] != len(ELECTRODES_1020):
        raise ValueError(f"Unexpected data shape: {data.shape}")

    info = mne.create_info(ch_names=ELECTRODES_1020, sfreq=SAMPLING_RATE, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')

    # Preprocessing: Bandpass 1.0 - 45.0 Hz & Average Reference
    raw.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin', verbose=False)
    raw.set_eeg_reference('average', projection=False, verbose=False)

    # Epoching: 2.0s non-overlapping epochs
    epochs = mne.make_fixed_length_epochs(
        raw, duration=EPOCH_DURATION, overlap=EPOCH_OVERLAP, preload=True, verbose=False
    )

    # wPLI Multitaper
    con = spectral_connectivity_epochs(
        epochs,
        method='wpli',
        mode='multitaper',
        sfreq=SAMPLING_RATE,
        fmin=FMIN,
        fmax=FMAX,
        faverage=True,
        mt_adaptive=False,
        verbose=False
    )

    con_data = con.get_data(output='dense')

    patient_features = []
    for b_idx in range(len(BAND_NAMES)):
        mat_b = con_data[:, :, b_idx]
        for i, j, _ in ELECTRODE_PAIRS:
            val = mat_b[i, j]
            patient_features.append(float(val))

    del raw, epochs, con
    gc.collect()

    return patient_features


def main():
    print("=" * 60)
    print("ASEEG Resting-State EEG wPLI Feature Extraction")
    print(f"Archive: {RAR_PATH}")
    print(f"Output:  {OUTPUT_CSV}")
    print("=" * 60)

    if not os.path.exists(RAR_PATH):
        raise FileNotFoundError(f"Archive not found: {RAR_PATH}")

    processed_ids = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            if 'Patient_ID' in existing_df.columns:
                processed_ids = set(existing_df['Patient_ID'].tolist())
                print(f"Found existing output with {len(processed_ids)} processed subjects.")
        except Exception:
            pass

    rf = rarfile.RarFile(RAR_PATH)
    mat_entries = [name for name in rf.namelist() if name.endswith('.mat')]
    mat_entries.sort()

    print(f"Total .mat files in archive: {len(mat_entries)}")

    if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
        header_df = pd.DataFrame(columns=ALL_COLUMNS)
        header_df.to_csv(OUTPUT_CSV, index=False)

    for idx, entry in enumerate(mat_entries, start=1):
        p_id, label = parse_patient_info(entry)
        if p_id in processed_ids:
            print(f"[{idx}/{len(mat_entries)}] Skipping already processed: {p_id}")
            continue

        print(f"[{idx}/{len(mat_entries)}] Processing {entry} -> ID: {p_id}, Label: {label}...")
        try:
            data_bytes = rf.read(entry)
            feats = process_single_mat(data_bytes)

            row = [p_id, label] + feats
            row_df = pd.DataFrame([row], columns=ALL_COLUMNS)
            row_df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

            processed_ids.add(p_id)
            print(f"    ✓ Successfully recorded features for {p_id}")
        except Exception as e:
            print(f"    ✗ Error processing {entry}: {e}")

    print("\n" + "=" * 60)
    print("Pipeline Execution Completed!")
    final_df = pd.read_csv(OUTPUT_CSV)
    print(f"Final CSV Shape: {final_df.shape}")
    print(f"Missing Values:  {final_df.isnull().sum().sum()}")
    print("First 5 rows preview:")
    print(final_df.iloc[:5, :6])
    print("=" * 60)


if __name__ == "__main__":
    main()
