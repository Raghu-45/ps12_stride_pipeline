import os
import json
import warnings

import joblib
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, lfilter
from scipy.stats import kurtosis

# ----------------- CONFIG -----------------
AUDIO_DIR = "/Users/haleshkt/Desktop/UDA Project/20251103_PS12"
OUT_JSON  = "/Users/haleshkt/Desktop/UDA Project/2025_11_03_EXPERIQS_PVT_LTD_PS12_STRIDErevised.json"

STAGE1_MODEL_PATH   = "/Users/haleshkt/Desktop/UDA Project/stage1_xgb_model.pkl"
STAGE1_SCALER_PATH  = "/Users/haleshkt/Desktop/UDA Project/stage1_scaler.pkl"
STAGE2_MODEL_PATH   = "/Users/haleshkt/Desktop/UDA Project/stage2_xgb_model.pkl"
STAGE2_SCALER_PATH  = "/Users/haleshkt/Desktop/UDA Project/stage2_scaler.pkl"
STAGE2_ENCODER_PATH = "/Users/haleshkt/Desktop/UDA Project/stage2_label_encoder.pkl"

SR = 32000          # target sample rate
WIN_LEN = 5.0       # window length (seconds)
HOP_LEN = 2.0       # stride / hop (seconds)
RMS_TARGET = 0.1    # target RMS for normalization

warnings.filterwarnings("ignore")

# ----------------- LOAD MODELS -----------------
stage1_model   = joblib.load(STAGE1_MODEL_PATH)
stage1_scaler  = joblib.load(STAGE1_SCALER_PATH)
stage2_model   = joblib.load(STAGE2_MODEL_PATH)
stage2_scaler  = joblib.load(STAGE2_SCALER_PATH)
label_encoder  = joblib.load(STAGE2_ENCODER_PATH)

print("Models and scalers loaded.")

# ----------------- UTILITIES -----------------
def highpass_filter(y, sr, cutoff=25, order=4):
    """Remove very low frequency rumble below `cutoff` Hz."""
    b, a = butter(order, cutoff / (0.5 * sr), btype="high")
    return lfilter(b, a, y)

def rms_normalize(y, target=RMS_TARGET):
    """Normalize audio to target RMS, then clip to [-1, 1]."""
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 0:
        y = y * (target / rms)
    return np.clip(y, -1.0, 1.0)

def extract_features(y, sr):
    """
    IMPORTANT: feature names MUST match training time exactly.
    This matches your original training code.
    """
    feats = {}

    # time-domain
    feats["rms_energy"] = float(np.mean(librosa.feature.rms(y=y)))
    feats["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # spectral features
    S, _ = librosa.magphase(librosa.stft(y))
    feats["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    feats["spectral_bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
    feats["spectral_rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)))
    feats["spectral_flatness"] = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    feats["flux"] = float(np.mean(librosa.onset.onset_strength(y=y, sr=sr)))

    # frequency-domain features
    mag = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)

    if np.sum(mag) > 0:
        peak_freq = freqs[np.argmax(mag)]
        feats["peak_freq_mean"] = float(peak_freq)
        feats["peak_freq_std"]  = float(np.std(mag))
    else:
        feats["peak_freq_mean"] = 0.0
        feats["peak_freq_std"]  = 0.0

    feats["spectral_kurtosis"] = float(kurtosis(mag))

    # MFCCs (20 x mean/std => 40 features)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(mfcc.shape[0]):
        feats[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i}_std"]  = float(np.std(mfcc[i]))

    return feats

def sliding_windows(total_dur, win_len=5.0, hop_len=2.0):
    """Return list of (start, end) for overlapping windows."""
    windows = []
    t = 0.0
    while t + win_len <= total_dur:
        windows.append((t, t + win_len))
        t += hop_len
    return windows

def majority_smoothing_binary(labels, window_size=3):
    """
    Simple majority smoothing on binary labels [0/1] in window domain.
    labels: list/array of 0/1, length N
    """
    labels = np.array(labels, dtype=int)
    N = len(labels)
    if N < 3:
        return labels

    smoothed = labels.copy()
    for i in range(1, N - 1):
        window = labels[i - 1 : i + 2]
        if np.sum(window) >= 2:
            smoothed[i] = 1
        else:
            smoothed[i] = 0
    return smoothed

def smooth_multiclass(preds):
    """
    Simple 3-window majority smoothing for class indices (0..3).
    """
    preds = np.array(preds, dtype=int)
    N = len(preds)
    if N < 3:
        return preds

    out = preds.copy()
    for i in range(1, N - 1):
        window = preds[i - 1 : i + 2]
        counts = np.bincount(window)
        if counts.max() >= 2:
            out[i] = np.argmax(counts)
    return out

def make_serializable(obj):
    """Helper for JSON dumping when numpy types appear."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

# ----------------- MAIN INFERENCE -----------------
files = sorted([f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav")])[:5]

categories = [
    {"id": 1, "name": "vessel"},
    {"id": 2, "name": "marine_animal"},
    {"id": 3, "name": "natural_sound"},
    {"id": 4, "name": "other_anthropogenic"},
]

audios = []
annotations = []
ann_id = 1

for idx, fname in enumerate(tqdm(files, desc="Processing audio files")):
    fpath = os.path.join(AUDIO_DIR, fname)

    # 1. load and preprocess audio
    y, sr = librosa.load(fpath, sr=SR, mono=True)
    y = highpass_filter(y, sr)
    y = rms_normalize(y)
    total_dur = librosa.get_duration(y=y, sr=sr)

    audio_id = idx + 1
    audios.append(
        {
            "id": audio_id,
            "file_name": fname,
            "file_path": f"{os.path.basename(AUDIO_DIR)}/{fname}",
            "duration": round(total_dur, 2),
        }
    )

    # 2. build overlapping windows
    win_list = sliding_windows(total_dur, win_len=WIN_LEN, hop_len=HOP_LEN)
    if not win_list:
        continue

    # dynamic threshold based on file duration
    if total_dur <= 60:
        THRESH = 0.50
    elif total_dur <= 120:
        THRESH = 0.45
    else:
        THRESH = 0.40

    # ---------- Stage 1: BG vs Event (window level) ----------
    win_probs = []
    win_labels = []   # 0 = bg, 1 = event

    for (s, e) in win_list:
        start_idx = int(s * sr)
        end_idx = int(e * sr)
        y_chunk = y[start_idx:end_idx]

        feats = extract_features(y_chunk, sr)
        X = pd.DataFrame([feats])   # DataFrame with correct feature names

        # scaler will align by column names
        X_scaled = stage1_scaler.transform(X)
        prob_evt = stage1_model.predict_proba(X_scaled)[0, 1]

        win_probs.append(prob_evt)
        win_labels.append(1 if prob_evt >= THRESH else 0)

    win_probs = np.array(win_probs, dtype=float)
    win_labels = np.array(win_labels, dtype=int)

    # smoothing over windows (majority)
    win_labels_smooth = majority_smoothing_binary(win_labels, window_size=3)

    # group consecutive windows with same label (for analysis if needed)
    segments_stage1 = []
    cur_label = win_labels_smooth[0]
    seg_start_idx = 0

    for i in range(1, len(win_labels_smooth)):
        if win_labels_smooth[i] != cur_label:
            s_win = win_list[seg_start_idx][0]
            e_win = win_list[i - 1][1]
            segments_stage1.append((cur_label, s_win, e_win))
            cur_label = win_labels_smooth[i]
            seg_start_idx = i

    # last segment
    s_win = win_list[seg_start_idx][0]
    e_win = win_list[-1][1]
    segments_stage1.append((cur_label, s_win, e_win))

    # collect all windows that are final "event" according to smoothed labels
    event_windows = []
    event_probs   = []

    for i, (s, e) in enumerate(win_list):
        if win_labels_smooth[i] == 1:
            event_windows.append((s, e))
            event_probs.append(win_probs[i])

    if not event_windows:
        # this file predicted as all BG, so continue without annotations
        continue

    # ---------- Stage 2: 4-class classification on event windows ----------
    cls_ids = []
    cls_scores = []
    times = []

    for (s, e) in event_windows:
        start_idx = int(s * sr)
        end_idx = int(e * sr)
        y_chunk = y[start_idx:end_idx]

        feats = extract_features(y_chunk, sr)
        X = pd.DataFrame([feats])

        X_scaled = stage2_scaler.transform(X)
        probs = stage2_model.predict_proba(X_scaled)[0]

        c_id = int(np.argmax(probs))
        cls_ids.append(c_id)
        cls_scores.append(float(np.max(probs)))
        times.append((s, e))

    # smooth class ids across neighboring event windows
    cls_ids = smooth_multiclass(cls_ids)

    # --------- FIXED MERGING: merge overlapping windows with same class ---------
    cur_cls = cls_ids[0]
    cur_start, cur_end = times[0]
    cur_scores = [cls_scores[0]]

    merged_events = []

    for i in range(1, len(cls_ids)):
        s, e = times[i]

        # if same class AND the next window starts before or at the current end,
        # treat as one continuous event and extend the end.
        if cls_ids[i] == cur_cls and s <= cur_end + 1e-6:
            cur_end = max(cur_end, e)
            cur_scores.append(cls_scores[i])
        else:
            merged_events.append((cur_cls, cur_start, cur_end, float(np.mean(cur_scores))))
            cur_cls = cls_ids[i]
            cur_start, cur_end = s, e
            cur_scores = [cls_scores[i]]

    merged_events.append((cur_cls, cur_start, cur_end, float(np.mean(cur_scores))))

    # build annotations
    for (cls, s, e, sc) in merged_events:
        cat_id = cls + 1  # 0-based to 1-based
        annotations.append(
            {
                "id": ann_id,
                "audio_id": audio_id,
                "category_id": int(cat_id),
                "start_time": round(float(s), 1),
                "end_time": round(float(e), 1),
                "duration": round(float(e - s), 1),
                "score": round(float(sc), 3),
            }
        )
        ann_id += 1

# ----------------- WRITE JSON -----------------
result = {
    "info": {
        "description": "Grand Challenge UDA",
        "version": "1.0",
        "year": 2025,
    },
    "audios": audios,
    "categories": categories,
    "annotations": annotations,
}

with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2, default=make_serializable)

print("DONE! JSON saved:", OUT_JSON)
print("Audios:", len(audios), "| Annotations:", len(annotations))
