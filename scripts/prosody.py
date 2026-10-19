"""Intonation comparison helpers (tools env): how closely an output follows the original's melody.

The pitch register differs on purpose (the target sits ~2 semitones above the source character), so contours are compared
in semitones with each one's median removed:
  contour_corr  DTW-align the two takes on MFCCs, then Pearson-correlate the pitch at aligned frames
                (1 = same rise and fall, 0 = unrelated; long lines that go flat score low)
  range_ratio   pitch range (IQR, semitones) of the output / of the original (1 = same liveliness)
"""
import numpy as np

STEP = 0.01  # 10 ms frames for both pitch and MFCC
SR = 16000


def _resample(wav, sr):
    import librosa

    return wav if sr == SR else librosa.resample(wav, orig_sr=sr, target_sr=SR)


def semitones(wav, sr):
    """Pitch track in semitones re 100 Hz at 10 ms steps (NaN = unvoiced) and its frame times."""
    import parselmouth

    p = parselmouth.Sound(wav.astype(np.float64), sr).to_pitch_ac(time_step=STEP, pitch_floor=75, pitch_ceiling=900)
    f0 = p.selected_array["frequency"]
    st = np.where(f0 > 0, 12 * np.log2(np.maximum(f0, 1e-3) / 100.0), np.nan)
    return st, np.asarray(p.xs())


def median_st(wav, sr):
    st, _ = semitones(wav, sr)
    return float(np.nanmedian(st)) if np.isfinite(st).sum() >= 5 else float("nan")


def range_st(wav, sr):
    st, _ = semitones(wav, sr)
    st = st[np.isfinite(st)]
    return float(np.percentile(st, 75) - np.percentile(st, 25)) if len(st) >= 5 else float("nan")


def range_ratio(orig, sr_o, out, sr_out):
    a, b = range_st(orig, sr_o), range_st(out, sr_out)
    return b / a if a and np.isfinite(a) and a > 0.1 else float("nan")


def _mfcc(wav):
    import librosa

    m = librosa.feature.mfcc(y=wav, sr=SR, n_mfcc=20, n_fft=400, hop_length=int(SR * STEP))
    return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-6)  # CMVN: less timbre


def _at_frames(st, xs, n):
    """Pitch value for MFCC frame k (centre k*STEP s)."""
    idx = np.clip(np.round((np.arange(n) * STEP - xs[0]) / STEP).astype(int), 0, len(st) - 1)
    return st[idx]


def contour_corr(orig, sr_o, out, sr_out):
    import librosa

    a, b = _resample(orig, sr_o), _resample(out, sr_out)
    ma, mb = _mfcc(a), _mfcc(b)
    if ma.shape[1] < 10 or mb.shape[1] < 10:
        return float("nan")
    _, wp = librosa.sequence.dtw(X=ma, Y=mb, metric="cosine")
    sa, xa = semitones(a, SR)
    sb, xb = semitones(b, SR)
    if len(sa) < 5 or len(sb) < 5:
        return float("nan")
    pa, pb = _at_frames(sa, xa, ma.shape[1]), _at_frames(sb, xb, mb.shape[1])
    x, y = pa[wp[::-1, 0]], pb[wp[::-1, 1]]
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20:
        return float("nan")
    x, y = x[ok] - np.median(x[ok]), y[ok] - np.median(y[ok])
    if x.std() < 1e-6 or y.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
