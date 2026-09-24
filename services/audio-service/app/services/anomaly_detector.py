import io
import librosa
import numpy as np


def detect_audio_anomalies(audio_bytes: bytes, ai_probability: float) -> list[str]:
    flags = []
    if ai_probability < 0.50:
        return flags
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=80, fmax=400)
    if voiced.sum() > 10:
        f0_std = np.nanstd(f0[voiced])
        if f0_std < 8.0:
            flags.append("unnatural_f0_continuity")
    rms = librosa.feature.rms(y=y)[0]
    silences = np.where(rms < rms.mean() * 0.1)[0]
    if len(silences) > 0:
        # spec variable was removed because it was unused
        harmonic_ratio = librosa.effects.harmonic(y).var() / (y.var() + 1e-8)
        if harmonic_ratio > 0.85:
            flags.append("harmonic_discontinuity")
    reverb_proxy = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].std()
    if reverb_proxy < 500:
        flags.append("no_room_acoustics_detected")
    return flags[:3]
