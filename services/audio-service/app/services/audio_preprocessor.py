import torch
import torchaudio
import torchaudio.transforms as AT
import io
import warnings


class AudioPreprocessor:
    """
    Standardized audio preprocessing pipeline.
    Ensures input is 16kHz, mono, and exactly 64,600 samples.

    Short audio policy: TILE (repeat the waveform cyclically until 64,600 samples,
    then crop).  This matches the official AASIST data_utils.pad() function and the
    project-level config (configs/audio_preprocessing.yaml: short_audio_policy: "tile").

    Long audio policy: CROP from the start (crop to exactly 64,600 samples after the
    60-second hard cap).
    """

    EXPECTED_SR = 16000
    EXPECTED_SAMPLES = 64600
    MAX_DURATION_S = 60.0

    @classmethod
    def process_file(cls, filepath: str) -> torch.Tensor:
        try:
            waveform, sr = torchaudio.load(filepath)
        except Exception as e:
            raise ValueError(f"Failed to load audio file {filepath}: {e}")

        return cls._process_tensor(waveform, sr)

    @classmethod
    def process_bytes(cls, audio_bytes: bytes) -> torch.Tensor:
        try:
            waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        except Exception as e:
            raise ValueError(f"Failed to load audio bytes: {e}")

        return cls._process_tensor(waveform, sr)

    @classmethod
    def _process_tensor(cls, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16 kHz if necessary
        if sr != cls.EXPECTED_SR:
            resampler = AT.Resample(orig_freq=sr, new_freq=cls.EXPECTED_SR)
            waveform = resampler(waveform)

        # Hard cap at 60 seconds
        max_samples = int(cls.MAX_DURATION_S * cls.EXPECTED_SR)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
            warnings.warn("Audio exceeds maximum duration of 60 seconds; truncated.")

        # Work in 1D from here
        waveform = waveform.squeeze(0)  # (n_samples,)
        n_samples = waveform.shape[0]

        if n_samples < cls.EXPECTED_SAMPLES:
            # --- TILE-REPEAT (matches official AASIST data_utils.pad) ---
            # Compute how many full repetitions are needed and repeat, then crop.
            num_repeats = (cls.EXPECTED_SAMPLES // n_samples) + 1
            waveform = waveform.repeat(num_repeats)[: cls.EXPECTED_SAMPLES]
        elif n_samples > cls.EXPECTED_SAMPLES:
            # Crop to exact window from the start
            waveform = waveform[: cls.EXPECTED_SAMPLES]

        # Return shape [1, 64600] as expected by AASIST
        return waveform.unsqueeze(0)
