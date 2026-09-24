import io
import asyncio
import numpy as np
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchaudio


class SpectrogramVisualizer:
    SR = 16_000

    async def generate(self, audio_bytes: bytes, ai_probability: float) -> bytes:
        return await asyncio.to_thread(self._gen_sync, audio_bytes, ai_probability)

    def _gen_sync(self, audio_bytes: bytes, ai_probability: float) -> bytes:
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        y = waveform.numpy().squeeze()
        if sr != self.SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.SR)
        y = y[: self.SR * 10]
        mel = librosa.feature.melspectrogram(y=y, sr=self.SR, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), facecolor="#1a1a2e")
        fig.suptitle(
            f"Audio Forensics | AI Probability: {ai_probability*100:.1f}%",
            color="white",
            fontsize=13,
            fontweight="bold",
        )
        for ax in (ax1, ax2):
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")
        librosa.display.waveshow(y, sr=self.SR, ax=ax1, color="#e94560")
        ax1.set_title("Waveform", color="white")
        img = librosa.display.specshow(
            mel_db,
            sr=self.SR,
            x_axis="time",
            y_axis="mel",
            fmax=8000,
            ax=ax2,
            cmap="magma",
        )
        fig.colorbar(img, ax=ax2, format="%+2.0f dB")
        ax2.set_title("Mel-Spectrogram (dB)", color="white")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(
            buf, format="png", dpi=120, bbox_inches="tight", facecolor="#1a1a2e"
        )
        plt.close(fig)
        return buf.getvalue()
