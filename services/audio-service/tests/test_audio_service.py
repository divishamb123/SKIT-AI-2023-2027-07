import io
import torch
import torchaudio
from app.services.audio_preprocessor import AudioPreprocessor


def test_audio_preprocessor_loads():
    assert hasattr(AudioPreprocessor, "process_bytes")


def test_audio_preprocessor_shape():
    # Create a dummy 1-second 16kHz mono audio waveform
    sample_rate = 16000
    duration = 1.0
    waveform = torch.zeros(1, int(sample_rate * duration))

    # Save to a bytes buffer
    buf = io.BytesIO()
    torchaudio.save(buf, waveform, sample_rate, format="wav")
    audio_bytes = buf.getvalue()

    # Process it
    processed = AudioPreprocessor.process_bytes(audio_bytes)

    # Check shape: should be (1, 64600)
    assert processed.shape == (1, 64600)


def test_audio_preprocessor_stereo_resample():
    # Create a dummy 2-second 44.1kHz stereo audio waveform
    sample_rate = 44100
    duration = 2.0
    waveform = torch.zeros(2, int(sample_rate * duration))

    buf = io.BytesIO()
    torchaudio.save(buf, waveform, sample_rate, format="wav")
    audio_bytes = buf.getvalue()

    processed = AudioPreprocessor.process_bytes(audio_bytes)

    # Check shape: should be (1, 64600) despite being stereo and 44.1kHz
    assert processed.shape == (1, 64600)
