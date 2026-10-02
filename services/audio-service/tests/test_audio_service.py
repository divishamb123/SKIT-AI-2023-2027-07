from app.services.audio_preprocessor import AudioPreprocessor


def test_audio_preprocessor_loads():
    assert hasattr(AudioPreprocessor, "process_bytes")
