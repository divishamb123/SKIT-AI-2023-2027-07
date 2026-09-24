import pytest
import os
from unittest.mock import MagicMock
from app.services.queue_consumer import QueueConsumer
from app.services.audio_preprocessor import AudioPreprocessor

def test_audio_preprocessor_loads():
    assert hasattr(AudioPreprocessor, "process_bytes")

