import pytest
import numpy as np
from pathlib import Path
from time_domain_signal import TimeDomainSignal


sin_full_scale = np.sin(np.linspace(0, np.pi, 124))

dummy_signal_data = np.array([-100, -100, 0, 100, 100], dtype=np.int16)

dummy_audio_signals_set = [TimeDomainSignal(np.array([-20, -20, 0, 20, 20], dtype=np.int16), fs=44100),
                           TimeDomainSignal(np.array([1, 1, 1], dtype=np.int16), fs=44100),
                           TimeDomainSignal(np.array([0, 10], dtype=np.int16), fs=44100)]


dummy_fs = 44100
dummy_sample_count = 124


audio_signals_different_fs = [TimeDomainSignal(np.array([-20, 1.20, 0, 10], dtype=np.int16), fs=44100),
                              TimeDomainSignal(np.array([1, 1, 1], dtype=np.int16), fs=48000)]


test_audio_signal_framerate = 44100  # That is known value of test audio file

test_file_filepath = str((Path(__file__).parent / "assets/white_noise_30s_44100kHz.wav").absolute())


@pytest.fixture()
def time_domain_signal_instance():
    audio_data = TimeDomainSignal(dummy_signal_data, dummy_fs)
    return audio_data


@pytest.fixture(params=[test_file_filepath])
def wav_data(request):
    return TimeDomainSignal.from_wav(request.param)