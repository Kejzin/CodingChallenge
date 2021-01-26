import pytest
import numpy

from time_domain_signal import TimeDomainSignal
from tests.conftest import dummy_sample_count, dummy_fs, test_audio_signal_framerate, dummy_audio_signals_set,\
    audio_signals_different_fs


def test_bad_datatype():
    with pytest.raises(ValueError):
        TimeDomainSignal(numpy.array([1, 2, 3], dtype=float), fs=40)


def test_from_wav_data(wav_data):
    assert isinstance(wav_data.data, numpy.ndarray)


def test_from_wav_framerate(wav_data):
    assert wav_data.fs == test_audio_signal_framerate


@pytest.mark.parametrize('sample_count', [dummy_sample_count])
@pytest.mark.parametrize('fs', [dummy_fs])
def test_unit_impulse_data(sample_count, fs):
    impulse_data = TimeDomainSignal.unit_impulse(sample_count, fs)
    signal_samples = impulse_data.data
    assert isinstance(signal_samples, numpy.ndarray)


@pytest.mark.parametrize('sample_count', [dummy_sample_count])
@pytest.mark.parametrize('fs', [dummy_fs])
def test_unit_impulse_length(sample_count, fs):
    impulse_data = TimeDomainSignal.unit_impulse(sample_count, fs)
    signal_samples = impulse_data.data
    assert len(signal_samples) == sample_count


@pytest.mark.parametrize('sample_count', [dummy_sample_count])
@pytest.mark.parametrize('fs', [dummy_fs])
def test_unit_impulse_fs(sample_count, fs):
    impulse_data = TimeDomainSignal.unit_impulse(sample_count, fs)
    signal_fs = impulse_data.fs
    assert signal_fs == fs


@pytest.mark.parametrize('sample_count', [dummy_sample_count])
@pytest.mark.parametrize('fs', [dummy_fs])
def test_unit_impulse_1(sample_count, fs):
    impulse_data = TimeDomainSignal.unit_impulse(sample_count, fs)
    signal_samples = impulse_data.data
    assert signal_samples[0] == 1


@pytest.mark.parametrize('sample_count', [dummy_sample_count])
@pytest.mark.parametrize('fs', [dummy_fs])
def test_unit_impulse_0s(sample_count, fs):
    impulse_data = TimeDomainSignal.unit_impulse(sample_count, fs)
    signal_samples = impulse_data.data
    assert numpy.all(signal_samples[1:] == 0)


@pytest.mark.parametrize('sample_count', [dummy_sample_count])
@pytest.mark.parametrize('fs', [dummy_fs])
def test_white_noise_samples_type(sample_count, fs):
    noise = TimeDomainSignal.white_noise(sample_count, fs)
    signal_samples = noise.data
    assert isinstance(signal_samples, numpy.ndarray)


@pytest.mark.parametrize('sample_count', [dummy_sample_count])
@pytest.mark.parametrize('fs', [dummy_fs])
def test_white_noise_samples_length(sample_count, fs):
    noise = TimeDomainSignal.white_noise(sample_count, fs)
    signal_samples = noise.data
    assert len(signal_samples) == sample_count


@pytest.mark.parametrize('sample_count', [dummy_sample_count])
@pytest.mark.parametrize('fs', [dummy_fs])
def test_white_noise_fs(sample_count, fs):
    noise = TimeDomainSignal.white_noise(sample_count, fs)
    signal_fs = noise.fs
    assert signal_fs == fs


@pytest.mark.parametrize('audio_signal_1', [dummy_audio_signals_set[0]])
@pytest.mark.parametrize('audio_signal_2', [dummy_audio_signals_set[1]])
@pytest.mark.parametrize('audio_signal_3', [dummy_audio_signals_set[2]])
def test_mix(audio_signal_1, audio_signal_2, audio_signal_3):
    mixed_signal = TimeDomainSignal.mix(audio_signal_1, audio_signal_2, audio_signal_3)
    assert numpy.array_equiv(mixed_signal.data,
                             numpy.array([-19, -9, 1, 20, 20], dtype=numpy.int16))


@pytest.mark.parametrize('audio_signal_1', [audio_signals_different_fs[0]])
@pytest.mark.parametrize('audio_signal_2', [audio_signals_different_fs[1]])
def test_mix_different_fs(audio_signal_1, audio_signal_2):
    with pytest.raises(ValueError):
        TimeDomainSignal.mix(audio_signal_1, audio_signal_2)


@pytest.mark.parametrize('audio_data', [dummy_audio_signals_set[0]])
def test_add(time_domain_signal_instance, audio_data):
    added_signals = time_domain_signal_instance.add(audio_data)
    assert numpy.array_equiv(added_signals.data,
                             numpy.array([-120, -120, 0, 120, 120], dtype=numpy.int16))


@pytest.mark.parametrize('gain_db', [-10])
def test_apply_gain(time_domain_signal_instance, gain_db):
    gained_audio = time_domain_signal_instance.apply_gain(gain_db)
    assert not numpy.array_equiv(gained_audio.data,
                                 numpy.array([-2603, -2603, 0, 2603, 2603], dtype=numpy.int16))


def test_apply_gain_0_db(time_domain_signal_instance):
    gained_audio = time_domain_signal_instance.apply_gain(0)
    assert numpy.array_equiv(gained_audio.data, time_domain_signal_instance.data)


@pytest.mark.parametrize('audio_data', [dummy_audio_signals_set[0]])
def test_rms(time_domain_signal_instance, audio_data):
    rms = time_domain_signal_instance.rms(audio_data)
    assert rms == 17


@pytest.mark.parametrize('audio_data', [dummy_audio_signals_set[0]])
@pytest.mark.parametrize('db', [True])
def test_rms_db_True(time_domain_signal_instance, audio_data, db):
    rms = time_domain_signal_instance.rms(audio_data, db)
    assert rms == -31


@pytest.mark.parametrize('audio_data', [dummy_audio_signals_set[0]])
@pytest.mark.parametrize('db', [False])
def test_rms_db_False(time_domain_signal_instance, audio_data, db):
    rms = time_domain_signal_instance.rms(audio_data, db)
    assert rms == 17


@pytest.mark.parametrize('audio_data', [dummy_audio_signals_set[0]])
@pytest.mark.parametrize('db', ['no'])
def test_rms_db_False(time_domain_signal_instance, audio_data, db):
    with pytest.raises(ValueError):
        time_domain_signal_instance.rms(audio_data, db)


@pytest.mark.parametrize('method', ['RMS'])
@pytest.mark.parametrize('target_dbfs', [-10])
def test_normalize_rms(time_domain_signal_instance, target_dbfs, method):
    normalized_data = time_domain_signal_instance.normalize(target_dbfs, method)
    assert numpy.array_equiv(normalized_data.data,
                             numpy.array([-2603, -2603, 0, 2603, 2603], dtype=numpy.int16))


@pytest.mark.parametrize('method', ['PEAK'])
@pytest.mark.parametrize('target_dbfs', [-10])
def test_normalize_peak(time_domain_signal_instance, target_dbfs, method):
    normalized_data = time_domain_signal_instance.normalize(target_dbfs, method)
    assert numpy.array_equiv(normalized_data.data,
                             numpy.array([-7326, -7326, 0, 7326, 7326], dtype=numpy.int16))


@pytest.mark.parametrize('method', ['Bad method'])
def test_normalize_bad_method(time_domain_signal_instance, method):
    with pytest.raises(ValueError):
        time_domain_signal_instance.normalize(-10, method)


def test_0_dbfs_RMS_reference(time_domain_signal_instance):
    value = time_domain_signal_instance.reference_0_dbfs_rms()
    return value.astype(numpy.int16) == 23169


@pytest.mark.parametrize('audio_data', [dummy_audio_signals_set[0]])
def test_convolved(time_domain_signal_instance, audio_data):
    convolved = time_domain_signal_instance.convolve(audio_data)
    assert numpy.array_equiv(convolved.data,
                             numpy.array([2000, 4000, 2000, -4000, -8000, -4000, 2000, 4000, 2000], dtype=numpy.int16))


@pytest.mark.parametrize('audio_data', [dummy_audio_signals_set[0]])
@pytest.mark.parametrize('fast_convolution', ["yes"])
def test_convolved_fast_not_bool(time_domain_signal_instance, audio_data, fast_convolution):
    with pytest.raises(ValueError):
        time_domain_signal_instance.convolve(audio_data, fast_convolution)


@pytest.mark.parametrize('audio_data', [dummy_audio_signals_set[0]])
@pytest.mark.parametrize('fast_convolution', [True])
def test_convolved_fast(time_domain_signal_instance, audio_data, fast_convolution):
    convolved = time_domain_signal_instance.convolve(audio_data, fast_convolution)
    convolved_direct = time_domain_signal_instance.convolve(audio_data)
    assert numpy.all(numpy.isclose(convolved.data,
                     convolved_direct.data,
                     atol=1))
