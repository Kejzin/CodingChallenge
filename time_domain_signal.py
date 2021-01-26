from __future__ import annotations

import wave

import numpy as np
from scipy import signal


class TimeDomainSignal:
    """"It's a structure to deal with simple operations with audio data."""

    def __init__(self, data: np.ndarray, fs: int):
        """      
        Args:
            data: audio data in np.int16 data type.
            fs: audio sampling frequency
        """
        if not data.dtype == np.int16:
            raise ValueError(f"Input array should have elements type np.int16. Actual type: {data.dtype}")
        self.data = data
        self.fs = fs

    @staticmethod
    def from_wav(filepath: str) -> TimeDomainSignal:
        """Read basic data from .wav file under filepath.

        Args:
            filepath: path to .wav file.

        Returns:
            TimeDomainSignal: audio data of .wav file.
        """
        with wave.open(filepath) as input_file:
            audio_frames_count = input_file.getnframes()

            wav_samples = input_file.readframes(audio_frames_count)
            wav_framerate = input_file.getparams().framerate

            wav_samples = np.frombuffer(wav_samples, dtype=np.int16)

        wav_data = TimeDomainSignal(wav_samples, wav_framerate)
        return wav_data

    @staticmethod
    def unit_impulse(sample_count: int, fs: int) -> TimeDomainSignal:
        """Create an unit impulse signal of defined number of samples.

        Resulting unit impulse contains 1 as a first sample and 0s for every other sample.
        Please notice that function is designed to simulate a Kronecker delta not a Dirac delta function.

        Args:
            sample_count: desired number of samples.
            fs: sampling frequency.

        Returns:
            TimeDomainSignal: audio data of created unit impulse.
        """
        samples = signal.unit_impulse(sample_count)
        samples = samples.astype(np.int16)

        impulse_data = TimeDomainSignal(samples, fs)
        return impulse_data

    @staticmethod
    def white_noise(sample_count: int, fs: int) -> TimeDomainSignal:
        """Create white noise signal with length of defined number of samples.

        Args:
            sample_count: desired number of samples.
            fs: sampling frequency.

        Returns:
            TimeDomainSignal: audio data of created white noise signal.
        """
        samples = np.random.random(size=sample_count)
        samples = np.multiply((samples - 0.5), np.iinfo(np.int16).max)
        samples = samples.astype(np.int16)

        noise_data = TimeDomainSignal(samples, fs)
        return noise_data

    @staticmethod
    def mix(*args: TimeDomainSignal) -> TimeDomainSignal:
        """Mix all audio files passed as an input.

        Args:
            *args: audio data as np.ndarray
        Returns:
            TimeDomainSignal: array containing mixed mono signal.
        """
        mixed_signal = np.empty(0)
        audio_signals = args
        audio_signals_data = []
        audio_signals_fs = []

        for audio_signal in audio_signals:
            audio_signals_data.append(audio_signal.data)
            audio_signals_fs.append(audio_signal.fs)

        if not np.all(np.array(audio_signals_fs) == audio_signals_fs[0]):
            raise ValueError(
                f"Sampling frequency isn't consistent in all signals. Audio signals fs are: {audio_signals_fs}")

        for audio_signal_data in audio_signals_data:

            length_dif = TimeDomainSignal._calc_length_diff(mixed_signal, audio_signal_data)

            if length_dif > 0:
                audio_signal_data = np.pad(audio_signal_data, (0, abs(length_dif)), 'constant', constant_values=0)
            elif length_dif < 0:
                mixed_signal = np.pad(mixed_signal, (0, abs(length_dif)), 'constant', constant_values=0)

            mixed_signal = np.add(mixed_signal, audio_signal_data)

        mixed_signal = TimeDomainSignal(mixed_signal.astype(np.int16), fs=44100)
        return mixed_signal

    @staticmethod
    def _calc_length_diff(array_1: np.ndarray, array_2: np.ndarray) -> int:
        """Function to calculate length difference between two arrays"""
        array_1_len = array_1.size
        array_2_len = array_2.size
        length_dif = array_1_len - array_2_len
        return length_dif

    def add(self, wav_data: TimeDomainSignal) -> TimeDomainSignal:
        """Add wav_data signal to self.data and return mixed mono file.

        Args:
            wav_data: audio data
        Returns:
            TimeDomainSignal: mixed signal audio data
        """
        audio_sum = self.mix(self, wav_data)
        return audio_sum

    def apply_gain(self, gain_db: int) -> TimeDomainSignal:
        """Apply defined gain in db to a signal stored in self.data.

        Args:
            gain_db: gain to apply
        Returns:
            TimeDomainSignal: gained audio data
        """
        linear_gain = np.power(20, (gain_db / 20.0))
        gained_data = self.data * linear_gain
        gained_data = gained_data.astype(np.int16)
        gained_audio = TimeDomainSignal(gained_data, self.fs)
        return gained_audio

    @staticmethod
    def rms(audio_data: TimeDomainSignal, db=False) -> np.int16:
        """Calculate RMS of a given audio signal.

        Args:
            audio_data: audio data
            db: bool value that indicate dbFS output when set to True

        Returns:
            np.int16: value of rms as a number value or in dB FS
        """
        data = audio_data.data
        powered = np.power(data, 2.0)
        mean = np.mean(powered)
        rms = np.sqrt(mean)

        if not isinstance(db, bool):
            raise ValueError(f"db parameter must be bool. Actual db parameter: {db}")

        if db is True:
            reference_rms = TimeDomainSignal.reference_0_dbfs_rms()
            rms = np.multiply(10, TimeDomainSignal._log_10_dealing_with_0(np.divide(rms, reference_rms)))

        rms = rms.astype(np.int16)
        return rms

    @staticmethod
    def _log_10_dealing_with_0(value: np.float) -> np.float:
        """Normal np.log10 but if value is 0 return dummy small value.

        Args:
            value: value to compute
        result: float
            computed value.
        """
        dummy_value = np.power(10, 0.00001)
        if value == 0:
            result = dummy_value
        else:
            result = np.log10(value)
        return result

    def normalize(self, target_dbfs: float, method: str) -> TimeDomainSignal:
        """Normalize an audio data stored in self.data.

        Please notice that 0 dB FS level is calculated according to IEC 61606-3 standard. Although there is possible to
        pass a positive value as a target_dbfs level, that may cause output signal to be distorted.

        Args:
            target_dbfs: target of loudness level in dB FS.
            method: method of normalization, can be "PEAK" or "RMS". Peak method base on peaks and normalize signal
            to reach maximum positive value in its peaks to desired dB FS level. RMS method is based on rms of the
            signal and normalize signal to have an RMS level of defined dB FS level.
        Returns:
            TimeDomainSignal: normalized audio data.
        """
        if method == "RMS":
            _0_db_rms_value = self.reference_0_dbfs_rms()
            actual_rms_value = self.rms(self.data)

            linear_rms_scaling_factor = np.power(10, (target_dbfs / 10.0))
            desired_rms_value = _0_db_rms_value * linear_rms_scaling_factor

            scaling_factor = desired_rms_value / actual_rms_value

            normalized_data = self.data * scaling_factor
        elif method == "PEAK":
            linear_rms_scaling_factor = np.power(20, (target_dbfs / 20.0))
            peak_value_0dbfs = np.iinfo(np.int16).max

            actual_peak_value = self.data.max()
            desired_peak_value = peak_value_0dbfs * linear_rms_scaling_factor

            scaling_factor = desired_peak_value / actual_peak_value

            normalized_data = self.data * scaling_factor
        else:
            raise ValueError(f"Method parameter must be either 'RMS' or 'PEAK', passed method is {method}")

        normalized_data = TimeDomainSignal(np.array(normalized_data, dtype=np.int16), self.fs)
        return normalized_data

    @staticmethod
    def reference_0_dbfs_rms():
        """Calculate 0 dB FS value for np.int16 data according to IEC 61606-3."""
        rms_0db_fs_value = np.divide(np.iinfo(np.int16).max, np.sqrt(2))
        return rms_0db_fs_value

    def convolve(self, wav_data: TimeDomainSignal, fast_convolution=False) -> TimeDomainSignal:
        """Perform convolution with audio data stored in self.data and wav_data.

        If fast_convolution is set to True convolution uses FFT method to convolve two signals.

        Args:
            wav_data: input audio data
            fast_convolution: Indicator value, if set to True indicates fast convolution.

        Returns:
            TimeDomainSignal: Numpy array with convolved data.
        """
        self._check_fs(wav_data)

        if not isinstance(fast_convolution, bool):
            raise ValueError(f"db parameter must be bool. Actual db parameter: {fast_convolution}")

        if fast_convolution is False:
            convolved = signal.convolve(self.data, wav_data.data)
        else:
            convolved = signal.fftconvolve(self.data, wav_data.data)
        convolved = TimeDomainSignal(convolved.astype(np.int16), self.fs)
        return convolved

    def _check_fs(self, wav_data):
        if not wav_data.fs == self.fs:
            raise (ValueError,
                   f"Sampling frequency must be consistent in both signals. Actual fs are: {wav_data} {self.fs}")
