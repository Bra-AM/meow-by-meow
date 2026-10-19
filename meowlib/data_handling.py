'''Scikit-learn compatible transformers that turn raw cat audio into model input.

Every class here follows the scikit-learn ``fit`` / ``transform`` interface, so
the steps can be chained with :class:`sklearn.pipeline.Pipeline`. The inference
chain used by the Streamlit app reproduces, exactly, the features the released
models were trained on (see src/data_handling/specgram_vectorization.ipynb)::

    audio file ──► (sample, rate)             FFMPEGLoader / WavLoader
               ──► resampled to 8 kHz         Resampler
               ──► 2.5 s overlapping windows  WaveformWindowSplitter
               ──► (n, 128, 67) mel dB        TrainingSpecgramTransformer
               ──► reshaped for the model     ReshapeTransformer

The chain SpecgramTransformer -> RollingWindowSplitter -> PadTransformer is
an alternative representation used in the exploratory notebooks.

Conventions used throughout:
    * A *sample* is a 1D float array of audio amplitudes in [-1, 1].
    * A *specgram* is a 2D array of shape ``(n_mel_bins, n_time_frames)``.
'''
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment
from scipy.io import wavfile
from sklearn.base import BaseEstimator, TransformerMixin
import tqdm

# Works both as part of the package (``from meowlib import data_handling``)
# and when meowlib/ itself is on sys.path (``import data_handling``), as in
# some of the notebooks.
try:
    from . import utils
except ImportError:
    import utils

# Value written into padded regions of a specgram. Specgrams are computed in dB
# relative to their own minimum (``ref=np.min``), so every real value is >= 0
# and -1 can never be confused with genuine signal.
PAD_VALUE = -1


class WavLoader(TransformerMixin, BaseEstimator):
    '''Load ``.wav`` files into ``(sample, rate)`` tuples.

    Only handles uncompressed WAV. Use :class:`FFMPEGLoader` for other formats.
    '''

    def fit(self, X, y=None):
        '''No-op. Present because scikit-learn pipelines require ``fit``.'''

        return self

    def transform(
        self,
        X: Union[list[str], pd.Series, np.ndarray],
    ) -> list[Tuple[np.ndarray, int]]:
        '''Read each file and rescale integer PCM data to floats in [-1, 1].

        Parameters
        ----------
        X :
            The input filepaths.

        Returns
        -------
        X_transformed :
            One ``(sample, rate)`` tuple per file.
        '''

        X = utils.check_filepaths_input(X)

        X_transformed = []
        for data_fp in tqdm.tqdm(X['filepath']):
            rate, data = wavfile.read(data_fp)

            # librosa expects floats in [-1, 1], not raw int16 PCM values.
            data = data / np.iinfo(data.dtype).max

            X_transformed.append((data, rate))

        return X_transformed


class FFMPEGLoader(TransformerMixin, BaseEstimator):
    '''Load any audio format ffmpeg understands (.m4a, .mp3, .wav, ...).

    Parameters
    ----------
    rate :
        Kept for backwards compatibility. Files are returned at their native
        rate; resampling is done downstream.
    '''

    def __init__(self, rate: int = 8000):
        self.rate = rate

    def fit(self, X, y=None):
        '''No-op. Present because scikit-learn pipelines require ``fit``.'''

        return self

    def transform(
        self,
        X: Union[list[str], pd.Series, np.ndarray],
    ) -> list[Tuple[np.ndarray, int]]:
        '''Decode each file to a mono float sample.

        Parameters
        ----------
        X :
            The input filepaths.

        Returns
        -------
        X_transformed :
            One ``(sample, rate)`` tuple per file.
        '''

        X = utils.check_filepaths_input(X)

        X_transformed = []
        for data_fp in tqdm.tqdm(X['filepath']):
            audio = AudioSegment.from_file(data_fp)
            X_transformed.append(audiosegment_to_array(audio))

        return X_transformed


def audiosegment_to_array(audio: AudioSegment) -> Tuple[np.ndarray, int]:
    '''Convert a pydub ``AudioSegment`` to a mono float sample in [-1, 1].

    Stereo recordings are downmixed first. Without this step pydub returns the
    two channels interleaved (L, R, L, R, ...), which would be misread as a
    single mono signal of twice the length.

    Parameters
    ----------
    audio :
        The decoded audio.

    Returns
    -------
    sample, rate :
        The mono sample and its sampling rate in Hz.
    '''

    audio = audio.set_channels(1)
    sample = np.array(audio.get_array_of_samples())
    sample = sample / np.iinfo(sample.dtype).max

    return sample, audio.frame_rate


def pad_or_truncate(sample: np.ndarray, n_samples: int) -> np.ndarray:
    '''Center a 1D signal in a length-``n_samples`` array.

    Long signals keep their middle ``n_samples`` values; short signals are
    padded with equal amounts of silence (zeros) on both sides. This mirrors
    ``pad_trunc`` in specgram_vectorization.ipynb, which produced the
    training data.
    '''

    length = sample.size
    if length > n_samples:
        start = (length - n_samples) // 2
        return sample[start:start + n_samples]

    pad_begin = (n_samples - length) // 2
    pad_end = n_samples - length - pad_begin
    return np.pad(sample, (pad_begin, pad_end), mode='constant')


class Resampler(TransformerMixin, BaseEstimator):
    '''Resample ``(sample, rate)`` tuples to a common rate.

    The training recordings are 8 kHz, while phones record at 44.1-48 kHz.
    The model only knows what 8 kHz audio looks like, so user audio must be
    converted first.

    Parameters
    ----------
    rate :
        Target sampling rate in Hz.
    '''

    def __init__(self, rate: int = 8000):
        self.rate = rate

    def fit(self, X, y=None):
        '''No-op. Present because scikit-learn pipelines require ``fit``.'''

        return self

    def transform(
        self,
        X: list[Tuple[np.ndarray, int]],
    ) -> list[Tuple[np.ndarray, int]]:
        '''Return every sample at ``self.rate``.'''

        return [
            (
                sample if rate == self.rate else
                librosa.resample(sample, orig_sr=rate, target_sr=self.rate),
                self.rate,
            )
            for sample, rate in X
        ]


class WaveformWindowSplitter(TransformerMixin, BaseEstimator):
    '''Slice one recording into overlapping fixed-length windows.

    The models were trained on 2.5 s clips (20000 samples at 8 kHz), but a
    user's recording can be any length. We slide a 2.5 s window across it
    with a 1 s stride and classify each window separately ("meow-by-meow"),
    then aggregate. Recordings shorter than one window yield one window,
    which is later centred and zero-padded exactly like short training clips.

    Parameters
    ----------
    window_seconds :
        Window length in seconds.
    hop_seconds :
        Stride between window starts, in seconds.
    '''

    def __init__(self, window_seconds: float = 2.5, hop_seconds: float = 1.):
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds

    def fit(self, X, y=None):
        '''No-op. Present because scikit-learn pipelines require ``fit``.'''

        return self

    def window_starts(self, n_samples: int, rate: int) -> np.ndarray:
        '''Start index of every window for a recording of ``n_samples``.

        The last window is aligned to the end of the recording so that the
        tail is never dropped.
        '''

        window = int(round(self.window_seconds * rate))
        hop = int(round(self.hop_seconds * rate))
        if n_samples <= window:
            return np.array([0])

        starts = np.arange(0, n_samples - window + 1, hop)
        if starts[-1] != n_samples - window:
            starts = np.append(starts, n_samples - window)
        return starts

    def window_centers_seconds(self, n_samples: int, rate: int) -> np.ndarray:
        '''Timestamp (s) at the middle of every window, for plotting.'''

        window = int(round(self.window_seconds * rate))
        starts = self.window_starts(n_samples, rate)
        return (starts + min(window, n_samples) / 2) / rate

    def transform(
        self,
        X: list[Tuple[np.ndarray, int]],
    ) -> list[Tuple[np.ndarray, int]]:
        '''Split every recording, returning one flat list of windows.'''

        windows = []
        for sample, rate in X:
            window = int(round(self.window_seconds * rate))
            for start in self.window_starts(sample.size, rate):
                windows.append((sample[start:start + window], rate))

        return windows


class TrainingSpecgramTransformer(TransformerMixin, BaseEstimator):
    '''Mel spectrograms computed exactly as for the released models' training.

    Reimplements, with librosa, the torchaudio pipeline in
    specgram_vectorization.ipynb (``pad_trunc`` + ``MelSpectrogram`` +
    ``AmplitudeToDB``). It matches the saved training features to within
    0.01 dB (verified in tests/test_data_handling.py) without needing
    PyTorch at inference time.

    Steps for each clip:
        1. Centre-pad/truncate to ``n_samples`` (2.5 s at 8 kHz).
        2. Short-time Fourier transform: 600-sample (75 ms) frames every
           300 samples (37.5 ms) -> 67 time frames.
        3. Power spectrum projected onto 128 mel bins (HTK mel scale, no
           filter normalisation, as torchaudio does by default).
        4. Convert power to dB (10 log10) and floor everything more than
           ``top_db`` below the loudest cell.

    Parameters
    ----------
    n_samples :
        Clip length (in samples) after padding/truncation.
    n_fft :
        FFT frame length in samples.
    hop_length :
        Stride between FFT frames in samples.
    n_mels :
        Number of mel frequency bins.
    top_db :
        Dynamic range kept below the maximum, in dB.
    '''

    def __init__(
        self,
        n_samples: int = 20000,
        n_fft: int = 600,
        hop_length: int = 300,
        n_mels: int = 128,
        top_db: float = 80.,
    ):
        self.n_samples = n_samples
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.top_db = top_db

    def fit(self, X, y=None):
        '''No-op. Present because scikit-learn pipelines require ``fit``.'''

        return self

    def transform(self, X: list[Tuple[np.ndarray, int]]) -> np.ndarray:
        '''Return an array of shape ``(len(X), n_mels, n_frames)``.'''

        specgrams = []
        for sample, rate in X:
            clip = pad_or_truncate(np.asarray(sample, dtype=float),
                                   self.n_samples)
            power = librosa.feature.melspectrogram(
                y=clip,
                sr=rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                htk=True,
                norm=None,
                power=2.0,
                center=True,
                pad_mode='reflect',
            )
            db = 10. * np.log10(np.maximum(power, 1e-10))
            specgrams.append(np.maximum(db, db.max() - self.top_db))

        return np.array(specgrams, dtype=np.float32)


class SpecgramTransformer(TransformerMixin, BaseEstimator):
    '''Turn ``(sample, rate)`` tuples into mel spectrograms measured in dB.

    This is the representation from the first half of
    specgram_vectorization.ipynb. The released models use
    :class:`TrainingSpecgramTransformer` features.

    Steps for each recording:
        1. Short-time Fourier transform (STFT): split the signal into
           overlapping frames and take the FFT of each, giving the energy at
           each frequency over time.
        2. Keep the magnitude and discard the phase.
        3. Project the linear frequency axis onto 128 mel bins. The mel scale
           is roughly logarithmic, matching how mammals perceive pitch.
        4. Convert amplitudes to decibels relative to the quietest bin of the
           recording. This normalises loudness per clip, so a quiet phone
           recording and a loud one look alike.

    Original code from specgram_vectorization.ipynb (Jinjing Yi,
    Brady Ali Medina).

    Parameters
    ----------
    sample_rate :
        The reference sampling rate (Hz) that ``base_n_fft`` is tuned for.
    base_n_fft :
        FFT window length at ``sample_rate``. It is rescaled for recordings
        at other rates so that each window always spans the same duration
        (2048 / 8000 Hz = 256 ms).
    '''

    def __init__(self, sample_rate: int = 8000, base_n_fft: int = 2048):

        self.sample_rate = sample_rate
        self.base_n_fft = base_n_fft

    def fit(self, X, y=None):
        '''No-op. Present because scikit-learn pipelines require ``fit``.'''

        return self

    def transform(
        self,
        X: list[Tuple[np.ndarray, int]],
    ) -> list[np.ndarray]:
        '''Compute one mel spectrogram per recording.

        Parameters
        ----------
        X :
            ``(sample, rate)`` tuples, e.g. the output of a loader.

        Returns
        -------
        X_transformed :
            Specgrams of shape ``(128, n_time_frames)``. The number of frames
            depends on the recording length, so a list is returned rather
            than a single array.
        '''

        arrs = []
        for datalib, ratelib in tqdm.tqdm(X):
            n_fft = int(self.base_n_fft * ratelib / self.sample_rate)
            specgram = librosa.stft(datalib, n_fft=n_fft)
            specmag, _ = librosa.magphase(specgram)
            mel_scale_sgram = librosa.feature.melspectrogram(
                S=specmag, sr=ratelib)
            mel_spec_db = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
            arrs.append(mel_spec_db)

        return arrs


class RollingWindowSplitter(TransformerMixin, BaseEstimator):
    '''Cut a long spectrogram into overlapping windows the model can digest.

    The models were trained on clips of ~4 s (67 frames). A user's recording
    can be any length, so we slide a window across it and classify every
    window separately ("meow-by-meow"). The per-window predictions are then
    aggregated.

    Window ``k`` is centred on frame ``c_k = h + k * window_spacing``, where
    ``h = (window_size - 1) / 2``, and covers frames ``[c_k - h, c_k + h]``.
    Windows near the end of the recording may be shorter than
    ``window_size``; :class:`PadTransformer` fills them to full width.

    Parameters
    ----------
    window_size :
        Width of each window in spectrogram frames. Must be odd so that each
        window has a well-defined centre frame.
    window_spacing :
        Stride between consecutive window centres, in frames.
        16 frames * 64 ms ≈ 1 s.
    concatenate :
        If True (and a single recording is passed), return a flat list of
        windows instead of a list of lists.
    '''

    def __init__(
        self,
        window_size: int = 63,
        window_spacing: int = 16,
        concatenate: bool = True,
    ):
        self.window_size = window_size
        self.window_spacing = window_spacing
        self.concatenate = concatenate

    def fit(self, X, y=None):
        '''Validate parameters and precompute the window half-width.'''

        assert self.window_size % 2 == 1, 'Window size must be odd.'
        self.window_halfsize_ = (self.window_size - 1) // 2

        return self

    def window_centers(self, n_frames: int) -> np.ndarray:
        '''Frame indices of the window centres for a specgram of given width.

        Exposed so that callers (e.g. the app) can place each window's
        prediction at the right point on a timeline.
        '''

        return np.arange(
            self.window_halfsize_,
            max(n_frames, self.window_halfsize_ + 1),
            self.window_spacing,
        )

    def transform(
        self,
        X: list[np.ndarray],
    ) -> Union[list[np.ndarray], list[list[np.ndarray]]]:
        '''Split every specgram into windows.

        Parameters
        ----------
        X :
            Specgrams of shape ``(n_mel_bins, n_time_frames)``.

        Returns
        -------
        X_transformed :
            For each specgram, a list of windows of shape
            ``(n_mel_bins, <= window_size)``.
        '''

        split_specgrams = []
        for full_specgram in X:
            specgrams = [
                full_specgram[
                    :,
                    max(wc - self.window_halfsize_, 0):
                    wc + self.window_halfsize_ + 1
                ]
                for wc in self.window_centers(full_specgram.shape[1])
            ]
            split_specgrams.append(specgrams)

        if self.concatenate:
            if len(split_specgrams) == 1:
                split_specgrams = split_specgrams[0]
            else:
                raise NotImplementedError(
                    'concatenate=True is only supported for one recording.'
                )

        return split_specgrams


class PadTransformer(TransformerMixin, BaseEstimator):
    '''Pad 2D arrays on the bottom/right so they all share one shape.

    Neural networks need fixed-size input, but spectrogram widths vary with
    recording length. Padded cells are filled with ``PAD_VALUE`` (-1), which
    never occurs in real data (see the note on ``PAD_VALUE``).

    Original code from specgram_vectorization.ipynb (Jinjing Yi,
    Brady Ali Medina).

    Parameters
    ----------
    shape :
        Target ``(n_rows, n_cols)``. Used when ``force_shape`` is True.
    force_shape :
        If True, pad to ``shape``. If False, learn the target shape in
        ``fit`` as the largest shape seen (plus one spare column, so every
        array ends in at least one padded column, which makes the padding
        easy to strip later).
    '''

    def __init__(
        self,
        shape: Optional[Tuple[int, int]] = None,
        force_shape: bool = True,
    ):
        self.shape = shape
        self.force_shape = force_shape

    def fit(self, X: list[np.ndarray], y=None):
        '''Learn the target shape from the data if ``force_shape`` is False.'''

        if not self.force_shape:
            self.max_shape0_ = np.max([_.shape[0] for _ in X])
            self.max_shape1_ = np.max([_.shape[1] for _ in X]) + 1

        return self

    def transform(self, X: list[np.ndarray]) -> np.ndarray:
        '''Pad every array and stack them.

        Parameters
        ----------
        X :
            2D arrays no larger than the target shape.

        Returns
        -------
        X_transformed :
            Array of shape ``(len(X), n_rows, n_cols)``.
        '''

        if self.force_shape:
            n_rows, n_cols = self.shape
        else:
            n_rows, n_cols = self.max_shape0_, self.max_shape1_

        spec_data = [
            np.pad(
                arr,
                ((0, n_rows - arr.shape[0]), (0, n_cols - arr.shape[1])),
                mode='constant',
                constant_values=PAD_VALUE,
            )
            for arr in X
        ]

        return np.array(spec_data)


class ReshapeTransformer(TransformerMixin, BaseEstimator):
    '''Reshape the stacked windows into the layout a given model expects.

    For example, the CNN wants ``(n, 128, 67, 1)``: a trailing "channel" axis,
    like a greyscale image. The KNN wants flat vectors ``(n, 128 * 67)``.

    Parameters
    ----------
    shape :
        Target shape, passed to ``np.ndarray.reshape``. Use -1 for the
        number of windows.
    '''

    def __init__(self, shape: Optional[Tuple[int, ...]] = None):

        self.shape = shape

    def fit(self, X: np.ndarray, y=None):
        '''No-op. Present because scikit-learn pipelines require ``fit``.'''

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        '''Return ``X`` reshaped to ``self.shape``.'''

        return X.reshape(self.shape)
