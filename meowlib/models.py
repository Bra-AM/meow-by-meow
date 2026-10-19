'''Loading the trained classifiers and turning their outputs into situations.

The app and the notebooks share this module so that model-specific details
(where the file lives, what each output neuron means) are defined once.
'''
from dataclasses import dataclass
import json
import io
from pathlib import Path
from typing import Optional
import zipfile

import numpy as np
from sklearn.pipeline import Pipeline

from . import data_handling

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_REPO_ID = 'zhafen/meow-by-meow'

# The three situations in the CatMeows dataset, in the order the models use,
# and the plain-English reading shown to users.
SITUATIONS = ['brushing', 'food', 'isolation']
SITUATION_MEANINGS = {
    'brushing': 'comfortable',
    'food': 'hungry',
    'isolation': 'uncomfortable',
}

# Output order of the 12-class model: sklearn's LabelEncoder sorts the string
# f'{sex}{breed}{situation}', so the classes run sex -> breed -> situation.
_BGS_CLASSES = [
    (sex, breed, situation)
    for sex in (0, 1)
    for breed in ('european_shorthair', 'maine_coon')
    for situation in SITUATIONS
]


@dataclass(frozen=True)
class ModelSpec:
    '''Where to find a trained model and what its outputs mean.

    Attributes
    ----------
    hf_filename :
        File name in the Hugging Face repository ``HF_REPO_ID``.
    output_situations :
        The situation that each output neuron corresponds to.
    local_path :
        Optional copy inside this repository, used if Hugging Face cannot be
        reached. (For the 12-class model this is an earlier training run of
        the same architecture, not a byte-identical copy.)
    description :
        One-line summary shown in the app.
    '''

    hf_filename: str
    output_situations: tuple
    local_path: Optional[str] = None
    description: str = ''


MODELS = {
    'CNN (situation + sex + breed)': ModelSpec(
        hf_filename='CNN_dataaug_with_2freqtime1rand_masking_bgs.keras',
        local_path=(
            'data/trained_models/CNN/'
            'CNN_dataaug_with_2freqtime1rand_masking_bgs.h5'
        ),
        output_situations=tuple(c[-1] for c in _BGS_CLASSES),
        description=(
            'Predicts 12 joint (sex, breed, situation) classes; we sum the '
            'probabilities belonging to each situation.'
        ),
    ),
    'CNN (situation only)': ModelSpec(
        hf_filename='CNN_dataaug_with_2freqtime1rand_masking_situation.keras',
        output_situations=tuple(SITUATIONS),
        description='Predicts the three situations directly.',
    ),
}


def resolve_model_file(spec: ModelSpec) -> str:
    '''Return a local path to the model, preferring the Hugging Face copy.

    The Hugging Face files are the ones the original app served. The local
    copy is a fallback for offline use or if the hub is unavailable.
    '''

    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id=HF_REPO_ID, filename=spec.hf_filename)
    except Exception:
        if spec.local_path is None or not (REPO_ROOT / spec.local_path).exists():
            raise
        return str(REPO_ROOT / spec.local_path)


def load_keras_model(path: str):
    '''Load a Keras model from an ``.h5`` or ``.keras`` file.

    In some ``.keras`` archives the layer names in the config (``conv2d_6``,
    ``conv2d_7``, ...) differ from the names the weights are stored under
    (``conv2d``, ``conv2d_1``, ...), and ``keras.models.load_model`` cannot
    match them. In that case we rebuild the architecture from the config and
    assign the stored weights in layer order.
    '''

    from tensorflow import keras

    try:
        return keras.models.load_model(path)
    except ValueError as error:
        if not str(path).endswith('.keras'):
            raise
        load_error = error

    import h5py

    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read('config.json'))
        weights_bytes = archive.read('model.weights.h5')

    model = keras.models.model_from_json(json.dumps(config))
    with h5py.File(io.BytesIO(weights_bytes), 'r') as weights:
        if 'layers' not in weights:
            # Older (Keras < 2.15) layout; nothing we know how to repair.
            raise load_error
        stored = weights['layers']
        for layer, kind in _layer_storage_names(model):
            group = stored[kind]['vars']
            layer.set_weights([group[str(i)][()] for i in range(len(group))])

    return model


def _layer_storage_names(model):
    '''Yield (layer, name used in model.weights.h5) for layers with weights.

    Keras stores the n-th layer of a given type as ``<type>``, ``<type>_1``,
    ``<type>_2``, ... in construction order.
    '''

    seen = {}
    for layer in model.layers:
        kind = type(layer).__name__
        kind = {'Conv2D': 'conv2d', 'Dense': 'dense'}.get(kind)
        if kind is None or not layer.weights:
            continue
        count = seen.get(kind, 0)
        seen[kind] = count + 1
        yield layer, kind if count == 0 else f'{kind}_{count}'


def build_preprocessing(
    input_shape: tuple,
    window_seconds: float = 2.5,
    hop_seconds: float = 1.,
) -> Pipeline:
    '''Pipeline from one ``(sample, rate)`` recording to model-ready windows.

    Parameters
    ----------
    input_shape :
        The model's input shape without the batch axis, e.g. (128, 67, 1).
    window_seconds, hop_seconds :
        Length of each analysed window and the stride between windows.
    '''

    return Pipeline([
        ('resample', data_handling.Resampler(rate=8000)),
        ('split into windows', data_handling.WaveformWindowSplitter(
            window_seconds=window_seconds, hop_seconds=hop_seconds)),
        ('mel spectrogram', data_handling.TrainingSpecgramTransformer()),
        ('reshape', data_handling.ReshapeTransformer(
            shape=(-1,) + tuple(input_shape))),
    ])


def situation_probabilities(
    model_outputs: np.ndarray,
    output_situations: tuple,
) -> np.ndarray:
    '''Collapse model outputs into P(situation) for every window.

    Parameters
    ----------
    model_outputs :
        Softmax outputs, shape ``(n_windows, n_outputs)``.
    output_situations :
        The situation that each output column belongs to.

    Returns
    -------
    probs :
        Shape ``(n_windows, 3)``, columns ordered as ``SITUATIONS``. Rows
        sum to one.
    '''

    output_situations = np.asarray(output_situations)
    return np.stack(
        [model_outputs[:, output_situations == s].sum(axis=1)
         for s in SITUATIONS],
        axis=1,
    )
