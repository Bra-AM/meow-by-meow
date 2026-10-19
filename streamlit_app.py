'''Meow-by-Meow: a Streamlit app that interprets a cat's meows.

Run locally from the repository root with::

    streamlit run streamlit_app.py

Flow of the page:
    1. Load a recording (user upload or a bundled example).
    2. Split it into overlapping 2.5 s windows and compute, for each window,
       the same mel spectrogram the models were trained on.
    3. Classify each window with a CNN downloaded from Hugging Face.
    4. Average the per-window probabilities ("soft voting") and show both the
       timeline and the overall interpretation.
'''
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import streamlit as st

from meowlib import data_handling, models

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDING = REPO_ROOT / 'data' / 'processed_data' / 'combined_meows.mp3'
MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 60.
PLOT_POINTS = 4000          # waveform points drawn; keeps the chart responsive
COLOR_SCHEME = 'tableau10'


# ---------------------------------------------------------------------------
# Cached helpers. Streamlit reruns this whole script on every interaction, so
# anything slow (downloading and building a model) is cached across reruns.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    '''Download (once) and load the named model.'''

    spec = models.MODELS[model_name]
    return models.load_keras_model(models.resolve_model_file(spec))


def load_recording(user_file) -> tuple[AudioSegment, bool]:
    '''Decode the uploaded file, falling back to the bundled example.

    Returns the audio and whether it is the user's own recording.
    '''

    if user_file is None:
        return AudioSegment.from_file(DEFAULT_RECORDING), False

    try:
        # Browsers report e.g. 'audio/x-m4a'; ffmpeg wants just 'm4a'.
        filetype = user_file.type.split('/')[-1].replace('x-', '')
        audio = AudioSegment.from_file(user_file, format=filetype)
    except (CouldntDecodeError, IndexError):
        st.error('Could not read that file, so we are using our example.')
        return AudioSegment.from_file(DEFAULT_RECORDING), False

    if audio.duration_seconds < MIN_DURATION_SECONDS:
        st.error(
            f'Please upload at least {MIN_DURATION_SECONDS:g} seconds of '
            'audio. Using our example instead.'
        )
        return AudioSegment.from_file(DEFAULT_RECORDING), False

    if audio.duration_seconds > MAX_DURATION_SECONDS:
        st.warning(
            f'Only the first {MAX_DURATION_SECONDS:g} seconds are analysed.')
        audio = audio[:int(MAX_DURATION_SECONDS * 1000)]

    st.success('File uploaded!')
    return audio, True


def waveform_frame(sample: np.ndarray, rate: int) -> pd.DataFrame:
    '''Downsample a waveform to its per-bin peak amplitude for plotting.'''

    n_bins = min(PLOT_POINTS, sample.size)
    edges = np.linspace(0, sample.size, n_bins + 1).astype(int)
    peaks = np.array([
        np.abs(sample[a:b]).max() if b > a else 0.
        for a, b in zip(edges[:-1], edges[1:])
    ])
    return pd.DataFrame({'time': edges[:-1] / rate, 'amplitude': peaks})


def technical_note(text: str):
    '''Show extra detail only when "Display details" is ticked.'''

    if technical:
        st.markdown(f':green[{text}]')


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title='Meow-by-Meow', page_icon='🐱', layout='wide')
st.title('Meow-by-Meow')

st.sidebar.header('Advanced settings')
technical = st.sidebar.checkbox('Display details')
model_name = st.sidebar.selectbox(
    'Select the ML model to use', options=list(models.MODELS))
technical_note(models.MODELS[model_name].description)

st.caption(
    'Created by Brady Ali, Jinjing Yi, Tantrik Mukerji, William Craig, '
    'and Zach Hafen-Saavedra during the Erdos Institute Fall 2023 Program.'
)
st.write(
    "Use machine learning to interpret your cat's "
    'chirps, complaints, yowls, yells, vocalizations, and meows.'
)
if technical:
    st.markdown(
        ':green[Highlighted contributions:]\n'
        '  * :green[Brady Ali: spectrogram data analysis, '
        'convolutional neural network (CNN) modeling]\n'
        '  * :green[Jinjing Yi: data augmentation, spectrogram data analysis, '
        'CNN modeling]\n'
        '  * :green[Tantrik Mukerji: classification data handling, '
        'CNN modeling]\n'
        '  * :green[William Craig: K-nearest neighbors modeling]\n'
        '  * :green[Zach Hafen-Saavedra: user interface, project and code '
        'management, asset management]\n'
    )
    technical_note(
        'Source code: [github](https://github.com/Bra-AM/meow-by-meow). '
        'The app is powered by [streamlit](https://streamlit.io/).'
    )
st.divider()

# --- 1. Input --------------------------------------------------------------
st.subheader('Upload a recording of your cat')
user_file = st.file_uploader(
    label='Accepted formats include .wav, .m4a, .mp3, and anything ffmpeg reads.'
)
technical_note(
    'Files are decoded with pydub/ffmpeg, mixed down to mono, and resampled '
    'to 8 kHz, the rate of the training recordings.'
)
audio, is_user_file = load_recording(user_file)
st.divider()

st.subheader('The recording to interpret')
if not is_user_file:
    st.write('No recording handy? We have one ready!')
    technical_note(
        'The example is stitched together from several dataset recordings '
        'so that it shows more than one situation.'
    )
st.audio(audio.export(format='wav').read(), format='audio/wav')
st.divider()

# --- 2-3. Analysis ---------------------------------------------------------
st.subheader('Our analysis')
technical_note(
    'The models are hosted on the [Hugging Face model hub]'
    '(https://huggingface.co/zhafen/meow-by-meow) and downloaded on first use.'
)
technical_note(
    'The models were trained on 2.5 s clips. Longer recordings are split '
    'into 2.5 s windows every 1 s, each window is classified, and the '
    'probabilities are averaged.'
)

with st.status('Interpreting...', expanded=technical) as status:
    st.write('Retrieving the trained model...')
    model = load_model(model_name)
    spec = models.MODELS[model_name]

    st.write('Preparing the audio...')
    sample, rate = data_handling.audiosegment_to_array(audio)
    preprocess = models.build_preprocessing(model.input_shape[1:])
    for step_name in preprocess.named_steps:
        st.write(f'* {step_name}')
    X = preprocess.fit_transform([(sample, rate)])

    st.write('Employing the model...')
    probs = models.situation_probabilities(
        model.predict(X, verbose=0), spec.output_situations)
    status.update(label='Done!', state='complete')

# Timestamps are in the 8 kHz time base used by the window splitter.
splitter = preprocess.named_steps['split into windows']
n_resampled = int(round(sample.size * 8000 / rate))
window_centers = splitter.window_centers_seconds(n_resampled, 8000)
behaviors = [models.SITUATION_MEANINGS[s] for s in models.SITUATIONS]
st.divider()

# --- 4. Results ------------------------------------------------------------
st.subheader('Our interpretation!')
technical_note(
    'Each part of the waveform is coloured by the window whose centre is '
    'closest to it. Cats rarely change mood from one meow to the next, so '
    'the overall verdict averages all windows.'
)

wave = waveform_frame(sample, rate)
nearest = np.abs(wave['time'].values[:, None] - window_centers).argmin(axis=1)
wave['behavior'] = np.array(behaviors)[probs.argmax(axis=1)][nearest]
color = alt.Color('behavior:N', title='sounds',
                  scale=alt.Scale(domain=behaviors, scheme=COLOR_SCHEME))
waveform_chart = alt.Chart(wave).mark_bar(size=2).encode(
    x=alt.X('time:Q', title='time (seconds)'),
    y=alt.Y('amplitude:Q', title='loudness', axis=alt.Axis(labels=False)),
    color=color,
).properties(height=220)
st.altair_chart(waveform_chart, use_container_width=True)

overall = probs.mean(axis=0)
best = int(overall.argmax())
st.markdown(
    "<p style='text-align: center; font-size: larger'>"
    f'<b>Overall</b>, your cat sounds <b><i>{behaviors[best]}</i></b> '
    f'({overall[best]:.0%} confidence).</p>',
    unsafe_allow_html=True,
)

confidence = pd.DataFrame({'sounds': behaviors, 'probability': overall})
st.altair_chart(
    alt.Chart(confidence).mark_bar().encode(
        x=alt.X('probability:Q', scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format='%')),
        y=alt.Y('sounds:N', sort=behaviors, title=''),
        color=alt.Color('sounds:N', legend=None,
                        scale=alt.Scale(domain=behaviors,
                                        scheme=COLOR_SCHEME)),
    ).properties(height=120),
    use_container_width=True,
)

technical_note(
    'The models were trained on recordings of 21 cats, so treat the result '
    'for your own cat as a fun guess, not a diagnosis.'
)
