import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

import torch
from trainer import Trainer, TrainerArgs

from TTS.bin.resample import resample_files
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.utils.downloaders import download_vctk

torch.set_num_threads(24)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# Name of the run for the Trainer
RUN_NAME = "VIPT-VCTK"

# Path where you want to save the models outputs (configs, checkpoints and tensorboard logs)
OUT_PATH = os.path.dirname(os.path.abspath(__file__))  # "/raid/coqui/Checkpoints/original-YourTTS/"

# If you want to do transfer learning and speedup your training you can set here the path to the original YourTTS model
RESTORE_PATH = None  # "/root/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/model_file.pth"

# This paramter is useful to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
SKIP_TRAIN_EPOCH = False

# Set here the batch size to be used in training and evaluation
BATCH_SIZE = 2

# Training Sampling rate and the target sampling rate for resampling the downloaded dataset (Note: If you change this you might need to redownload the dataset !!)
# Note: If you add new datasets, please make sure that the dataset sampling rate and this parameter are matching, otherwise resample your audios
SAMPLE_RATE = 24000

# Max audio length in seconds to be used in training (every audio bigger than it will be ignored)
MAX_AUDIO_LEN_IN_SECONDS = float("inf")

DATASET_BASE_DIR = "/raid/alefiury/prosody_transfer/data"

### Download VCTK dataset
VCTK_DOWNLOAD_PATH = os.path.join(DATASET_BASE_DIR, "VCTK")
### Resampled VCTK dataset
VCTK_RESAMPLED_PATH = os.path.join(DATASET_BASE_DIR, "VCTK_24KHz")
# Define the number of threads used during the audio resampling
NUM_RESAMPLE_THREADS = 10
# Check if VCTK dataset is not already downloaded, if not download it
if not os.path.exists(VCTK_DOWNLOAD_PATH):
    print(">>> Downloading VCTK dataset:")
    download_vctk(VCTK_DOWNLOAD_PATH)
    resample_files(VCTK_DOWNLOAD_PATH, SAMPLE_RATE, file_ext="flac", n_jobs=NUM_RESAMPLE_THREADS)

# init configs
vctk_config = BaseDatasetConfig(
    formatter="vctk",
    dataset_name="vctk",
    meta_file_train="",
    meta_file_val="",
    path=VCTK_DOWNLOAD_PATH,
    language="en",
    ignored_speakers=[
        "p261",
        "p225",
        "p294",
        "p347",
        "p238",
        "p234",
        "p248",
        "p335",
        "p245",
        "p326",
        "p302",
    ],  # Ignore the test speakers to full replicate the paper experiment
)

# Add here all datasets configs, in our case we just want to train with the VCTK dataset then we need to add just VCTK. Note: If you want to add new datasets, just add them here and it will automatically compute the speaker embeddings (d-vectors) for this new dataset :)
DATASETS_CONFIG_LIST = [vctk_config]

# Audio config used in training.
audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=None,
    num_mels=80,
)

# Init VITSArgs setting the arguments that are needed for the YourTTS model
model_args = VitsArgs(
    num_layers_text_encoder=10,
    resblock_type_decoder="1",  # In the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the ResNet blocks type 1 like the VITS model
    use_language_embedding=True,
    embedded_language_dim=4,
    use_speaker_embedding=True,
    speaker_embedding_channels=256,
    use_sdp=False,
    # use_prosody_embedding=True,
    # embedded_prosody_dim=512,
    use_prosody_embedding=False,
    embedded_prosody_dim=0,
)

# General training config, here you can change the batch size and others useful parameters
config = VitsConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name="VIPT-VCTK",
    run_description="""- VIPT using VCTK dataset""",
    dashboard_logger="tensorboard",
    logger_uri=None,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=8,
    eval_split_max_size=256,
    print_step=50,
    plot_step=100,
    log_model_step=1000,
    save_step=5000,
    save_n_checkpoints=2,
    save_checkpoints=True,
    target_loss="loss_1",
    print_eval=False,
    use_phonemes=False,
    phonemizer="multi_phonemizer",
    phoneme_language=None,
    compute_input_seq_cache=True,
    add_blank=True,
    use_language_weighted_sampler=True,
    text_cleaner="multilingual_cleaners",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\u00af\u00b7\u00df\u00e0\u00e1\u00e2\u00e3\u00e4\u00e6\u00e7\u00e8\u00e9\u00ea\u00eb\u00ec\u00ed\u00ee\u00ef\u00f1\u00f2\u00f3\u00f4\u00f5\u00f6\u00f9\u00fa\u00fb\u00fc\u00ff\u0101\u0105\u0107\u0113\u0119\u011b\u012b\u0131\u0142\u0144\u014d\u0151\u0153\u015b\u016b\u0171\u017a\u017c\u01ce\u01d0\u01d2\u01d4\u0430\u0431\u0432\u0433\u0434\u0435\u0436\u0437\u0438\u0439\u043a\u043b\u043c\u043d\u043e\u043f\u0440\u0441\u0442\u0443\u0444\u0445\u0446\u0447\u0448\u0449\u044a\u044b\u044c\u044d\u044e\u044f\u0451\u0454\u0456\u0457\u0491\u2013!'(),-.:;? ",
        punctuations="\u2014!'(),-.:;?\u00bf ",
        phonemes="iy\u0268\u0289\u026fu\u026a\u028f\u028ae\u00f8\u0258\u0259\u0275\u0264o\u025b\u0153\u025c\u025e\u028c\u0254\u00e6\u0250a\u0276\u0251\u0252\u1d7b\u0298\u0253\u01c0\u0257\u01c3\u0284\u01c2\u0260\u01c1\u029bpbtd\u0288\u0256c\u025fk\u0261q\u0262\u0294\u0274\u014b\u0272\u0273n\u0271m\u0299r\u0280\u2c71\u027e\u027d\u0278\u03b2fv\u03b8\u00f0sz\u0283\u0292\u0282\u0290\u00e7\u029dx\u0263\u03c7\u0281\u0127\u0295h\u0266\u026c\u026e\u028b\u0279\u027bj\u0270l\u026d\u028e\u029f\u02c8\u02cc\u02d0\u02d1\u028dw\u0265\u029c\u02a2\u02a1\u0255\u0291\u027a\u0267\u025a\u02de\u026b'\u0303' ",
        is_unique=True,
        is_sorted=True,
    ),
    phoneme_cache_path=os.path.join(OUT_PATH, "phoneme_cache"),
    precompute_num_workers=12,
    start_by_longest=True,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=False,
    test_sentences=[
        [
            "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
            "VCTK_p277",
            None,
            "en",
        ],
        [
            "Be a voice, not an echo.",
            "VCTK_p239",
            None,
            "en",
        ],
        [
            "I'm sorry Dave. I'm afraid I can't do that.",
            "VCTK_p258",
            None,
            "en",
        ],
        [
            "This cake is great. It's so delicious and moist.",
            "VCTK_p244",
            None,
            "en",
        ],
        [
            "Prior to November 22, 1963.",
            "VCTK_p305",
            None,
            "en",
        ],
    ],
    # Enable the weighted sampler
    use_weighted_sampler=True,
    # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
    weighted_sampler_attrs={"speaker_name": 1.0},
    weighted_sampler_multipliers={},
    # It defines the Speaker Consistency Loss (SCL) α to 9 like the paper
    speaker_encoder_loss_alpha=9.0,
)

# Load all the datasets samples and split traning and evaluation sets
train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers

language_manager = LanguageManager(config=config)
config.model_args.num_languages = language_manager.num_languages

tokenizer, config = TTSTokenizer.init_from_config(config)

# Init the model
model = Vits(config, ap, tokenizer, speaker_manager, language_manager)

model.load_checkpoint(
    config=None,
    checkpoint_path="checkpoints/best_model.pth",
    strict=False,
)

# Init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
