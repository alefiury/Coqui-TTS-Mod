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
BATCH_SIZE = 95

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
if os.path.exists(VCTK_DOWNLOAD_PATH) and not os.path.exists(VCTK_RESAMPLED_PATH):
    print(">>> Resampling VCTK dataset to {}KHz".format(SAMPLE_RATE / 1000))
    resample_files(VCTK_DOWNLOAD_PATH, SAMPLE_RATE, output_dir=VCTK_RESAMPLED_PATH, file_ext="flac", n_jobs=NUM_RESAMPLE_THREADS)

# init configs
vctk_config = BaseDatasetConfig(
    formatter="vctk",
    dataset_name="vctk",
    meta_file_train="",
    meta_file_val="",
    # path=VCTK_DOWNLOAD_PATH,
    path=VCTK_RESAMPLED_PATH,
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
    ],  # Ignore the test speakers to full replicate the paper experiment (yourtts)
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
    resblock_type_decoder="2",  # In the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the ResNet blocks type 1 like the VITS model
    use_language_embedding=True,
    embedded_language_dim=4,
    use_speaker_embedding=True,
    speaker_embedding_channels=256,
    use_sdp=False,
    use_prosody_embedding=True,
    embedded_prosody_dim=512,
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
    use_phonemes=True,
    phonemizer="multi_phonemizer",
    phoneme_language=None,
    compute_input_seq_cache=True,
    add_blank=True,
    use_language_weighted_sampler=True,
    text_cleaner="multilingual_cleaners",
    phoneme_cache_path=os.path.join(OUT_PATH, "phoneme_cache"),
    precompute_num_workers=12,
    start_by_longest=True,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=False,
    test_sentences=[
        [
            "Ask her to bring these things with her from the store.",
            "VCTK_p277",
            None,
            "en",
            os.path.join(VCTK_RESAMPLED_PATH, "wav48_silence_trimmed", "p277", "p277_002_mic1.flac")
        ],
        [
            "People look, but no one ever finds it.",
            "VCTK_p239",
            None,
            "en",
            os.path.join(VCTK_RESAMPLED_PATH, "wav48_silence_trimmed", "p239", "p239_010_mic1.flac")
        ],
        [
            "Because it's a waste of time for both sides.",
            "VCTK_p258",
            None,
            "en",
            os.path.join(VCTK_RESAMPLED_PATH, "wav48_silence_trimmed", "p258", "p258_029_mic1.flac")
        ],
        [
            "Or it would have been.",
            "VCTK_p244",
            None,
            "en",
            os.path.join(VCTK_RESAMPLED_PATH, "wav48_silence_trimmed", "p244", "p244_133_mic1.flac")
        ],
        [
            "It shows that painting is still relevant.",
            "VCTK_p305",
            None,
            "en",
            os.path.join(VCTK_RESAMPLED_PATH, "wav48_silence_trimmed", "p305", "p305_218_mic1.flac")
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
