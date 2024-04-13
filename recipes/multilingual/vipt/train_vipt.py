import os

import torch
import wandb
from tqdm import tqdm
from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.bin.resample import resample_files
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.utils.downloaders import download_libri_tts

torch.set_num_threads(16)

wandb.login()

run = wandb.init(
    project="VIPT"
)

# Change run name to facilitate train statistics traceability
wandb.run.name = "VIPT-CML"
wandb.run.save()

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# Name of the run for the Trainer
RUN_NAME = "VIPT-CML"

# Path where you want to save the models outputs (configs, checkpoints and tensorboard logs)
OUT_PATH = os.path.dirname(os.path.abspath(__file__))  # "/raid/coqui/Checkpoints/original-YourTTS/"

RESTORE_PATH = "/hadatasets/alef.ferreira/translation/Coqui-TTS-Mod/recipes/multilingual/vipt/VIPT-CML-April-01-2024_04+16PM-0000000/checkpoint_250000.pth"

# This paramter is useful to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
# SKIP_TRAIN_EPOCH = True
SKIP_TRAIN_EPOCH = False

# Set here the batch size to be used in training and evaluation
BATCH_SIZE = 48

# Training Sampling rate and the target sampling rate for resampling the downloaded dataset (Note: If you change this you might need to redownload the dataset !!)
# Note: If you add new datasets, please make sure that the dataset sampling rate and this parameter are matching, otherwise resample your audios
SAMPLE_RATE = 24000

# Max audio length in seconds to be used in training (every audio bigger than it will be ignored)
MAX_AUDIO_LEN_IN_SECONDS = float("inf")

### Download CML-TTS dataset
# You need to download the dataset for all languages manually and extract it to a path and then set the CML_DATASET_PATH to this path: https://github.com/freds0/CML-TTS-Dataset#download
CML_DATASET_PATH = "/hadatasets/alef.ferreira/svc/data/cml/cml_2"


### Download LibriTTS dataset
# it will automatic download the dataset, if you have problems you can comment it and manually donwload and extract it ! Download link: https://www.openslr.org/resources/60/train-clean-360.tar.gz
LIBRITTS_DOWNLOAD_PATH = "./datasets/LibriTTS/LibriTTS/"
# Check if LibriTTS dataset is not already downloaded, if not download it
if not os.path.exists(LIBRITTS_DOWNLOAD_PATH):
    print(">>> Downloading LibriTTS dataset:")
    download_libri_tts(LIBRITTS_DOWNLOAD_PATH, subset="libri-tts-clean-360")

# init LibriTTS configs
libritts_config = BaseDatasetConfig(
    formatter="libri_tts",
    dataset_name="libri_tts",
    meta_file_train="",
    meta_file_val="",
    path=os.path.join(LIBRITTS_DOWNLOAD_PATH, "train-clean-360"),
    language="en",
)

# init CML-TTS configs
pt_config = BaseDatasetConfig(
    formatter="cml_tts",
    dataset_name="cml_tts",
    meta_file_train="train.csv",
    meta_file_val="",
    path=os.path.join(CML_DATASET_PATH, "cml_tts_dataset_portuguese_v0.1/"),
    language="pt-br",
)

pl_config = BaseDatasetConfig(
    formatter="cml_tts",
    dataset_name="cml_tts",
    meta_file_train="train.csv",
    meta_file_val="",
    path=os.path.join(CML_DATASET_PATH, "cml_tts_dataset_polish_v0.1/"),
    language="pl",
)

it_config = BaseDatasetConfig(
    formatter="cml_tts",
    dataset_name="cml_tts",
    meta_file_train="train.csv",
    meta_file_val="",
    path=os.path.join(CML_DATASET_PATH, "cml_tts_dataset_italian_v0.1/"),
    language="it",
)

fr_config = BaseDatasetConfig(
    formatter="cml_tts",
    dataset_name="cml_tts",
    meta_file_train="train.csv",
    meta_file_val="",
    path=os.path.join(CML_DATASET_PATH, "cml_tts_dataset_french_v0.1/"),
    language="fr",
)

du_config = BaseDatasetConfig(
    formatter="cml_tts",
    dataset_name="cml_tts",
    meta_file_train="train.csv",
    meta_file_val="",
    path=os.path.join(CML_DATASET_PATH, "cml_tts_dataset_dutch_v0.1/"),
    language="du",
)

ge_config = BaseDatasetConfig(
    formatter="cml_tts",
    dataset_name="cml_tts",
    meta_file_train="train.csv",
    meta_file_val="",
    path=os.path.join(CML_DATASET_PATH, "cml_tts_dataset_german_v0.1/"),
    language="ge",
)

sp_config = BaseDatasetConfig(
    formatter="cml_tts",
    dataset_name="cml_tts",
    meta_file_train="train.csv",
    meta_file_val="",
    path=os.path.join(CML_DATASET_PATH, "cml_tts_dataset_spanish_v0.1/"),
    language="sp",
)

# Add here all datasets configs Note: If you want to add new datasets, just add them here and it will automatically compute the speaker embeddings (d-vectors) for this new dataset :)
DATASETS_CONFIG_LIST = [libritts_config, pt_config, pl_config, it_config, fr_config, du_config, ge_config, sp_config]

# DATASETS_CONFIG_LIST = [pl_config]

test_sentences_paths = [
    os.path.join(CML_DATASET_PATH, "cml_tts_dataset_portuguese_v0.1", "test/audio/3050/2941/3050_2941_000020.wav"),
    os.path.join(CML_DATASET_PATH, "cml_tts_dataset_portuguese_v0.1", "test/audio/3050/2941/3050_2941_000020.wav"),
    os.path.join(CML_DATASET_PATH, "cml_tts_dataset_portuguese_v0.1", "test/audio/3050/2941/3050_2941_000019-0001.wav"),
    os.path.join(CML_DATASET_PATH, "cml_tts_dataset_portuguese_v0.1", "test/audio/3050/2941/3050_2941_000019-0001.wav"),
    os.path.join(CML_DATASET_PATH, "cml_tts_dataset_french_v0.1", "test/audio/2223/1745/2223_1745_000039.wav"),
    os.path.join(CML_DATASET_PATH, "cml_tts_dataset_french_v0.1", "test/audio/2223/1745/2223_1745_000039.wav"),
    os.path.join(CML_DATASET_PATH, "cml_tts_dataset_french_v0.1", "test/audio/2223/1745/2223_1745_000039.wav"),
    os.path.join(CML_DATASET_PATH, "cml_tts_dataset_spanish_v0.1", "test/audio/3468/1378/3468_1378_000036-0001.wav"),
    os.path.join(CML_DATASET_PATH, "cml_tts_dataset_spanish_v0.1", "test/audio/3468/1378/3468_1378_000036-0001.wav"),
    os.path.join(CML_DATASET_PATH, "cml_tts_dataset_spanish_v0.1", "test/audio/3468/1378/3468_1378_000036-0001.wav"),
    os.path.join(CML_DATASET_PATH, "cml_tts_dataset_polish_v0.1", "test/audio/1889/1447/1889_1447_000033.wav")
]

### Extract speaker embeddings
SPEAKER_ENCODER_CHECKPOINT_PATH = (
    "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
)
SPEAKER_ENCODER_CONFIG_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"

D_VECTOR_FILES = []  # List of speaker embeddings/d-vectors to be used during the training

# Iterates all the dataset configs checking if the speakers embeddings are already computated, if not compute it
for dataset_conf in DATASETS_CONFIG_LIST:
    # Check if the embeddings weren't already computed, if not compute it
    embeddings_file = os.path.join(dataset_conf.path, "speakers.pth")
    if not os.path.isfile(embeddings_file):
        print(f">>> Computing the speaker embeddings for the {dataset_conf.dataset_name} dataset")
        compute_embeddings(
            SPEAKER_ENCODER_CHECKPOINT_PATH,
            SPEAKER_ENCODER_CONFIG_PATH,
            embeddings_file,
            old_speakers_file=None,
            config_dataset_path=None,
            formatter_name=dataset_conf.formatter,
            dataset_name=dataset_conf.dataset_name,
            dataset_path=dataset_conf.path,
            meta_file_train=dataset_conf.meta_file_train,
            meta_file_val=dataset_conf.meta_file_val,
            disable_cuda=False,
            no_eval=False,
        )
    D_VECTOR_FILES.append(embeddings_file)

#check if test_sentences_paths exists
for test_sentence_path in tqdm(test_sentences_paths, desc="Checking test sentences paths"):
    assert os.path.exists(test_sentence_path), f"Test sentence path {test_sentence_path} does not exist, please make sure to download the CML-TTS dataset and set the CML_DATASET_PATH to the correct path"

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
    spec_segment_size=62,
    hidden_channels=192,
    hidden_channels_ffn_text_encoder=768,
    num_heads_text_encoder=2,
    num_layers_text_encoder=10,
    kernel_size_text_encoder=3,
    dropout_p_text_encoder=0.1,
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=True,
    d_vector_dim=512,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    resblock_type_decoder="2",  # In the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the ResNet blocks type 1 like the VITS model
    # Useful parameters to enable the Speaker Consistency Loss (SCL) described in the paper
    use_speaker_encoder_as_loss=False,
    # Useful parameters to enable multilingual training
    use_language_embedding=True,
    embedded_language_dim=4,
    # use_speaker_embedding=True,
    # speaker_embedding_channels=256,
    use_sdp=False,
    use_prosody_embedding=True,
    embedded_prosody_dim=512,
)

# General training config, here you can change the batch size and others useful parameters
config = VitsConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name="VIPT-CML",
    run_description="""
            - - VIPT training using CML-TTS and LibriTTS datasets
        """,
    dashboard_logger="tensorboard",
    logger_uri=None,
    # run_eval_steps=1000,
    epochs=100000,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=12,
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
    phonemizer="espeak",
    phoneme_language="en",
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="multilingual_cleaners",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\u00a1\u00a3\u00b7\u00b8\u00c0\u00c1\u00c2\u00c3\u00c4\u00c5\u00c7\u00c8\u00c9\u00ca\u00cb\u00cc\u00cd\u00ce\u00cf\u00d1\u00d2\u00d3\u00d4\u00d5\u00d6\u00d9\u00da\u00db\u00dc\u00df\u00e0\u00e1\u00e2\u00e3\u00e4\u00e5\u00e7\u00e8\u00e9\u00ea\u00eb\u00ec\u00ed\u00ee\u00ef\u00f1\u00f2\u00f3\u00f4\u00f5\u00f6\u00f9\u00fa\u00fb\u00fc\u0101\u0104\u0105\u0106\u0107\u010b\u0119\u0141\u0142\u0143\u0144\u0152\u0153\u015a\u015b\u0161\u0178\u0179\u017a\u017b\u017c\u020e\u04e7\u05c2\u1b20",
        punctuations="\u2014!'(),-.:;?\u00bf ",
        phonemes="iy\u0268\u0289\u026fu\u026a\u028f\u028ae\u00f8\u0258\u0259\u0275\u0264o\u025b\u0153\u025c\u025e\u028c\u0254\u00e6\u0250a\u0276\u0251\u0252\u1d7b\u0298\u0253\u01c0\u0257\u01c3\u0284\u01c2\u0260\u01c1\u029bpbtd\u0288\u0256c\u025fk\u0261q\u0262\u0294\u0274\u014b\u0272\u0273n\u0271m\u0299r\u0280\u2c71\u027e\u027d\u0278\u03b2fv\u03b8\u00f0sz\u0283\u0292\u0282\u0290\u00e7\u029dx\u0263\u03c7\u0281\u0127\u0295h\u0266\u026c\u026e\u028b\u0279\u027bj\u0270l\u026d\u028e\u029f\u02c8\u02cc\u02d0\u02d1\u028dw\u0265\u029c\u02a2\u02a1\u0255\u0291\u027a\u0267\u025a\u02de\u026b'\u0303' ",
        is_unique=True,
        is_sorted=True,
    ),
    phoneme_cache_path=None,
    precompute_num_workers=12,
    start_by_longest=True,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=False,
    test_sentences=[
        [
            "E o sol, quando aparece, Já o encontra, robusto e manso, a trabalhar. Forte e meigo animal! Que bondade serena Tem na doce expressão da face resignada.",
            3050,
            None,
            "pt-br",
            os.path.join(CML_DATASET_PATH, "cml_tts_dataset_portuguese_v0.1", "test/audio/3050/2941/3050_2941_000020.wav")
        ],
        [
            "And the sun, when it appears, Already finds him, robust and gentle, at work. Strong and tender animal! What serene kindness Lies in the sweet expression of the resigned face.",
            3050,
            None,
            "en",
            os.path.join(CML_DATASET_PATH, "cml_tts_dataset_portuguese_v0.1", "test/audio/3050/2941/3050_2941_000020.wav")
        ],
        [
            "É que, às vezes, amarrado Me deixam durante o dia.",
            3050,
            None,
            "pt-br",
            os.path.join(CML_DATASET_PATH, "cml_tts_dataset_portuguese_v0.1", "test/audio/3050/2941/3050_2941_000019-0001.wav")
        ],
        [
            "It's just that, sometimes, they leave me tied up during the day.",
            3050,
            None,
            "en",
            os.path.join(CML_DATASET_PATH, "cml_tts_dataset_portuguese_v0.1", "test/audio/3050/2941/3050_2941_000019-0001.wav")
        ],
        [
            "O être humain! te voilà, maintenant, nu comme un ver, en présence de mon glaive de diamant",
            2223,
            None,
            "fr",
            os.path.join(CML_DATASET_PATH, "cml_tts_dataset_french_v0.1", "test/audio/2223/1745/2223_1745_000039.wav")
        ],
        [
            "Oh human being! Here you are, now, naked as a worm, in the presence of my diamond sword.",
            2223,
            None,
            "en",
            os.path.join(CML_DATASET_PATH, "cml_tts_dataset_french_v0.1", "test/audio/2223/1745/2223_1745_000039.wav")
        ],
        [
            "Ó ser humano! Aqui estás tu, agora, nu como uma minhoca, na presença da minha espada de diamante.",
            2223,
            None,
            "pt-br",
            os.path.join(CML_DATASET_PATH, "cml_tts_dataset_french_v0.1", "test/audio/2223/1745/2223_1745_000039.wav")
        ],
        [
            "Finalmente, el doctor Pedro Recio Agüero de Tirteafuera prometió de darle de cenar aquella noche, aunque excediese de todos los aforismos de Hipócrates.",
            3468,
            None,
            "sp",
            os.path.join(CML_DATASET_PATH, "cml_tts_dataset_spanish_v0.1", "test/audio/3468/1378/3468_1378_000036-0001.wav")
        ],
        [
            "Finalmente, o doutor Pedro Recio Agüero de Tirteafuera prometeu dar-lhe jantar naquela noite, mesmo que excedesse todos os aforismos de Hipócrates.",
            3468,
            None,
            "pt-br",
            os.path.join(CML_DATASET_PATH, "cml_tts_dataset_spanish_v0.1", "test/audio/3468/1378/3468_1378_000036-0001.wav")
        ],
        [
            "Finalement, le docteur Pedro Recio Agüero de Tirteafuera a promis de lui donner à dîner cette nuit-là, même si cela dépassait tous les aphorismes d'Hippocrate.",
            3468,
            None,
            "fr",
            os.path.join(CML_DATASET_PATH, "cml_tts_dataset_spanish_v0.1", "test/audio/3468/1378/3468_1378_000036-0001.wav")
        ],
        [
            "Pokazywało się wówczas, że magazyn konfekcji był tylko fasadš, za którš kryła się antykwarnia, zbiór wysoce dwuznacznych wydawnictw i druków prywatnych",
            1890,
            None,
            "pl",
            os.path.join(CML_DATASET_PATH, "cml_tts_dataset_polish_v0.1", "test/audio/1889/1447/1889_1447_000033.wav")
        ]
    ],
    # Enable the weighted sampler
    use_weighted_sampler=True,
    # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
    # weighted_sampler_attrs={"language": 1.0, "speaker_name": 1.0},
    weighted_sampler_attrs={"language": 1.0},
    weighted_sampler_multipliers={
        # "speaker_name": {
        # you can force the batching scheme to give a higher weight to a certain speaker and then this speaker will appears more frequently on the batch.
        # It will speedup the speaker adaptation process. Considering the CML train dataset and "new_speaker" as the speaker name of the speaker that you want to adapt.
        # The line above will make the balancer consider the "new_speaker" as 106 speakers so 1/4 of the number of speakers present on CML dataset.
        # 'new_speaker': 106, # (CML tot. train speaker)/4 = (424/4) = 106
        # }
    },
    # It defines the Speaker Consistency Loss (SCL) α to 9 like the YourTTS paper
    speaker_encoder_loss_alpha=9.0,
)

# Load all the datasets samples and split traning and evaluation sets
train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# Init the model
model = Vits.init_from_config(config)

# Init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

# try:
trainer.fit()
# except Exception as e:
#     # write file .log with complete error
#     with open('error.log', 'w') as f:
#         f.write(str(e))