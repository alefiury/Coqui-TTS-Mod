import os
import json
import torchaudio
import numpy as np
import torch
from tqdm import tqdm
import shutil
import phonemizer
from nltk.tokenize import word_tokenize
from librosa.filters import mel as librosa_mel_fn
import pandas as pd

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, wav_to_spec

hann_window = {}
mel_basis = {}


class TextCleaner:
    def __init__(self, dummy=None):
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»“” '
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

        # Export all symbols:
        symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

        dicts = {}
        for i in range(len((symbols))):
            dicts[symbols[i]] = i

        self.word_index_dictionary = dicts

    def __call__(self, text):
        indexes = []
        # make sure that the text is a string
        text = str(text)
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(f"Phoneme Character: '{char}' cannot be found in the dictionary.")
                pass
        indexes.insert(0, 0)
        indexes.append(0)

        return indexes


def _amp_to_db(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def _db_to_amp(x, C=1):
    return torch.exp(x) / C


def amp_to_db(magnitudes):
    output = _amp_to_db(magnitudes)
    return output


def db_to_amp(magnitudes):
    output = _db_to_amp(magnitudes)
    return output


def spec_to_mel(spec, n_fft, num_mels, sample_rate, fmin, fmax):
    """
    Args Shapes:
        - spec : :math:`[B,C,T]`

    Return Shapes:
        - mel : :math:`[B,C,T]`
    """
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    mel = torch.matmul(mel_basis[fmax_dtype_device], spec)
    mel = amp_to_db(mel)
    return mel


def load_audio(file_path):
    """Load the audio file normalized in [-1, 1]

    Return Shapes:
        - x: :math:`[1, T]`
    """
    x, sr = torchaudio.load(file_path)
    assert (x > 1).sum() + (x < -1).sum() == 0
    return x, sr


def id_to_torch(aux_id, cuda=False, device="cpu"):
    if cuda:
        device = "cuda"
    if aux_id is not None:
        aux_id = np.asarray(aux_id)
        aux_id = torch.from_numpy(aux_id).to(device)
    return aux_id


def numpy_to_torch(np_array, dtype, cuda=False, device="cpu"):
    if cuda:
        device = "cuda"
    if np_array is None:
        return None
    tensor = torch.as_tensor(np_array, dtype=dtype, device=device)
    return tensor


def embedding_to_torch(d_vector, cuda=False, device="cpu"):
    if cuda:
        device = "cuda"
    if d_vector is not None:
        d_vector = np.asarray(d_vector)
        d_vector = torch.from_numpy(d_vector).type(torch.FloatTensor)
        d_vector = d_vector.squeeze().unsqueeze(0).to(device)
    return d_vector


def inference(model, ref_wav, text, language_id_code=None, device="cuda"):
    espeak_language_codes = {
        "en": "en-us",
        "pt-br": "pt-br",
        "pl": "pl",
        "it": "it",
        "fr": "fr-fr",
        "du": "nl",
        "ge": "de",
        "sp": "es"
    }

    text_cleaner = TextCleaner()

    d_vector = model.speaker_manager.compute_embedding_from_clip(ref_wav)
    wav, sr = load_audio(ref_wav)
    if sr != model.config.audio["sample_rate"]:
        transform = torchaudio.transforms.Resample(sr, model.config.audio["sample_rate"])
        wav = transform(wav)
        sr = model.config.audio["sample_rate"]
    spec = wav_to_spec(
        wav,
        model.config.audio.fft_size,
        model.config.audio.hop_length,
        model.config.audio.win_length,
        center=False,
    )

    spec = torch.tensor(spec, dtype=torch.float32, device=device)

    mel = spec_to_mel(
        spec=spec,
        n_fft=model.config.audio.fft_size,
        num_mels=model.config.audio.num_mels,
        sample_rate=model.config.audio.sample_rate,
        fmin=model.config.audio.mel_fmin,
        fmax=model.config.audio.mel_fmax,
    )

    language_id = model.language_manager.name_to_id.get(language_id_code, None)

    language_name = None
    if language_id is not None:
        language = [k for k, v in model.language_manager.name_to_id.items() if v == language_id]
        assert len(language) == 1, "language_id must be a valid language"
        language_name = language[0]

    print("Language Name: ", language_name)
    print(f"Language ID: {language_id}")

    global_phonemizer = phonemizer.backend.EspeakBackend(
        language=espeak_language_codes[language_id_code],
        preserve_punctuation=True,
        with_stress=True
    )

    text = text.strip()

    phonemes = global_phonemizer.phonemize([text])
    tokenized_phonemes = word_tokenize(phonemes[0])
    tokenized_phonemes = ' '.join(tokenized_phonemes)

    token_ids_phonems_styletts = text_cleaner(tokenized_phonemes)
    token_ids_phonems_styletts = numpy_to_torch(token_ids_phonems_styletts, torch.long, device=device)
    token_ids_phonems_styletts = token_ids_phonems_styletts.unsqueeze(0)

    print("Tokenized Phonemes: ", tokenized_phonemes)
    print("Token IDs Phonems StyleTTS: ", token_ids_phonems_styletts)

    d_vectors = embedding_to_torch(d_vector, device=device)

    if language_id is not None:
        language_id = id_to_torch(language_id, device=device)

    return model.inference(
        token_ids_phonems_styletts,
        aux_input={
            "x_lengths": None,
            "d_vectors": d_vectors,
            "speaker_ids": None,
            "language_ids": language_id,
            "durations": None,
            "spec": spec,
            "mel": mel,
        },
    )["model_outputs"]


def main():
    CONFIG_PATH = "/raid/alefiury/translation/Coqui-TTS-Mod/recipes/multilingual/vipt/VIPT-CML-PL-BERT-June-16-2024_03+45PM-36c6b3554/config.json"
    CHECKPOINT_PATH = "/raid/alefiury/translation/Coqui-TTS-Mod/recipes/multilingual/vipt/VIPT-CML-PL-BERT-June-16-2024_03+45PM-36c6b3554/checkpoint_350000.pth"
    LANGUAGE_ID_PATH = "/raid/alefiury/translation/Coqui-TTS-Mod/recipes/multilingual/vipt/VIPT-CML-PL-BERT-June-16-2024_03+45PM-36c6b3554/language_ids.json"
    ALC_DATASET_PATH = "/raid/fred/DATASETS/dataset_alc_new_24khz"
    OUTPUT_PATH = "/raid/alefiury/translation/Coqui-TTS-Mod/recipes/multilingual/vipt/output_alc_cml_mel_plbert"

    TEST_METADATA_PATH = "/raid/alefiury/translation/Coqui-TTS-Mod/recipes/multilingual/vipt/test_metadata_complete.csv"

    with open(LANGUAGE_ID_PATH, "r") as f:
        language_ids = json.load(f)

    df = pd.read_csv(TEST_METADATA_PATH, header=None, names=["filename", "transcription", "transcription_en"], sep="|")
    test_sentences = []

    for row in df.iterrows():
        filename = row[1]["filename"]
        pt_transcription = row[1]["transcription"]
        en_transcription = row[1]["transcription_en"]

        test_sentences.append(
                [
                    pt_transcription,
                    "pt-br",
                    os.path.join(ALC_DATASET_PATH, filename),
                ]
        )

        test_sentences.append(
            [
                en_transcription,
                "en",
                os.path.join(ALC_DATASET_PATH, filename),
            ]
        )

    config = VitsConfig()
    config.load_json(CONFIG_PATH)
    # print(config["d_vector_file"] )
    config["d_vector_file"] = None # Remove speakers_file from config
    config["speakers_file"] = None
    # print(config["model_args"])
    config["model_args"]["speakers_file"] = None
    config["model_args"]["d_vector_file"] = None
    config["model_args"]["language_ids_file"] = LANGUAGE_ID_PATH
    config["language_ids_file"] = LANGUAGE_ID_PATH
    print(config["model_args"]["speakers_file"])
    print(config["model_args"]["d_vector_file"])
    # exit()
    model = Vits.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=CHECKPOINT_PATH, eval=True)
    model.cuda()

    for test_sentence in tqdm(test_sentences):
        sentence, langid, ref_wav = test_sentence
        wav = inference(model, ref_wav, sentence, langid)
        sub_dir = os.path.basename(ref_wav).replace(".wav", "")
        OUTPUT_PATH_EXT = os.path.join(OUTPUT_PATH, sub_dir)
        os.makedirs(OUTPUT_PATH_EXT, exist_ok=True)
        shutil.copy2(ref_wav, os.path.join(OUTPUT_PATH_EXT, "ref.wav"))
        torchaudio.save(os.path.join(OUTPUT_PATH_EXT, f"{langid}-{os.path.basename(CHECKPOINT_PATH).replace('.pth', '').replace('checkpoint_', '')}.wav"), wav.squeeze(0).cpu(), 24000)


if __name__ == "__main__":
    main()