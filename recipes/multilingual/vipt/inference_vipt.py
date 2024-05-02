import os
import json
import torchaudio
import numpy as np
import torch
from tqdm import tqdm
import shutil

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, wav_to_spec


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


def inference(model, ref_wav, text, language_id=None, device="cuda"):
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

    print("Spec.shape", spec.shape)

    spec = torch.tensor(spec, dtype=torch.float32, device=device)

    print("Spec2.shape", spec.shape)

    language_id = model.language_manager.name_to_id.get(language_id, None)

    language_name = None
    if language_id is not None:
        language = [k for k, v in model.language_manager.name_to_id.items() if v == language_id]
        assert len(language) == 1, "language_id must be a valid language"
        language_name = language[0]

    print("Language Name: ", language_name)
    print(f"Language ID: {language_id}")

    text_inputs = np.asarray(
        model.tokenizer.text_to_ids(text, language=language_name),
        dtype=np.int32,
    )

    text_inputs = numpy_to_torch(text_inputs, torch.long, device=device)
    text_inputs = text_inputs.unsqueeze(0)

    d_vectors = embedding_to_torch(d_vector, device=device)

    if language_id is not None:
        language_id = id_to_torch(language_id, device=device)

    return model.inference(
        text_inputs,
        aux_input={
            "x_lengths": None,
            "d_vectors": d_vectors,
            "speaker_ids": None,
            "language_ids": language_id,
            "durations": None,
            "spec": spec
        },
    )["model_outputs"]


def main():
    CONFIG_PATH = "/raid/alefiury/translation/Coqui-TTS-Mod/recipes/multilingual/vipt/VIPT-ALC-April-23-2024_08+10PM-3a7f2229/config.json"
    CHECKPOINT_PATH = "/raid/alefiury/translation/Coqui-TTS-Mod/recipes/multilingual/vipt/VIPT-ALC-April-23-2024_08+10PM-3a7f2229/checkpoint_560000.pth"
    LANGUAGE_ID_PATH = "/raid/alefiury/translation/Coqui-TTS-Mod/recipes/multilingual/vipt/VIPT-ALC-April-23-2024_08+10PM-3a7f2229/language_ids.json"
    ALC_DATASET_PATH = "/raid/fred/DATASETS/dataset_alc_new_24khz"
    OUTPUT_PATH = "/raid/alefiury/translation/Coqui-TTS-Mod/recipes/multilingual/vipt/output_alc"

    TEST_METADATA_PATH = "/raid/fred/DATASETS/dataset_alc_new_24khz/test_metadata.csv"

    with open(LANGUAGE_ID_PATH, "r") as f:
        language_ids = json.load(f)

    test_sentences=[
        [
            "Cuidado! Só estamos invadindo um navio pirata pra achar uma ovelha. Não se preocupa! Essa não.",
            "pt-br",
            os.path.join(ALC_DATASET_PATH, "d750d6/100/2f22ed5224.wav")
        ],
        [
            "Careful! We're just raiding a pirate ship to find a sheep. Do not worry! Not this one.",
            "en",
            os.path.join(ALC_DATASET_PATH, "d750d6/100/2f22ed5224.wav")
        ],
        [
            "A partir de então, o sonho se repetia toda noite. Ela sonhava no avião, indo para Calcutá e acordava gritando.",
            "pt-br",
            os.path.join(ALC_DATASET_PATH, "ee23b4/100/ae5311b74d.wav")
        ],
        [
            "From then on, the dream repeated itself every night. She would dream on the plane, going to Calcutta and wake up screaming.",
            "en",
            os.path.join(ALC_DATASET_PATH, "ee23b4/100/ae5311b74d.wav")
        ],
    ]

    config = VitsConfig()
    config.load_json(CONFIG_PATH)
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