import os
import glob
import argparse
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor

import torchaudio
import numpy as np
from tqdm import tqdm

Info = namedtuple("Info", ["length", "sample_rate", "channels"])


def get_audio_info(path: str) -> namedtuple:
    """
    Get basic information related to number of frames,
    sample rate and number of channels.
    """

    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)


def process_file(file_path: str) -> namedtuple:
    return get_audio_info(file_path), file_path


def get_total_dataset_length(base_dir: str, workers: int) -> None:
    """
    Gets information related to the length of the audios
    and the amount of data in dataset itself, using parallel processing.
    """
    length = []
    srs = []
    channels = []

    extensions = ["wav", "mp3", "flac", "ogg"]

    file_paths = []
    for ext in tqdm(extensions, desc="Collecting files", total=len(extensions)):
        fs = glob.glob(os.path.join(base_dir, "**", f"*.{ext}"), recursive=True)
        print(f"Found {len(fs)} files with extension {ext}")
        file_paths.extend(fs)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(process_file, file_paths), total=len(file_paths)))

    for audio_info, file_path in results:
        srs.append(audio_info.sample_rate)
        channels.append(audio_info.channels)
        length.append(audio_info.length / audio_info.sample_rate)
        l = audio_info.length / audio_info.sample_rate
        if l < 1:
            with open("short_audios.txt", "a") as f:
                f.write(f"{file_path}\n")

    print(f"Min audio length (in seconds): {min(length)} | Max audio length (in seconds): {max(length)}")
    print(f"Mean audio length (in seconds): {np.mean(length)} | Median: {np.median(length)} | Std: {np.std(length)}")
    print(f"Total amount of data (in minutes): {np.sum(length)/60}")
    print('-'*50)
    print(f"Different samplerates in audios in the dataset: {set(srs)}")
    print(f"Different number of channels in the audios in the dataset: {set(channels)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data-base-dir',
        default='../data/train',
        type=str,
        help="Base directory where the audio data is stored"
    )
    parser.add_argument(
        '-w',
        '--workers',
        default=8,
        type=int,
        help="Number of parallel workers"
    )
    args = parser.parse_args()

    get_total_dataset_length(args.data_base_dir, args.workers)

if __name__ == '__main__':
    main()
