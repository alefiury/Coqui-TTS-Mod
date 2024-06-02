import os
import glob
import argparse
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
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


def get_total_dataset_length(base_dir: str, metadata_path: str, workers: int) -> None:
    """
    Gets information related to the length of the audios
    and the amount of data in dataset itself, using parallel processing.
    """
    df = pd.read_csv(metadata_path)

    file_paths = [os.path.join(base_dir, row['audio_segment_path']) for index, row in df.iterrows()]
    # file_paths = file_paths[:10]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(process_file, file_paths), total=len(file_paths)))

    metadata = []

    for audio_info, filepath in results:
        length = audio_info.length / audio_info.sample_rate

        metadata.append({"path": filepath.replace(base_dir+"/", ""), "length": length})

    return metadata


def main() -> None:
    metadata_path = "train_sim.csv"
    base_dir = "/raid/fred/DATASETS"
    num_workers = 12

    metadata = get_total_dataset_length(base_dir, metadata_path, num_workers)

    df = pd.read_csv(metadata_path)

    metadata_dict = {item['path']: item['length'] for item in metadata}
    # Map the duration from metadata_dict to the DataFrame
    df['duration'] = df['audio_segment_path'].map(metadata_dict)

    print(df.columns)
    print(df[['audio_segment_path', 'duration']])

    df.to_csv("train_sim_duration.csv", index=False)


if __name__ == '__main__':
    main()
