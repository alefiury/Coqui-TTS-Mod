import os
import pandas as pd


def main():
    metadata_path = "dataset/transcriptions_cleaned.csv"

    df = pd.read_csv(metadata_path, sep="|", header=None, names=["wav_filename", "transcript"])

    print(df)

    # Split into train and test, take 100 samples for test
    test_df = df.sample(n=100)
    train_df = df.drop(test_df.index)

    print(train_df)
    print(test_df)

    train_df.to_csv("dataset/train_metadata.csv", sep="|", header=False, index=False)
    test_df.to_csv("dataset/test_metadata.csv", sep="|", header=False, index=False)


if __name__ == '__main__':
    main()