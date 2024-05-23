import os

import pandas as pd

def main():
    metadata_path = "train_sim.csv"

    df = pd.read_csv(metadata_path)

    print(df)

    # show the distribution of the "cer" column
    print(df["cer"].describe())
    print(df["normalized_sim"].describe())
    print(df.columns)

    # filter the dataframe by the "normalized_sim" column, remove values that are less than 0.5
    new_df = df[df["cer"] < 0.05]

    print(new_df)

    # try to find the best "normalized_sim" threshold for filtering sentences
    # The threashold is between 0.1 and 1.0
    for i in range(1, 10):
        threshold = i / 10
        print(f"Threshold: {threshold}")
        new_df = df[df["normalized_sim"] > threshold]
        # print number of samples
        print(new_df.shape)
        # write 100 samples at random to a file
        new_df.sample(100).to_csv(f"sim/train_sim_{threshold}.csv", index=False)




if __name__ == "__main__":
    main()