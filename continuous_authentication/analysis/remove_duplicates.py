import pandas as pd
from pathlib import PurePath

def main():
    files = [95977, 731275, 867544, 808884]

    for file in files: 
        path = PurePath(f"../../data/clarkson2_files/{file}")
        df = pd.read_csv(path, sep = "\t", header = None)
        length = df.shape[0]
        true_length = int(length / 2) if file != 731275 else int(length / 3)
        df_actual = df.iloc[:true_length]

        df_actual.to_csv(path, sep = "\t", index = False, header = False)
        print(f"Finished writing to {path}")

if __name__ == "__main__":
    main()