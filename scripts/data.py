# descarga de datos
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.25
RANDOM_STATE = 123456
URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DATA_FOLDER = "data/winequality-red/"


def main():

    df = pd.read_csv(URL, sep=";")

    # preparacion de datos
    y = df["quality"]
    x = df.copy()
    x.pop("quality")

    # dividir los datos en entrenamiento y testing
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # crear la carpeta "data" si no existe

    if os.path.exists(DATA_FOLDER) == False:
        os.makedirs(DATA_FOLDER)
    # guardar los datos en la carpeta "data"
    x_train.to_csv(Path(DATA_FOLDER) / "x_train.csv", index=False)
    x_test.to_csv(Path(DATA_FOLDER) / "x_test.csv", index=False)
    y_train.to_csv(Path(DATA_FOLDER) / "y_train.csv", index=False)
    y_test.to_csv(Path(DATA_FOLDER) / "y_test.csv", index=False)


if __name__ == "__main__":
    main()
