import numpy as np


if __name__ == "__main__":
    abalones = np.genfromtxt(
        "abalone.data",
        delimiter=",",
        dtype=(np.float32, (9,)),
        converters={
            0: lambda s: {b"M": 0.0, b"F": 1.0}.get(s, 2.0)
        },
    )
    for abalone in abalones:
        print(abalone)
