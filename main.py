import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=150)


if __name__ == "__main__":
    abalones = np.genfromtxt(
        "abalone.data",
        delimiter=",",
        dtype=(np.float32, (9,)),
        converters={
            0: lambda s: {b"M": 0.0, b"F": 1.0}.get(s, 2.0),
            8: lambda r: 1 if int(r) <= 8 else 2 if int(r) <= 10 else 3,
        },
    )
    for abalone in abalones:
        print(abalone)
