#!/usr/bin/env python3

def clocks(n):
    P = []
    for x in range(2, n + 1):
        if all(x % q for q in P):
            P.append(x)
    return P


def level(p, z):
    P = clocks(p)

    X = [
        p + 1 + y*p + x
        for y in range(p)
        for x in range(p)
        if all((p + 1 + y*p + x) % q for q in P)
    ]

    XY = {x*y for x in X for y in X}

    L = []

    for _ in range(p):
        while not (
            all(
                z % q not in (1, q - 1)
                or z - 1 == q
                or z + 1 == q
                for q in P
            )
            and z - 1 not in XY
            and z + 1 not in XY
        ):
            z += 2 * 3

        L.append(z)
        z += 2 * 3

    return tuple(L)


def run(n):
    z = 2 * 3

    for p in clocks(n)[2:]:
        L = level(p, z)
        print(p, L)
        z = L[-1]


if __name__ == "__main__":
    import sys
    run(int(sys.argv[1]) if len(sys.argv) > 1 else int(input("Prime ceiling: ")))
