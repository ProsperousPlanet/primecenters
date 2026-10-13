#!/usr/bin/env python3
from itertools import combinations_with_replacement

def clocks(n):
    P = []
    for x in range(2, n + 1):
        if all(x % q for q in P):
            P.append(x)
    return P

def level(p, z, step=2*3, gap=1):
    P = clocks(p)

    # One complete forward p-cycle.
    phase = tuple((z + step*k) % p for k in range(p))

    # Two synchronized copies give only the diagonal of the p × p square.
    XY = tuple((x, x) for x in phase)

    # New surviving factors inside the p² counting domain.
    X = [x for x in range(p + 1, p*p + p + 1)
         if all(x % q for q in P)]

    # Fold X × X across x=y instead of computing both halves.
    C = {x*y for x, y in combinations_with_replacement(X, 2)}

    L = []

    while len(L) < len(XY):
        if (all(z % q not in (gap % q, (-gap) % q)
                or z - gap == q or z + gap == q
                for q in P)
                and z - gap not in C
                and z + gap not in C):
            L.append(z)

        z += step

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
