#!/usr/bin/env python3


def add_phase(R, step, P):
    return tuple((r + step) % q for q, r in zip(P, R))


def clocks(n):
    P = [2]
    R = [0]

    for x in range(3, n + 1):
        R = [(r + 1) % q for q, r in zip(P, R)]

        if all(R):
            P.append(x)
            R.append(0)

    return tuple(P)


def phases(p):
    P = clocks(p)
    x = p
    R = tuple(x % q for q in P)
    A = []

    for _ in range(p * p):
        x += 1
        R = add_phase(R, 1, P)

        if all(R):
            A.append(x)

    return P, tuple(A)


def level(p, start):
    P, A = phases(p)
    C = {a * b for a in A for b in A}

    out = []
    c = start
    R = tuple(c % q for q in P)

    while len(out) < p:
        if (all(r not in (1, q - 1) or c - 1 == q or c + 1 == q
                for q, r in zip(P, R))
                and c - 1 not in C
                and c + 1 not in C):
            out.append(c)

        c += 6
        R = add_phase(R, 6, P)

    return tuple(out)


def run(n):
    start = 2 * 3

    for p in clocks(n)[2:]:
        L = level(p, start)
        print(p, L)
        start = L[-1]


if __name__ == "__main__":
    import sys
    run(int(sys.argv[1]) if len(sys.argv) > 1 else int(input("Prime ceiling: ")))
