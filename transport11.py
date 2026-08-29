#!/usr/bin/env python3

def clocks(n):
    P = [2]
    for x in range(3, n + 1):
        if all(x % q for q in P):
            P.append(x)
    return tuple(P)


def phases(p):
    P, x, A = clocks(p), p, []

    for _ in range(p):
        for _ in range(p):
            x += 1
            if all(x % q for q in P):
                A.append(x)

    return P, tuple(A)


def level(p, start):
    P, A = phases(p)
    C = {a * b for a in A for b in A}
    out = []

    while len(out) < p:
        R = tuple(start % q for q in P)

        if (all(r not in (1, q - 1) or start - 1 == q or start + 1 == q
                for q, r in zip(P, R))
                and start - 1 not in C
                and start + 1 not in C):
            out.append(start)

        start += P[0] * P[1]

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
