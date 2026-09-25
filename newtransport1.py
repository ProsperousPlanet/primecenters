#!/usr/bin/env python3
BATCH = 3 * 8

def primes():
    P, n = [], 2
    while True:
        for p in P:
            if p*p > n: break
            if n % p == 0: break
        else: p = 0
        if not p or p*p > n:
            P.append(n); yield n
        n = 3 if n == 2 else n + 2

from itertools import count

def centers():
    source = primes()
    next(source); next(source)
    incoming, clocks = next(source), []

    for n in count(1):
        while incoming**2 <= 6*n + 1:
            clocks.append([incoming, (6*n-1) % incoming, (6*n+1) % incoming])
            incoming = next(source)

        if all(left and right for _, left, right in clocks):
            yield 6*n

        for clock in clocks:
            clock[1] = (clock[1] + 6) % clock[0]
            clock[2] = (clock[2] + 6) % clock[0]

if __name__ == '__main__':
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else int(input('Prime ceiling: '))
    stream, opening = centers(), None
    for p in primes():
        if p < 5: continue
        if p > limit: break
        row = [next(stream)] if opening is None else [opening]
        row += [next(stream) for _ in range(p-len(row))]
        opening = row[-1]
        print(f'**{p}**', tuple(row))
