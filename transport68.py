#!/usr/bin/env python3

BATCH = 3 * 8

def primes():
    P, n = [], 2
    while True:
        if all(n % p for p in P if p*p <= n):
            P.append(n); yield n
        n = 3 if n == 2 else n + 2

def centers():
    source = primes()
    next(source); next(source)
    p, clocks, lo = next(source), [], 1

    while True:
        for _ in range(BATCH):
            square = (p*p - 1)//6
            gap = (p + 1)//3 if p % 6 == 5 else (2*p + 1)//3
            clocks.append([p, square, square + gap])
            p = next(source)

        hi = (p*p - 1)//6
        schedule = bytearray(hi - lo)

        for clock in clocks:
            q = clock[0]
            for side in (1, 2):
                x = clock[side]
                if x < hi:
                    n = (hi - 1 - x)//q + 1
                    schedule[x-lo::q] = b'\1'*n
                    clock[side] = x + n*q

        i = schedule.find(0)
        while i >= 0:
            yield 6*(lo + i)
            i = schedule.find(0, i + 1)
        lo = hi

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
