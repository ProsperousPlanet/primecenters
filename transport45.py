#!/usr/bin/env python3
from heapq import heappop, heappush
from math import isqrt

class TwinCenters:
    STEP = 2 * 3

    def __init__(self):
        self.primes = [2, 3, self.STEP - 1]
        self.base_count = len(self.primes)

    def primes_to(self, limit):
        if self.primes[-1] >= limit: return
        limit = max(limit, 2*self.primes[-1])
        sieve = bytearray(b'\1')*((limit + 1)//2); sieve[0] = 0
        for p in range(3, isqrt(limit) + 1, 2):
            if sieve[p//2]:
                start = p*p//2
                sieve[start::p] = b'\0'*((len(sieve) - 1 - start)//p + 1)
        self.primes = [2] + [2*i + 1 for i in range(1, len(sieve)) if sieve[i]]

    def children(self, residue, period, prime, floor):
        span = period*prime
        cycle, phase = divmod(floor, span)
        first = ((phase - residue + period - 1)//period) % prime
        inverse = pow(self.STEP*period, -1, prime)

        left = ((1 - self.STEP*residue)*inverse - first) % prime
        right = ((-1 - self.STEP*residue)*inverse - first) % prime
        left, right = sorted((left, right))

        for rank in range(prime - 2):
            child = (first + rank + (rank >= left) + (rank >= right - 1)) % prime
            next_residue = residue + period*child
            yield (cycle*span + next_residue
                   + (next_residue < phase)*span,
                   next_residue, span)

    def take(self, start, amount):
        heap, output = [], []

        def push(stream, index):
            try:
                value, residue, period = next(stream)
                heappush(heap, (value, index, residue, period, stream))
            except StopIteration:
                pass

        push(self.children(0, 1, self.STEP - 1, start), self.base_count)

        while len(output) < amount:
            value, index, residue, period, stream = heappop(heap)
            if stream: push(stream, index)

            while index >= len(self.primes):
                self.primes_to(2*self.primes[-1])

            prime = self.primes[index]

            if self.STEP*value + 1 < prime*prime:
                output.append(value)
                heappush(heap, (value + period, index, residue, period, None))
            else:
                push(self.children(residue, period, prime, value), index + 1)

        return output

    def levels(self, ceiling):
        step = self.STEP
        first = (step, 2*step, 3*step, (2+3)*step, (2+3+2)*step)

        if ceiling >= step - 1:
            yield step - 1, first

        opening = first[-1]
        self.primes_to(ceiling)

        for prime in self.primes[self.base_count:]:
            if prime > ceiling: break
            centers = tuple(step*n for n in self.take(opening//step, prime))
            yield prime, centers
            opening = centers[-1]

if __name__ == '__main__':
    import sys

    quiet = '--noprint' in sys.argv
    args = [x for x in sys.argv[1:] if x != '--noprint']
    ceiling = int(args[0]) if args else int(input('Prime ceiling: '))

    for prime, centers in TwinCenters().levels(ceiling):
        if not quiet:
            print(f'**{prime}**', centers)
