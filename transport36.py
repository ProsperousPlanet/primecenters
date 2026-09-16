#!/usr/bin/env python3
from bisect import bisect_right
from math import ceil, isqrt
import numpy as np

class TwinCenterGenerator:
    STEP = 2 * 3
    FRAME = STEP - 2

    def __init__(self):
        self.primes = [2, 3, self.STEP - 1]
        self.scan_start, self.pending = 2 * self.STEP, []
        self.wheel, self.wheel_period, self.wheel_prime_index = b'\0', 1, 1
        self.density, self.density_prime_index = 1.0, 1

    def extend_primes(self, limit):
        if self.primes[-1] >= limit: return
        limit = max(limit, self.primes[-1] * 2)
        sieve = bytearray(b'\1') * ((limit + 1) // 2); sieve[0] = 0
        for prime in range(3, isqrt(limit) + 1, 2):
            if sieve[prime // 2]:
                start = prime * prime // 2
                sieve[start::prime] = b'\0' * ((len(sieve) - 1 - start) // prime + 1)
        self.primes = [2] + [2 * i + 1 for i in range(1, len(sieve)) if sieve[i]]

    def density_to(self, limit):
        stop = bisect_right(self.primes, limit) - 1
        for index in range(self.density_prime_index + 1, stop + 1):
            prime = self.primes[index]
            self.density *= (prime - 2) / prime
        self.density_prime_index = max(self.density_prime_index, stop)
        return self.density

    def batch(self, required):
        count = self.wheel_period
        while count < required: count += count
        return count

    def fold_wheel(self, first, count):
        start = self.wheel_prime_index + 1
        for index, prime in enumerate(self.primes[start:], start):
            period = self.wheel_period * prime
            if prime + 1 >= first or period > count: break
            wheel = bytearray(self.wheel * prime)
            inverse = ((self.STEP - prime % self.STEP) * prime + 1) // self.STEP
            for hit in (inverse, -inverse % prime):
                wheel[hit::prime] = b'\1' * ((period - 1 - hit) // prime + 1)
            self.wheel, self.wheel_period, self.wheel_prime_index = bytes(wheel), period, index

    def generate(self, needed):
        while len(self.pending) < needed:
            first = self.scan_start
            target = self.FRAME * (needed - len(self.pending))
            count = self.batch(ceil(target / self.density))
            reset = first * (first - 2)

            while True:
                end = min(first + self.STEP * count, reset)
                count, limit = (end - first) // self.STEP, isqrt(end)
                self.extend_primes(limit)
                required = ceil(target / self.density_to(limit))
                if count >= required or end == reset: break
                count = self.batch(required)

            self.scan_start = end; self.fold_wheel(first, count)
            address, offset = first // self.STEP, (first // self.STEP) % self.wheel_period
            repeats = (offset + count + self.wheel_period - 1) // self.wheel_period
            blocked = bytearray((self.wheel * repeats)[offset:offset + count])
            for prime in self.primes[self.wheel_prime_index + 1:bisect_right(self.primes, limit)]:
                inverse = ((self.STEP - prime % self.STEP) * prime + 1) // self.STEP
                for hit in ((inverse - address) % prime, (-inverse - address) % prime):
                    blocked[hit::prime] = b'\1' * ((count - 1 - hit) // prime + 1)
            survivors = np.flatnonzero(np.frombuffer(blocked, dtype=np.uint8) == 0)
            self.pending.extend((first + self.STEP * survivors).tolist())

    def levels(self, ceiling, output=True):
        self.extend_primes(ceiling)
        start = self.STEP
        for prime in self.primes[2:bisect_right(self.primes, ceiling)]:
            self.generate(prime - 1)
            if output:
                centers = (start, *self.pending[:prime - 1])
                yield prime, centers
                start = centers[-1]
            else:
                start = self.pending[prime - 2]
            del self.pending[:prime - 1]

if __name__ == '__main__':
    import sys
    quiet = '--noprint' in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != '--noprint']
    ceiling = int(args[0]) if args else int(input('Prime ceiling: '))
    for prime, centers in TwinCenterGenerator().levels(ceiling, not quiet):
        print(f'**{prime}**', centers)
