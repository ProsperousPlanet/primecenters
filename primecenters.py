#!/usr/bin/env python3
"""
structural_prime_center_explorer.py

Purpose
-------
Study the exact middle of the primes inside the square-domain ladder

    8 = 2^3
    64 = 2^6
    4096 = 2^12
    16,777,216 = 2^24
    2^48
    2^96
    ...

while keeping the experimental algebraic structure separate from the
external prime-certification step.

The program tests:

1. Exact prime center of each domain.
2. Exact half-gap of the two middle primes when pi(D) is even.
3. Fibonacci half-gap hypothesis:
       F_2, F_4, F_6, [F_8 hidden at odd reset?], F_10, ...
4. Nearest-square correction:
       center = r^2 + correction
5. Triangular correction branch:
       doorway index = previous power-of-2 exponent - 2
       T_n = n(n+1)/2
       giving n = 1,4,10,22,... and T_n = 1,10,55,253,...
6. Gold/Fibonacci alternatives after the 55 junction:
       89, 144, T_22=253, T_55=1540, F_34=5,702,887
7. Nearest Fibonacci, triangular number, and power of 3 to the
   observed correction.
8. The basic x/y/z construction:
       x=1
       y=x+x=2
       z=x+y=3
       A=y+y=y*y=y^y=4
       T_4=10
       T_10=55
       R(n)=2n+2, so R(4)=10 and R(10)=22

Prime certification
-------------------
For domains through 2^24 the program uses its own ordinary sieve.

For larger domains it uses an installed `primecount` executable, but it
only relies on the simple command:

    primecount N

which returns pi(N).  It then locates the n-th prime itself by binary
searching pi(x).  This avoids depending on any special nth-prime CLI flag.

This program deliberately does NOT use the experimental Gold/triangular
rules to decide what is prime.  First measure the exact center; then test
the structural hypotheses against it.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
from functools import lru_cache


# ---------------------------------------------------------------------------
# Primitive construction
# ---------------------------------------------------------------------------

X = 1
Y = X + X          # 2
Z = X + Y          # 3
A = Y + Y          # 4


def triangular(n: int) -> int:
    return n * (n + 1) // 2


def reset(n: int) -> int:
    """The experimental +2 reset: n -> y*n + y = 2n+2."""
    return Y * n + Y


def fib(n: int) -> int:
    """F_1=1, F_2=1."""
    if n <= 0:
        return 0
    a, b = 1, 1
    if n <= 2:
        return 1
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b


def fibonacci_up_to_value(limit: int):
    out = [(1, 1), (2, 1)]
    a, b = 1, 1
    i = 2
    while b < limit:
        a, b = b, a + b
        i += 1
        out.append((i, b))
    return out


def nearest_fibonacci(value: int):
    value = abs(int(value))
    if value <= 1:
        return 1, 1, abs(value - 1)
    seq = fibonacci_up_to_value(value)
    # include one term above
    i, last = seq[-1]
    if last < value:
        seq.append((i + 1, seq[-1][1] + seq[-2][1]))
    return min(
        ((i, f, abs(value - f)) for i, f in seq),
        key=lambda t: t[2]
    )


def nearest_triangular(value: int):
    value = abs(int(value))
    # solve n(n+1)/2 ~= value
    n0 = max(0, (math.isqrt(1 + 8 * value) - 1) // 2)
    candidates = []
    for n in range(max(0, n0 - 2), n0 + 4):
        t = triangular(n)
        candidates.append((n, t, abs(value - t)))
    return min(candidates, key=lambda t: t[2])


def nearest_power3(value: int):
    value = abs(int(value))
    if value <= 1:
        return 0, 1, abs(value - 1)
    k = max(0, round(math.log(value, 3)))
    candidates = []
    for j in range(max(0, k - 2), k + 3):
        p = 3 ** j
        candidates.append((j, p, abs(value - p)))
    return min(candidates, key=lambda t: t[2])


# ---------------------------------------------------------------------------
# Exact primes through 2^24: ordinary sieve
# ---------------------------------------------------------------------------

LOCAL_SIEVE_LIMIT = 1 << 24


def odd_sieve(n: int):
    if n < 2:
        return []
    size = n // 2 + 1
    s = bytearray(b"\x01") * size
    s[0] = 0  # number 1

    for p in range(3, math.isqrt(n) + 1, 2):
        if s[p // 2]:
            start = (p * p) // 2
            step = p
            count = ((size - 1 - start) // step) + 1
            s[start::step] = b"\x00" * count

    return [2] + [
        2 * i + 1
        for i in range(1, size)
        if s[i] and 2 * i + 1 <= n
    ]


# ---------------------------------------------------------------------------
# primecount wrapper for large domains
# ---------------------------------------------------------------------------

class PrimeCounter:
    def __init__(self):
        self.exe = shutil.which("primecount")
        self.local_primes = None

    @lru_cache(maxsize=None)
    def pi(self, n: int) -> int:
        n = int(n)
        if n <= LOCAL_SIEVE_LIMIT:
            if self.local_primes is None:
                self.local_primes = odd_sieve(LOCAL_SIEVE_LIMIT)
            # binary search count
            import bisect
            return bisect.bisect_right(self.local_primes, n)

        if not self.exe:
            raise RuntimeError(
                "This domain is larger than 2^24 and `primecount` was not found.\n"
                "Install primecount, then rerun this program."
            )

        raw = subprocess.check_output(
            [self.exe, str(n)],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

        # Accept either a bare integer or text containing one final integer.
        nums = re.findall(r"\d+", raw.replace(",", ""))
        if not nums:
            raise RuntimeError(f"Could not parse primecount output:\n{raw}")
        return int(nums[-1])

    def nth_prime(self, k: int) -> int:
        """Smallest n with pi(n) >= k."""
        if k < 1:
            raise ValueError("Prime rank must be >= 1")

        if k == 1:
            return 2

        # If the desired prime is local, return it directly.
        if self.local_primes is None:
            self.local_primes = odd_sieve(LOCAL_SIEVE_LIMIT)
        if k <= len(self.local_primes):
            return self.local_primes[k - 1]

        if not self.exe:
            raise RuntimeError(
                "Need `primecount` to locate a prime of this rank."
            )

        # Safe broad bracket based on the standard n(log n + log log n)
        # upper scale. Expand if necessary rather than trusting the estimate.
        kk = float(k)
        logk = math.log(kk)
        loglogk = math.log(logk)

        lo = max(2, int(kk * max(1.0, logk + loglogk - 1.25)))
        hi = int(kk * (logk + loglogk + 0.50)) + 100

        while self.pi(lo) >= k:
            lo //= 2
        while self.pi(hi) < k:
            hi *= 2

        # Binary search the first integer whose prime count reaches k.
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.pi(mid) >= k:
                hi = mid
            else:
                lo = mid
        return hi

    def next_prime_after_rank(self, p: int, rank: int) -> int:
        """Find prime of rank rank+1, using p as the prime of rank rank."""
        target = rank + 1
        lo = p
        step = 2
        hi = p + step

        while self.pi(hi) < target:
            step *= 2
            hi = p + step

        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.pi(mid) >= target:
                hi = mid
            else:
                lo = mid
        return hi


# ---------------------------------------------------------------------------
# Center measurement
# ---------------------------------------------------------------------------

def exact_center_for_domain(D: int, pc: PrimeCounter):
    count = pc.pi(D)

    if count % 2 == 1:
        rank = (count + 1) // 2
        p = pc.nth_prime(rank)
        return {
            "count": count,
            "kind": "single",
            "lower": p,
            "upper": p,
            "center_num": p,
            "center_den": 1,
            "half_gap_num": None,
            "half_gap_den": None,
        }

    k = count // 2
    p1 = pc.nth_prime(k)
    p2 = pc.next_prime_after_rank(p1, k)

    # p1,p2 are odd except tiny cases, so this is an integer in our domains.
    return {
        "count": count,
        "kind": "pair",
        "lower": p1,
        "upper": p2,
        "center_num": p1 + p2,
        "center_den": 2,
        "half_gap_num": p2 - p1,
        "half_gap_den": 2,
    }


def rational_to_int(num, den):
    if num % den == 0:
        return num // den
    return None


def nearest_square(center: int):
    r0 = math.isqrt(center)
    r1 = r0 + 1
    if abs(center - r0 * r0) <= abs(center - r1 * r1):
        r = r0
    else:
        r = r1
    return r, center - r * r


# ---------------------------------------------------------------------------
# Structural report
# ---------------------------------------------------------------------------

def compare_candidate(label: str, actual: int, candidate: int):
    diff = actual - candidate
    exact = " EXACT" if diff == 0 else ""
    print(
        f"      {label:<24} {candidate:>18,}   "
        f"actual-candidate={diff:+,}{exact}"
    )


def domain_tier(exponent: int):
    """Return tier s for exponent = 3*2^s, otherwise None."""
    if exponent < 3 or exponent % 3:
        return None
    q = exponent // 3
    if q & (q - 1):
        return None
    return q.bit_length() - 1


def report_stage(stage: int, exponent: int, pc: PrimeCounter):
    D = 1 << exponent
    tier = domain_tier(exponent)
    data = exact_center_for_domain(D, pc)

    center = rational_to_int(data["center_num"], data["center_den"])
    if center is None:
        center_float = data["center_num"] / data["center_den"]
        raise RuntimeError(f"Noninteger center encountered: {center_float}")

    print("\n" + "=" * 88)
    print(f"STAGE {stage}: D = 2^{exponent} = {D:,}")
    print("=" * 88)
    print(f"pi(D) = {data['count']:,}")

    if data["kind"] == "single":
        print(f"prime center = {center:,} (unique median prime)")
        half_gap = None
    else:
        half_gap = rational_to_int(
            data["half_gap_num"], data["half_gap_den"]
        )
        print(
            f"middle primes = {data['lower']:,}, {data['upper']:,}\n"
            f"exact center  = {center:,}\n"
            f"half-gap      = {half_gap:,}"
        )

        # Every-second-Fibonacci hypothesis tied to the actual square-domain tier:
        # exponent 3,6,12,24,48,... corresponds to F_2,F_4,F_6,F_8,F_10,...
        print("\n  Fibonacci half-gap test")
        if tier is not None:
            expected_fib_index = 2 * (tier + 1)
            expected_half_gap = fib(expected_fib_index)
            compare_candidate(
                f"F_{expected_fib_index}",
                half_gap,
                expected_half_gap,
            )
        else:
            print("      domain is outside the standard 3*2^s exponent ladder")

        fi, fv, fd = nearest_fibonacci(half_gap)
        print(
            f"      nearest Fibonacci         F_{fi}={fv:,}; "
            f"difference={half_gap-fv:+,}"
        )

    r, correction = nearest_square(center)
    print("\n  Nearest-square structure")
    print(f"      r = {r:,}")
    print(f"      r^2 = {r*r:,}")
    print(f"      center - r^2 = {correction:+,}")

    if correction != 0:
        fi, fv, fd = nearest_fibonacci(correction)
        tn, tv, td = nearest_triangular(correction)
        pk, pv, pd = nearest_power3(correction)

        print("\n  What is closest to |correction|?")
        print(
            f"      Fibonacci:    F_{fi}={fv:,}, "
            f"difference={abs(correction)-fv:+,}"
        )
        print(
            f"      Triangular:   T_{tn}={tv:,}, "
            f"difference={abs(correction)-tv:+,}"
        )
        print(
            f"      Power of 3:   3^{pk}={pv:,}, "
            f"difference={abs(correction)-pv:+,}"
        )

    # Triangular branch derived only from the previous exponent.
    if stage >= 1:
        previous_exponent = exponent // 2
        doorway_index = previous_exponent - Y
        tri_value = triangular(doorway_index)

        # Observed pre-reset sign pattern:
        # 2^6: +T1
        # 2^12: -T4
        # 2^24: +T10
        # 2^48: -T22 (candidate only)
        # For the standard ladder, observed signs at exponents 6,12,24 are +,-,+.
        # Continue that alternation by the true domain tier, not by command-line order.
        sign = None
        if tier is not None and tier >= 1:
            sign = 1 if tier % 2 == 1 else -1
        predicted = None if sign is None else sign * tri_value

        print("\n  Derived triangular doorway branch")
        print(
            f"      previous exponent = {previous_exponent}\n"
            f"      previous exponent - y = {previous_exponent}-{Y} "
            f"= {doorway_index}\n"
            f"      T_{doorway_index} = {tri_value:,}"
        )
        if predicted is not None:
            print(f"      alternating-sign candidate = {predicted:+,}")
            compare_candidate("signed triangular", correction, predicted)
        else:
            print("      sign prediction not assigned outside the standard ladder")

    # Explicit post-55 branches for the first stage after 2^24.
    if exponent >= 48 and correction != 0:
        print("\n  Post-55 branch tests")
        candidates = [
            ("Gold next 89", 89),
            ("Gold next-next 144", 144),
            ("reset T_22", triangular(22)),
            ("pair branch T_55", triangular(55)),
            ("nested Fibonacci F_34", fib(34)),
        ]
        for label, candidate in candidates:
            # Compare signed and magnitude because the doorway sign itself
            # is one of the things under investigation.
            compare_candidate(label + " (+)", correction, candidate)
            compare_candidate(label + " (-)", correction, -candidate)

    return {
        "stage": stage,
        "exponent": exponent,
        "domain": D,
        "count": data["count"],
        "center": center,
        "kind": data["kind"],
        "lower": data["lower"],
        "upper": data["upper"],
        "half_gap": half_gap,
        "radius": r,
        "correction": correction,
    }


def print_foundation():
    print("=" * 88)
    print("STRUCTURAL FOUNDATION")
    print("=" * 88)
    print(f"x = {X}")
    print(f"y = x+x = {Y}")
    print(f"z = x+y = {Z}")
    print(f"A = y+y = y*y = y^y = {A}")
    print()
    print(f"T_4  = {triangular(4)}")
    print(f"T_10 = {triangular(10)}")
    print(f"R(4)=2*4+2   = {reset(4)}")
    print(f"R(10)=2*10+2 = {reset(10)}")
    print()
    print("Gold row:")
    print("  ", [fib(i) for i in range(3, 13)])  # 2,3,5,...
    print("y x Gold row:")
    print("  ", [Y * fib(i) for i in range(3, 13)])


def main():
    parser = argparse.ArgumentParser(
        description="Exact prime-center + Gold/triangular structural explorer."
    )
    parser.add_argument(
        "--exponents",
        nargs="*",
        type=int,
        default=[3, 6, 12, 24, 48],
        help=(
            "Powers of 2 to test. Default: 3 6 12 24 48, "
            "corresponding to 8,64,4096,2^24,2^48."
        ),
    )
    parser.add_argument(
        "--stop-before-large",
        action="store_true",
        help="Skip exponents above 24, useful if primecount is not installed.",
    )
    args = parser.parse_args()

    exponents = args.exponents
    if args.stop_before_large:
        exponents = [e for e in exponents if e <= 24]

    print_foundation()

    pc = PrimeCounter()

    if any(e > 24 for e in exponents):
        print("\nprimecount executable:", pc.exe or "NOT FOUND")
        if not pc.exe:
            print(
                "\nLarge-domain tests need primecount. "
                "Run with --stop-before-large for the built-in exact tests "
                "through 2^24."
            )
            sys.exit(2)

    rows = []
    for stage, exponent in enumerate(exponents):
        rows.append(report_stage(stage, exponent, pc))

    print("\n" + "=" * 88)
    print("COMPACT SUMMARY")
    print("=" * 88)
    print(
        f"{'2^e':>8} {'pi(D)':>18} {'center':>18} "
        f"{'half-gap':>12} {'radius':>14} {'corr':>14}"
    )
    for row in rows:
        hg = "-" if row["half_gap"] is None else f"{row['half_gap']:,}"
        print(
            f"{('2^'+str(row['exponent'])):>8} "
            f"{row['count']:>18,} "
            f"{row['center']:>18,} "
            f"{hg:>12} "
            f"{row['radius']:>14,} "
            f"{row['correction']:>+14,}"
        )

    print("\nInterpretation discipline:")
    print("  * primecount/sieve decides the exact prime data.")
    print("  * Gold/triangular/reset rules are tested only AFTER the center is known.")
    print("  * an EXACT label is a genuine equality; a small difference is only a clue.")


if __name__ == "__main__":
    main()
