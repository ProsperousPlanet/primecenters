#!/usr/bin/env python3
"""
RECURSIVE PHASE--TRANSPORT TWIN-CENTER ENGINE

Usage:

    python3 transport3.py 70
    python3 transport3.py 100
    python3 transport3.py 100 --show-last-sequence
    python3 transport3.py 100 --show-sequences
    python3 transport3.py 100 --stats

The positional number is an UPPER INTEGER BOUND for the recursive prime
levels, not a requirement that the number itself be prime.

For example:

    70  -> generate levels 5,7,11,...,61,67 and stop at integer address 70
    100 -> generate levels 5,7,11,...,89,97 and stop at integer address 100

Generation is split into two forward memories.

1. PrimeCycleClock
   Generates prime labels without calling a primality test. A new prime opens
   whenever the current integer address is not already closed by an earlier
   prime cycle. Its first closure is p^2, then p^2+p, p^2+2p, ...

2. TwinCenterPhaseClock
   Walks only the six-spaced center lattice

       6, 12, 18, 24, 30, ...

   Every generated prime p >= 5 contributes two recurring closure streams:
       center - 1 is divisible by p
       center + 1 is divisible by p

   Each stream repeats after p center-index steps, equivalently after 6p on
   the center number line.

A center survives when no required phase stream closes there.

No stored prime table, no stored twin-center table, and no primality test is
used by generation.  The optional --verify mode is a completely separate
post-generation audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import Iterator
import argparse


SIX = 6


# ============================================================
# 1. PRIME-CYCLE EVENT MEMORY
# ============================================================

class PrimeCycleClock:
    """
    Forward prime-cycle clock.

    The clock never asks "is this number prime?"

    Instead, already-open prime cycles schedule their future composite
    closures.  If an integer address arrives with no scheduled closure, that
    address opens a new prime cycle.
    """

    def __init__(self) -> None:
        self.position = 1
        self.closures: dict[int, list[int]] = {}
        self.primes: list[int] = []

    def _schedule(self, position: int, prime: int) -> None:
        self.closures.setdefault(position, []).append(prime)

    def tick(self) -> int | None:
        """Advance exactly one integer address."""
        self.position += 1
        position = self.position

        closing = self.closures.pop(position, None)

        if closing is None:
            prime = position
            self.primes.append(prime)

            # Earlier multiples are already closed by earlier prime cycles.
            # The newly opened p-cycle first contributes new information at p^2.
            self._schedule(prime * prime, prime)
            return prime

        # Every cycle that closes here continues one p-step forward.
        for prime in closing:
            self._schedule(position + prime, prime)

        return None

    def advance_to(self, target: int) -> tuple[int, ...]:
        """
        Advance forward through exactly target.

        This never scans beyond target.
        """
        if target < self.position:
            raise ValueError("PrimeCycleClock cannot move backward.")

        opened: list[int] = []

        while self.position < target:
            prime = self.tick()
            if prime is not None:
                opened.append(prime)

        return tuple(opened)


# ============================================================
# 2. TWIN-CENTER PHASE MEMORY
# ============================================================

@dataclass(frozen=True)
class PhaseStream:
    prime: int
    side: int  # -1 -> center-1; +1 -> center+1

    @property
    def label(self) -> str:
        return "center-1" if self.side < 0 else "center+1"


@dataclass(frozen=True)
class CenterEvent:
    index: int
    center: int
    closures: tuple[PhaseStream, ...]

    @property
    def survives(self) -> bool:
        return not self.closures


class TwinCenterPhaseClock:
    """
    Forward phase clock on center = 6*n.

    Only the next scheduled occurrence of each active phase stream is stored.
    """

    def __init__(self) -> None:
        self.index = 0
        self.events: dict[int, list[PhaseStream]] = {}

        # This prime clock is independent of the level clock.
        # It opens a prime only when that prime can become relevant to the
        # center currently being resolved.
        self.prime_clock = PrimeCycleClock()
        self.installed_primes: set[int] = set()

        self.survivor_count = 0
        self.event_count = 0

    def _schedule(self, index: int, stream: PhaseStream) -> None:
        self.events.setdefault(index, []).append(stream)

    def _install_prime(self, prime: int) -> None:
        """Install the two recurring six-lattice closure phases for p."""
        if prime < 5 or prime in self.installed_primes:
            return

        self.installed_primes.add(prime)

        # --------------------------------------------------------
        # RIGHT CLOSURE STREAM
        #
        # p^2 == 1 (mod 6) for every prime p >= 5.
        #
        # Therefore:
        #
        #     center + 1 = p^2
        #     center     = p^2 - 1 = 6*n
        # --------------------------------------------------------
        right_index = (prime * prime - 1) // SIX
        self._schedule(
            right_index,
            PhaseStream(prime=prime, side=+1),
        )

        # --------------------------------------------------------
        # LEFT CLOSURE STREAM
        #
        # Find the first multiplier k >= p for which
        #
        #     p*k == -1 (mod 6).
        #
        # If p == 1 (mod 6), k must be 5 (mod 6), so k=p+4.
        # If p == 5 (mod 6), k must be 1 (mod 6), so k=p+2.
        # --------------------------------------------------------
        residue = prime % SIX

        if residue == 1:
            multiplier = prime + 4
        elif residue == 5:
            multiplier = prime + 2
        else:
            raise RuntimeError(
                f"Unexpected prime residue for p={prime}"
            )

        left_composite = prime * multiplier
        left_index = (left_composite + 1) // SIX

        self._schedule(
            left_index,
            PhaseStream(prime=prime, side=-1),
        )

    def _ensure_required_prime_cycles(self, center: int) -> None:
        """
        Lazily open only the prime cycles required at this center.

        If center-1 or center+1 is composite, one of its prime factors is no
        larger than sqrt(center+1).  No larger prime cycle is needed yet.
        """
        required = isqrt(center + 1)

        newly_opened = self.prime_clock.advance_to(required)

        for prime in newly_opened:
            self._install_prime(prime)

    def tick(self) -> CenterEvent:
        """Advance exactly one six-spaced center address."""
        self.index += 1
        center = SIX * self.index

        # Install all and only the prime cycles that can matter here.
        self._ensure_required_prime_cycles(center)

        closing = tuple(
            self.events.pop(self.index, ())
        )

        # Each closure orientation repeats after p center-index positions:
        #
        #     6(n+p) +/- 1
        #       =
        #     (6n +/- 1) + 6p
        #
        # so its number-line period is exactly the completed state 6p.
        for stream in closing:
            self._schedule(
                self.index + stream.prime,
                stream,
            )

        self.event_count += 1

        event = CenterEvent(
            index=self.index,
            center=center,
            closures=closing,
        )

        if event.survives:
            self.survivor_count += 1

        return event

    def next_center(self) -> int:
        """Return the next center surviving every required active phase."""
        while True:
            event = self.tick()
            if event.survives:
                return event.center

    def centers(self) -> Iterator[int]:
        while True:
            yield self.next_center()


# ============================================================
# 3. PRIME-LEVEL / THREE-STATE RECURSION
# ============================================================

@dataclass(frozen=True)
class LevelState:
    prime: int
    previous_prime: int | None
    inherited: int | None
    mediated: int | None
    new: int
    opening: int
    closing: int
    centers: tuple[int, ...]

    def residues(
        self,
        center: int,
    ) -> tuple[int | None, int | None, int]:
        inherited_phase = (
            None
            if self.inherited is None
            else center % self.inherited
        )

        mediated_phase = (
            None
            if self.mediated is None
            else center % self.mediated
        )

        new_phase = center % self.new

        return (
            inherited_phase,
            mediated_phase,
            new_phase,
        )


class RecursivePhaseTransport:
    """
    Progressive recursive generator.

    The level clock advances integer-by-integer from the beginning.
    Whenever it opens a prime p >= 5, that p-level is built immediately.

    It does NOT first generate a list of all primes up to the user's bound.
    """

    def __init__(self) -> None:
        self.center_clock = TwinCenterPhaseClock()
        self.level_clock = PrimeCycleClock()

        self.previous_prime: int | None = None
        self.carry: int | None = None
        self.levels: list[LevelState] = []

    def _build_level(self, prime: int) -> LevelState:
        """Build one newly opened prime level."""
        new_state = SIX * prime

        if self.previous_prime is None:
            # The 5-level is the base:
            # 6,12,18,30,42
            inherited = None
            mediated = None

            centers = tuple(
                self.center_clock.next_center()
                for _ in range(prime)
            )

        else:
            inherited = SIX * self.previous_prime
            mediated = (inherited + new_state) // 2

            if self.carry is None:
                raise RuntimeError(
                    "Missing carried closing center."
                )

            # Previous close becomes the new opening.
            # Then generate p-1 new consecutive survivors.
            centers = (
                self.carry,
            ) + tuple(
                self.center_clock.next_center()
                for _ in range(prime - 1)
            )

        state = LevelState(
            prime=prime,
            previous_prime=self.previous_prime,
            inherited=inherited,
            mediated=mediated,
            new=new_state,
            opening=centers[0],
            closing=centers[-1],
            centers=centers,
        )

        self.previous_prime = prime
        self.carry = state.closing
        self.levels.append(state)

        return state

    def advance_to(self, maximum: int) -> tuple[LevelState, ...]:
        """
        Start from the beginning and advance the level address through maximum.

        maximum does NOT need to be prime.

        Example:
            maximum=70
            -> builds every prime level through 67
            -> level clock stops exactly at integer address 70

        Primes and levels are opened progressively.  No future prime-level list
        is built in advance.
        """
        if maximum < 5:
            raise ValueError(
                "maximum must be at least 5"
            )

        if maximum < self.level_clock.position:
            raise ValueError(
                "The recursive level clock cannot move backward."
            )

        while self.level_clock.position < maximum:
            prime = self.level_clock.tick()

            if prime is not None and prime >= 5:
                self._build_level(prime)

        return tuple(self.levels)


# ============================================================
# 4. OPTIONAL EXTERNAL AUDIT -- NEVER USED BY GENERATION
# ============================================================

def is_prime_trial(number: int) -> bool:
    """
    Conventional trial division.

    This function exists ONLY for --verify and is never called by generation.
    """
    if number < 2:
        return False

    if number == 2:
        return True

    if number % 2 == 0:
        return False

    divisor = 3
    limit = isqrt(number)

    while divisor <= limit:
        if number % divisor == 0:
            return False
        divisor += 2

    return True


def verify_levels(
    levels: tuple[LevelState, ...],
) -> None:
    """
    Re-scan each completed interval independently after generation.
    """
    print("\nINDEPENDENT POST-GENERATION AUDIT")
    print("=" * 78)

    for state in levels:
        externally_found = tuple(
            center
            for center in range(
                state.opening,
                state.closing + SIX,
                SIX,
            )
            if (
                is_prime_trial(center - 1)
                and is_prime_trial(center + 1)
            )
        )

        exact = externally_found == state.centers

        print(
            f"p={state.prime:<5} "
            f"generated={len(state.centers):<5} "
            f"exact consecutive match="
            f"{'YES' if exact else 'NO'}"
        )

        if not exact:
            generated_only = sorted(
                set(state.centers)
                - set(externally_found)
            )

            audit_only = sorted(
                set(externally_found)
                - set(state.centers)
            )

            print(
                "  generated only:",
                generated_only,
            )
            print(
                "  audit only:    ",
                audit_only,
            )

            raise AssertionError(
                "Post-generation audit failed."
            )


# ============================================================
# 5. REPORTS
# ============================================================

def print_levels(
    levels: tuple[LevelState, ...],
    show_sequences: bool = False,
    show_last_sequence: bool = False,
) -> None:
    print(
        "RECURSIVE PHASE--TRANSPORT LEVELS"
    )
    print("=" * 100)

    print(
        f"{'p':>5} "
        f"{'opening':>12} "
        f"{'inherited':>12} "
        f"{'mediated':>12} "
        f"{'new':>10} "
        f"{'count':>8} "
        f"{'closing':>12}"
    )

    print("-" * 100)

    for state in levels:
        inherited = (
            "-"
            if state.inherited is None
            else str(state.inherited)
        )

        mediated = (
            "-"
            if state.mediated is None
            else str(state.mediated)
        )

        print(
            f"{state.prime:>5} "
            f"{state.opening:>12} "
            f"{inherited:>12} "
            f"{mediated:>12} "
            f"{state.new:>10} "
            f"{len(state.centers):>8} "
            f"{state.closing:>12}"
        )

        if show_sequences:
            print(
                "     ",
                ", ".join(
                    str(value)
                    for value in state.centers
                ),
            )

    if show_last_sequence and levels:
        last = levels[-1]

        print(
            f"\nLAST GENERATED SEQUENCE: p={last.prime}"
        )
        print("=" * 100)
        print(
            ", ".join(
                str(value)
                for value in last.centers
            )
        )


def print_stats(
    engine: RecursivePhaseTransport,
    requested_maximum: int,
) -> None:
    levels = engine.levels
    center_clock = engine.center_clock

    print("\nFORWARD-MEMORY STATISTICS")
    print("=" * 78)

    print(
        f"requested integer bound:      "
        f"{requested_maximum:,}"
    )

    print(
        f"level clock stopped exactly:  "
        f"{engine.level_clock.position:,}"
    )

    if levels:
        print(
            f"highest generated prime level:"
            f" {levels[-1].prime:,}"
        )
        print(
            f"number of generated levels:   "
            f"{len(levels):,}"
        )
        print(
            f"last generated center:        "
            f"{levels[-1].closing:,}"
        )

    print(
        f"six-lattice addresses walked: "
        f"{center_clock.index:,}"
    )

    print(
        f"surviving centers generated:  "
        f"{center_clock.survivor_count:,}"
    )

    print(
        f"center prime clock stopped at:"
        f" {center_clock.prime_clock.position:,}"
    )

    if center_clock.installed_primes:
        print(
            f"highest installed closure p:  "
            f"{max(center_clock.installed_primes):,}"
        )

    active_streams = sum(
        len(streams)
        for streams in center_clock.events.values()
    )

    print(
        f"active future phase streams:  "
        f"{active_streams:,}"
    )


# ============================================================
# 6. COMMAND LINE
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursive phase-transport generator. "
            "The positional number is an upper integer bound; "
            "it does not need to be prime."
        )
    )

    parser.add_argument(
        "maximum",
        nargs="?",
        type=int,
        default=31,
        help=(
            "advance from the beginning through this integer "
            "prime-level address (default: 31)"
        ),
    )

    parser.add_argument(
        "--show-sequences",
        action="store_true",
        help=(
            "print every generated center in every completed level"
        ),
    )

    parser.add_argument(
        "--show-last-sequence",
        action="store_true",
        help=(
            "print the complete center sequence only for the "
            "highest prime level reached"
        ),
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help=(
            "show how far the independent recursive clocks advanced"
        ),
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "AFTER generation, run a separate conventional primality audit"
        ),
    )

    args = parser.parse_args()

    if args.maximum < 5:
        parser.error(
            "maximum must be at least 5"
        )

    engine = RecursivePhaseTransport()

    levels = engine.advance_to(
        args.maximum
    )

    print_levels(
        levels,
        show_sequences=args.show_sequences,
        show_last_sequence=args.show_last_sequence,
    )

    if levels:
        print(
            f"\nStopped at integer address {args.maximum}. "
            f"Highest prime level generated: {levels[-1].prime}."
        )

    if args.stats:
        print_stats(
            engine,
            args.maximum,
        )

    if args.verify:
        verify_levels(
            levels
        )


if __name__ == "__main__":
    main()
