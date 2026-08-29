#!/usr/bin/env python3
"""
GOLD--SIMPLEX RECURSIVE PRIME-CENTER MEMORY ENGINE

This version removes the dimension-by-dimension carry lookup.

The construction has three recursive clocks:

1. Gold clock
       G_0 = 2
       G_1 = 3
       G_(n+1) = G_n + G_(n-1)

2. Domain / simplex clock
       D_(k,d) = 2^(2^k d)

   A block runs
       d, d+1, ..., 2d

   and the dimensional carry is
       d -> 2d.

3. Prime-cycle event clock

   When a prime p is generated, all multiples p*m with 2 <= m < p are
   already closed by an earlier prime factor of m.  Therefore the first
   closure that the newly opened p-cycle itself needs to schedule is

       p^2.

   From there the clock advances one p-step at a time:

       p^2, p^2+p, p^2+2p, ...

   The program stores the next required closure of every open prime cycle.
   When the number line reaches a stored closure, that address is closed.
   When no earlier cycle closes, the address opens a new prime cycle.

The prime center is therefore NOT supplied to the carry formula.

For a requested domain D, the program:

    previous memory
        -> advance all internal prime-cycle clocks to D
        -> count the generated prime addresses
        -> extract the middle prime(s)
        -> obtain C(D)
        -> derive the transport carry R

For consecutive center states,

    C_next = M*C_previous + R

so after C_next has been generated internally by the clocks,

    R = C_next - M*C_previous.

At square-coordinate states,

    C = r^2 + rho,

and if M is a square,

    r_next = sqrt(M)*r_previous + q

with

    q = r_next - sqrt(M)*r_previous.

Thus R, q and rho are OUTPUTS of the evolved recursive memory, not inputs
taken from an externally known next center.

No primality library and no primecount call is used by the construction.

Important computational note
----------------------------
The domain / Gold / simplex hierarchy can be generated to arbitrarily large
finite exponents almost instantly.

The exact prime-cycle clock walks the integer addresses of the domain.
That is an exact finite algorithm, but its runtime and memory grow with the
domain.  The default exact limit is therefore modest.  Increase
EXACT_MAX_EXPONENT if desired.

Optional primecount code is kept only as a completely separate audit and is
disabled by default.  It never feeds data back into the recursive memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isqrt
import shutil
import subprocess


# ============================================================
# SETTINGS
# ============================================================

# 2^24 completes the first three recursively squared blocks.
# Lower this value for a faster demonstration, or raise it experimentally.
EXACT_MAX_EXPONENT = 24

# Structural hierarchy only; this does not enumerate the domains.
STRUCTURE_MAX_EXPONENT = 384

VERIFY_WITH_PRIMECOUNT = False
PRIMECOUNT_MAX_EXPONENT = 64


# ============================================================
# 1. BUILD SMALL INTEGERS FROM ONE
# ============================================================

ONE = 1
ZERO = ONE - ONE


def successor(number: int) -> int:
    return number + ONE


TWO = successor(ONE)
THREE = successor(TWO)
FOUR = successor(THREE)
FIVE = successor(FOUR)
SIX = successor(FIVE)
SEVEN = successor(SIX)
EIGHT = successor(SEVEN)


def power(base: int, exponent: int) -> int:
    answer = ONE

    for _ in range(exponent):
        answer *= base

    return answer


# ============================================================
# 2. SIMPLEX HIERARCHY
# ============================================================

@lru_cache(maxsize=None)
def simplex(order: int, position: int) -> int:
    """
    Repeated accumulation:

        S_0(n) = 1

        S_d(n) = S_d(n-1) + S_(d-1)(n)
    """
    if order == ZERO:
        return ONE

    if position == ZERO:
        return ZERO

    return (
        simplex(order, position - ONE)
        + simplex(order - ONE, position)
    )


# ============================================================
# 3. GOLD CLOCK
# ============================================================

_gold = [TWO, THREE]


def G(index: int) -> int:
    """
    G_0 = 2, G_1 = 3.
    """
    if index < ZERO:
        if index == -ONE:
            return ONE
        raise ValueError("Gold index below -1 is not used.")

    while len(_gold) <= index:
        _gold.append(
            _gold[-ONE] + _gold[-TWO]
        )

    return _gold[index]


def gamma(dimension: int) -> tuple[int, int]:
    """
    Gamma_d = (G_(2d-2), G_(2d-1))
    """
    return (
        G(TWO * dimension - TWO),
        G(TWO * dimension - ONE),
    )


def doubled_gamma(
    pair: tuple[int, int],
) -> tuple[int, int]:
    """
    Exact Gold dimensional lift.

    If Gamma_d = (a,b), then

        Gamma_(2d)
        =
        (a^2 + (b-a)^2, a(2b-a)).
    """
    a, b = pair

    return (
        a * a + (b - a) * (b - a),
        a * (TWO * b - a),
    )


def gold_address(number: int) -> tuple[int, tuple[tuple[int, int], ...]]:
    """
    Greedy nonconsecutive Gold/Fibonacci address.

    We include G_-1 = 1.

    Returns:
        sign,
        ((Gold index, Gold value), ...)
    """
    if number == ZERO:
        return ONE, tuple()

    sign = -ONE if number < ZERO else ONE
    remaining = abs(number)

    terms: list[tuple[int, int]] = [
        (-ONE, ONE),
        (ZERO, G(ZERO)),
        (ONE, G(ONE)),
    ]

    index = TWO

    while terms[-ONE][ONE] <= remaining:
        terms.append(
            (index, G(index))
        )
        index = successor(index)

    chosen: list[tuple[int, int]] = []

    for index, value in reversed(terms):
        if value <= remaining:
            chosen.append((index, value))
            remaining -= value

        if remaining == ZERO:
            break

    if remaining != ZERO:
        raise RuntimeError(
            "Gold address construction failed."
        )

    return sign, tuple(chosen)


def format_gold_address(number: int) -> str:
    sign, address = gold_address(number)

    if not address:
        return "0"

    body = " + ".join(
        (
            "G_-1"
            if index == -ONE
            else f"G_{index}"
        )
        for index, _ in address
    )

    return (
        f"-({body})"
        if sign < ZERO
        else body
    )


# ============================================================
# 4. DOMAIN / DIMENSION CLOCK
# ============================================================

@dataclass(frozen=True)
class DomainAddress:
    k: int
    dimension: int

    @property
    def scale_exponent(self) -> int:
        """
        log_2(M_k) = 2^k
        """
        return power(TWO, self.k)

    @property
    def scale(self) -> int:
        return power(
            TWO,
            self.scale_exponent,
        )

    @property
    def exponent(self) -> int:
        """
        D_(k,d) = 2^(2^k d).
        """
        return (
            self.scale_exponent
            * self.dimension
        )


@dataclass(frozen=True)
class DimensionBlock:
    k: int
    start_dimension: int

    @property
    def scale_exponent(self) -> int:
        return power(TWO, self.k)

    @property
    def opening_exponent(self) -> int:
        return (
            self.scale_exponent
            * self.start_dimension
        )

    @property
    def closing_exponent(self) -> int:
        return (
            self.scale_exponent
            * TWO
            * self.start_dimension
        )

    def addresses(self) -> tuple[DomainAddress, ...]:
        return tuple(
            DomainAddress(
                self.k,
                dimension,
            )
            for dimension in range(
                self.start_dimension,
                TWO * self.start_dimension + ONE,
            )
        )

    def scale_carry(self) -> "DimensionBlock":
        """
        M -> M^2, d unchanged.
        """
        return DimensionBlock(
            k=successor(self.k),
            start_dimension=self.start_dimension,
        )

    def dimensional_carry(self) -> "DimensionBlock":
        """
        M unchanged, d -> 2d.
        """
        return DimensionBlock(
            k=self.k,
            start_dimension=(
                TWO * self.start_dimension
            ),
        )


def equivalent_addresses(
    exponent: int,
) -> tuple[DomainAddress, ...]:
    """
    All exact integer addresses satisfying

        exponent = 2^k * d.
    """
    result: list[DomainAddress] = []

    k = ZERO
    scale_exponent = ONE

    while exponent % scale_exponent == ZERO:
        result.append(
            DomainAddress(
                k=k,
                dimension=(
                    exponent // scale_exponent
                ),
            )
        )

        k = successor(k)
        scale_exponent *= TWO

    return tuple(result)


def dimensional_boundaries(
    count: int = EIGHT,
) -> tuple[int, ...]:
    """
    3 -> 6 -> 12 -> 24 -> ...
    """
    values = [THREE]

    while len(values) < count:
        values.append(
            TWO * values[-ONE]
        )

    return tuple(values)


def structural_blocks(
    maximum_exponent: int,
) -> tuple[DimensionBlock, ...]:
    """
    Follow the paper's main hierarchy.

    First:
        M = 2,4,16,256
        with d = 3.

    At the S_6 boundary:
        keep M = 256
        and carry d: 3 -> 6 -> 12 -> 24 -> ...

    No numerical center is required to generate these blocks.
    """
    blocks: list[DimensionBlock] = []

    block = DimensionBlock(
        k=ZERO,
        start_dimension=THREE,
    )

    # Scale carries:
    # (2,3) -> (4,3) -> (16,3) -> (256,3)
    while block.k <= THREE:
        if block.opening_exponent <= maximum_exponent:
            blocks.append(block)

        block = block.scale_carry()

    # Dimensional carries at M=256.
    block = DimensionBlock(
        k=THREE,
        start_dimension=SIX,
    )

    while block.opening_exponent <= maximum_exponent:
        blocks.append(block)
        block = block.dimensional_carry()

    return tuple(blocks)


def structural_exponents(
    maximum_exponent: int,
) -> tuple[int, ...]:
    values: set[int] = set()

    for block in structural_blocks(
        maximum_exponent
    ):
        for address in block.addresses():
            if address.exponent <= maximum_exponent:
                values.add(address.exponent)

    return tuple(sorted(values))


# ============================================================
# 5. PRIME-CYCLE EVENT MEMORY
# ============================================================

@dataclass(frozen=True)
class CycleEvent:
    position: int
    closing_primes: tuple[int, ...]
    opened_prime: int | None


class PrimeCycleMemory:
    """
    Exact recursive number-line clock.

    No divisibility query is made to decide whether a number is prime.

    A generated prime p begins contributing new closure information at p^2.

    Earlier multiples p*m with 2 <= m < p are already closed by an earlier
    prime factor of m, so scheduling them again is unnecessary.

    Once p^2 is reached, that cycle is rescheduled p positions later.

    Therefore every composite address is closed by already-open cycle
    memory, while an address with no closure opens a new prime cycle.
    """

    def __init__(
        self,
        ceiling: int,
    ) -> None:
        if ceiling < TWO:
            raise ValueError(
                "ceiling must be at least 2"
            )

        self.ceiling = ceiling
        self.position = ONE

        # position -> list of prime cycles closing there
        self.closures: dict[int, list[int]] = {}

        # Generated prime addresses in increasing order.
        self.primes: list[int] = []

    def _schedule(
        self,
        position: int,
        prime: int,
    ) -> None:
        if position > self.ceiling:
            return

        self.closures.setdefault(
            position,
            [],
        ).append(prime)

    def tick(self) -> CycleEvent:
        """
        Advance the natural-number clock by one address.
        """
        if self.position >= self.ceiling:
            raise StopIteration

        position = successor(
            self.position
        )

        closing = self.closures.pop(
            position,
            None,
        )

        if closing is None:
            # No prior cycle closed here.
            # This address therefore opens a new prime cycle.
            opened_prime = position
            self.primes.append(
                opened_prime
            )

            first_nontrivial_closure = (
                opened_prime
                * opened_prime
            )

            self._schedule(
                first_nontrivial_closure,
                opened_prime,
            )

            closing_primes: tuple[int, ...] = tuple()

        else:
            opened_prime = None
            closing_primes = tuple(
                sorted(closing)
            )

            # Every closing prime cycle continues to its next winding.
            for prime in closing_primes:
                self._schedule(
                    position + prime,
                    prime,
                )

        self.position = position

        return CycleEvent(
            position=position,
            closing_primes=closing_primes,
            opened_prime=opened_prime,
        )

    def advance_to(
        self,
        position: int,
    ) -> None:
        if position > self.ceiling:
            raise ValueError(
                "Requested position exceeds this clock's ceiling."
            )

        if position < self.position:
            raise ValueError(
                "Prime-cycle memory only advances forward."
            )

        while self.position < position:
            self.tick()

    @property
    def prime_count(self) -> int:
        return len(self.primes)

    def center(self) -> tuple[int, tuple[int, ...]]:
        """
        Return:
            center,
            middle prime or middle prime pair.
        """
        count = self.prime_count

        if count == ZERO:
            raise RuntimeError(
                "No prime population has been generated yet."
            )

        middle = count // TWO

        if count % TWO:
            prime = self.primes[middle]
            return prime, (prime,)

        left = self.primes[middle - ONE]
        right = self.primes[middle]

        return (
            (left + right) // TWO,
            (left, right),
        )


# ============================================================
# 6. CENTER AND CARRY MEMORY
# ============================================================

def nearest_square_state(
    center: int,
) -> tuple[int, int]:
    """
    C = r^2 + rho
    using the nearest integer square.
    """
    lower = isqrt(center)
    upper = successor(lower)

    lower_residual = (
        center - lower * lower
    )

    upper_residual = (
        center - upper * upper
    )

    if abs(lower_residual) <= abs(upper_residual):
        return lower, lower_residual

    return upper, upper_residual


@dataclass(frozen=True)
class CenterSnapshot:
    exponent: int
    domain: int
    prime_count: int
    center: int
    middle_primes: tuple[int, ...]
    radius: int
    residual: int
    addresses: tuple[DomainAddress, ...]


@dataclass(frozen=True)
class CarryMemory:
    previous_exponent: int
    exponent: int
    multiplier: int
    carry: int

    previous_center: int
    center: int

    previous_radius: int
    radius: int

    root_multiplier: int | None
    radius_carry: int | None

    residual: int

    gold_carry: str
    gold_radius_carry: str | None


class RecursiveCenterMemory:
    """
    Stores the inherited center state.

    The next center is never passed into a carry rule.

    Instead:

        1. prime-cycle memory advances to the next domain;
        2. that evolved state yields the next center;
        3. the carry is read out from the two completed states.
    """

    def __init__(self) -> None:
        self.snapshots: list[CenterSnapshot] = []
        self.carries: list[CarryMemory] = []

    def absorb(
        self,
        snapshot: CenterSnapshot,
    ) -> None:
        if self.snapshots:
            previous = self.snapshots[-ONE]

            exponent_step = (
                snapshot.exponent
                - previous.exponent
            )

            multiplier = power(
                TWO,
                exponent_step,
            )

            carry = (
                snapshot.center
                - multiplier
                * previous.center
            )

            root = isqrt(multiplier)

            if root * root == multiplier:
                radius_carry = (
                    snapshot.radius
                    - root
                    * previous.radius
                )
                root_multiplier: int | None = root
            else:
                radius_carry = None
                root_multiplier = None

            self.carries.append(
                CarryMemory(
                    previous_exponent=previous.exponent,
                    exponent=snapshot.exponent,
                    multiplier=multiplier,
                    carry=carry,
                    previous_center=previous.center,
                    center=snapshot.center,
                    previous_radius=previous.radius,
                    radius=snapshot.radius,
                    root_multiplier=root_multiplier,
                    radius_carry=radius_carry,
                    residual=snapshot.residual,
                    gold_carry=format_gold_address(
                        carry
                    ),
                    gold_radius_carry=(
                        None
                        if radius_carry is None
                        else format_gold_address(
                            radius_carry
                        )
                    ),
                )
            )

        self.snapshots.append(
            snapshot
        )


def snapshot_from_clock(
    exponent: int,
    clock: PrimeCycleMemory,
) -> CenterSnapshot:
    domain = power(
        TWO,
        exponent,
    )

    clock.advance_to(
        domain
    )

    center, middle = clock.center()
    radius, residual = nearest_square_state(
        center
    )

    return CenterSnapshot(
        exponent=exponent,
        domain=domain,
        prime_count=clock.prime_count,
        center=center,
        middle_primes=middle,
        radius=radius,
        residual=residual,
        addresses=equivalent_addresses(
            exponent
        ),
    )


def build_exact_center_memory(
    maximum_exponent: int,
) -> RecursiveCenterMemory:
    """
    Exact internal construction.

    The requested domain endpoints come from the recursive structural
    hierarchy.  PrimeCycleMemory then evolves to each endpoint and
    generates the center internally.
    """
    targets = tuple(
        exponent
        for exponent in structural_exponents(
            maximum_exponent
        )
        if exponent >= THREE
    )

    if not targets:
        raise ValueError(
            "No structural domain lies inside the requested range."
        )

    ceiling = power(
        TWO,
        max(targets),
    )

    clock = PrimeCycleMemory(
        ceiling=ceiling
    )

    memory = RecursiveCenterMemory()

    for exponent in targets:
        snapshot = snapshot_from_clock(
            exponent,
            clock,
        )

        memory.absorb(
            snapshot
        )

    return memory


# ============================================================
# 7. SMALL PRIME-CYCLE / PRIMORIAL STAIRCASE
# ============================================================

@dataclass(frozen=True)
class PrimeCycleStage:
    prime: int
    previous_period: int
    copies: int
    period: int


def prime_cycle_staircase(
    number_of_new_primes: int,
) -> tuple[PrimeCycleStage, ...]:
    """
    Generate

        6 --x5--> 30 --x7--> 210 --x11--> ...

    The primes themselves are produced by the same event-clock mechanism.
    """
    # 100 is much more than enough for a short printed staircase.
    clock = PrimeCycleMemory(
        ceiling=100
    )

    clock.advance_to(100)

    generated = clock.primes

    period = TWO * THREE
    stages: list[PrimeCycleStage] = []

    primes_after_three = [
        prime
        for prime in generated
        if prime > THREE
    ]

    for prime in primes_after_three[
        :number_of_new_primes
    ]:
        previous_period = period
        period = (
            prime * previous_period
        )

        stages.append(
            PrimeCycleStage(
                prime=prime,
                previous_period=previous_period,
                copies=prime,
                period=period,
            )
        )

    return tuple(stages)


# ============================================================
# 8. OPTIONAL PRIMECOUNT AUDIT
# ============================================================

def primecount_available() -> bool:
    return (
        shutil.which("primecount")
        is not None
    )


def primecount_pi(
    number: int,
) -> int:
    output = subprocess.check_output(
        ["primecount", str(number)],
        text=True,
    ).strip()

    return int(
        output.splitlines()[-ONE]
        .replace(",", "")
    )


def optional_primecount_audit(
    snapshots: tuple[CenterSnapshot, ...],
) -> None:
    """
    Independent audit only.

    No result from this function is ever returned to the recursive
    construction.
    """
    if not VERIFY_WITH_PRIMECOUNT:
        return

    if not primecount_available():
        print(
            "\nprimecount audit requested, "
            "but primecount is not installed."
        )
        return

    print(
        "\nOPTIONAL PRIMECOUNT AUDIT"
    )
    print("=" * 78)

    for snapshot in snapshots:
        if (
            snapshot.exponent
            > PRIMECOUNT_MAX_EXPONENT
        ):
            continue

        external_count = primecount_pi(
            snapshot.domain
        )

        print(
            f"2^{snapshot.exponent:<3} "
            f"internal pi={snapshot.prime_count:,}  "
            f"external pi={external_count:,}  "
            f"{'MATCH' if external_count == snapshot.prime_count else 'DIFFER'}"
        )


# ============================================================
# 9. REPORTS
# ============================================================

def print_structure() -> None:
    print(
        "\nRECURSIVE GOLD--SIMPLEX DOMAIN MEMORY"
    )
    print("=" * 78)

    print(
        "Dimensional boundary clock:",
        " -> ".join(
            str(value)
            for value in dimensional_boundaries()
        ),
    )

    print(
        "\nStructural blocks:"
    )

    for block in structural_blocks(
        STRUCTURE_MAX_EXPONENT
    ):
        exponents = tuple(
            address.exponent
            for address in block.addresses()
            if (
                address.exponent
                <= STRUCTURE_MAX_EXPONENT
            )
        )

        if not exponents:
            continue

        print(
            f"  k={block.k:<2} "
            f"d={block.start_dimension}.."
            f"{TWO * block.start_dimension:<3} "
            f"-> "
            + ", ".join(
                f"2^{exponent}"
                for exponent in exponents
            )
        )


def print_gold_states() -> None:
    print(
        "\nGOLD / SIMPLEX MEMORY STATES"
    )
    print("=" * 78)

    dimension = THREE

    while (
        EIGHT * dimension
        <= STRUCTURE_MAX_EXPONENT
    ):
        pair = gamma(
            dimension
        )

        print(
            f"d={dimension:<3} "
            f"Gamma_d=({pair[ZERO]:,}, {pair[ONE]:,})"
        )

        dimension = successor(
            dimension
        )

        if dimension > 16:
            # Keep the default report readable.
            break


def print_prime_cycle_staircase() -> None:
    print(
        "\nPRIME-CYCLE REPLICATION MEMORY"
    )
    print("=" * 78)

    print(
        "W_3 = 6"
    )

    for stage in prime_cycle_staircase(
        FIVE
    ):
        print(
            f"{stage.copies} copies of "
            f"{stage.previous_period:,} "
            f"-> W_{stage.prime}="
            f"{stage.period:,}"
        )


def print_exact_centers(
    memory: RecursiveCenterMemory,
) -> None:
    print(
        "\nEXACT CENTERS GENERATED BY INTERNAL CYCLE CLOCKS"
    )
    print("=" * 78)

    for state in memory.snapshots:
        pair_text = (
            str(state.middle_primes[ZERO])
            if len(state.middle_primes) == ONE
            else (
                f"{state.middle_primes[ZERO]}, "
                f"{state.middle_primes[ONE]}"
            )
        )

        print(
            f"2^{state.exponent:<3} "
            f"pi={state.prime_count:<8,} "
            f"C={state.center:<12,} "
            f"middle=({pair_text})"
        )

        print(
            f"      r={state.radius:,}, "
            f"rho={state.residual:+,}"
        )


def print_carry_memory(
    memory: RecursiveCenterMemory,
) -> None:
    print(
        "\nRECURSIVE CARRY MEMORY"
    )
    print("=" * 78)

    for carry in memory.carries:
        print(
            f"2^{carry.previous_exponent} "
            f"-> 2^{carry.exponent}"
        )

        print(
            f"      M = {carry.multiplier:,}"
        )

        print(
            f"      R = C_next - M*C_previous "
            f"= {carry.carry:+,}"
        )

        print(
            f"      Gold address of R: "
            f"{carry.gold_carry}"
        )

        if carry.radius_carry is not None:
            print(
                f"      r_next = "
                f"{carry.root_multiplier}*r_previous "
                f"+ q"
            )

            print(
                f"      q = "
                f"{carry.radius_carry:+,}"
            )

            print(
                f"      Gold address of q: "
                f"{carry.gold_radius_carry}"
            )

        print(
            f"      rho_next = "
            f"{carry.residual:+,}"
        )


# ============================================================
# 10. SELF-TESTS
# ============================================================

def run_tests(
    memory: RecursiveCenterMemory,
) -> None:
    assert simplex(
        THREE,
        TWO,
    ) == FOUR

    assert G(ZERO) == TWO
    assert G(ONE) == THREE
    assert G(TWO) == FIVE
    assert G(THREE) == EIGHT

    assert doubled_gamma(
        gamma(THREE)
    ) == gamma(SIX)

    # The clock must generate the familiar first primes without a
    # divisibility test.
    clock = PrimeCycleMemory(
        ceiling=30
    )

    clock.advance_to(30)

    assert clock.primes == [
        2, 3, 5, 7, 11,
        13, 17, 19, 23, 29,
    ]

    known_centers = {
        3: 4,
        4: 6,
        5: 13,
        6: 26,
        8: 105,
        10: 446,
        12: 1839,
        16: 30256,
        20: 493243,
        24: 7986331,
    }

    for snapshot in memory.snapshots:
        expected = known_centers.get(
            snapshot.exponent
        )

        if expected is not None:
            assert snapshot.center == expected

    print(
        "\nALL INTERNAL RECURSIVE TESTS PASSED"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print_structure()
    print_gold_states()
    print_prime_cycle_staircase()

    memory = build_exact_center_memory(
        EXACT_MAX_EXPONENT
    )

    print_exact_centers(
        memory
    )

    print_carry_memory(
        memory
    )

    run_tests(
        memory
    )

    optional_primecount_audit(
        tuple(memory.snapshots)
    )


if __name__ == "__main__":
    main()
