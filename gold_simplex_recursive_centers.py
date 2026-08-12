#!/usr/bin/env python3
"""
GOLD--SIMPLEX RECURSIVE PRIME-CENTER CONSTRUCTION

Purpose
-------
Build the number, Gold, simplex, prime-cycle, and dyadic-domain structures
from the smallest seeds and carry them recursively into arbitrarily large
domains.

The construction itself does NOT call primecount.

Optional primecount verification can be enabled at the bottom, but it is
strictly secondary, never feeds values back into the construction, and is
hard-capped at 2^64.

Important distinction
---------------------
The domain hierarchy is completely recursive:

    D(k,d) = 2^(2^k d)

and therefore the same interval can be viewed at several nested resolutions.

For example, 2^48 ... 2^96 appears as

    step 16 : 48, 64, 80, 96
    step  8 : 48, 56, 64, 72, 80, 88, 96
    step  4 : 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96
    step  2 : 48, 50, 52, ..., 96
    step  1 : 48, 49, 50, ..., 96

The prime-cycle hierarchy is also recursive:

    6 --x5--> 30 --x7--> 210 --x11--> 2310 --x13--> 30030 ...

These are coupled structural hierarchies, but they are not the same list.

For the prime-center coordinate, the program reproduces every explicit
numerical construction presently supplied by the paper through 2^64.
Beyond that point it continues the recursive center ADDRESS exactly:

    C_next = M*C_previous + R(S_d, Gamma_d)

or, in the M=256 square-coordinate description,

    r_next = 16*r_previous + q_d
    C      = r_next^2 + rho_d

where Gamma_d is generated automatically.

No numerical q_d or rho_d is invented when the paper has not supplied one.
The program nevertheless continues the domain, simplex, Gold, prime-cycle,
and symbolic center recursion to any requested exponent.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import shutil
import subprocess


# ============================================================
# SETTINGS
# ============================================================

MAX_EXPONENT = 192

# How far to refine a displayed dyadic interval.
# 0 -> only the coarsest row
# 1 -> one halving
# 2 -> two halvings, etc.
REFINEMENT_LEVELS = 4

VERIFY_WITH_PRIMECOUNT = False
PRIMECOUNT_MAX_EXPONENT = 64


# ============================================================
# 1. BUILD EVERYTHING FROM ONE
# ============================================================

ONE = 1
ZERO = ONE - ONE


def successor(n: int) -> int:
    return n + ONE


def natural(index: int) -> int:
    """Build a non-negative integer by repeated successor."""
    value = ZERO
    for _ in range(index):
        value = successor(value)
    return value


TWO = successor(ONE)
THREE = successor(TWO)
FOUR = successor(THREE)
FIVE = successor(FOUR)
SIX = successor(FIVE)
SEVEN = successor(SIX)
EIGHT = successor(SEVEN)


# ============================================================
# 2. BASIC RECURSIVE ARITHMETIC
# ============================================================

def power(base: int, exponent: int) -> int:
    answer = ONE
    count = ZERO

    while count < exponent:
        answer *= base
        count = successor(count)

    return answer


@lru_cache(maxsize=None)
def simplex(order: int, position: int) -> int:
    """
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
# 3. GOLD SEQUENCE
# ============================================================

_gold = [TWO, THREE]


def G(index: int) -> int:
    while len(_gold) <= index:
        _gold.append(_gold[-ONE] + _gold[-TWO])

    return _gold[index]


def gamma(dimension: int) -> tuple[int, int]:
    """
    Gamma_d = (G_(2d-2), G_(2d-1))
    """
    return (
        G(TWO * dimension - TWO),
        G(TWO * dimension - ONE),
    )


def doubled_gamma(pair: tuple[int, int]) -> tuple[int, int]:
    """
    Exact dimensional lift:

        (a,b) ->
        (a^2 + (b-a)^2, a(2b-a))
    """
    a, b = pair

    return (
        a * a + (b - a) * (b - a),
        a * (TWO * b - a),
    )


# ============================================================
# 4. DYADIC DOMAIN ADDRESSES
# ============================================================

@dataclass(frozen=True)
class DomainAddress:
    k: int
    dimension: int

    @property
    def scale_exponent(self) -> int:
        return power(TWO, self.k)

    @property
    def scale(self) -> int:
        return power(TWO, self.scale_exponent)

    @property
    def exponent(self) -> int:
        """
        D_(k,d) = 2^(2^k d)
        """
        return self.scale_exponent * self.dimension

    def label(self) -> str:
        return (
            f"D({self.k},{self.dimension})"
            f" = 2^{self.exponent}"
        )


def equivalent_addresses(exponent: int) -> tuple[DomainAddress, ...]:
    """
    Return every integer address (k,d) satisfying

        exponent = 2^k * d.
    """
    addresses: list[DomainAddress] = []

    k = ZERO
    divisor = ONE

    while exponent % divisor == ZERO:
        addresses.append(
            DomainAddress(
                k=k,
                dimension=exponent // divisor,
            )
        )

        k = successor(k)
        divisor *= TWO

    return tuple(addresses)


def dimensional_block(
    k: int,
    start_dimension: int,
) -> tuple[DomainAddress, ...]:
    """
    M^d, M^(d+1), ..., M^(2d)
    """
    return tuple(
        DomainAddress(k, d)
        for d in range(
            start_dimension,
            TWO * start_dimension + ONE,
        )
    )


def exponent_grid(
    start_exponent: int,
    end_exponent: int,
    step: int,
) -> tuple[int, ...]:
    return tuple(
        range(
            start_exponent,
            end_exponent + ONE,
            step,
        )
    )


def recursive_refinement_grids(
    start_exponent: int,
    end_exponent: int,
    first_step: int,
    levels: int,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """
    Repeatedly halve the exponent step.

    Example:
        48..96, first_step=8

        8 : 48,56,...,96
        4 : 48,52,...,96
        2 : ...
        1 : ...
    """
    rows: list[tuple[int, tuple[int, ...]]] = []

    step = first_step

    for _ in range(levels):
        rows.append(
            (
                step,
                exponent_grid(
                    start_exponent,
                    end_exponent,
                    step,
                ),
            )
        )

        if step == ONE:
            break

        step //= TWO

    return tuple(rows)


def local_refinement(
    start_exponent: int,
    end_exponent: int,
    step: int,
) -> tuple[int, ...]:
    """
    A convenient helper for examining a smaller inherited interval.

    Example:
        local_refinement(56,72,4)
        -> (56,60,64,68,72)
    """
    return exponent_grid(
        start_exponent,
        end_exponent,
        step,
    )


# ============================================================
# 5. AUTOMATIC DIMENSIONAL CARRY
# ============================================================

def dimensional_boundaries(
    first_dimension: int = THREE,
    count: int = EIGHT,
) -> tuple[int, ...]:
    """
    3, 6, 12, 24, 48, ...
    """
    values = [first_dimension]

    while len(values) < count:
        values.append(TWO * values[-ONE])

    return tuple(values)


def post_s6_blocks(
    count: int,
) -> tuple[tuple[DomainAddress, ...], ...]:
    """
    After the S6 carry the paper retains k=3, i.e. M=256,
    while the active dimension doubles:

        6..12
        12..24
        24..48
        ...
    """
    blocks: list[tuple[DomainAddress, ...]] = []

    k = THREE
    d = SIX

    for _ in range(count):
        blocks.append(dimensional_block(k, d))
        d *= TWO

    return tuple(blocks)


# ============================================================
# 6. PRIME-CYCLE REPLICATION
# ============================================================

def generated_prime(candidate: int, earlier_primes: tuple[int, ...]) -> bool:
    """
    Internal cycle test.

    A candidate is rejected when an already-open prime cycle closes on it.
    No external primality library is used.
    """
    for p in earlier_primes:
        if p * p > candidate:
            break

        if candidate % p == ZERO:
            return False

    return True


def next_generated_prime(
    current_prime: int,
    earlier_primes: tuple[int, ...],
) -> int:
    candidate = successor(current_prime)

    while not generated_prime(candidate, earlier_primes):
        candidate = successor(candidate)

    return candidate


@dataclass(frozen=True)
class PrimeCycle:
    prime: int
    previous_period: int
    copies: int
    period: int


def prime_cycle_ladder(
    number_of_steps: int,
) -> tuple[PrimeCycle, ...]:
    """
    Build

        6 --x5--> 30 --x7--> 210 --x11--> 2310 ...

    The multiplier prime is generated from earlier cycles rather than
    entered as a list.
    """
    primes = [TWO, THREE]

    period = TWO * THREE
    current_prime = THREE

    result: list[PrimeCycle] = []

    for _ in range(number_of_steps):
        next_prime = next_generated_prime(
            current_prime,
            tuple(primes),
        )

        previous = period
        period = next_prime * previous

        result.append(
            PrimeCycle(
                prime=next_prime,
                previous_period=previous,
                copies=next_prime,
                period=period,
            )
        )

        primes.append(next_prime)
        current_prime = next_prime

    return tuple(result)


# ============================================================
# 7. CENTER STATES
# ============================================================

@dataclass(frozen=True)
class CenterState:
    exponent: int
    center: int | None
    radius: int | None
    residual: int | None
    dimension: int | None
    gold_pair: tuple[int, int] | None
    expression: str
    status: str


def square_center(radius: int, residual: int) -> int:
    return radius * radius + residual


def first_center_states() -> dict[int, CenterState]:
    """
    Derive the early chain from the primitive, simplex, and Gold rules.
    """
    states: dict[int, CenterState] = {}

    c8 = simplex(THREE, TWO)

    states[THREE] = CenterState(
        exponent=THREE,
        center=c8,
        radius=None,
        residual=None,
        dimension=THREE,
        gold_pair=gamma(THREE),
        expression="S_3(2)",
        status="constructed",
    )

    center = c8
    exponent = THREE

    first_carries = (
        -G(ZERO),
        simplex(ZERO, ONE),
        ZERO,
    )

    for carry in first_carries:
        center = TWO * center + carry
        exponent = successor(exponent)

        states[exponent] = CenterState(
            exponent=exponent,
            center=center,
            radius=None,
            residual=None,
            dimension=exponent,
            gold_pair=None,
            expression=f"2*C(2^{exponent-1}) + ({carry})",
            status="constructed",
        )

    c64 = center

    scale = FOUR
    center = c64
    d = THREE

    second_carries = (
        simplex(ZERO, ONE),
        c64,
        G(SEVEN),
    )

    for carry in second_carries:
        d = successor(d)
        center = scale * center + carry
        exponent = TWO * d

        states[exponent] = CenterState(
            exponent=exponent,
            center=center,
            radius=None,
            residual=None,
            dimension=d,
            gold_pair=gamma(d),
            expression=f"4*C(previous) + ({carry})",
            status="constructed",
        )

    # 2^12 square coordinate.
    r12 = TWO * G(FIVE) + ONE
    rho12 = -simplex(TWO, FOUR)

    assert square_center(r12, rho12) == states[12].center

    states[12] = CenterState(
        exponent=12,
        center=states[12].center,
        radius=r12,
        residual=rho12,
        dimension=THREE,
        gold_pair=gamma(THREE),
        expression="(2*G_5+1)^2 - S_2(4)",
        status="constructed",
    )

    # M=16 block.  q begins at G_0=2 and triples:
    # 2, 6, 18.
    r = r12
    q = G(ZERO)

    residuals = (
        -simplex(THREE, FOUR),
        G(FIVE) * G(FIVE) - G(ZERO),
        G(SEVEN),
    )

    for d, rho in zip(
        range(FOUR, SEVEN),
        residuals,
    ):
        r = FOUR * r + q
        exponent = FOUR * d
        center = square_center(r, rho)

        states[exponent] = CenterState(
            exponent=exponent,
            center=center,
            radius=r,
            residual=rho,
            dimension=d,
            gold_pair=gamma(d),
            expression=f"r={FOUR}*r_previous+{q}; C=r^2+({rho})",
            status="constructed",
        )

        q *= THREE

    # Independent Gold--simplex identity at 2^24.
    r24 = (
        G(SIX)
        * (
            G(EIGHT)
            - simplex(TWO, THREE)
        )
        + c8
    )

    assert r24 == states[24].radius

    return states


def m256_known_refinement(
    dimension: int,
    states: dict[int, CenterState],
) -> tuple[int, int] | None:
    """
    Explicit Gold--simplex refinements appearing in the paper.

    Nothing here is obtained from primecount.
    """
    c8 = states[THREE].center
    c64 = states[SIX].center

    assert c8 is not None
    assert c64 is not None

    scale16 = power(TWO, FOUR)
    scale256 = power(TWO, EIGHT)

    gamma6 = gamma(SIX)
    gamma7 = gamma(SEVEN)
    gamma8 = gamma(EIGHT)

    if dimension == FOUR:
        q = gamma6[ONE] - G(SEVEN) - G(TWO)

        rho = scale16 * (
            G(NINE := successor(EIGHT))
            + G(SEVEN)
            - simplex(TWO, THREE)
        )

        return q, rho

    if dimension == FIVE:
        q = (
            gamma8[ONE]
            + gamma6[ZERO]
            + G(FOUR)
            + ONE
        )

        rho = -(
            G(ZERO)
            * (G(FOUR) - G(ZERO))
            * (G(FIVE) - c8)
            * (G(FIVE) + G(ZERO))
            * (G(EIGHT) - scale16)
        )

        return q, rho

    if dimension == SIX:
        q32 = gamma6[ONE] - G(SEVEN) - G(TWO)

        gamma11 = gamma(
            natural(11)
        )
        gamma12 = gamma(
            natural(12)
        )
        gamma9 = gamma(
            natural(9)
        )

        q = gamma11[ZERO] + q32 + c8

        factor83 = simplex(SIX, FOUR) - ONE

        factor68219 = (
            gamma12[ZERO]
            - gamma9[ONE]
            - G(SIX)
            - G(THREE)
            + ONE
        )

        rho = -factor83 * factor68219

        return q, rho

    if dimension == SEVEN:
        nineteen = G(FIVE) - G(ZERO)
        seventy_three = G(EIGHT) - scale16
        forty_four = TWO * G(FIVE) + TWO

        q = nineteen * (
            gamma6[ZERO] * seventy_three
            - forty_four
        )

        sixty_seven = (
            G(EIGHT)
            - (
                THREE * power(TWO, THREE)
                - TWO
            )
        )

        nine_sixty_seven = (
            gamma7[ONE]
            - simplex(THREE, FOUR)
        )

        rho = -(
            G(TWO)
            * sixty_seven
            * seventy_three
            * nine_sixty_seven
        )

        return q, rho

    if dimension == EIGHT:
        nineteen = G(FIVE) - G(ZERO)

        g9 = G(successor(EIGHT))

        eight_fifty_seven = (
            gamma7[ONE]
            - g9
            + G(FOUR)
            + ONE
        )

        q = nineteen * (
            gamma6[ZERO] * eight_fifty_seven
            + c64
        )

        fifty_three = G(SEVEN) - G(ZERO)

        five_seventy_two = (
            gamma7[ZERO]
            - (G(SIX) + c8)
        )

        two_thirteen = (
            gamma6[ZERO]
            - simplex(THREE, FOUR)
        )

        rho = -(
            scale256
            * fifty_three
            * (
                gamma6[ONE] * five_seventy_two
                + two_thirteen
            )
        )

        return q, rho

    return None


def build_center_chain(
    max_exponent: int,
) -> dict[int, CenterState]:
    """
    Build every numerical center explicitly supplied by the construction,
    then continue the formal recursion automatically.

    The function never silently substitutes measured prime data.
    """
    states = first_center_states()

    r = states[24].radius
    assert r is not None

    d = FOUR

    while EIGHT * d <= max_exponent:
        exponent = EIGHT * d

        refinement = m256_known_refinement(
            d,
            states,
        )

        if refinement is not None:
            q, rho = refinement
            r = power(TWO, FOUR) * r + q
            center = square_center(r, rho)

            states[exponent] = CenterState(
                exponent=exponent,
                center=center,
                radius=r,
                residual=rho,
                dimension=d,
                gold_pair=gamma(d),
                expression=(
                    f"r=16*r_previous+{q}; "
                    f"C=r^2+({rho})"
                ),
                status="constructed",
            )

        elif exponent not in states:
            pair = gamma(d)

            states[exponent] = CenterState(
                exponent=exponent,
                center=None,
                radius=None,
                residual=None,
                dimension=d,
                gold_pair=pair,
                expression=(
                    f"r_{exponent}=16*r_previous+q(S_{d},Gamma_{d}); "
                    f"C(2^{exponent})=r_{exponent}^2+rho(S_{d},Gamma_{d})"
                ),
                status="recursive prediction address",
            )

        d = successor(d)

    return states


# ============================================================
# 8. COMPOSE SQUARE-COORDINATE TRANSPORTS
# ============================================================

@dataclass(frozen=True)
class AffineTransport:
    multiplier: int
    offset: int

    def apply(self, value: int) -> int:
        return (
            self.multiplier * value
            + self.offset
        )

    def then(self, later: "AffineTransport") -> "AffineTransport":
        """
        later(self(r))
        """
        return AffineTransport(
            multiplier=(
                later.multiplier
                * self.multiplier
            ),
            offset=(
                later.multiplier
                * self.offset
                + later.offset
            ),
        )


def known_radius_transport(
    q: int,
) -> AffineTransport:
    return AffineTransport(
        multiplier=power(TWO, FOUR),
        offset=q,
    )


# ============================================================
# 9. OPTIONAL EXTERNAL VERIFICATION
# ============================================================

def primecount_available() -> bool:
    return shutil.which("primecount") is not None


def primecount_pi(number: int) -> int:
    output = subprocess.check_output(
        ["primecount", str(number)],
        text=True,
    ).strip()

    return int(
        output.splitlines()[-ONE]
        .replace(",", "")
    )


def verify_center_with_primecount(
    exponent: int,
    predicted_center: int,
) -> None:
    """
    Secondary audit only.

    This function is deliberately unable to run above 2^64.
    It never feeds anything into build_center_chain().
    """
    if exponent > PRIMECOUNT_MAX_EXPONENT:
        raise ValueError(
            "External validation is disabled above 2^64."
        )

    domain_value = power(TWO, exponent)
    count = primecount_pi(domain_value)

    print(
        f"2^{exponent}: pi(D)={count:,}, "
        f"constructed center={predicted_center:,}"
    )


# ============================================================
# 10. REPORT
# ============================================================

def format_exponents(values: tuple[int, ...]) -> str:
    return ", ".join(
        f"2^{value}"
        for value in values
    )


def print_domain_hierarchy() -> None:
    print("\nRECURSIVE DOMAIN HIERARCHY")
    print("=" * 78)

    print(
        "Dimensional boundaries:",
        " -> ".join(
            str(value)
            for value in dimensional_boundaries()
        ),
    )

    print("\nOrder-six block:")
    block = dimensional_block(
        THREE,
        SIX,
    )

    print(
        format_exponents(
            tuple(
                address.exponent
                for address in block
            )
        )
    )

    print("\nNested refinements of 2^48 ... 2^96:")

    for step, row in recursive_refinement_grids(
        48,
        96,
        EIGHT,
        REFINEMENT_LEVELS,
    ):
        print(
            f"  exponent step {step:>2}: "
            + ", ".join(
                str(value)
                for value in row
            )
        )

    print("\nLocal refinement 2^56 ... 2^72:")
    print(
        "  "
        + ", ".join(
            str(value)
            for value in local_refinement(
                56,
                72,
                FOUR,
            )
        )
    )

    print("\nLocal refinement 2^72 ... 2^96:")
    print(
        "  "
        + ", ".join(
            str(value)
            for value in local_refinement(
                72,
                96,
                FOUR,
            )
        )
    )


def print_prime_cycles() -> None:
    print("\nPRIME-CYCLE REPLICATION")
    print("=" * 78)

    print("start: W_3 = 6")

    for state in prime_cycle_ladder(FIVE):
        print(
            f"{state.copies} copies of "
            f"{state.previous_period:,} "
            f"-> W_{state.prime}="
            f"{state.period:,}"
        )


def print_gold_lift() -> None:
    print("\nGOLD / SIMPLEX DIMENSIONAL LIFT")
    print("=" * 78)

    for d in range(SIX, natural(13)):
        pair = gamma(d)
        exponent = EIGHT * d

        print(
            f"d={d:>2}  "
            f"2^{exponent:<3}  "
            f"Gamma=({pair[0]:,}, {pair[1]:,})"
        )

    print("\nExact dimension doubling check:")
    print(
        "Gamma_6 -> Gamma_12:",
        doubled_gamma(gamma(SIX)),
        "=",
        gamma(natural(12)),
    )


def print_centers(states: dict[int, CenterState]) -> None:
    print("\nPRIME-CENTER CONSTRUCTION")
    print("=" * 78)

    for exponent in sorted(states):
        state = states[exponent]

        if exponent > MAX_EXPONENT:
            continue

        if state.center is not None:
            print(
                f"2^{exponent:<3} "
                f"C={state.center:,}"
            )

            if state.radius is not None:
                print(
                    f"      r={state.radius:,}, "
                    f"rho={state.residual:,}"
                )

        elif exponent >= 72:
            print(
                f"2^{exponent:<3} "
                f"d={state.dimension:<3} "
                f"Gamma={state.gold_pair}"
            )
            print(
                f"      {state.expression}"
            )


def print_composed_transport(states: dict[int, CenterState]) -> None:
    """
    Demonstrate that fine recursive steps compose into a larger inherited
    domain step.

    2^48 -> 2^56 -> 2^64 becomes one 2^48 -> 2^64 transform.
    """
    q56 = m256_known_refinement(
        SEVEN,
        states,
    )
    q64 = m256_known_refinement(
        EIGHT,
        states,
    )

    if q56 is None or q64 is None:
        return

    t56 = known_radius_transport(q56[ZERO])
    t64 = known_radius_transport(q64[ZERO])

    combined = t56.then(t64)

    r48 = states[48].radius
    r64 = states[64].radius

    assert r48 is not None
    assert r64 is not None
    assert combined.apply(r48) == r64

    print("\nNESTED TRANSPORT COMPOSITION")
    print("=" * 78)
    print(
        "48 -> 56 -> 64:"
    )
    print(
        f"  r64 = {combined.multiplier}*r48 "
        f"+ {combined.offset:,}"
    )
    print(
        f"  = {r64:,}"
    )


def run_tests(states: dict[int, CenterState]) -> None:
    assert simplex(THREE, TWO) == FOUR

    assert G(ZERO) == TWO
    assert G(ONE) == THREE
    assert G(TWO) == FIVE
    assert G(THREE) == EIGHT

    assert doubled_gamma(
        gamma(THREE)
    ) == gamma(SIX)

    assert doubled_gamma(
        gamma(SIX)
    ) == gamma(natural(12))

    assert DomainAddress(
        THREE,
        SIX,
    ).exponent == 48

    assert DomainAddress(
        FOUR,
        THREE,
    ).exponent == 48

    assert local_refinement(
        56,
        72,
        FOUR,
    ) == (
        56,
        60,
        64,
        68,
        72,
    )

    expected = {
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
        32: 2073257177,
        40: 534885358935,
        48: 137609832979107,
        56: 35349221681200784,
        64: 9072244146347642849,
    }

    for exponent, value in expected.items():
        assert states[exponent].center == value

    print("\nALL STRUCTURAL TESTS PASSED")


def main() -> None:
    states = build_center_chain(
        MAX_EXPONENT
    )

    print_domain_hierarchy()
    print_prime_cycles()
    print_gold_lift()
    print_centers(states)
    print_composed_transport(states)
    run_tests(states)

    if VERIFY_WITH_PRIMECOUNT:
        if not primecount_available():
            print(
                "\nprimecount is not installed; "
                "construction completed without it."
            )
            return

        print("\nOPTIONAL PRIMECOUNT AUDIT")
        print("=" * 78)

        for exponent in (48, 56, 64):
            state = states[exponent]

            assert state.center is not None

            verify_center_with_primecount(
                exponent,
                state.center,
            )


if __name__ == "__main__":
    main()
