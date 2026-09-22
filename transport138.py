#!/usr/bin/env python3
"""
This file is a work in progress, some areas require further implementation and optimization.
Over the next few weeks this file will continue to have better optimizations, and stronger 
functional logic connected to it's underlying geometry.

Recursive twin-prime-center transport.

The program has three mathematical layers:

1. GEOMETRY
   Build the 6n±1 center lattice, the first 3-dimensional/8-vertex parent,
   and a prefix of completed 3-clock cube masks.

2. TRANSPORT
   Move that geometry across the number line.  Gray code orders the eight
   cube vertices; completed CRT masks are streamed as periodic objects; every
   clock not yet absorbed into a completed cube remains in one square-open
   frontier and is carried by its two reflected residues.

3. LEVELS
   Count surviving center positions.  A p-level contains p centers and shares
   its final center with the next level, so each new level consumes p-1 new
   survivors.  Consequently the closing center of one level is literally the
   opening center of the next.

The bit-packing constants below describe the machine word only.  They are not
part of the number-theory geometry.
"""

from math import isqrt
import os
import numpy as np
from numba import njit, prange, set_num_threads


# =============================================================================
# 1. FUNDAMENTAL GEOMETRY
# =============================================================================

# 2 and 3 establish the first completed divisibility lattice.
SEED = (2, 3)
LATTICE = SEED[0] * SEED[1]          # 6; surviving prime branches are 6n±1

# A 3-dimensional binary cube has 2^3 = 8 vertices.
# DIM comes from the two seed clocks plus the new axis opened beyond them.
DIM = len(SEED) + 1
CUBE = 1 << DIM                      # vertex count; never typed as a magic 8
RESET_SCALE = CUBE ** DIM            # three cube scales: 8^3 = 2^9 = 512

# The total number of ordinary cube edges would be DIM*CUBE/2.
# We do NOT enumerate those edges.  Gray order below traverses the vertices so
# successive branch labels differ in exactly one axis bit.


# =============================================================================
# 2. STORAGE GEOMETRY (COMPUTER REPRESENTATION, NOT NUMBER THEORY)
# =============================================================================

WORD = np.uint64
WORD_BYTES = np.dtype(WORD).itemsize
BYTE_BITS = np.iinfo(np.uint8).bits
WORD_BITS = WORD_BYTES * BYTE_BITS
WORD_SHIFT = WORD_BITS.bit_length() - 1
WORD_LOW = WORD_BITS - 1
FULL_WORD = np.uint64((1 << WORD_BITS) - 1)

# Masks for the branch-free SWAR population count used by resolve().
# They are derived from WORD_BITS instead of being hard-coded 64-bit literals.
POPCOUNT_1 = np.uint64(int(FULL_WORD) // 3)
POPCOUNT_2 = np.uint64(int(FULL_WORD) // 5)
POPCOUNT_4 = np.uint64(int(FULL_WORD) // 17)
POPCOUNT_BYTE = np.uint64(int(FULL_WORD) // ((1 << BYTE_BITS) - 1))
INT32_MAX = np.iinfo(np.int32).max

try:
    set_num_threads(min(CUBE, os.cpu_count() or 1))
except ValueError:
    pass


# =============================================================================
# 3. PRIME CLOCKS AND CRT GEOMETRY
# =============================================================================

def prime_state(limit):
    """Return primes and their signed 6n±1 coordinates through *limit*.

    For p > 3, write p = 6a-1 or 6a+1.  H stores one signed coordinate:

        H = +a  for p = 6a-1
        H = -a  for p = 6a+1

    The sign records handedness; abs(H) supplies the coordinate magnitude.
    This is the only per-prime geometric state carried by the program.
    """
    sieve = np.ones((limit + 1)//2, np.bool_)
    sieve[0] = False

    for p in range(3, isqrt(limit) + 1, 2):
        if sieve[p//2]:
            sieve[p*p//2::p] = False

    odd = 2*np.flatnonzero(sieve) + 1
    dtype = np.int32 if limit <= INT32_MAX else np.int64

    primes = np.empty(len(odd) + 1, dtype)
    primes[0] = SEED[0]
    primes[1:] = odd

    hand = np.zeros(len(primes), np.int32)
    q = primes[len(SEED):].astype(np.int64)
    hand[len(SEED):] = np.where(
        q % LATTICE == LATTICE - 1,
        (q + 1)//LATTICE,
        -(q - 1)//LATTICE,
    ).astype(np.int32)

    return primes, hand


def pack_bits(bits):
    """Pack one byte-per-position field into the native unsigned word type."""
    packed = np.packbits(bits, bitorder='little')
    padding = (-len(packed)) % WORD_BYTES
    if padding:
        packed = np.pad(packed, (0, padding))
    return packed.view(WORD).copy()


def square_and_gap(hand):
    """Return the square-opening center index and reflected-face gap.

    If p = 6a±1, the first self-composite event is p².  On the center lattice
    its index is (p²-1)/6, which becomes the formulas below in signed a.

    The second value is the displacement to the reflected closure belonging to
    the opposite ±1 neighbor.  These two positions are the two forbidden
    residues contributed by that prime clock once its square has opened.
    """
    a = abs(hand)
    if hand > 0:                    # p = 6a-1
        return LATTICE*a*a - 2*a, 2*a
    return LATTICE*a*a + 2*a, 4*a + 1  # p = 6a+1


def build_geometry(primes, hand):
    """Build the reusable parent and the completed CRT cube objects.

    CRT interpretation
    ------------------
    Every prime p>3 excludes exactly two center-index residues modulo p.
    Because the prime moduli are pairwise coprime, their combined pattern has
    product period.  We therefore realize the Chinese Remainder Theorem as a
    periodic bit mask instead of explicitly solving each CRT congruence.

    Cube interpretation
    -------------------
    The first CUBE global prime states define the reusable parent.  After that,
    every DIM consecutive prime clocks form one 3-clock cube object.  A leading
    prefix of those cube periods is compressed while the total stored period
    still fits inside one live parent window.  The rest stay in the raw square
    frontier and need no second representation.
    """
    parent = bytearray(1)

    # 2 and 3 are already absorbed into the 6n±1 lattice, so the parent begins
    # with the remaining prime states up to the first 2^DIM completion.
    for index in range(len(SEED), CUBE):
        p = int(primes[index])
        coordinate = abs(int(hand[index]))
        parent *= p

        # CRT faces: the two forbidden residues ±coordinate (mod p).
        for residue in (coordinate, (-coordinate) % p):
            parent[residue::p] = b'\1' * (
                (len(parent) - 1 - residue)//p + 1
            )

    # Rotate once so local center index 1 is first, then expose all cube vertices.
    parent = (parent[1:] + parent[:1]) * CUBE
    template = pack_bits(np.frombuffer(parent, np.uint8))
    window = len(parent)

    masks = []
    offsets = []
    periods = []
    maturity = []
    flat_offset = 0
    used_period = 0
    index = CUBE

    while index + DIM <= len(primes):
        period = 1
        for axis in range(DIM):
            period *= int(primes[index + axis])

        # Compression is self-limiting: completed cube masks are retained only
        # while their total CRT period fits inside one reusable parent window.
        if used_period + period > window:
            break

        pattern = np.zeros(period, np.uint8)
        last_opening = 0

        for axis in range(DIM):
            p = int(primes[index + axis])
            h = int(hand[index + axis])
            coordinate = abs(h)

            pattern[coordinate::p] = 1
            pattern[(-coordinate) % p::p] = 1

            square, gap = square_and_gap(h)
            last_opening = max(last_opening, square + gap)

        # periodic() may read across one word boundary; duplicating the mask and
        # appending one word gives that read a safe contiguous tail.
        words = pack_bits(
            np.concatenate((pattern, pattern, pattern[:WORD_BITS]))
        )

        masks.append(words)
        offsets.append(flat_offset)
        periods.append(period)
        maturity.append(last_opening + 1)

        flat_offset += len(words)
        used_period += period
        index += DIM

    return (
        template,
        window,
        np.concatenate(masks),
        np.asarray(offsets, np.int64),
        np.asarray(periods, np.int64),
        np.asarray(maturity, np.int64),
    )


# =============================================================================
# 4. HOT TRANSPORT KERNEL
# =============================================================================

@njit(cache=True, inline='always')
def set_bit(bits, index):
    """Close one center address inside a packed bit field."""
    bits[index >> WORD_SHIFT] |= (
        np.uint64(1) << np.uint64(index & WORD_LOW)
    )


@njit(cache=True, inline='always')
def popcount(word):
    """Count closed addresses in one packed word without a lookup table."""
    word = word - ((word >> np.uint64(1)) & POPCOUNT_1)
    word = (word & POPCOUNT_2) + ((word >> np.uint64(2)) & POPCOUNT_2)
    word = (word + (word >> np.uint64(4))) & POPCOUNT_4
    return int(
        (word * POPCOUNT_BYTE) >> np.uint64(WORD_BITS - BYTE_BITS)
    )


@njit(cache=True, inline='always')
def periodic_word(flat, offset, start):
    """Read one machine word from a periodic CRT mask at arbitrary phase."""
    word = offset + (start >> WORD_SHIFT)
    shift = start & WORD_LOW

    if shift == 0:
        return flat[word]

    return (
        (flat[word] >> np.uint64(shift))
        | (flat[word + 1] << np.uint64(WORD_BITS - shift))
    )


@njit(cache=True, parallel=True)
def transport(bits, primes, hand, lower, window, flat, offsets, periods, maturity):
    """Transport completed geometry and the unresolved square frontier.

    Gray code
    ---------
    k ^ (k >> 1) maps the CUBE worker indices into reflected Gray order.  The
    eight resulting chunks are disjoint, so they can be processed in parallel.
    Consecutive Gray labels differ by one axis bit; this is the orientation
    ordering of the cube.  This implementation uses that ordering to assign
    branches, but it does not explicitly enumerate all DIM*CUBE/2 cube edges.

    Square / cube transport
    -----------------------
    A prime clock does nothing before its square.  After p² opens, its two
    reflected closure residues repeat every p center indices.  Three successive
    clocks may eventually mature into one precompiled cube mask.  Therefore the
    only changing state needed here is `done`: how many leading cube objects
    have matured at this local position.

        completed cubes -> streamed periodic masks
        unfinished cubes -> one contiguous square frontier

    The frontier ends at sqrt(high), because every composite neighbor <= high
    has a prime factor no greater than its square root.  This is why the older
    special cube-exterior q*r routine is no longer necessary.
    """
    words = (window + WORD_LOW)//WORD_BITS
    chunk_words = (words + CUBE - 1)//CUBE
    cube_count = len(periods)

    for worker in prange(CUBE):
        gray = worker ^ (worker >> 1)
        first_word = gray * chunk_words

        if first_word >= words:
            continue

        final_word = min(words, first_word + chunk_words)
        local_start = first_word << WORD_SHIFT
        local_stop = min(window, final_word << WORD_SHIFT)
        local_lower = lower + local_start
        local_upper = lower + local_stop

        high_neighbor = LATTICE*(local_upper - 1) + 1
        root = int(np.sqrt(high_neighbor))
        while (root + 1)*(root + 1) <= high_neighbor:
            root += 1
        while root*root > high_neighbor:
            root -= 1

        # Maturity is monotone: once a cube object is complete, all earlier
        # cube objects are complete too.  `done` therefore determines both the
        # compressed prefix and the exact first raw prime clock.
        done = 0
        while done < cube_count and local_lower >= maturity[done]:
            done += 1

        absolute_word_start = lower + (first_word << WORD_SHIFT)

        if done == cube_count:
            # Fast path: every compressed cube is mature.  OR all of their CRT
            # masks together before touching the destination word.
            phase = np.empty(cube_count, np.int64)
            for cube in range(cube_count):
                phase[cube] = absolute_word_start % periods[cube]

            for word in range(first_word, final_word):
                combined = np.uint64(0)
                for cube in range(cube_count):
                    combined |= periodic_word(
                        flat, offsets[cube], phase[cube]
                    )
                    phase[cube] += WORD_BITS
                    if phase[cube] >= periods[cube]:
                        phase[cube] -= periods[cube]
                bits[word] |= combined

        else:
            # Only the completed prefix is streamed.  Everything after it is
            # handled below by the same raw square-frontier rule.
            for cube in range(done):
                period = periods[cube]
                phase = absolute_word_start % period
                offset = offsets[cube]

                for word in range(first_word, final_word):
                    bits[word] |= periodic_word(flat, offset, phase)
                    phase += WORD_BITS
                    if phase >= period:
                        phase -= period

        # Each completed cube absorbed DIM prime clocks.  This equation is the
        # state hand-off: cube maturity directly identifies the raw frontier.
        first_raw_clock = CUBE + DIM*done
        stop_clock = np.searchsorted(primes, root, side='right')

        for index in range(first_raw_clock, stop_clock):
            p = int(primes[index])
            h = int(hand[index])
            square, gap = square_and_gap_numba(h)

            # The two reflected faces generated by this prime clock.
            for initial_hit in (square, square + gap):
                hit = initial_hit
                if hit < local_lower:
                    hit += ((local_lower - hit + p - 1)//p)*p

                for center in range(hit, local_upper, p):
                    set_bit(bits, center - lower)


@njit(cache=True, inline='always')
def square_and_gap_numba(hand):
    """Compiled form of square_and_gap() for the hot frontier loop."""
    coordinate = hand if hand > 0 else -hand
    if hand > 0:
        return (
            LATTICE*coordinate*coordinate - 2*coordinate,
            2*coordinate,
        )
    return (
        LATTICE*coordinate*coordinate + 2*coordinate,
        4*coordinate + 1,
    )


# =============================================================================
# 5. COMBINATORIAL LEVEL COUNTING
# =============================================================================

@njit(cache=True)
def resolve(bits, window, seen, targets, target_index):
    """Locate only the survivor ranks required to close prime levels.

    Combinatorial interpretation
    ----------------------------
    A zero bit is one surviving twin-center state.  We count survivors word by
    word instead of enumerating all of them.  `targets` contains cumulative
    survivor ranks at which a prime level closes.

    The first level needs p survivors.  Thereafter consecutive levels overlap
    in one vertex/center -- previous close == next opening -- so a level of
    size p contributes only p-1 *new* survivors.  edges() constructs exactly
    those cumulative target ranks.
    """
    output = np.empty(len(targets) - target_index, np.int64)
    output_count = 0
    words = (window + WORD_LOW)//WORD_BITS

    for word in range(words):
        base = word << WORD_SHIFT
        valid = min(WORD_BITS, window - base)
        valid_mask = (
            FULL_WORD
            if valid == WORD_BITS
            else (np.uint64(1) << np.uint64(valid)) - np.uint64(1)
        )

        closed = bits[word] & valid_mask
        open_mask = (~bits[word]) & valid_mask
        available = valid - popcount(closed)
        after = seen + available

        while target_index < len(targets) and targets[target_index] <= after:
            rank = int(targets[target_index] - seen)
            candidate = open_mask

            # Remove rank-1 lowest surviving bits; the next set bit is the
            # exact closing center requested by this level.
            for _ in range(rank - 1):
                candidate &= candidate - np.uint64(1)

            bit = 0
            while (candidate & np.uint64(1)) == 0:
                candidate >>= np.uint64(1)
                bit += 1

            output[output_count] = base + bit
            output_count += 1
            target_index += 1

        seen = after

    return seen, target_index, output[:output_count]


# =============================================================================
# 6. LEVEL CARRY / PUBLIC GENERATOR
# =============================================================================

def edges(limit):
    """Yield (prime, opening center, closing center) through *limit*."""
    primes, hand = prime_state(max(RESET_SCALE, limit))
    rows = primes[(primes >= 5) & (primes <= limit)]

    if not len(rows):
        return

    # Target ranks encode the overlapping level combinatorics:
    # first close at p, then add p-1 because the old close is the new opening.
    targets = np.empty(len(rows), np.int64)
    targets[0] = int(rows[0])
    for index in range(1, len(rows)):
        targets[index] = targets[index - 1] + int(rows[index]) - 1

    template, window, flat, offsets, periods, maturity = build_geometry(
        primes, hand
    )

    lower = 1
    opening = LATTICE
    seen = 0
    row_index = 0

    while row_index < len(rows):
        # The transport frontier only needs prime clocks through sqrt(high).
        high_neighbor = LATTICE*(lower + window - 1) + 1
        required_prime = isqrt(high_neighbor)

        if primes[-1] < required_prime:
            primes, hand = prime_state(
                max(required_prime, 2*int(primes[-1]))
            )

        closed = template.copy()

        # The first local block contains the seed/self positions.  Clear the
        # first CUBE-1 bits so the reusable parent starts at the true opening.
        if lower == 1:
            closed[0] &= np.uint64(~(CUBE - 1) & int(FULL_WORD))

        transport(
            closed,
            primes,
            hand,
            lower,
            window,
            flat,
            offsets,
            periods,
            maturity,
        )

        old_row = row_index
        seen, row_index, positions = resolve(
            closed, window, seen, targets, row_index
        )

        for offset, position in enumerate(positions):
            closing = LATTICE*(lower + int(position))
            yield int(rows[old_row + offset]), opening, closing

            # Fundamental carry invariant of the level recursion.
            opening = closing

        lower += window


# =============================================================================
# 7. COMMAND LINE
# =============================================================================

if __name__ == '__main__':
    import sys

    argv = sys.argv[1:]
    quiet = '--noprint' in argv

    if '--threads' in argv:
        index = argv.index('--threads')
        set_num_threads(int(argv[index + 1]))
        del argv[index:index + 2]

    argv = [value for value in argv if value != '--noprint']
    limit = int(argv[0]) if argv else int(input('Prime ceiling: '))

    for prime, opening, closing in edges(limit):
        if not quiet:
            print(f'**{prime}** ({opening}, {closing})')
