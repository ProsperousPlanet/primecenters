#!/usr/bin/env python3
from math import isqrt
import os
import numpy as np
from numba import njit, prange, set_num_threads

DIM = 3
CUBE = 1 << DIM
MASK64 = (1 << 64) - 1
try:
    set_num_threads(min(CUBE, os.cpu_count() or 1))
except ValueError:
    pass

# ---------- 1. geometry -------------------------------------------------------

def prime_state(limit):
    if limit < 2:
        return np.empty(0, np.int32), np.empty(0, np.int32)
    sieve = np.ones((limit + 1)//2, np.bool_); sieve[0] = False
    for p in range(3, isqrt(limit) + 1, 2):
        if sieve[p//2]: sieve[p*p//2::p] = False
    odd = 2*np.flatnonzero(sieve) + 1
    dtype = np.int32 if limit < 2147483647 else np.int64
    P = np.empty(len(odd) + 1, dtype); P[0] = 2; P[1:] = odd
    H = np.zeros(len(P), np.int32); q = P[2:].astype(np.int64)
    H[2:] = np.where(q % 6 == 5, (q + 1)//6, -(q - 1)//6).astype(np.int32)
    return P, H

def geometry(P, H):
    # First global 2^3 completion: indices 0..7, with 2 and 3 already absorbed.
    parent = bytearray(1)
    for z in range(2, CUBE):
        p = int(P[z]); parent *= p; a = abs(int(H[z]))
        for r in (a, (-a) % p):
            parent[r::p] = b'\1' * ((len(parent) - 1 - r)//p + 1)
    parent = (parent[1:] + parent[:1]) * CUBE
    raw = np.frombuffer(parent, np.uint8); packed = np.packbits(raw, bitorder='little')
    if len(packed) & 7: packed = np.pad(packed, (0, 8 - (len(packed) & 7)))
    template, window = packed.view(np.uint64).copy(), len(parent)

    # Every DIM later clocks form one cube.  Fold the largest leading set whose
    # stored periods fit inside one live parent window; no prime identities are fixed.
    arrays=[]; offs=[]; periods=[]; mature=[]; folded=[]; offset=used=0; z=CUBE
    while z + DIM <= len(P):
        triple = [int(x) for x in P[z:z + DIM]]
        period = triple[0] * triple[1] * triple[2]
        if used + period > window: break
        pat = np.zeros(period, np.uint8); last = 0
        for j, p in enumerate(triple):
            h = int(H[z + j]); a = abs(h)
            pat[a::p] = 1; pat[(-a) % p::p] = 1
            square = 6*a*a - 2*a if h > 0 else 6*a*a + 2*a
            gap = 2*a if h > 0 else 4*a + 1
            last = max(last, square + gap)
        dup = np.concatenate((pat, pat, pat[:64])); pack = np.packbits(dup, bitorder='little')
        if len(pack) & 7: pack = np.pad(pack, (0, 8 - (len(pack) & 7)))
        words = pack.view(np.uint64).copy()
        arrays.append(words); offs.append(offset); periods.append(period); mature.append(last + 1); folded.extend(triple)
        offset += len(words); used += period; z += DIM
    return (template, window, np.asarray(folded, P.dtype), np.concatenate(arrays),
            np.asarray(offs, np.int64), np.asarray(periods, np.int64), np.asarray(mature, np.int64), z)

# ---------- 2. transport ------------------------------------------------------

@njit(cache=True, inline='always')
def bound(a, x, right=False):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi)//2
        if a[mid] < x or (right and a[mid] == x): lo = mid + 1
        else: hi = mid
    return lo

@njit(cache=True, inline='always')
def setbit(bits, i):
    bits[i >> 6] |= np.uint64(1) << np.uint64(i & 63)

@njit(cache=True, inline='always')
def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((x * np.uint64(0x0101010101010101)) >> np.uint64(56))

@njit(cache=True, inline='always')
def icbrt(n):
    x = int(n ** (1.0/3.0))
    while (x + 1)**3 <= n: x += 1
    while x**3 > n: x -= 1
    return x

@njit(cache=True, inline='always')
def periodic_word(flat, off, start):
    j = off + (start >> 6); s = start & 63
    if s == 0: return flat[j]
    return (flat[j] >> np.uint64(s)) | (flat[j + 1] << np.uint64(64 - s))

@njit(cache=True, parallel=True)
def close_field(bits, P, H, lower, window, folded, flat, offs, periods, mature, fold_end):
    words = (window + 63)//64; chunk = (words + CUBE - 1)//CUBE
    for k in prange(CUBE):
        g = k ^ (k >> 1); w0 = g * chunk
        if w0 >= words: continue
        w1 = min(words, w0 + chunk); a = w0 << 6; b = min(window, w1 << 6)
        local_lo = lower + a; local_hi = lower + b
        low_n = 6*local_lo - 1; high_n = 6*(local_hi - 1) + 1
        cube_root = icbrt(high_n); root = int(np.sqrt(high_n))
        while (root + 1)*(root + 1) <= high_n: root += 1
        while root*root > high_n: root -= 1

        first = lower + (w0 << 6); nt = len(periods); all_mature = True
        for t in range(nt):
            if local_lo < mature[t]: all_mature = False; break
        if all_mature:
            pos = np.empty(nt, np.int64)
            for t in range(nt): pos[t] = first % periods[t]
            for w in range(w0, w1):
                mask = np.uint64(0)
                for t in range(nt):
                    mask |= periodic_word(flat, offs[t], pos[t]); pos[t] += 64
                    if pos[t] >= periods[t]: pos[t] -= periods[t]
                bits[w] |= mask
        else:
            for t in range(nt):
                if local_lo >= mature[t]:
                    period = periods[t]; pos = first % period; off = offs[t]
                    for w in range(w0, w1):
                        bits[w] |= periodic_word(flat, off, pos); pos += 64
                        if pos >= period: pos -= period
                else:
                    z0 = bound(P, folded[DIM*t])
                    for z in range(z0, z0 + DIM):
                        q = int(P[z]); h = int(H[z]); aa = h if h > 0 else -h
                        square = 6*aa*aa - 2*aa if h > 0 else 6*aa*aa + 2*aa; gap = 2*aa if h > 0 else 4*aa + 1
                        for hit0 in (square, square + gap):
                            hit = hit0
                            if hit < local_lo: hit += ((local_lo - hit + q - 1)//q)*q
                            for n in range(hit, local_hi, q): setbit(bits, n - lower)

        for z in range(fold_end, bound(P, cube_root, True)):
            q = int(P[z]); h = int(H[z]); aa = h if h > 0 else -h
            square = 6*aa*aa - 2*aa if h > 0 else 6*aa*aa + 2*aa; gap = 2*aa if h > 0 else 4*aa + 1
            for hit0 in (square, square + gap):
                hit = hit0
                if hit < local_lo: hit += ((local_lo - hit + q - 1)//q)*q
                for n in range(hit, local_hi, q): setbit(bits, n - lower)

        qs = bound(P, cube_root + 1); qe = bound(P, root, True)
        if qs < qe:
            q = int(P[qs]); rmin = max(q, (low_n + q - 1)//q); rmax = high_n//q
            lo = bound(P, rmin); hi = bound(P, rmax, True)
            for qi in range(qs, qe):
                q = int(P[qi]); rmax = high_n//q
                while hi > 0 and int(P[hi - 1]) > rmax: hi -= 1
                if q*q < low_n:
                    rmin = (low_n + q - 1)//q
                    while lo > 0 and int(P[lo - 1]) >= rmin: lo -= 1
                else: lo = qi
                if lo < qi: lo = qi
                hq = int(H[qi]); aq = hq if hq > 0 else -hq; sq = 1 if hq > 0 else -1
                for ri in range(lo, hi):
                    hr = int(H[ri]); ar = hr if hr > 0 else -hr; sr = 1 if hr > 0 else -1
                    i = 6*aq*ar - aq*sr - ar*sq - lower
                    if a <= i < b: setbit(bits, i)

# ---------- 3. levels ---------------------------------------------------------

@njit(cache=True)
def resolve(bits, window, seen, targets, ti):
    out = np.empty(len(targets) - ti, np.int64); outn = 0
    for w in range((window + 63)//64):
        base = w << 6; valid = min(64, window - base)
        mask = np.uint64(MASK64) if valid == 64 else (np.uint64(1) << np.uint64(valid)) - np.uint64(1)
        openmask = (~bits[w]) & mask; available = valid - popcount64(bits[w] & mask); after = seen + available
        while ti < len(targets) and targets[ti] <= after:
            rank = int(targets[ti] - seen); m = openmask
            for _ in range(rank - 1): m &= m - np.uint64(1)
            bit = 0
            while (m & np.uint64(1)) == 0: m >>= np.uint64(1); bit += 1
            out[outn] = base + bit; outn += 1; ti += 1
        seen = after
    return seen, ti, out[:outn]

def edges(limit):
    P, H = prime_state(max(256, limit))
    rows = P[(P >= 5) & (P <= limit)]; targets = np.empty(len(rows), np.int64)
    if len(rows):
        targets[0] = int(rows[0])
        for i in range(1, len(rows)): targets[i] = targets[i - 1] + int(rows[i]) - 1
    template, window, folded, flat, offs, periods, mature, fold_end = geometry(P, H)
    lower = 1; opening = 6; seen = rowi = 0
    while rowi < len(rows):
        upper = lower + window; high = 6*(upper - 1) + 1; cr = int(high ** (1/3))
        while (cr + 1)**3 <= high: cr += 1
        while cr**3 > high: cr -= 1
        need = high//(cr + 1) + 2
        if P[-1] < need:
            P, H = prime_state(max(need, 2*int(P[-1])))
        bits = template.copy()
        if lower == 1: bits[0] &= np.uint64(~7 & MASK64)
        close_field(bits, P, H, lower, window, folded, flat, offs, periods, mature, fold_end)
        old = rowi; seen, rowi, pos = resolve(bits, window, seen, targets, rowi)
        for j, x in enumerate(pos):
            close = 6*(lower + int(x)); yield int(rows[old + j]), opening, close; opening = close
        lower = upper

if __name__ == '__main__':
    import sys
    argv = sys.argv[1:]; quiet = '--noprint' in argv
    if '--threads' in argv:
        i = argv.index('--threads'); set_num_threads(int(argv[i + 1])); del argv[i:i + 2]
    argv = [x for x in argv if x != '--noprint']
    limit = int(argv[0]) if argv else int(input('Prime ceiling: '))
    for p, opening, close in edges(limit):
        if not quiet: print(f'**{p}** ({opening}, {close})')
