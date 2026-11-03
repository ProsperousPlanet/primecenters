from argparse import ArgumentParser
from heapq import heappop, heappush
from math import isqrt


def signed_centers(limit):
    """Yield (center, displacement, x2, x3, dx2, dx3, left_hits, right_hits)."""
    yield 4, 0, 0, 0, 0, 0, 0, 0  # Exceptional pair (3, 5).
    c, mid, rad, jump, left, right = 6, 36, 12, 108, 30, 42
    events = []
    x2 = x3 = 0

    while c <= limit:
        while mid - rad <= c:
            # Tag L/R refers to the composite flank, not row orientation.
            for start, step, flank in (
                (mid-rad, left, 'R'),
                (mid, left, 'L'),
                (mid, right, 'L'),
                (mid+rad, right, 'R'),
            ):
                heappush(events, (start, step, flank))
            mid, rad, jump = mid+jump, rad+12, jump+72
            left, right = left+36, right+36

        left_hits = right_hits = 0
        while events and events[0][0] <= c:
            start, step, flank = heappop(events)
            if start == c:
                if flank == 'L':
                    left_hits += 1
                else:
                    right_hits += 1
            heappush(events, (start+step, step, flank))

        # Independently maintained left/right coordinates: duplicates count
        # once per flank. Their transitions are genuine signed movements.
        new_x2 = 2 if left_hits else 0
        new_x3 = -3 if right_hits else 0
        dx2, dx3 = new_x2-x2, new_x3-x3
        assert dx2 in (-2, 0, 2) and dx3 in (-3, 0, 3)
        x2, x3 = x2+dx2, x3+dx3

        yield c, x2+x3, x2, x3, dx2, dx3, left_hits, right_hits
        c += 6


def audit(limit, show):
    rows = list(signed_centers(limit))
    zeros = {c for c, d, *_ in rows if d == 0}
    truth = independent_truth(limit)
    errors = sorted(zeros ^ truth)

    raw_false_zeros = []
    duplicates = []
    state_examples = {}
    for c, d, x2, x3, dx2, dx3, lh, rh in rows:
        state_examples.setdefault((bool(lh), bool(rh)), (c, d))
        if lh+rh > 2:
            duplicates.append((c, lh, rh))
        if c != 4 and 2*lh-3*rh == 0 and lh+rh and d:
            raw_false_zeros.append((c, lh, rh, d))

    print(f'Checked every center through {limit:,}: '
          f'{len(rows):,} candidates, {len(zeros):,} zero returns.')
    print(f'Independent twin-prime oracle mismatches: {len(errors)}')
    print(f'Four signed states (L blocked, R blocked): {state_examples}')
    print(f'Addresses with 3+ raw event hits: {len(duplicates):,}; '
          f'first {duplicates[:show]}')
    print(f'False zeros if raw +2*L -3*R counts were used: '
          f'{len(raw_false_zeros):,}; first {raw_false_zeros[:show]}')
    print('First signed movements:')
    for c, d, x2, x3, dx2, dx3, lh, rh in rows[:show]:
        print(f'  c={c:>4}  movement=({dx2:+d},{dx3:+d}) '
              f'position=({x2:+d},{x3:+d})  D={d:+d}  hits=({lh},{rh})')
    assert not errors, errors[:show]


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=100_000_000)
    parser.add_argument('--show', type=int, default=15)
    args = parser.parse_args()
    audit(args.limit, args.show)


