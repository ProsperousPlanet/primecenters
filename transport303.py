from heapq import heappop, heappush
from itertools import islice

def centers():
    yield 4
    c, mid, rad, jump, left, right, events = 6, 36, 12, 108, 30, 42, []
    while True:
        while mid - rad <= c:
            for start, step in ((mid-rad, left), (mid, left), (mid, right), (mid+rad, right)): heappush(events, (start, step))
            mid, rad, jump, left, right = mid+jump, rad+12, jump+72, left+36, right+36
        blocked = False
        while events and events[0][0] <= c:
            start, step = heappop(events)
            blocked |= start == c
            heappush(events, (start+step, step))
        if not blocked: yield c
        c += 6

if __name__ == "__main__":
    stream = centers()
    try:
        while True: print(*islice(stream, 25), sep=", ", flush=True)
    except KeyboardInterrupt: pass
