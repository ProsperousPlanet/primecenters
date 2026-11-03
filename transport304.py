from heapq import heappop, heappush
from itertools import islice

def centers():
    yield 2+2
    unit = 2+2+2
    radius = unit+unit
    square = radius+radius+radius
    c, mid, rad, jump = unit, square, radius, square+square+square
    left, right, events = square-radius+unit, square+radius-unit, []
    while True:
        while mid - rad <= c:
            for start, step in ((mid-rad, left), (mid, left), (mid, right), (mid+rad, right)): heappush(events, (start, step))
            mid, rad, jump = mid+jump, rad+radius, jump+square+square
            left, right = left+square, right+square
        blocked = False
        while events and events[0][0] <= c:
            start, step = heappop(events)
            blocked |= start == c
            heappush(events, (start+step, step))
        if not blocked: yield c
        c += unit

if __name__ == "__main__":
    stream = centers()
    try:
        while True: print(*islice(stream, 25), sep=", ", flush=True)
    except KeyboardInterrupt: pass
