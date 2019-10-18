import pstats
from pstats import SortKey
p = pstats.Stats('p2.txt')
p.sort_stats(SortKey.TIME).print_stats(20)