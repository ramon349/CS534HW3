import pstats
from pstats import SortKey
p = pstats.Stats('log')
p.sort_stats(SortKey.TIME).print_stats(20)