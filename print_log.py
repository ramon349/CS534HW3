import pstats
from pstats import SortKey
p = pstats.Stats('log')
p.sort_stats(SortKey.CUMULATIVE).print_stats(20)