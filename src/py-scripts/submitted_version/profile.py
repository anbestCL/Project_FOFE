import pstats

p = pstats.Stats("tagger.cprof")
p.strip_dirs().sort_stats('cumulative').print_stats(20)
