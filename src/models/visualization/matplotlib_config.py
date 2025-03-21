import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
matplotlib.rcParams['figure.figsize'] = (8, 6)
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 100
# Set a hard limit on maximum pixel dimensions of plots
matplotlib.rcParams['agg.path.chunksize'] = 10000