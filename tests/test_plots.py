import inspect

import quantstats as qs

#
# import matplotlib
# matplotlib.rcParams.update({
#     'font.family': 'DejaVu Sans',
# })
#
# def test_plot(returns, benchmark):
#     for fname in dir(qs.plots):
#         if not fname.startswith('_'):
#             func = getattr(qs.plots, fname)
#             if callable(func):
#                 print(f"Calling {fname}")
#                 try:
#                     func(returns, benchmark)
#                 except Exception as e:
#                     print(f"Plot {fname} failed with error: {e}")


def test_stats(returns, benchmark):
    for fname in dir(qs.stats):
        if not fname.startswith("_"):
            func = getattr(qs.stats, fname)
            if callable(func):
                print(f"Testing {fname}")
                args = inspect.signature(func).parameters
                arg_names = list(args.keys())
                try:
                    if "benchmark" in arg_names:
                        func(returns, benchmark=benchmark)
                    else:
                        func(returns)
                except Exception as e:
                    print(f"Try {fname} failed: {e}")
