
from . import plot

# # ---------------------------------
# def html_tearsheet(backtest, benchmark=None):

#     data = pd.DataFrame(index=backtest.index, data={
#         "backtest": backtest.fillna(0)
#     })
#     if benchmark is not None:
#         data['benchmark'] = benchmark.fillna(0)


#     f = open('tearsheet.html')
#     tpl = f.read()
#     f.close()

#     # figfile = file_stream()
#     # plot_timeseries(data['backtest'], data['benchmark'],
#     #                 'Cumulative Returns vs Benchmark',
#     #                 match_volatility=False, log_scale=False, resample=None,
#     #                 compound=True, lw=2, figsize=(8, 5), ylabel="Cumulative Returns",
#     #                 savefig={'fname': figfile, 'format': 'svg'}, show=False)
#     # tpl = tpl.replace('{{returns}}', figfile.getvalue().decode())

#     # figfile = file_stream()
#     # plot_timeseries(data['backtest'], data['benchmark'],
#     #                 'Cumulative Returns vs Benchmark (Log Scaled)',
#     #                 match_volatility=False, log_scale=True, resample=None,
#     #                 compound=True, lw=2, figsize=(8, 4), ylabel="Cumulative Returns",
#     #                 savefig={'fname': figfile, 'format': 'svg'}, show=False)
#     # tpl = tpl.replace('{{log_returns}}', figfile.getvalue().decode())


#     # figfile = file_stream()
#     # plot_timeseries(data['backtest'], data['benchmark'],
#     #                 title='Cumulative Returns vs Benchmark (Volatility Matched)',
#     #                 match_volatility=True, log_scale=False, resample=None,
#     #                 compound=True, lw=2, figsize=(8, 4), ylabel="Cumulative Returns",
#     #                savefig={'fname': figfile, 'format': 'svg'}, show=False)
#     # tpl = tpl.replace('{{vol_returns}}', figfile.getvalue().decode())


#     # figfile = file_stream()
#     # annual = data[['backtest', 'benchmark']].resample('A').apply(compsum).resample('A').last()
#     # plot_returns_bars(annual['backtest'], annual['benchmark'],
#     #                   hline=annual['backtest'].mean(), hlw=2, hllabel="Average",
#     #                   match_volatility=False, log_scale=False, resample=None,
#     #                   title='EOY Returns vs Benchmark', figsize=(8, 4.37),
#     #                   savefig={'fname': figfile, 'format': 'svg'}, show=False)
#     # tpl = tpl.replace('{{eoy_returns}}', figfile.getvalue().decode())


#     # figfile = file_stream()
#     # plot_returns_histogram(data['backtest'], resample="M",
#     #                        title="Distribution of Monthly Returns", figsize=(8, 4),
#     #                        savefig={'fname': figfile, 'format': 'svg'}, show=False)
#     # tpl = tpl.replace('{{monthly_dist}}', figfile.getvalue().decode())


#     # figfile = file_stream()
#     # plot_timeseries(data['backtest'],
#     #                 title='Daily Returns',
#     #                 match_volatility=False, log_scale=False, resample=None,
#     #                 compound=False, lw=1, figsize=(8, 3), ylabel="Returns",
#     #                 savefig={'fname': figfile, 'format': 'svg'}, show=False)
#     # tpl = tpl.replace('{{daily_returns}}', figfile.getvalue().decode())


#     # figfile = file_stream()
#     # plot_rolling_beta(data['backtest'], data['benchmark'],
#     #                   window1=126, window1_label="6-Months",
#     #                   window2=252, window2_label="12-Months",
#     #                   title="Rolling Beta to Benchmark",
#     #                   hlcolor="red", figsize=(8, 3),
#     #                   savefig={'fname': figfile, 'format': 'svg'}, show=False)
#     # tpl = tpl.replace('{{rolling_beta}}', figfile.getvalue().decode())


#     figfile = file_stream()
#     rolling_vol = data[['backtest', 'benchmark']
#                        ].rolling(126).std() * np.sqrt(252)
#     plot_rolling_stats(rolling_vol['backtest'], rolling_vol['benchmark'],
#                        hline=rolling_vol['backtest'].mean(),
#                        hlw=2, hlcolor="red", hllabel="Average",
#                        title='Rolling Volatility (6-Months)',
#                        figsize=(8, 3), ylabel="Volatility",
#                        savefig={'fname': figfile, 'format': 'svg'}, show=False)
#     tpl = tpl.replace('{{rolling_vol}}', figfile.getvalue().decode())


#     figfile = file_stream()
#     rolling_sharpe = data['backtest'].fillna(0).rolling(126).apply(sharpe, raw=True)
#     plot_rolling_stats(rolling_sharpe, hline=rolling_sharpe.mean(),
#                        stats_label="Sharpe",
#                        hlw=2, hlcolor="red", hllabel="Average",
#                        title='Rolling Sharpe Ratio (6-Months)',
#                        figsize=(8, 3), ylabel="Sharpe",
#                        savefig={'fname': figfile, 'format': 'svg'}, show=False)
#     tpl = tpl.replace('{{rolling_sharpe}}', figfile.getvalue().decode())


#     # figfile = file_stream()
#     # plot_longest_drawdowns(data['backtest'], lw=3, figsize=(8, 4),
#     #                        savefig={'fname': figfile, 'format': 'svg'}, show=False)
#     # tpl = tpl.replace('{{dd_periods}}', figfile.getvalue().decode())


#     # figfile = file_stream()
#     # dd = drawdown(data['backtest'])
#     # plot_timeseries(dd, title='Underwater Plot',
#     #                 hline=dd.mean(), hlw=2, hllabel="Average",
#     #                 returns_label="Drawdown",
#     #                 compound=False, match_volatility=False, log_scale=False, resample=None,
#     #                 fill=True, lw=2, figsize=(8, 4), ylabel="Drawdown",
#     #                 savefig={'fname': figfile, 'format': 'svg'}, show=False)
#     # tpl = tpl.replace('{{dd_plot}}', figfile.getvalue().decode())


#     figfile = file_stream()
#     mh = mrh.plot(data['backtest'], compounded=True, figsize=(figsize[0].5, 5), show=False)[0]
#     mh.savefig(fname=figfile, format='svg')
#     tpl = tpl.replace('{{monthly_heatmap}}', figfile.getvalue().decode())


#     # figfile = file_stream()
#     # plot_returns_distribution(data[['backtest']], figsize=(8, 4),
#     #                           savefig={'fname': figfile, 'format': 'svg'}, show=False)
#     # tpl = tpl.replace('{{returns_dist}}', figfile.getvalue().decode())

#     return tpl.replace("svg height", "svg height-disabled")


def full_tearsheet(returns, benchmark=None, grayscale=False, figsize=(8, 5)):

    plot.returns(returns, benchmark, grayscale=grayscale,
                 figsize=(figsize[0], figsize[0]*.6))

    plot.log_returns(returns, benchmark, grayscale=grayscale,
                     figsize=(figsize[0], figsize[0]*.5))

    if benchmark is not None:
        plot.returns(returns, benchmark, match_volatility=True,
                     grayscale=grayscale,
                     figsize=(figsize[0], figsize[0]*.5))

    plot.yearly_returns(returns, benchmark,
                        grayscale=grayscale,
                        figsize=(figsize[0], figsize[0]*.5))

    plot.histogram(returns, grayscale=grayscale,
                   figsize=(figsize[0], figsize[0]*.5))

    plot.daily_returns(returns, grayscale=grayscale,
                       figsize=(figsize[0], figsize[0]*.3))

    if benchmark is not None:
        plot.rolling_beta(returns, benchmark, grayscale=grayscale,
                          figsize=(
                              figsize[0], figsize[0]*.3))

    plot.rolling_volatility(
        returns, benchmark, grayscale=grayscale,
        figsize=(figsize[0], figsize[0]*.3))

    plot.rolling_sharpe(returns, grayscale=grayscale,
                        figsize=(figsize[0], figsize[0]*.3))

    plot.rolling_sortino(returns, grayscale=grayscale,
                         figsize=(figsize[0], figsize[0]*.3))

    plot.drawdowns_periods(returns, grayscale=grayscale,
                           figsize=(figsize[0], figsize[0]*.5))

    plot.drawdown(returns, grayscale=grayscale,
                  figsize=(figsize[0], figsize[0]*.5))

    plot.monthly_heatmap(returns, grayscale=grayscale,
                         figsize=(figsize[0], figsize[0]*.5))

    plot.distribution(returns, grayscale=grayscale,
                      figsize=(figsize[0], figsize[0]*.5))
