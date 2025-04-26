import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import quantstats as qs

    return (qs,)


@app.cell
def _(qs):
    stock = qs.utils.download_returns("META")
    stock.to_csv("meta.csv")
    return


@app.cell
def _(qs):
    spy = qs.utils.download_returns("SPY")
    spy.to_csv("benchmark.csv")
    return


if __name__ == "__main__":
    app.run()
