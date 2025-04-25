import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import quantstats as qs

    qs.extend_pandas()
    return (qs,)


@app.cell
def _(qs):
    stock = qs.utils.download_returns("META")
    return (stock,)


@app.cell
def _(stock):
    stock.sharpe()
    return


@app.cell
def _(stock):
    stock
    return


@app.cell
def _(stock):
    stock.to_csv("meta.csv")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
