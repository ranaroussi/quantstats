#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# QuantStats: Portfolio analytics for quants
# https://github.com/ranaroussi/quantstats
#
# Copyright 2019 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import quantstats.stats as stats


def help():
    print("""# run allocator
a = allocator.run(data, runs=100)

# reveal best index by returns/sharpe/drawdown/volatility
ix = a.reveal(by='returns')

# print stats for index
print(a.stats(ix))

# show weights per asset
print(a.weights(ix))

# get combined returns and plot
a.returns(ix).add(1).cumprod().plot()""")


def create_random_weights(n_assets):
    '''
    returns randomly choosen portfolio weights that sum to one
    '''
    w = np.random.rand(n_assets)
    return w / w.sum()


class Weights(dict):

    def __repr__(self):
        return "<portfolio weights object>"

    def reveal(self, by='return'):
        if by == 'sharpe':
            ix = self['sharpes'].argmax()
        elif by == 'dd' or by == 'drawdown':
            ix = self['drawdowns'].argmax()
        elif by == 'std' or by == 'volatility':
            ix = self['volatility'].argmin()
        else:
            ix = self['returns'].argmax()

        return ix

    def stats(self, ix):
        return {
            'Sharpe': self['sharpes'][ix],
            'Return': self['returns'][ix] * 100,
            'Drawdown': self['drawdowns'][ix] * 100,
            'Volatility': self['volatility'][ix] * 100
        }

    def weights(self, ix):
        assets = list(self['data'].columns)
        weights = {}
        for i, wi in enumerate(self['weights'][ix]):
            weights[assets[i]] = wi

        weights = pd.DataFrame(index=[0], data=weights).T
        weights.columns = ['Weight']
        return(weights)

    def returns(self, ix):
        return (self['data'] * self['weights'][ix]).sum(axis=1)


def run(data, runs=1000):
    weights = []
    sharpes = np.zeros(runs)
    returns = np.zeros(sharpes.shape)
    drawdowns = np.zeros(sharpes.shape)
    volatility = np.zeros(sharpes.shape)

    for i in range(runs):
        w = create_random_weights(len(data.columns))
        r = (data * w).sum(axis=1)

        weights.append(w)
        returns[i] = r.add(1).prod()
        sharpes[i] = stats.sharpe(r)
        drawdowns[i] = stats.max_drawdown(r)
        volatility[i] = stats.volatility(r)

    return Weights({
        'data': data,
        'weights': weights,
        'sharpes': sharpes,
        'returns': returns,
        'drawdowns': drawdowns,
        'volatility': volatility
    })
