---
layout: post
title: Detecting anomalies in periodic timeseries 
---

In periodic time series it's posible make an aproximation of it using discrete fourier transform given by the following equation:
$$S_{t} = \sum_{k = 1}^{N-1} A_{k}\sin(\frac{2k\pi t}{N}) + \sum_{k = 1}^{N-1} B_{k}\cos(\frac{2k\pi t}{N})$$

