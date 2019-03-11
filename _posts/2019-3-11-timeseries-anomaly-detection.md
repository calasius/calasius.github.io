---
layout: post
title: Detecting anomalies in periodic timeseries 
---

In periodic time series it's posible make an aproximation of them using discrete fourier transform given by the following equation:


$$S_{t} = \sum_{k = 1}^{N-1} A_{k}\sin(\frac{2k\pi t}{N}) + \sum_{k = 1}^{N-1} B_{k}\cos(\frac{2k\pi t}{N})$$


Also it's possible express them using sin functions only:

$$S_{t} = \sum_{k = 1}^{N-1} A_{k}\sin(\frac{2k\pi t}{N} + P_{k})$$


Then we can express this model for example on tensorflow and find the parameters  $$A_{k}$$, $$B_{k}$$ for the first model or $$A_{k}$$, $$P_{k}$$ for the last model.

In my particular problem the tiemeseries has two seasonings daily and weekly and timeseries buckets are 10 minutes long, then $$N$$ has to be the amount of buckets of 10 minutes in one week.

One interesting thing is that if applying box-cox transformation it makes the learnign proccess more quikly.

The box-cox transformation is given by the following equation:

$$X_{\lambda}^{'} = \frac{X^{\lambda} - 1}{\lambda}$$

When $$\lambda = 0$$ then box-cox transformation is like $$\log$$ function.

In my experiments I found that $$\lambda = 0.5$$ is good choise because the transformation maintain the anomalies and reduce the magnitude of the noise.
I think that is the reason the learning proccess is more quickly.

The tricky part to detect anomalies is make the learned function smoother than the original. We can acomplish that not taking into account the lower frquencies in the fourier model. In my case 

## Code to learn the parameters in tensorflow

{% highlight css %}
x_train = tf.constant(x_ts_train,dtype=tf.float32)

y_train = tf.constant(y_ts_train,dtype=tf.float32)

x = tf.constant(x_ts,dtype=tf.float32)

y = tf.constant(y_ts,dtype=tf.float32)

multipliers = tf.constant([(x*2*PI)/PERIOD for x in range(THRESHOLD,PERIOD)], dtype=tf.float32)

amplitudes_cos = tf.Variable(tf.random_uniform([1, PERIOD - THRESHOLD], -10, 10, seed=42))

phases = tf.Variable(tf.random_uniform([1, PERIOD - THRESHOLD], -10, 10, seed=42))

amplitudes_sin = tf.Variable(tf.random_uniform([1, PERIOD - THRESHOLD], -10, 10, seed=42))

bias = tf.Variable(tf.zeros([1]))

sin_part = tf.reduce_sum(tf.sin(tf.transpose(x_train) * multipliers) * amplitudes_sin, 1)

cos_part = tf.reduce_sum(tf.cos(tf.transpose(x_train) * multipliers) * amplitudes_cos, 1)

y_pred = sin_part + cos_part + bias

error = y_pred - y_train

regularizer = ALPHA_REG * tf.nn.l2_loss(amplitudes_cos) + ALPHA_REG * tf.nn.l2_loss(amplitudes_sin)

mse = tf.reduce_mean(tf.square(error) + regularizer, name="mse")

op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(mse)

#Evaluate the model on future dates
sin_part = tf.reduce_sum(tf.sin(tf.transpose(x) * multipliers) * amplitudes_sin, 1)

cos_part = tf.reduce_sum(tf.cos(tf.transpose(x) * multipliers) * amplitudes_cos, 1)

y_ext = sin_part + cos_part + bias
{% endhighlight %}

