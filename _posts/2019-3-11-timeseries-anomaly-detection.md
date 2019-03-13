---
layout: post
title: Detecting anomalies in periodic timeseries usign gradient descent 
---

In periodic time series it's possible make an aproximation of them using discrete fourier transform given by the following equation:


$$S_{t} = \sum_{k = 1}^{N-1} A_{k}\sin(\frac{2k\pi t}{N}) + \sum_{k = 1}^{N-1} B_{k}\cos(\frac{2k\pi t}{N})$$


Also it's possible express them using sin functions only:

$$S_{t} = \sum_{k = 1}^{N-1} A_{k}\sin(\frac{2k\pi t}{N} + P_{k})$$


Then we can express this model for example on tensorflow and find the parameters  $$A_{k}$$, $$B_{k}$$ for the first model or $$A_{k}$$, $$P_{k}$$ for the last model.

In my particular problem the tiemeseries has two seasonings daily and weekly and timeseries buckets are 10 minutes long, then $$N$$ has to be the amount of buckets of 10 minutes in one week.

One interesting thing is that if applying box-cox transformation it makes the learnign proccess faster. 

The box-cox transformation is given by the following equation:

$$X_{\lambda}^{'} = \frac{X^{\lambda} - 1}{\lambda}$$

When $$\lambda = 0$$ then box-cox transformation is like $$\log$$ function. $$\lambda$$ must take values between 0 and 1.

The box-cox transformation maintains the anomalies and reduce the magnitude of the noise, because reduce the scale of timeseries. When using the transformed timeseries the function to learn in simpler than the original. The question is why the learning proccess is faster if applying box-cox transformation? I think that is similar to standarize the input, the gradient is more direct.


The tricky part to detect anomalies is make the learned function smoother. We can acomplish that not taking into account the high frequencies in the fourier model. Doing in that way the learned function has less noice, then is more easy to detect anomalies in the original timeseries.
Filter frequencies has another benefit for learning, the amount of features decrease, then the learning  is faster.

## Code to learn the parameters in tensorflow

Here is the code to vectorize the fourier model in tensorflow.

{% highlight  css %}
x_train = tf.constant(x_ts_train,dtype=tf.float32)
y_train = tf.constant(y_ts_train,dtype=tf.float32)
x = tf.constant(x_ts,dtype=tf.float32)
y = tf.constant(y_ts,dtype=tf.float32)
multipliers = tf.constant([(x*2*PI)/PERIOD for x in range(MIN_FREQUENCY,PERIOD)], dtype=tf.float32)
amplitudes_cos = tf.Variable(tf.random_uniform([1, PERIOD - MIN_FREQUENCY], -10, 10, seed=42))
phases = tf.Variable(tf.random_uniform([1, PERIOD - MIN_FREQUENCY], -10, 10, seed=42))
amplitudes_sin = tf.Variable(tf.random_uniform([1, PERIOD - MIN_FREQUENCY], -10, 10, seed=42))
bias = tf.Variable(tf.zeros([1]))
sin_part = tf.reduce_sum(tf.sin(tf.transpose(x_train) * multipliers) * amplitudes_sin, 1)
cos_part = tf.reduce_sum(tf.cos(tf.transpose(x_train) * multipliers) * amplitudes_cos, 1)
y_pred = sin_part + cos_part + bias
error = y_pred - y_train
regularizer = ALPHA_REG * tf.nn.l2_loss(amplitudes_cos) + ALPHA_REG * tf.nn.l2_loss(amplitudes_sin)
mse = tf.reduce_mean(tf.square(error) + regularizer, name="mse")
op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(mse)
//Evaluate the model on future dates
sin_part = tf.reduce_sum(tf.sin(tf.transpose(x) * multipliers) * amplitudes_sin, 1)
cos_part = tf.reduce_sum(tf.cos(tf.transpose(x) * multipliers) * amplitudes_cos, 1)
y_ext = sin_part + cos_part + bias
//Learning
n_epochs = 400
init = tf.global_variables_initializer()
sess = tf.Session()
with tf.Session() as sess:
  sess.run(init)
  for epoch in range(n_epochs):
    sess.run(op)
    print("epoch %s, mse = %s " % (epoch, mse.eval()))
  y_p = y_pred.eval()
  y_extended = y_ext.eval()
{% endhighlight %}

## Plot timeseries transformed, learned function and anomalies 

To detect anomalies I calculated the noise (difference between timeseries and learned function), then calculate noise standard deviation and calculate the point where the noice obsolute value is greater that 3 times std. The blue line is the learned function, the red one is the transformed timeseries and the green point are the anomalies.

![Anomalies](/images/learned_function.png){:class="img-responsive"}


## The noise

![noise](/images/noice.png){:class="img-responsive"}

## Plot anomalies in the original tiemseries

Here is the plot of the previous anomalies on the original timeseries

![original timeseries](/images/original-timeseries.png){:class="img-responsive"}


## Forecasting

After learned parameters of fourier model it's possible evaluate on futures points in time.

![Forecasting](/images/forecasting.png){:class="img-responsive"}

All the code is [here](https://github.com/calasius/timeseries)
