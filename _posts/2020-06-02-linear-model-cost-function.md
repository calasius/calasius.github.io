---
layout: post
title: Linear model cost function and gradient descent
---

In this post I want to talk about linear regression cost function and the relationship between the solution using normal equations and gradient descent when there is more features than samples in our dataset.

I belonged to a group of people thas was studiying the book "Introduction to statistical learning with R" and in the chapter 3 it talk about linear regression and the authors give the following example:

![noise](/images/linear_cost_function/linear_regression.png){:class="img-responsive"}

Then they give us the loss function

![cost_function](/images/linear_cost_function/cost_function.png){:class="img-responsive"}

$$RSS$$ is a funtion of the dataset, different datasets give us different cost functions. In this case is a function of two parameters $$\beta_{1}$$ y $$\beta_{2}$$.

The then plot the two dimensional cost function in this way

![cost_function](/images/linear_cost_function/plot_cost_function.png){:class="img-responsive"}

We can see that the function has only one minimum. Equating the gradient of the cost function to zero and doing some calculation we can arrive at this expression:

![cost_function](/images/linear_cost_function/analytical_solution.png){:class="img-responsive"}





