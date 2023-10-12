# Differential Deep Learning

<div>
    <a href="https://github.com/brightonm/notebooks/blob/main/Differential%20Deep%20Learning%20in%20Pytorch.ipynb" style="text-decoration: none; color: black;">
        <i class="fa fa-book fa" style="color: darkorange; font-size: 42px;"></i>
    </a>
    <a href="https://www.example1.com" style="text-decoration: none; color: black;">
        <span style="font-size: 38px;">🤗</span>
    </a>  
</div>

<div style="text-align: center; margin-bottom: 20px;">
  <img src="/docs/assets/images/diff_ml_paper.png" alt="Image description">
  <figcaption>Fig. 1. Differential Machine Learning paper by <a href="https://arxiv.org/abs/2005.02347" style="text-decoration: underline; color: #888;">Savine et al., 2020</a></figcaption>
</div>

This post presents an annotated version of the **Differential Machine Learning** paper (<a href="https://arxiv.org/abs/2005.02347" style="text-decoration: underline; color: #111">Savine et al., 2020</a>) with PyTorch implementation.

*I emphasize key code sections; the full code and documentation are in the notebooks* <a href="https://github.com/brightonm/notebooks/blob/main/Differential%20Deep%20Learning%20in%20Pytorch.ipynb" style="text-decoration: none; color: black;"><i class="fa fa-book fa" style="color: darkorange; font-size: 18px;"></i>

* TOC
{:toc}

```python
# imports
import torch
import torch.nn as nn
from torch.optim import Adam

# for reproducibility
torch.manual_seed(7)
```

## Supervised Learning

<!-- TODO : add standardization -->

Let $X \in \mathbb{R}^{n \times d}$ be a dataset of inputs with $n$ samples and $d$ features, and let $y \in \mathbb{R}^{n}$ represent the corresponding labels. In the context of pricing a financial derivative, $X$ contains information about the derivative's payoff, such as the strike price, and the model used to compute the price, along with market data like spot prices and interest rates. $y$ represents the corresponding prices. Depending on the pricing model and payoff structure, these prices can be generated either analytically or through computationally expensive numerical methods, such as Monte Carlo simulations. The pricing function, denoted as $f$, maps $X$ to $y$ as follows:

$$
\begin{align*}
  f: \mathbb{R}^{n \times d} &\to \mathbb{R}^{n}\\
  X &\mapsto y
\end{align*}
$$

```python
# generate random inputs and outputs simulating a single batch
n_samples = 256
d_features = 5
X = torch.rand(n_samples, d_features)
y = torch.rand(n_samples, 1)
```

The idea is to approximate the pricing function $f$ by employing a neural network denoted as $f_{\theta}$, where $\theta$ represents its weights and biases. Inference with a neural network is fast and amenable to batching. Therefore, in situations where traditional pricing methods prove to be slow, the neural network approximation can serve as a promising alternative.

```python
# define neural network
f_theta = nn.Sequential(
            nn.Linear(5, 64),
            nn.Softplus(),
            nn.Linear(64, 1)
          )
```

To closely replicate the pricer $f$, the neural network must be trained on $(X=(x_1, \dots, x_n), y)$. his is accomplished by minimizing the Mean Squared Error (MSE) as the loss function:

$$
\theta^* = \min_{\theta} J(\theta) = \min_{\theta} \frac{1}{n} \sum_{i=0}^{n}(y_i - f_{\theta}(x_i))^2
$$

```python
# define loss function
loss_func = nn.MSE()
```

Various numerical algorithms based on mini-batch stochastic gradient descent, such as Adam (<a href="https://arxiv.org/abs/1412.6980" style="text-decoration: underline; color: #111">Kingma et al., 2014</a>), can be employed to solve this optimization problem. These algorithms are typically implemented in deep learning libraries like PyTorch, JAX or TensorFlow, that leverage Automatic Adjoint Differentiation (AAD) to efficiently compute the quantity $\frac{\partial J(\theta)}{\partial \theta}$ (<a href=" https://openreview.net/pdf?id=BJJsrmfCZ" style="text-decoration: underline; color: #111">Paszke et al., 2017</a>).

```python
# define optimizer
optimizer = Adam(f_theta.parameters(), lr=1e-3)

# forward pass
predictions = f_theta(X)

# penalize errors
loss = loss_func(predictions, outputs)

# compute derivatives with respect to weights and biases of the neural network f_theta
# they are stored in the .grad attribute of weights tensors
loss.backward()

# perform one optimization step on this single batch in the Adam algorithm
optimizer.step()
```

## Differential Machine Learning : Training with Derivatives

*Unlike the article, I am adopting an offline learning approach here. I will be employing differential machine learning by training on ground truth prices instead of noisy sampled payoffs, as seen in the article. Differential deep learning can also be beneficial in this context. While the data generation phase and training can be expensive, the neural network is trained just once on a predefined domain of market data.*


### Differential labels

In the field of quantitative finance, assessing price sensitivities to market data, commonly referred to as "the Greeks", is crucial for effective hedging and robust risk management. Differential machine learning leverages these sensitivities by incorporating them into the training process of supervised learning techniques. This approach involves working with augmented datasets that include differentials of labels with respect to inputs, denoted as

$$
Z = \frac{\partial y}{\partial X} \in \mathbb{R}^{n \times d}\\
$$

Each element $Z_{ij}$ in the matrix corresponds to the partial derivative of the $i$-th sample of the vector $y$ with respect to the $i$ sample and $j$-th feature of the dataset $X$.
This can be expressed as:

$$
Z_{ij} = \frac{\partial y_i}{\partial X_{ij}}
$$

```python
# generate random labels derivatives associated with inputs and outputs
# simulating a single batch
Z = torch.rand(256, 5)
```

 The derivation of these differentials, which depends on the specific model and payoff structure, can be accomplished through various methods, including analytical calculations, numerical techniques such as Monte Carlo simulations, or the application of Automatic Adjoint Differentiation (AAD). (<a href="https://books.google.fr/books?hl=en&lr=&id=eZZxDwAAQBAJ&oi=fnd&pg=PR11&ots=VT7YWs35Du&sig=L9sgoh4lEZJYwghXWFbuIcauG4w&redir_esc=y#v=onepage&q&f=false" style="text-decoration: underline; color: #111">Savine, 2018</a>).

### Differential predictions

<div style="text-align: center; margin-bottom: 20px;">
  <img src="/docs/assets/images/diff_ml_twin_network.PNG" alt="Image description">
  <figcaption>Fig. 2. Twin network. (Image source: <a href="https://arxiv.org/abs/2005.02347" style="text-decoration: underline; color: #888;">Savine et al., 2020</a>)</figcaption>
</div>

They introduce a twin network to illustrate a fundamental concept: similar to how you can calculate derivatives of outputs with respect to weights, you can also compute derivatives of outputs with respect to inputs using the same technique, known as AAD. Under the hood, every deep learning library employs this method to propagate gradients during backpropagation.

In practical terms, it involves tracking the computational process during the forward pass. This means that for each variable, whether it is an input, weight, or bias within the neural network, there is a sequence of straightforward operations that lead from these variables to the final outputs. Every step along this path contains information about the forward operation as well as the corresponding reverse operation needed for gradient backpropagation.

Consequently, you have the ability to determine the derivative of any variable 'b' within one node of the graph concerning any other node in the graph that contains variable 'a,' provided that 'b' is positioned ahead of 'a' in the computational graph. This powerful capability allows you to compute the derivatives of the network's outputs with respect to its inputs.

In PyTorch, to begin dynamically tracking the computation graph, you need to set the `requires_grad` attribute of the variable for which you want to compute derivatives. In this context, the inputs require this attribute to be set to `True`. The weights and biases, on the other hand, have this attribute set to True by default. Subsequently, to perform backpropagation through the graph up to a specific node, you only need a single line of PyTorch code, as demonstrated in the example below using `torch.autograd` (<a href=" https://openreview.net/pdf?id=BJJsrmfCZ" style="text-decoration: underline; color: #111">Paszke et al., 2017</a>).

```python
# flush previous computed gradients with respect to weights
optimizer.zero_grad()
X.requires_grad = True

# forward pass
predictions = f_theta(X)

# create_graph argument creates the graph of calculation of the derivatives
# needs here to be set to True, to backpropagate the loss through this new graph

# grad_outputs argument must be specified when not differentiating a scalar
# must as the same shape of the starting tensor of the backpropagation graph (here predictions)
# contains the weights of the different terms of the sum of the gradients (here all ones)
predictions_differentials = torch.autograd.grad(predictions,
                                                X, 
                                                create_graph=True, 
                                                grad_outputs=torch.ones_like(predictions))
```

### New loss function

Now that you have both the true differential labels and their approximations produced by the neural network, you can penalize the approximation errors using the same metric (MSE) that you used for penalizing errors in values.

Keep in mind that the $Z$ differential dataset is built with Greeks of significantly varying magnitudes. Take, for instance, the case of European Call prices in the Black-Scholes model, where the theta Greek is much smaller compared to the gamma. Think of it as trying to combine "apples and oranges", as the saying goes. To address this, you need to make sure that errors in one derivative are penalized in a similar way to errors in another. You can achieve this by normalizing the differentials, which involves multiplying them by a scaling coefficient denoted as $\lambda_j=\frac{1}{||Z_j||}$.

$$
\begin{align*}
MSE_{value}(\theta) &= \frac{1}{n} \sum_{i=0}^{n}\left(y_i - f_{\theta}(x_i)\right)^2 \\
MSE_{differentials}(\theta) &= \frac{1}{n} \sum_{i=0}^{n}\sum_{j=0}^{m} \lambda_j^2 \left(Z_{ij} - \left(\frac{\partial f_{\theta}(X)}{\partial X}\right)_{ij}\right)^2
\end{align*}
$$

```python
# loss on values
loss_values = loss_func(predictions, outputs)

# compute scaling factor
lambdas = 1.0 / (torch.sqrt((Z * Z).mean(dim=0)))
# loss on differentials
loss_differentials = loss_func(lambdas * predictions_differentials, lambdas * Z)
```

These two losses can be combined in a convex manner by introducing an additional hyperparameter, denoted as $\alpha$, which controls how much you want to penalize the derivatives. By default, set $\alpha$ to $\frac{1}{m+1}$, where $m$ represents the number of features with respect to which derivatives are calculated. This choice considers one error in the price to be as important as an error in one of the Greeks.

$$
J(\theta) = \alpha \times MSE_{value}(\theta) + (1-\alpha) \times MSE_{differentials}(\theta)
$$

```python
# set new hyperparameter
alpha = 0.01
loss = alpha * loss_values + (1 - alpha) * loss_diffentials

# compute the derivatives with respect to weights and biases of the f_theta.
# they are store in the .grad attribute of weights tensors
loss.backward()
# perform on optimization step in the Adam algorithm
optimizer.step()
```

Remark : When training with Monte-Carlo paths using pathwise derivatives as differential labels (as in the article), this additional term can be regarded as a form of regularization, akin to Tikhonov or Lasso regularization. It helps in avoiding the overfitting that can occur when fitting noisy samples.

## Application on Black-Scholes Example

Unlike the article, I propose a simple example that does not use pathwise derivatives, which is the approximation of the price of a European call under Black-Scholes diffusion.

### Supervised Learning as Benchmark

### Supervised Learning with Differential Machine Learning

## Notebook and demo on Hugging Face Spaces

## Further Research

## Citation

Cited as:

<blockquote>
Muffat, Brighton. (Oct 2023). Differential Deep Learning. https://brightonmuffat.com/2023/10/03/differential_machine_learning.html.
</blockquote>

Or

```bibtex
@article{muffat2023differentialdeeplearning,
  title   = "Differential Deep Learning",
  author  = "Muffat, Brighton",
  journal = "brightonmuffat.com",
  year    = "2023",
  month   = "Oct",
  url     = "https://brightonmuffat.com/2023/10/03/differential_machine_learning.html"
}
```

## References

[1] Huge, Brian & Antoine Savine. <a href="https://arxiv.org/abs/2005.02347" style="text-decoration: underline; color: #111">"Differential machine learning: The shape of things to come."</a> Risk 10 2020.

[2] Kingma et al. <a href="https://arxiv.org/abs/1412.6980" style="text-decoration: underline; color: #111"> "Adam: A Method for Stochastic Optimization
"</a> arXiv:1412.6980 (2014).

[3] Paszke et al. <a href="https://openreview.net/pdf?id=BJJsrmfCZ" style="text-decoration: underline; color: #111">"Automatic differentiation in PyTorch"</a>  NIPS 2017 Workshop.

[4] Savine, Antoine. <a href="https://books.google.fr/books?hl=en&lr=&id=eZZxDwAAQBAJ&oi=fnd&pg=PR11&ots=VT7YWs35Du&sig=L9sgoh4lEZJYwghXWFbuIcauG4w&redir_esc=y#v=onepage&q&f=false" style="text-decoration: underline; color: #111">"Modern computational finance: AAD and parallel simulations."</a> John Wiley & Sons, 2018.

<!-- [5] <a href="" style="text-decoration: underline; color: #111">""</a> 
[5] <a href="" style="text-decoration: underline; color: #111">""</a> 
[5] <a href="" style="text-decoration: underline; color: #111">""</a> 
[5] <a href="" style="text-decoration: underline; color: #111">""</a> 
[5] <a href="" style="text-decoration: underline; color: #111">""</a> 
[5] <a href="" style="text-decoration: underline; color: #111">""</a> 
[5] <a href="" style="text-decoration: underline; color: #111">""</a> 
[5] <a href="" style="text-decoration: underline; color: #111">""</a>  -->
