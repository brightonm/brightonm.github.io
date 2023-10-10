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

This post presents an annotated version of the Differential Machine Lenrning paper (<a href="https://arxiv.org/abs/2005.02347" style="text-decoration: underline; color: #111">Savine et al., 2020</a>) with PyTorch implementation. 

* TOC
{:toc}


## Supervised Learning

Let $X \in \mathbb{R}^{n \times d}$ be a dataset of inputs with $n$ samples and $d$ features, and let $y \in \mathbb{R}^{n}$ represent the corresponding labels. In the context of pricing a financial derivative, $X$ contains information about the derivative's payoff, such as the strike price, and the model used to compute the price, along with market data like spot prices and interest rates. $y$ represents the corresponding prices. Depending on the pricing model and payoff structure, these prices can be generated either analytically or through computationally expensive numerical methods, such as Monte Carlo simulations. The pricing function, denoted as $f$, maps $X$ to $y$ as follows:

$$
\begin{align*}
  f: \mathbb{R}^{n \times d} &\to \mathbb{R}^{n}\\
  X &\mapsto y
\end{align*}
$$

The idea is to approximate the pricing function $f$ by employing a neural network denoted as $f_{\theta}$, where $\theta$ represents its weights and biases. Inference with a neural network is fast and amenable to batching. Therefore, in situations where traditional pricing methods prove to be slow, the neural network approximation can serve as a promising alternative.

To closely replicate the pricer $f$, the neural network must be trained on $(X=(x_1, \dots, x_n), y)$. his is accomplished by minimizing the Mean Squared Error (MSE) as the loss function:

$$
\theta^* = \min_{\theta} J(\theta) = \min_{\theta} \frac{1}{n} \sum_{i=0}^{n}(y_i - f_{\theta}(x_i))^2
$$

Various numerical algorithms based on mini-batch stochastic gradient descent, such as Adam (<a href="https://arxiv.org/abs/1412.6980" style="text-decoration: underline; color: #111">Kingma et al., 2014</a>), can be employed to solve this optimization problem. These algorithms are typically implemented in deep learning libraries like PyTorch, JAX or TensorFlow, that leverage Automatic Adjoint Differentiation (AAD) to efficiently compute the quantity $\frac{\partial J(\theta)}{\partial \theta}$ (<a href=" https://openreview.net/pdf?id=BJJsrmfCZ" style="text-decoration: underline; color: #111">Paszke et al., 2017</a>).

### PyTorch implementation

*In this blog post, I emphasize key code sections; the full code and documentation are in the notebook* <a href="https://github.com/brightonm/notebooks/blob/main/Differential%20Deep%20Learning%20in%20Pytorch.ipynb" style="text-decoration: none; color: black;"><i class="fa fa-book fa" style="color: darkorange; font-size: 18px;"></i>
</a>.

{% highlight python %}
import torch
import torch.nn as nn
from torch.optimizer import Adam

# for reproducibility
torch.manual_seed(7)

# define neural network
f_theta = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
          )

# define the loss function
loss_func = nn.MSE()

# define the optimizer
optimizer = Adam(f_theta.parameters(), lr=1e-3)

# generate random inputs and outputs simulating one batch
n_samples = 256
d_features = 5
X = torch.rand(n_samples, d_features)
y = torch.rand(n_samples, 1)
...

**`loss = loss_func(f_theta(X), outputs)`**

# compute the derivatives with respect to weights and biases of the f_theta.
# they are store in the .grad attribute of weights tensors
loss.backward()

# perform on optimization step in the Adam algorithm
optimizer.step()
{% endhighlight %}

## Supervised Learning with differentials

### Training with Derivatives

In the field of quantitative finance, assessing price sensitivities to market data, commonly referred to as "the Greeks", is crucial for effective hedging and robust risk management. Differential machine learning leverages these sensitivities by incorporating them into the training process of supervised learning techniques. This approach involves working with augmented datasets that include differentials of labels with respect to inputs, denoted as 

$$
Z = \frac{\partial y}{\partial X} \in \mathbb{R}^{n \times d}\\
$$

Each element $Z_{ij}$ in the matrix corresponds to the partial derivative of the $i$-th element of the vector $y$ with respect to the $j$-th element of the matrix $X$.
So, if you want to calculate a specific element of $Z$, you would compute the partial derivative of the $i$-th element of $y$ with respect to the $j$-th feature of $X$. This can be expressed as:

$$
Z_{ij} = \frac{\partial y_i}{\partial X_{ij}}
$$

 The derivation of these differentials, which depends on the specific model and payoff structure, can be accomplished through various methods, including analytical calculations, numerical techniques such as Monte Carlo simulations, or the application of Automatic Adjoint Differentiation (AAD). (<a href="https://books.google.fr/books?hl=en&lr=&id=eZZxDwAAQBAJ&oi=fnd&pg=PR11&ots=VT7YWs35Du&sig=L9sgoh4lEZJYwghXWFbuIcauG4w&redir_esc=y#v=onepage&q&f=false" style="text-decoration: underline; color: #111">Savine, 2018</a>).







### PyTorch implementation

*In this blog post, I emphasize key code sections; the full code and documentation are in the notebook* <a href="https://github.com/brightonm/notebooks/blob/main/Differential%20Deep%20Learning%20in%20Pytorch.ipynb" style="text-decoration: none; color: black;"><i class="fa fa-book fa" style="color: darkorange; font-size: 18px;"></i>
</a>.

## Application on Black-Scholes Example

Unlike the article, I'm going to take an offline learning approach here. I'm going to apply differential machine learning by learning on ground truth prices and not on noisy sample payoffs as in the article. Differential deep learning can also help with this approach. The data generation phase and training are expensive, but the neural network is trained only once on a large predefined domain of market data.

Unlike the article, we propose a simple example that does not use pathwise derivatives, which is the approximation of the price of a European call under Black-Scholes diffusion.

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
