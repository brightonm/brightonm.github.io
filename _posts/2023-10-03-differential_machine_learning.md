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

This post presents an annotated version of the Differential Machine Leanrning paper (<a href="https://arxiv.org/abs/2005.02347" style="text-decoration: underline; color: #111">Savine et al., 2020</a>) with PyTorch implementation. 

* TOC
{:toc}

## Training with Derivatives

### Supervised Learning

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
\theta^* = \min_{\theta} J(\theta) = \min_{\theta} \sum_{i=0}^{n}(y_i - f_{\theta}(x_i))^2
$$

Various numerical algorithms based on mini-batch stochastic gradient descent, such as Adam (quote paper), can be employed to solve this optimization problem. These algorithms are typically implemented in deep learning libraries like PyTorch, JAX or TensorFlow, that leverage Automatic Adjoint Differentiation (AAD) to efficiently compute the quantity $\frac{\partial J(\theta)}{\partial \theta}$ (quote paper torch).

### Adding differentials

Differential machine learning is used on augmented datasets with differentials of labels with respect to inputs that we note $Z = \frac{\partial y}{\partial X}$. In the context of pricing, Z are sensitivies or the Greeks. Depending on the model and payoff, they can be obtained analytically, numerically (Monte-Carlo for instance) or using AAD (quote Savine).  



Unlike the article, I'm going to take an offline learning approach here. I'm going to apply differential machine learning by learning on ground truth prices and not on noisy sample payoffs as in the article. Differential deep learning can also help with this approach. The data generation phase and training are expensive, but the neural network is trained only once on a large predefined domain of market data.

## Application on Black-Scholes Example

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