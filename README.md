# gmlp
A gated MLP implementation

Consider a gated feed-forward network operating on batched inputs of shape (batch_size × input_dim). The architecture consists of an input projection, a spatial gating unit (SGU), and an output projection:

$$
\begin{align*}
Z &= \text{GELU}(XW_{\text{in}}) \\
G &= \sigma(ZW_{\text{gate}}) \\
P &= ZW_{\text{proj}} \\
A &= G \odot P \\
Y &= AW_{\text{out}}
\end{align*}
$$

The spatial gating unit uses a sigmoid function $\sigma$ to create gates that modulate the projection pathway, enabling the network to selectively route information. During backpropagation, the gradients flow through both the gate and projection pathways:

$$
\begin{align*}
\frac{\partial L}{\partial Y} &= Y - Y_{\text{true}} \\
\frac{\partial L}{\partial W_{\text{out}}} &= A^\top(\frac{\partial L}{\partial Y}) \\
\frac{\partial L}{\partial A} &= (\frac{\partial L}{\partial Y})(W_{\text{out}})^\top \\
\frac{\partial L}{\partial G} &= \frac{\partial L}{\partial A} \odot P \\
\frac{\partial L}{\partial P} &= \frac{\partial L}{\partial A} \odot G \\
\frac{\partial L}{\partial \sigma(ZW_{\text{gate}})} &= \frac{\partial L}{\partial G} \odot \sigma(ZW_{\text{gate}})(1-\sigma(ZW_{\text{gate}})) \\
\frac{\partial L}{\partial ZW_{\text{gate}}} &= \frac{\partial L}{\partial \sigma(ZW_{\text{gate}})} \\
\frac{\partial L}{\partial W_{\text{gate}}} &= Z^\top(\frac{\partial L}{\partial ZW_{\text{gate}}}) \\
\frac{\partial L}{\partial W_{\text{proj}}} &= Z^\top(\frac{\partial L}{\partial P}) \\
\frac{\partial L}{\partial Z} &= (\frac{\partial L}{\partial ZW_{\text{gate}}})(W_{\text{gate}})^\top + (\frac{\partial L}{\partial P})(W_{\text{proj}})^\top \\
\frac{\partial L}{\partial \text{GELU}(XW_{\text{in}})} &= \frac{\partial L}{\partial Z} \odot \text{GELU}'(XW_{\text{in}}) \\
\frac{\partial L}{\partial W_{\text{in}}} &= X^\top(\frac{\partial L}{\partial \text{GELU}(XW_{\text{in}})})
\end{align*}
$$

The AdamW optimizer maintains exponential moving averages of gradients and their squares through $\beta_1$ and $\beta_2$, while simultaneously applying L2 regularization through weight decay $\lambda$. The learning rate is denoted by $\eta$, $t$ is the current training iteration, and $\epsilon$ is a small constant for numerical stability. For each weight matrix $W$, the update rule is:

$$
\begin{align*}
m &= \beta_1m + (1-\beta_1)(\frac{\partial L}{\partial W}) \\
v &= \beta_2v + (1-\beta_2)(\frac{\partial L}{\partial W})^2 \\
W &= (1-\lambda\eta)W - \eta\cdot\frac{m}{1-\beta_1^t}/\sqrt{\frac{v}{1-\beta_2^t} + \epsilon}
\end{align*}
$$

The implementation leverages BLAS for matrix operations, enabling efficient computation on modern hardware.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
make run
```
