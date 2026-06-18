# Generative Model Theory: VAE, GAN, Flow, and Diffusion Derivations

**Date:** 2026-06-19
**Category:** Algorithm Theory
**Scope:** Complete mathematical derivations for VAE, GAN, Flow-based, and Diffusion models with practical LLM/RTX 4090 relevance

---

## 1. Variational Autoencoder (VAE)

### 1.1 Latent Variable Model and Intractability

The generative process models data $\mathbf{x}$ via latent variables $\mathbf{z}$:

$$p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z})\,p(\mathbf{z})\,d\mathbf{z}$$

This marginal is **intractable** — the integral over all $\mathbf{z}$ has no closed form, and $p(\mathbf{x}|\mathbf{z})$ is a complex neural network. Posterior inference $p(\mathbf{z}|\mathbf{x}) = p(\mathbf{x}|\mathbf{z})p(\mathbf{z})/p(\mathbf{x})$ is also intractable since $p(\mathbf{x})$ is the denominator.

★★★★★★★★ **Critical insight:** This intractability is the *exact same problem* as computing posterior uncertainty in RLHF reward models — the reward $r(\mathbf{x})$ given human preferences is a latent variable model where marginal preference likelihood is intractable.

### 1.2 Evidence Lower Bound (ELBO) Derivation

Starting from the log-evidence, introduce an approximate posterior $q(\mathbf{z}|\mathbf{x})$:

$$\log p(\mathbf{x}) = \log \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z})\,d\mathbf{z} = \log \int \frac{p(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\,q(\mathbf{z}|\mathbf{x})\,d\mathbf{z}$$

Apply Jensen's inequality (log is concave, so $\log E[\cdot] \geq E[\log \cdot]$):

$$\log p(\mathbf{x}) \geq \int q(\mathbf{z}|\mathbf{x})\,\log\frac{p(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\,d\mathbf{z} = \text{ELBO}$$

Decompose the ELBO:

$$\text{ELBO} = E_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})] - \text{KL}(q(\mathbf{z}|\mathbf{x})\,\|\,p(\mathbf{z}))$$

The gap between log-evidence and ELBO is exactly the KL divergence:

$$\log p(\mathbf{x}) = \text{ELBO} + \text{KL}(q(\mathbf{z}|\mathbf{x})\,\|\,p(\mathbf{z}|\mathbf{x}))$$

Therefore maximizing ELBO simultaneously minimizes $\text{KL}(q\,\|\,p(\mathbf{z}|\mathbf{x}))$, bringing $q$ closer to the true posterior.

### 1.3 Reparameterization Trick

Sampling $\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}),\,\boldsymbol{\sigma}_\phi^2(\mathbf{x})\mathbf{I})$ blocks gradient flow through $\boldsymbol{\mu}, \boldsymbol{\sigma}$. Reparameterize:

$$\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \cdot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

Now $\boldsymbol{\epsilon}$ is the random variable, and $\boldsymbol{\mu}, \boldsymbol{\sigma}$ are deterministic transformations — gradients flow through them.

★★★★★★★★ **This trick is foundational for all stochastic gradient estimation in deep RL.** GRPO's advantage estimation and verl's reward model training use the same principle: separate stochastic noise from deterministic parameters.

### 1.4 KL Divergence for Gaussians — Full Derivation

For $q = \mathcal{N}(\boldsymbol{\mu}_q,\,\boldsymbol{\sigma}_q^2\mathbf{I})$ and $p = \mathcal{N}(0, \mathbf{I})$:

$$\text{KL}(q\,\|\,p) = \frac{1}{2}\sum_{j=1}^{J}\left(1 + \log\sigma_{q,j}^2 - \sigma_{q,j}^2 - \mu_{q,j}^2\right)$$

**Derivation:** For diagonal Gaussians, the general formula is:

$$\text{KL}(q\,\|\,p) = \frac{1}{2}\left[\text{tr}\left(\boldsymbol{\Sigma}_p^{-1}\boldsymbol{\Sigma}_q\right) + (\boldsymbol{\mu}_p - \boldsymbol{\mu}_q)^T\boldsymbol{\Sigma}_p^{-1}(\boldsymbol{\mu}_p - \boldsymbol{\mu}_q) - J + \log\frac{\det\boldsymbol{\Sigma}_p}{\det\boldsymbol{\Sigma}_q}\right]$$

Setting $\boldsymbol{\mu}_p = 0$, $\boldsymbol{\Sigma}_p = \mathbf{I}$, $\boldsymbol{\Sigma}_q = \text{diag}(\sigma_{q,j}^2)$:

$$\text{tr}(\mathbf{I}^{-1}\boldsymbol{\Sigma}_q) = \sum_j \sigma_{q,j}^2,\quad \boldsymbol{\mu}_q^T\mathbf{I}^{-1}\boldsymbol{\mu}_q = \sum_j \mu_{q,j}^2,\quad \log\frac{\det\mathbf{I}}{\det\boldsymbol{\Sigma}_q} = -\sum_j \log\sigma_{q,j}^2$$

Combining: $\text{KL} = \frac{1}{2}\sum_j(\sigma_{q,j}^2 + \mu_{q,j}^2 - 1 - \log\sigma_{q,j}^2)$, rearranged as stated.

### 1.5 Why VAE Produces Blurry Images

The KL term penalizes $q$ for deviating from $p(\mathbf{z}) = \mathcal{N}(0,\mathbf{I})$. This creates an **information bottleneck** — the encoder compresses information through a narrow Gaussian channel. The reconstruction term $E_q[\log p(\mathbf{x}|\mathbf{z})]$ is typically a Gaussian likelihood (MSE), which minimizes per-pixel expected error.

★★★★★★★★ **Mode averaging vs. mode covering:** VAE's objective encourages covering all modes of $p(\mathbf{x})$ (since KL penalizes missing any region where $p$ has mass), but the Gaussian reconstruction loss averages over modes → blurry outputs. GAN does the opposite: mode dropping (covers few modes sharply).

### 1.6 $\beta$-VAE (Higgins et al., 2017)

Modified objective with $\beta > 1$:

$$\text{ELBO}_\beta = E_{q}[\log p(\mathbf{x}|\mathbf{z})] - \beta\,\text{KL}(q(\mathbf{z}|\mathbf{x})\,\|\,p(\mathbf{z}))$$

Higher $\beta$ tightens the information bottleneck, forcing each latent dimension to capture independent generative factors → **disentangled representations**. Tradeoff: reconstruction quality degrades as $\beta$ increases.

### 1.7 VAE → LLM Connection

- **RLHF reward model:** Bradley-Terry preference model $P(y_1 > y_2) = \sigma(r(y_1) - r(y_2))$ — the reward $r$ is a latent variable; uncertainty estimation requires variational inference over reward functions
- **Diffusion models:** DDPM training is a variational inference problem (see Sec. 4) — the ELBO structure reappears
- **verl bypass_mode:** The concept of approximate posterior ($q$ approximating true $p$) mirrors FSDP sharding approximating full-parameter access

---

## 2. Generative Adversarial Network (GAN)

### 2.1 Minimax Objective

$$\min_G\max_D\; V(D,G) = E_{\mathbf{x}\sim p_{\text{data}}}[\log D(\mathbf{x})] + E_{\mathbf{z}\sim p_z}[\log(1 - D(G(\mathbf{z})))]$$

The discriminator $D$ tries to distinguish real from fake; the generator $G$ tries to fool $D$.

### 2.2 Optimal Discriminator Derivation

For fixed $G$, maximize $V$ over $D$. The objective per data point is:

$$p_{\text{data}}(\mathbf{x})\log D(\mathbf{x}) + p_G(\mathbf{x})\log(1 - D(\mathbf{x}))$$

Taking derivative w.r.t. $D(\mathbf{x})$ and setting to zero:

$$\frac{p_{\text{data}}(\mathbf{x})}{D(\mathbf{x})} - \frac{p_G(\mathbf{x})}{1 - D(\mathbf{x})} = 0$$

$$D^*_G(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_G(\mathbf{x})}$$

### 2.3 Generator Optimum — Jensen-Shannon Divergence

Substituting $D^*$ back into $V$:

$$C(G) = E_{p_{\text{data}}}\left[\log\frac{p_{\text{data}}}{p_{\text{data}}+p_G}\right] + E_{p_G}\left[\log\frac{p_G}{p_{\text{data}}+p_G}\right]$$

This equals $-2\,\text{JSD}(p_{\text{data}}\,\|\,p_G) + 2\log 2$ where:

$$\text{JSD}(P\,\|\,Q) = \frac{1}{2}\text{KL}(P\,\|\,M) + \frac{1}{2}\text{KL}(Q\,\|\,M),\quad M = \frac{P+Q}{2}$$

Global minimum at $p_G = p_{\text{data}}$ where $\text{JSD} = 0$ and $D^*(\mathbf{x}) = 1/2$ everywhere.

### 2.4 Training Instability

★★★★★★★★ **Mode collapse:** When $G$ finds a few outputs that fool $D$, it collapses to generating only those — $p_G$ concentrates on limited modes. **Vanishing gradients:** When $D$ is too strong, $\log(1-D(G(\mathbf{z}))) \to 0$ rapidly → gradient for $G$ vanishes. The non-saturating heuristic replaces $G$'s loss with $-\log D(G(\mathbf{z}))$ for better gradients, but this changes the game equilibrium.

### 2.5 WGAN — Wasserstein Distance

Replace JS divergence with Earth-Mover (Wasserstein-1) distance:

$$W(p_{\text{data}}, p_G) = \sup_{\|f\|_L \leq 1}\left\{E_{p_{\text{data}}}[f(\mathbf{x})] - E_{p_G}[f(\mathbf{x})]\right\}$$

★★★★★★★★ **Key advantage:** Wasserstein distance is continuous and differentiable even when distributions have disjoint support (unlike JS which jumps to $\log 2$). The critic $f$ (1-Lipschitz) provides meaningful gradients everywhere. Practical loss: train critic to maximize $E_{\text{real}}[f(\mathbf{x})] - E_{\text{fake}}[f(\mathbf{x})]$, train generator to minimize $-E_{\text{fake}}[f(\mathbf{x})]$.

### 2.6 WGAN-GP (Gulrajani et al., 2017)

Enforce Lipschitz constraint via gradient penalty on interpolated points:

$$\hat{\mathbf{x}} = \epsilon\,\mathbf{x}_{\text{real}} + (1-\epsilon)\mathbf{x}_{\text{fake}},\quad \epsilon \sim U[0,1]$$

$$\text{GP} = \lambda\,E\left[(\|\nabla_{\hat{\mathbf{x}}}f(\hat{\mathbf{x}})\|_2 - 1)^2\right]$$

Weight clipping (original WGAN) is crude; gradient penalty is smooth and preserves critic capacity.

### 2.7 StyleGAN (Karras et al., 2019)

- **Progressive growing:** Start training at low resolution, gradually add layers
- **Style modulation:** Mapping network $f: \mathbf{z} \to \mathbf{w}$; $\mathbf{w}$ modulates each convolution via $y_{s,i} = \text{style}_i / \text{std}(\text{activation}_i)$ (instance normalization + affine transform)
- **Stochastic variation:** Per-pixel noise injected after style modulation → local detail variation without affecting global structure

### 2.8 GAN → LLM Connection

- **RLHF as adversarial training:** The reward model acts as "discriminator" judging LLM outputs; the policy (generator) is optimized against it via PPO/GRPO
- **AI alignment:** Constitutional AI uses a "critique" model adversarially — same minimax structure
- **Mode collapse analogy:** RLHF reward hacking is the LLM equivalent of GAN mode collapse — the policy finds outputs that score high on reward but are narrow/repetitive

---

## 3. Flow-based Models

### 3.1 Exact Likelihood via Change of Variables

Given invertible transformation $\mathbf{z} = f(\mathbf{x})$ with $f: \mathbb{R}^D \to \mathbb{R}^D$ and base density $p(\mathbf{z})$:

$$p(\mathbf{x}) = p(\mathbf{z})\,\left|\det\frac{\partial f}{\partial \mathbf{x}}\right|^{-1},\quad \mathbf{z} = f(\mathbf{x})$$

For a composition of $K$ invertible layers $f = f_K \circ f_{K-1} \circ \cdots \circ f_1$:

$$\log p(\mathbf{x}) = \log p(\mathbf{z}) + \sum_{i=1}^{K}\log\left|\det\frac{\partial f_i}{\partial \mathbf{h}_{i-1}}\right|$$

where $\mathbf{h}_0 = \mathbf{x}$ and $\mathbf{h}_i = f_i(\mathbf{h}_{i-1})$.

★★★★★★★★ **Flow models are the ONLY generative model class that provides exact likelihood computation** — no lower bound approximation. This makes them uniquely suited for tasks requiring calibrated probability estimates (anomaly detection, Bayesian inference).

### 3.2 NICE — Additive Coupling Layers (Dinh et al., 2015)

Split $\mathbf{x} = (\mathbf{x}_1, \mathbf{x}_2)$. Forward:

$$\mathbf{y}_1 = \mathbf{x}_1,\quad \mathbf{y}_2 = \mathbf{x}_2 + m(\mathbf{x}_1)$$

Inverse: $\mathbf{x}_2 = \mathbf{y}_2 - m(\mathbf{y}_1)$, $\mathbf{x}_1 = \mathbf{y}_1$. Jacobian:

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \begin{pmatrix} \mathbf{I} & \mathbf{0} \\ \frac{\partial m}{\partial \mathbf{x}_1} & \mathbf{I} \end{pmatrix},\quad \det = 1$$

★★★★★★★★ **Log-determinant = 0** — completely free! But expressive power is limited since $\mathbf{x}_1$ is copied unchanged. Only $\mathbf{x}_2$ is transformed, and the Jacobian contribution is zero (volume-preserving).

### 3.3 RealNVP — Affine Coupling (Dinh et al., 2017)

$$\mathbf{y}_1 = \mathbf{x}_1,\quad \mathbf{y}_2 = \mathbf{x}_2 \cdot s(\mathbf{x}_1) + t(\mathbf{x}_1)$$

where $s(\cdot), t(\cdot)$ are neural networks. Jacobian is block-triangular:

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \begin{pmatrix} \mathbf{I} & \mathbf{0} \\ \frac{\partial \mathbf{y}_2}{\partial \mathbf{x}_1} & \text{diag}(s(\mathbf{x}_1)) \end{pmatrix}$$

$$\det = \prod_j s_j(\mathbf{x}_1),\quad \log|\det| = \sum_j \log|s_j(\mathbf{x}_1)|$$

★★★★★★★★ **Affine coupling adds non-zero log-determinant** → scale parameters $s$ allow volume changes, dramatically improving expressive power. The price: computing $\log|\det|$ costs only a sum over $|D/2|$ elements, not a full $D\times D$ determinant.

### 3.4 Glow (Kingma & Dhariwal, 2018)

Adds three invertible operations before each coupling layer:
1. **ActNorm:** per-channel scale/shift (like BatchNorm but invertible, data-dependent init)
2. **Invertible 1×1 convolution:** $\mathbf{y} = W\mathbf{x}$ where $W$ is a learned permutation-like matrix; $\log|\det W| = \log|\det(W)|$ (LU decomposition for tractability)
3. **Affine coupling layer** (same as RealNVP)

The 1×1 conv mixes channels across the split, ensuring all dimensions eventually get transformed (fixing NICE/RealNVP's "always copy $\mathbf{x}_1$" limitation).

### 3.5 Continuous Normalizing Flows (CNF / Neural ODE)

Define flow as ODE: $\frac{\partial \mathbf{x}(t)}{\partial t} = f(\mathbf{x}(t), t)$. The instantaneous change of log-density follows:

$$\frac{\partial \log p(\mathbf{x}(t))}{\partial t} = -\text{tr}\left(\frac{\partial f}{\partial \mathbf{x}(t)}\right)$$

★★★★★★★★ **This is the continuous analog of $\log|\det(\partial f_i/\partial \mathbf{h}_{i-1})|$.** Computing $\text{tr}(\partial f/\partial \mathbf{x})$ costs $O(D)$ via Hutchinson's trace estimator instead of $O(D^3)$ for full determinant — crucial for high-dimensional flows.

Integrating from $t=0$ to $t=1$:

$$\log p(\mathbf{x}(1)) = \log p(\mathbf{x}(0)) - \int_0^1 \text{tr}\left(\frac{\partial f}{\partial \mathbf{x}(t)}\right)dt$$

This is solved with black-box ODE solvers (no discretization of layers needed).

### 3.6 Flow → LLM Connection

- **Uncertainty estimation:** Normalizing flows can model complex posterior distributions over LLM outputs — exact likelihood enables calibrated confidence scores
- **Diffusion foundation:** Continuous normalizing flows (Neural ODE) directly generalize to diffusion models — see Sec. 4
- **Probabilistic programming:** Flow-based density estimation appears in Bayesian optimization for hyperparameter search (relevant to GRPO hyperparameters on RTX 4090)

---

## 4. Diffusion Models (Bridge to Modern LLM Relevance)

### 4.1 Forward Process

Gradually add Gaussian noise over $T$ steps:

$$q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t;\,\sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\,\beta_t\mathbf{I})$$

Closed-form marginal at any step using $\alpha_t = 1-\beta_t$, $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$:

$$q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\,\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\,(1-\bar{\alpha}_t)\mathbf{I})$$

### 4.2 Reverse Process and Training

Learn reverse $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1};\,\boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\,\sigma_t^2\mathbf{I})$.

The variational lower bound decomposes as $L = L_T + \sum_{t>1}L_{t-1} + L_0$. Ho et al. (2020) showed simplifying to:

$$L_{\text{simple}} = E_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \boldsymbol{\epsilon}\|^2\right]$$

where $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(0,\mathbf{I})$.

★★★★★★★★ **This is just noise prediction** — the network learns to predict the noise added at each step, not the clean image. The reparameterization trick (Sec. 1.3) appears again: noise is the random variable, network parameters are deterministic.

### 4.3 DDIM — Deterministic Sampling (Song et al., 2021)

DDIM generalizes diffusion to non-Markovian processes with a "jump" parameter $\eta$:

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\,\hat{\mathbf{x}}_0}{\sqrt{1-\bar{\alpha}_t}} + \sigma_t\,\boldsymbol{\epsilon}$$

where $\sigma_t = \eta\sqrt{(1-\bar{\alpha}_{t-1})/(1-\bar{\alpha}_t)}\sqrt{1-\bar{\alpha}_t/\bar{\alpha}_{t-1}}$. Setting $\eta=0$ gives **deterministic** ODE-like sampling → fewer steps needed (10-50 vs. 1000).

### 4.4 Diffusion → LLM Connection

- **Discrete diffusion (D3PM, DDPM for text):** Replace Gaussian noise with uniform transition over token vocabulary — enables diffusion-based text generation
- **Multimodal LLMs:** DALL-E, Stable Diffusion serve as vision components in LLM systems → vLLM/SGLang must serve diffusion models efficiently
- **Score matching in RL:** GRPO advantage estimation can be viewed as learning a "score function" $\nabla_\mathbf{x}\log p(\text{reward}|\mathbf{x})$ — same mathematical structure as diffusion score functions

---

## 5. Comparison Table

| Property | VAE | GAN | Flow | Diffusion |
|---|---|---|---|---|
| **Likelihood** | Lower bound (ELBO) | No explicit likelihood | **Exact** | Lower bound |
| **Sample quality** | Blurry (mode averaging) | Sharp but mode collapse | Good, but limited expressiveness per layer | **Best** (diverse + sharp) |
| **Training stability** | **Stable** (single objective) | Unstable (two-player game) | **Stable** (exact likelihood) | **Stable** (noise prediction) |
| **Invertibility** | Approximate ($q$ approximates $p(\mathbf{z}|\mathbf{x})$) | No ($G$ is not invertible) | **Exact** ($f$ invertible by construction) | Approximate |
| **Mode coverage** | Good (KL covers all modes) | Poor (mode collapse) | Good (exact density) | **Best** (Langevin dynamics explores) |
| **Latent structure** | Explicit $\mathbf{z}$ with semantics | Implicit, no control | $\mathbf{z} = f(\mathbf{x})$ explicit | Implicit in noise trajectory |
| **Math foundation** | Variational inference | Game theory (minimax) | Change of variables | Stochastic processes / SDEs |
| **Sampling speed** | Fast (one decode) | Fast (one forward) | Medium (sequential inverse) | Slow (many denoising steps) |

★★★★★★★★ **Practical takeaway for RTX 4090:** Diffusion models dominate quality but are slow; Flow models are theoretically elegant but underused; VAE is the conceptual foundation for all probabilistic deep learning; GAN's adversarial framework directly maps to RLHF.

---

## 6. RTX 4090 Relevance

### 6.1 Diffusion Model Inference on RTX 4090

Stable Diffusion (SD 1.5/2.1/XL) inference requirements:
- **FP16:** ~2-4 GiB VRAM for SD 1.5, ~6-8 GiB for SDXL → **fits on RTX 4090**
- **FP8 (RTX 4090 Ada FP8 support):** ~1-2 GiB for SD 1.5 via quantization → relates to vLLM #45656 (GPTQ/CT quantization guards)
- **Batch size:** RTX 4090 can run 4-8 concurrent SD generations in FP16 → relevant for multimodal LLM serving

★★★★★★★★ **SD on RTX 4090 is the FP8 quantization testbed** — the same quantization issues (FP8→BF16 prologue fusion on SM89, PyTorch #184119) apply to both LLM inference and diffusion inference. The Inductor SM<90 Fusion Guard (P9) protects both pathways.

### 6.2 Flow Models as Probabilistic Programming Foundation

Normalizing flows provide exact likelihood → enable:
- **Calibrated uncertainty** for LLM outputs (rare in practice but theoretically important)
- **Bayesian optimization** for GRPO hyperparameters (learning rate, KL penalty, clip range)
- **Anomaly detection** in training: if model likelihood drops, detect training instability early

### 6.3 Practical: Stable Diffusion on RTX 4090

| Config | VRAM | Speed (steps/s) | Quality |
|---|---|---|---|
| SD 1.5 FP16 | 2.4 GiB | ~50 DDIM/20s | Good |
| SD 1.5 FP8 | 1.2 GiB | ~60 DDIM/18s | Acceptable |
| SDXL FP16 | 7.2 GiB | ~15 DDIM/30s | Excellent |
| SDXL FP8 | 3.6 GiB | ~18 DDIM/25s | Good |
| SD 3.0 FP16 | ~12 GiB | ~8 DDIM/45s | Best (flow matching!) |

★★★★★★★★ **SD 3.0 uses Rectified Flow (flow matching) — not traditional diffusion!** This is a continuous-time flow model variant, directly connecting Sec. 3 (Flow) to Sec. 4 (Diffusion). The generative model field is converging.

---

## 7. Connection to Framework Issues We Track

| Generative Concept | Framework Issue | Mechanism |
|---|---|---|
| VAE variational inference | **RLHF reward model** | Bradley-Terry model is a latent variable model; reward uncertainty requires variational methods |
| VAE reparameterization trick | **GRPO advantage estimation** | Stochastic gradients through sampling — same principle as $\mathbf{z} = \mu + \sigma\epsilon$ |
| VAE mode averaging | **verl #6782 LoRA rank** | Low-rank LoRA averages over weight modes → rank=64 breaks EOS (too narrow bottleneck) |
| GAN adversarial training | **RLHF reward hacking** | Policy "fools" reward model → mode collapse equivalent → needs KL penalty (like WGAN-GP) |
| GAN discriminator optimum | **verl reward model** | $D^* = p_{\text{real}}/(p_{\text{real}}+p_{\text{fake}})$ → reward model should output probability of human preference |
| GAN mode collapse | **DeepSpeed #8061 NaN** | Adversarial dynamics cause training instability → overlap_comm creates multi-stream conflict (two-player game on GPU) |
| Flow exact likelihood | **LLM confidence estimation** | Normalizing flows enable calibrated probability → rare but important for safety-critical deployments |
| Flow invertibility | **verl #6794 weight sync** | Delta weight sync is conceptually a flow: base→LoRA weight transformation → must be invertible for rollback |
| Flow change of variables | **verl #6512 per-unit summon** | FSDP unit discovery changes the "latent space" of parameter groups → different Jacobian structure |
| Diffusion noise prediction | **GRPO noise in rewards** | Human preference labels are noisy; GRPO learns to predict the "noise" (variance) in group rewards |
| Diffusion discrete variant | **vLLM #45656 quantization** | Discrete diffusion over token vocab → same quantization issues as continuous diffusion over pixels |
| Diffusion multimodal | **SGLang #28588 security** | Image decompression bomb in diffusion serving → nano_nemotron_vl.py disables PIL guard → same vulnerability |
| CNF Neural ODE | **verl HYBRID sleep/wake** | Continuous weight transfer (sleep→wake) is a Neural ODE trajectory → weight updates are the "flow" |
| Diffusion score matching | **verl #6699 detach** | Score function $\nabla\log p$ requires gradient flow → detach blocks gradient → same as breaking score estimation |

★★★★★★★★ **The deepest connection:** RLHF is a *GAN on text* — the reward model is the discriminator, the LLM policy is the generator, and KL penalty is the Lipschitz constraint (analogous to WGAN-GP). Understanding GAN theory is essential for understanding RLHF failure modes (reward hacking = mode collapse).

---

## References

1. Kingma, D.P. & Welling, M. (2014). "Auto-Encoding Variational Bayes." *ICLR*.
2. Higgins, I. et al. (2017). "$\beta$-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework." *ICLR*.
3. Goodfellow, I. et al. (2014). "Generative Adversarial Nets." *NeurIPS*.
4. Arjovsky, M. et al. (2017). "Wasserstein GAN." *ICML*.
5. Gulrajani, I. et al. (2017). "Improved Training of Wasserstein GANs." *NeurIPS*.
6. Karras, T. et al. (2019). "A Style-Based Generator Architecture for GANs." *CVPR*.
7. Dinh, L. et al. (2015). "NICE: Non-linear Independent Components Estimation." *ICLR Workshop*.
8. Dinh, L. et al. (2017). "Density Estimation Using Real NVP." *ICLR*.
9. Kingma, D.P. & Dhariwal, P. (2018). "Glow: Generative Flow with Invertible 1×1 Convolutions." *NeurIPS*.
10. Chen, R.T.Q. et al. (2018). "Neural Ordinary Differential Equations." *NeurIPS*.
11. Ho, J. et al. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*.
12. Song, J. et al. (2021). "Denoising Diffusion Implicit Models." *ICLR*.
13. Lipman, Y. et al. (2023). "Flow Matching for Generative Modeling." *ICLR*.
14. Austin, J. et al. (2021). "Structured Denoising Diffusion Models in Discrete State Spaces." *NeurIPS*.
15. Bradley-Terry, R.A. (1952). "Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons." *Biometrika*.
