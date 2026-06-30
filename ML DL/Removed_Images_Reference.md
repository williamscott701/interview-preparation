# Removed Images Reference

All images removed from the notes because their content was text, tables, or formulas written accurately in Markdown/LaTeX. Click any image name to open it.

---

## Mathematical Foundations

| Image | Content | Placed In Section |
|:---|:---|:---|
| [image46.gif](images/image46.gif) | Characteristic polynomial derivation: $\|A-\lambda I\|=0 \Rightarrow \lambda^2+3\lambda+2=0 \Rightarrow \lambda_1=-1,\lambda_2=-2$ | §1.8 Eigenvalues & Eigenvectors |
| [image17.gif](images/image17.gif) | Eigenvector for $\lambda_1=-1$: $v_{1,1}+v_{1,2}=0 \Rightarrow v_{1,1}=-v_{1,2}$, giving $\mathbf{v}_1=k_1[+1,-1]^T$ | §1.8 Eigenvalues & Eigenvectors |
| [image3.gif](images/image3.gif) | Eigenvector for $\lambda_2=-2$: solve $(A-\lambda_2 I)\mathbf{v}_2=0$, giving $\mathbf{v}_2=k_2[+1,-2]^T$ | §1.8 Eigenvalues & Eigenvectors |
| [image78.png](images/image78.png) | Entropy formula: $H(S) = -\sum_i p_i \log_2 p_i$ | §1.7 Entropy & Information Gain |
| [image57.png](images/image57.png) | Covariance formula: $\text{Cov}(X,Y) = \frac{\sum(X_i-\bar{X})(Y_i-\bar{Y})}{N}$ | §1.6 Covariance vs Correlation |
| [image61.png](images/image61.png) | Correlation formula: $\text{Corr}(X,Y) = \frac{\text{Cov}(X,Y)}{\sigma_X \cdot \sigma_Y}$ | §1.6 Covariance vs Correlation |

---

## Statistical Inference

| Image | Content | Placed In Section |
|:---|:---|:---|
| [image39.jpg](images/image39.jpg) | MAP→MLE derivation: 3-step proof showing that with uniform prior $P(\theta)=\text{const}$, MAP reduces to MLE | §2.6 MAP vs MLE |

---

## Core ML Concepts

| Image | Content | Placed In Section |
|:---|:---|:---|
| [image67.png](images/image67.png) | ERM formula text: True Risk $R(h)$, Empirical Risk $\hat{R}(h)$, ERM objective $h^*=\arg\min \hat{R}(h)$ | §3.1 Empirical Risk Minimization |
| [image14.png](images/image14.png) | Accuracy, Precision, Recall formulas in terms of TP/TN/FP/FN | §3.3 Evaluation Metrics |
| [image47.png](images/image47.png) | F1 harmonic mean in both forms: $\frac{2}{1/P + 1/R} = \frac{2PR}{P+R}$ | §3.3 Evaluation Metrics |
| [image23.jpg](images/image23.jpg) | F-Beta score formula: $F_\beta = \frac{(1+\beta^2)}{\beta^2/\text{Recall} + 1/\text{Precision}}$ | §3.3 Evaluation Metrics |
| [image32.png](images/image32.png) | Micro-Precision, Micro-Recall, Macro-Precision, Macro-Recall formulas (multi-class) | §3.5 Micro vs Macro Averaging |

---

## Classical ML Models

| Image | Content | Placed In Section |
|:---|:---|:---|
| [image75.png](images/image75.png) | Bayes theorem with all 4 terms labeled: Posterior, Likelihood, Class Prior, Predictor Prior (evidence) | §5.2 Naive Bayes |
| [image76.png](images/image76.png) | Naive Bayes posterior: $P(c\|X) = P(x_1\|c) \cdot P(x_2\|c) \cdots P(x_n\|c) \cdot P(c)$ | §5.2 Naive Bayes |
| [image82.jpg](images/image82.jpg) | RBF kernel formula: $k(x,y) = \exp(-\|x-y\|^2 / 2\sigma^2)$ | §5.6 SVM — Kernels table |
| [image74.png](images/image74.png) | SVM objective with $\text{cost}_0$/$\text{cost}_1$ terms, and hypothesis $h_\theta(x)=1$ if $\theta^Tx \geq 0$ | §5.6 Support Vector Machines |

---

## Ensemble Methods

| Image | Content | Placed In Section |
|:---|:---|:---|
| [image19.png](images/image19.png) | Bagging vs Boosting comparison table including Similarities section and Weighting row | §6.2 Bagging vs Boosting |

---

## Regularization & Loss Functions

| Image | Content | Placed In Section |
|:---|:---|:---|
| [image45.png](images/image45.png) | L1 vs L2 comparison: 8 properties covering sparsity, prior, differentiability, solutions, robustness, complexity, and efficiency | §7.1 L1 vs L2 Regularization |
| [image41.jpg](images/image41.jpg) | All 10 loss functions: 6 regression (MBE/MAE/MSE/RMSE/Huber/Log-Cosh) + 4 classification (BCE/Hinge/Cross-Entropy/KL) with formulas | §7.2 Loss Functions |
| [image85.png](images/image85.png) | Softmax→Cross-Entropy flow: softmax converts scores to probabilities $f(s)_i = e^{s_i}/\sum e^{s_j}$, then CE $= -\sum t_i\log f(s)_i$ | §7.2 Loss Functions |
| [image33.png](images/image33.png) | KL Divergence formula: $D_{KL}(p\|q) = \sum_x p(x)\ln\frac{p(x)}{q(x)}$ | §12.8 KL Divergence |
| [image12.png](images/image12.png) | Triplet Loss: $\mathcal{L}(A,P,N) = \max(\|f(A)-f(P)\|_2^2 - \|f(A)-f(N)\|_2^2 + \alpha, 0)$ | §7.2 Loss Functions |

---

## Optimization

| Image | Content | Placed In Section |
|:---|:---|:---|
| [image6.png](images/image6.png) | GD cost function $J(\theta_0,\theta_1) = \frac{1}{2m}\sum(h_\theta-y)^2$ and per-parameter update rules with partial derivatives | §8.1 Gradient Descent Variants |
| [image72.png](images/image72.png) | Gradient Descent variants comparison table (Batch/SGD/Mini-Batch columns with descriptions) | §8.1 Gradient Descent Variants |

---

## Neural Network Fundamentals

| Image | Content | Placed In Section |
|:---|:---|:---|
| [image87.png](images/image87.png) | Batch Normalization formula: $h_\text{norm} = (h - \mu) / (\sigma + \epsilon)$ and network placement order | §9.8 Batch Normalization |

---

## Dimensionality Reduction

| Image | Content | Placed In Section |
|:---|:---|:---|
| [image59.png](images/image59.png) | SVD derivation: $A^TA = V\Sigma^T\Sigma V^T$, let $D=\Sigma^T\Sigma$, boxed result $A^TA = VDV^T$ | §10.1 PCA vs SVD |

---

## NLP & Transformers

| Image | Content | Placed In Section |
|:---|:---|:---|
| [image15.png](images/image15.png) | BERT vs GPT vs BART comparison table (Architecture / Direction / Pre-training / Fine-tuning / Best for / Original org) | §12.7 BERT vs GPT vs BART |

---

## Mathematical Foundations (continued)

| Image | Content | Placed In Section |
|:---|:---|:---|
| [image53.png](images/image53.png) | Standard Normal PDF formula: $f(x) = \frac{1}{\sqrt{2\pi}} \exp(-x^2/2)$ | §1.5 Gaussian Distribution |
| [image71.png](images/image71.png) | Two-column layout: Entropy definition + $H(S)=-\sum p_i\log_2 p_i$ with legend (S, N, $p_i$); IG definition + $Gain(A,S)=H(S)-\sum_v\frac{\|S_v\|}{\|S\|}H(S_v)$ with legend ($H(S),\|S_v\|,\|S\|,v,S_v,H(A,S)$) | §1.7 Entropy & Information Gain |
| [image10.png](images/image10.png) | Entropy example: 14 instances (9 yes, 5 no) → $H(S)=-\frac{9}{14}\log_2\frac{9}{14}-\frac{5}{14}\log_2\frac{5}{14}=0.940$ | §1.7 Entropy & Information Gain |

---

## Classical ML Models (continued)

| Image | Content | Placed In Section |
|:---|:---|:---|
| [image13.png](images/image13.png) | SVM Kernels table: Linear $k=x^Ty$, Polynomial $k=(x^Ty)^p$, RBF $k=\exp(-\|x-y\|^2/2\sigma^2)$, Sigmoid $k=\tanh(\alpha x^Ty+c)$ | §5.6 Support Vector Machines |

---

## Optimization (continued)

| Image | Content | Placed In Section |
|:---|:---|:---|
| [image27.png](images/image27.png) | Batch GD vs SGD comparison table (12 rows): whole-dataset vs single-sample gradient; slow/expensive vs fast/cheap; deterministic vs stochastic; optimal vs good solution; no shuffle vs shuffle needed; can't escape vs can escape local minima; fixed vs adjustable LR; converges to global/local min; overfitting vs regularization effect | §8.1 Gradient Descent Variants |

---

## Wrong Section (fixed this session)

| Image | Actual Content | Was Placed In | Action Taken |
|:---|:---|:---|:---|
| [image24.png](images/image24.png) | Classification metrics text: Error Rate, Accuracy, Sensitivity, Specificity, Precision, F-Score formulas | §4.1 Feature Engineering (mislabeled "Feature Scaling Methods") | Removed — all formulas already present as LaTeX in §3.3 |
| [image25.png](images/image25.png) | Feature Scaling tree diagram: Normalization and Standardization with formulas | §8.1 Gradient Descent Variants (mislabeled "Gradient Descent Visualization") | Moved to §4.1 Feature Engineering (correct section) |
| [image29.png](images/image29.png) | Diagonal BiLSTM Architecture diagram | §1.5 Gaussian Distribution (mislabeled "Normal Distribution PDF") | Moved to §11.3 Diagonal BiLSTM (correct section) |
| [image28.png](images/image28.png) | Multi-Head Attention: Attention Head #0 and #1 showing Q/K/V weight matrices ($W^Q_0, W^K_0, W^V_0$, $W^Q_1, W^K_1, W^V_1$) | §11.3 Diagonal BiLSTM (mislabeled "Diagonal BiLSTM Architecture") | Moved to §12.5 Transformer Architecture (correct section) |
| [image27.png](images/image27.png) | Batch GD vs SGD comparison table (11 rows) | §12.4 Multi-Head Attention (mislabeled "Multi-Head Attention Head Components") | Transcribed into §8.1 Gradient Descent Variants (correct section) |

---

*Total: 35 images — 31 replaced with LaTeX/Markdown • 2 moved to correct section • 2 removed as duplicate*
