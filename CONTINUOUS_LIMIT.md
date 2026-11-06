# Continuous-time limit: the DS geometric Brownian motion (diffusive scaling)

**Status: extension of the thesis, not a thesis result.** Code: `src/dsbidask/continuous.py`. Tests: `tests/test_continuous.py`.
Reproduce: `python scripts/continuous_market_test.py`. Proofs below are complete except where a cited theorem is used (marked **cited**).

## 0. Definition

Take the DS-multiplicative random walk of thesis Sec. 3.1 with $n$ steps of length $\Delta=T/n$ and
$$u_n=e^{\sigma\sqrt\Delta},\quad d_n=1/u_n,\quad R_n=e^{r\Delta},\quad q_n=\frac{R_n-d_n}{u_n-d_n},\quad \varepsilon_n=1-b_u-b_d=\frac{\theta\sqrt\Delta}{\sigma},$$
with $\sigma>0$ the volatility and $\theta\ge 0$ the **drift ambiguity per year**. Two parametrisations of the one-step masses are used:

| kind | $b_u$ | $b_d$ | central drift offset $m_c$ |
|---|---|---|---|
| `thesis` | $q_n$ | $1-q_n-\varepsilon_n$ | $\theta$ |
| `symmetric` | $q_n-\varepsilon_n/2$ | $1-q_n-\varepsilon_n/2$ | $0$ |

The **DS geometric Brownian motion** $(\sigma,\theta)$ is defined as the limit $n\to\infty$ of the bid/ask recursions (19) and (21):
$$Y_k=R_n^{-1}\big[b_uY_u+b_dY_d+\varepsilon_n\min(Y_u,Y_d)\big],\qquad \bar Y_k=R_n^{-1}\big[b_u\bar Y_u+b_d\bar Y_d+\varepsilon_n\max(\bar Y_u,\bar Y_d)\big].$$
It is a limit of *nonlinear expectations of Markov functionals of $S_T$*. Weak convergence of the belief function on path space
(the Feng-Nguyen route mentioned in the thesis, Sec. 6) is **not** shown here.

## 1. Two lemmas

**Lemma 1 (centred form).** With $\pi=b_u+\varepsilon/2$: $\;b_uf_u+b_df_d+\varepsilon\min(f_u,f_d)=\pi f_u+(1-\pi)f_d-\tfrac{\varepsilon}{2}|f_u-f_d|$, and with $\max$ the sign of the last term is $+$.
*Proof.* $\min(a,b)=\frac{a+b}{2}-\frac{|a-b|}{2}$ and $\max(a,b)=\frac{a+b}{2}+\frac{|a-b|}{2}$. $\square$

**Lemma 2 (monotone payoffs).** If $\varphi$ is non-decreasing, every $Y_k$ is non-decreasing in $S$ (induction: a non-negative combination of $Y_{k+1}(Su)$ and $Y_{k+1}(Sd)$), so $\min=Y_d$, $\max=Y_u$ and
$$\text{bid}=e^{-rT}E_{p_L}\varphi(S_T),\quad \text{ask}=e^{-rT}E_{p_U}\varphi(S_T),\qquad p_L=b_u,\;p_U=1-b_d,$$
where $S_T=S_0u^Nd^{n-N}$, $N\sim\mathrm{Bin}(n,p)$. For non-increasing $\varphi$ the two probabilities are swapped. $\square$

## 2. Which scaling gives a limit

**Proposition 1 (Poisson scaling is vacuous).** If $b_u=\lambda\Delta$, $b_d=\mu\Delta$ and $u>1>d$ are fixed (the limit of Sec. 3.1 of the report, applied to prices), then for a call $\text{bid}\to0$ and $\text{ask}\to\infty$.
*Proof.* By Lemma 2, $E_pS_T=S_0\,(d+p(u-d))^n$. Bid: $p\le\lambda'\Delta$ gives $E_pS_T\le S_0d^n e^{\lambda'T(u-d)/d}\to0$ and $(S-K)^+\le S$. Ask: $p=1-\mu\Delta$ gives $E_pS_T\ge S_0u^n e^{-\mu'T(u-d)/u}\to\infty$ and $(S-K)^+\ge S-K$. $\square$
(In the report this limit is used for the $t$-step *transition belief*, not for prices; the proposition says that pricing needs $(u,d)$ rescaled.)

**Proposition 2 (trichotomy).** Let $r\ge0$ and $\varepsilon'_n=\varepsilon_n$ (thesis) or $\varepsilon_n/2$ (symmetric), so $p_U=q_n+\varepsilon'_n$.
(i) If $\varepsilon_n/\sqrt\Delta\to\infty$ (e.g. $\varepsilon_n$ constant), the call ask tends to $\infty$.
(ii) If $\varepsilon_n/\sqrt\Delta\to0$, bid and ask both tend to Black-Scholes.
(iii) If $\varepsilon_n=\theta\sqrt\Delta/\sigma$, the limit is non-degenerate (Theorem 1).
*Proof of (i).* $q_nu+(1-q_n)d=R_n$, so $E_{p_U}S_T=S_0(R_n+x_n)^n$ with $x_n=\varepsilon'_n(u-d)\ge2\sigma\varepsilon'_n\sqrt\Delta$. Since $R_n\ge1$ and $\ln(1+x)\ge x-x^2/2$, $E_{p_U}S_T\ge S_0\exp(nx_n-nx_n^2/2)$. Now $nx_n\ge2\sigma T\varepsilon'_n/\sqrt\Delta\to\infty$ while $nx_n^2\le 4\sigma^2T(1+o(1))$ because $\varepsilon'_n\le1$. Hence $E_{p_U}S_T\to\infty$ and $\text{ask}\ge e^{-rT}(E_{p_U}S_T-K)\to\infty$. (ii) is Theorem 1 with $m=0$ plus the same estimate. $\square$

## 3. Closed forms for monotone payoffs

**Theorem 1.** Let $p_n=q_n+\dfrac{m\sqrt\Delta}{2\sigma}$ and let $\varphi$ be continuous with linear growth. Then
$$e^{-rT}E_{p_n}\varphi(S_T)\;\longrightarrow\;e^{-rT}E\,\varphi\!\big(S_0e^{(r+m-\sigma^2/2)T+\sigma\sqrt T\,Z}\big),\quad Z\sim N(0,1).$$
For a call this equals $\mathrm{BS}_C(S_0e^{mT},K)$.

*Proof.* Write $\ln(S_T/S_0)=\sigma\sqrt\Delta\sum_{i\le n}\xi_i$, $\xi_i=\pm1$, $P(\xi_i=1)=p_n$.
1. *Mean.* $2q_n-1=\dfrac{e^{r\Delta}-\cosh\sigma\sqrt\Delta}{\sinh\sigma\sqrt\Delta}=\dfrac{r-\sigma^2/2}{\sigma}\sqrt\Delta+O(\Delta^{3/2})$ and $2(p_n-q_n)=m\sqrt\Delta/\sigma$, so $E\ln(S_T/S_0)=n\sigma\sqrt\Delta(2p_n-1)\to(r+m-\sigma^2/2)T$.
2. *Variance.* $n\sigma^2\Delta\cdot4p_n(1-p_n)\to\sigma^2T$ since $p_n\to\frac12$.
3. *Normality.* The summands are bounded by $\sigma\sqrt\Delta\to0$, so the Lindeberg condition holds trivially and $\ln(S_T/S_0)$ converges in law to the normal with these moments.
4. *Uniform integrability.* $E e^{\gamma\ln(S_T/S_0)}=\big(\cosh(\gamma\sigma\sqrt\Delta)+(2p_n-1)\sinh(\gamma\sigma\sqrt\Delta)\big)^n$ has a finite limit for every $\gamma$ ($n\gamma^2\sigma^2\Delta/2$ and $n(2p_n-1)\gamma\sigma\sqrt\Delta$ are bounded). With $\gamma=2$ this gives uniform integrability of $S_T$, hence of any payoff of linear growth, so convergence in law implies convergence of expectations.
5. For the call, $e^{-rT}E[(S_0e^{(r+m-\sigma^2/2)T+\sigma\sqrt TZ}-K)^+]=\mathrm{BS}_C(S_0e^{mT},K)$ (the Black-Scholes formula with forward $S_0e^{(r+m)T}$). $\square$

With $C(S)=\mathrm{BS}_C(S,K)$, $P(S)=\mathrm{BS}_P(S,K)$, Lemma 2 gives (Theorem 1 with $m\in\{0,\pm\theta,2\theta\}$):

| | call bid | call ask | put bid | put ask |
|---|---|---|---|---|
| `thesis` | $C(S)$ | $C(Se^{2\theta T})$ | $P(Se^{2\theta T})$ | $P(S)$ |
| `symmetric` | $C(Se^{-\theta T})$ | $C(Se^{\theta T})$ | $P(Se^{\theta T})$ | $P(Se^{-\theta T})$ |

**Corollary 1 (parity and the underlying).** (a) Call bid and put ask share one spot, so $C_{\rm bid}-P_{\rm ask}=Se^{m_b T}-Ke^{-rT}$ and $C_{\rm ask}-P_{\rm bid}=Se^{m_aT}-Ke^{-rT}$ ($m_b,m_a$ the spot shifts of the call). (b) The underlying is the $K=0$ call: its quote is $[Se^{m_bT},Se^{m_aT}]$ (symmetric: $[Se^{-\theta T},Se^{\theta T}]$).
**Corollary 2 (spread bound).** For every strike, the call and put spreads are at most the underlying's spread.
*Proof.* $\partial_S\mathrm{BS}_C=\Phi(d_1)\in(0,1)$ and $|\partial_S\mathrm{BS}_P|=\Phi(-d_1)\in(0,1)$, so $C(Se^{m_aT})-C(Se^{m_bT})\le S(e^{m_aT}-e^{m_bT})$, and likewise for puts. $\square$

## 4. General payoffs

**Theorem 2.** Let $m_c$ be the central drift offset of the table in Sec. 0. With $\tau=T-t$ and $x=\ln S$, the limit of the recursions solves
$$v_\tau=\tfrac12\sigma^2v_{xx}+\big(r+m_c-\tfrac12\sigma^2\big)v_x\pm\theta|v_x|-rv,\qquad v(0,x)=\varphi(e^x),$$
with $+$ for the ask and $-$ for the bid; equivalently, the ask is $\sup_{|\vartheta_s|\le\theta}E[e^{-rT}\varphi(S_T^\vartheta)]$ with $dS=(r+m_c+\vartheta_s)S\,ds+\sigma S\,dW$ (bid: $\inf$). The drift lies in the band $[r+m_c-\theta,\,r+m_c+\theta]$.

*Proved here.* (1) **Consistency.** By Lemma 1, one step is $R_n^{-1}[\pi f(Su)+(1-\pi)f(Sd)\pm\frac{\varepsilon}2|f(Su)-f(Sd)|]$ with $\pi=q_n+m_c\sqrt\Delta/(2\sigma)$. Taylor: $E_{q_n}f(S_1)=f+\Delta[rSf_S+\frac12\sigma^2S^2f_{SS}]+o(\Delta)$; the $\pi-q_n$ term adds $m_c\Delta Sf_S$; and $\frac\varepsilon2|f(Su)-f(Sd)|=\frac{\theta\sqrt\Delta}{2\sigma}\cdot2\sigma\sqrt\Delta\,S|f_S|+o(\Delta)=\theta\Delta S|f_S|+o(\Delta)$. Discounting subtracts $r\Delta f$. (2) **Monotonicity.** The weights on $Y_u,Y_d$ are $\pi\pm\varepsilon/2\ge0$ (e.g. $b_u+\varepsilon\mathbf 1\{Y_u\ge Y_d\}$), which are non-negative once $n$ is large. (3) **Stability.** The moment bound of step 4 of Theorem 1, applied to the extreme drifts, gives $|V_n(t,S)|\le C(1+S)$ uniformly. (4) **Reduction.** For monotone $\varphi$, $v_x$ has a fixed sign, $|v_x|=\pm v_x$, the PDE is linear, and its solution is Theorem 1.
*Cited, not re-proved.* Strong comparison for this Hamiltonian ($H(p)=\theta|p|$ is Lipschitz; Crandall-Ishii-Lions 1992) and the Barles-Souganidis (1991) theorem for monotone, stable, consistent schemes then give locally uniform convergence of the trees to the unique viscosity solution of polynomial growth. The control representation is the HJB/verification statement of Fleming-Soner; with $m_c=0$ it is the $\kappa$-ignorance of Chen-Epstein (2002, Econometrica 70(4)), $\kappa=\theta/\sigma$ (their theorems were not re-checked here). $\square$

## 5. Numerical verification ($\sigma=\theta=0.3$, $T=0.08$, $S_0=300.11$, $r=\ln1.0555$)

**Tree to closed form** (max abs error over calls and puts, $K\in\{270,300,330\}$, bid and ask):

| $n$ | 20 | 50 | 100 | 250 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|---|---|---|
| thesis | 1.5e-1 | 9.5e-2 | 2.9e-2 | 1.2e-2 | 5.0e-3 | 2.2e-3 | 2.6e-3 | 3.1e-4 |
| symmetric | 1.3e-1 | 5.7e-2 | 2.4e-2 | 8.5e-3 | 3.8e-3 | 1.5e-3 | 1.6e-3 | 1.7e-4 |

The error decays roughly like $1/n$ with the usual strike-grid oscillation (visible at $n=2000$).

**Straddle $|S-300|$ (non-monotone, no closed form), symmetric kind:** tree $n=4000$ gives bid 16.943, ask 24.216; an independent upwind PDE solver with Richardson extrapolation from $dx=0.002,0.001$ gives 16.943 and 24.217; classical Black-Scholes gives 20.296. Thesis kind: tree 17.892 / 26.130 versus PDE 17.892 / 26.131.

## 6. Test on META quotes (2023-09-29 snapshot, 42 calls + 33 puts, expiry 2023-10-27)

`results/continuous_market_test.csv`; strike-wise 4-fold cross-validation, price loss (thesis eq. 30), flat $\sigma$:

| model | $\sigma$ | $\theta$ (per year) | in-sample RMSE | held-out RMSE | held-out model spread (market 0.202) |
|---|---|---|---|---|---|
| Black-Scholes ($\theta=0$) | 0.460 | 0 | 0.377 | 0.378 | 0 |
| continuous, `thesis` | 0.460 | 0.005 | 0.374 | 0.375 | 0.035 |
| continuous, `symmetric` | 0.460 | 0.018 | 0.356 | 0.357 | 0.117 |

1. **Prices.** The closed form fits about as well as a flat-volatility model can; it is worse than the free-$d$ discrete fit (held-out RMSE 0.256, `docs/AMBIGUITY.md`) because $u=1/d$ removes the skew parameter and $\sigma$ is flat (no smile).
2. **Spread shape is supported.** Regressing market spreads (64 quotes with bid $>0$) on $|\Delta|$ gives $R^2=0.62$; vega gives 0.55; adding vega to $|\Delta|$ and a constant adds nothing (0.67 either way). The two regressors are correlated, so this discriminates only weakly.
3. **Spread level is rejected.** Corollary 2 says the slope on $|\Delta|$ should equal the underlying's spread (0.19 USD); the data give 1.03.
   - $\theta$ fixed from the stock quote predicts a mean at-the-money spread of 0.093 against 0.500 in the market.
   - $\theta$ fitted to the options alone (symmetric) implies an underlying spread of 1.03 against 0.19 observed (thesis kind: 0.27, but its fitted option spreads are only 0.035).
   - 25 of 64 quotes (39%) have a spread above the underlying's spread, violating Corollary 2; the violations are the near-the-money quotes.
4. A diagnostic additive floor ($\pm c/2$, $c\approx0.20$) lifts the mean model spread to 0.230 and the interval overlap to 0.59, but it is not a DS result and wider intervals overlap more trivially.

**Reading.** A single DS belief about the underlying's drift produces option spreads inherited through delta, so it can reproduce the *shape* of option spreads but not an amplitude larger than the underlying's. Amplitudes above the underlying's spread need an ingredient not tied to the underlying (ambiguity about volatility, or an option-specific belief); neither is derived here.

## 7. Caveats

* One snapshot, one expiry; cross-validation by strike is not out-of-time validation.
* The underlying's bid/ask (300.11 / 300.30) may not be time-aligned with the option quotes.
* Flat $\sigma$: no smile. The mid error changes sign across moneyness (`docs/AMBIGUITY.md`).
* Not shown: weak convergence of the path-space belief, continuous-time replication / self-financing analogue of thesis Prop. 4.4, and any arbitrage statement.
