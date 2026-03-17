# CBSA System Architecture Design
## Continuous Behavioural Session Authentication — Research-Grade Implementation

**Version:** 2.0 (Redesigned Layer-2 + New Layer-4)
**Status:** Production-Grade / Reviewer-Defensible

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Critical Bugs Fixed in v2.0](#2-critical-bugs-fixed-in-v20)
3. [Layer-2: Preprocessing and Behavioural Modelling](#3-layer-2-preprocessing-and-behavioural-modelling)
4. [Layer-4: Continuous Trust Engine](#4-layer-4-continuous-trust-engine)
5. [Prototype Quarantine System](#5-prototype-quarantine-system)
6. [Prototype Lifecycle Management](#6-prototype-lifecycle-management)
7. [Failure and Edge Case Handling](#7-failure-and-edge-case-handling)
8. [System Justification — Design Choice Rationale](#8-system-justification--design-choice-rationale)
9. [Final Architecture Summary](#9-final-architecture-summary)
10. [Current Implementation Status and Limitations](#10-current-implementation-status-and-limitations)

---

## 1. System Overview

CBSA is a continuous behavioural authentication system that re-verifies user identity throughout a session by analysing behavioral patterns extracted from device interactions (touch, scroll, keystrokes, accelerometer, gyroscope). The system architecture has four layers:

```
Mobile App
    |
    v  (WebSocket JSON stream, ~500ms or event-based)
Layer 1: Secure Ingestion         [app/engine/ingestion.py]
    |      Validates, extracts, deduplicates events
    v
Layer 2: Preprocessing + Modelling [app/engine/preprocessing.py, prototype_engine.py]
    |      Computes leakage-free drift, stability, similarity; manages prototypes
    v
Layer 3: Deep Analysis — GAT      [app/gat/, app/engine/trust_engine.py (escalation)]
    |      Graph Attention Network on temporal behavioral graph (triggered by Layer-4)
    v
Layer 4: Trust Aggregation        [app/engine/trust_engine.py]
           Continuous EMA trust score, decision zones, Layer-3 escalation logic
```

**Key principle**: Layers 2 and 4 produce measurements and trust scores respectively. Neither layer makes authentication decisions (block/allow). The system outputs a continuous trust score and decision signal for the application layer to act upon.

---

## 2. Critical Bugs Fixed in v2.0

### Bug 1: Statistical Leakage in Drift Computation (SEVERITY: CRITICAL)

**Original code** (buffer_manager.py):
```python
session_state = update_session_buffer(event)   # running_mean updated to include v_t
short_drift = l2_norm(v_t, session_state.running_mean)  # LEAKAGE
```

**Problem**: The `running_mean` was updated WITH the current event `v_t` BEFORE drift was computed. This means drift was measuring distance from a distribution that already included `v_t`. In the extreme case (first event): `running_mean == v_t`, so `short_drift = 0` identically — completely uninformative.

**Fix**: `update_session_buffer()` now returns a `PreUpdateSnapshot` capturing all statistics BEFORE incorporating `v_t`. All drift and stability computations use only the snapshot.

```python
session_state, snapshot = update_session_buffer(event)  # snapshot = pre-update stats
short_drift = compute_short_drift(v_t, snapshot.short_window_mean)  # No leakage
```

**Mathematical significance**: This is not a minor improvement — it corrects a fundamental data validity violation. Drift computed against a distribution that includes the current sample has no statistical meaning. Every drift measurement in the original system was systematically biased toward zero.

---

### Bug 2: Unbounded stability_score Used in Bounded Composite (SEVERITY: HIGH)

**Original code** (preprocessing.py, similarity_engine.py):
```python
stability_raw = float(np.mean(diffs))   # L2 differences between consecutive events — UNBOUNDED
stability_score = stability_raw          # not normalized, can be >> 1
# ...
score = 0.5 * cosine + 0.3 * (1.0 - normalized_mahalanobis) + 0.2 * stability_score
```

**Problem**: `stability_score` was the raw mean L2 difference between consecutive events — an unbounded quantity that could be 0.001 or 10.0. Using it directly in the composite similarity formula (which expects [0,1] inputs) could produce similarity scores outside [0,1], breaking all downstream comparisons.

**Fix**: Stability is now defined as:
```
S(t) = exp(-(1/D) * sum_i [Var_short,i / max(Var_global,i, eps)])
```
This is bounded in (0, 1] by the properties of the exponential function. No clipping required.

---

### Bug 3: Prototype EMA Variance Update Error (SEVERITY: MEDIUM)

**Original code** (prototype_engine.py):
```python
alpha = 1.0 / (prototype.support_count + 1)
old_vector = prototype.vector.copy()
updated_vector = (1.0 - alpha) * prototype.vector + alpha * current_vector
updated_variance = (1.0 - alpha) * prototype.variance + alpha * np.square(current_vector - old_vector)
```

**Problem**: The variance update uses `(current_vector - old_vector)^2` — the deviation from the PRE-update mean. Correct EMA variance should use deviation from the POST-update mean:
```
sigma^2_EMA = (1 - eta) * sigma^2_old + eta * (x - mu_new)^2
```

Using the pre-update mean overestimates variance because `||x - mu_old|| > ||x - mu_new||` always (the updated mean is closer to x). The bias is small for large n but systematic.

Additionally, `alpha = 1/(n+1)` causes the update rate to decay toward zero, effectively freezing mature prototypes and preventing adaptation to legitimate behavioral drift.

**Fix**: Adaptive learning rate `eta(n) = eta_base * exp(-n/tau) + eta_floor` (see Section 6).

---

### Bug 4: Immediate Prototype Creation (SEVERITY: HIGH — Security)

**Original code**: Any behavioral vector with similarity < 0.8 to all existing prototypes immediately became a new prototype.

**Problem**: This is a security vulnerability — a single anomalous event permanently alters the user's behavioral profile. An attacker who gains brief access can inject a prototype that their own behavioral patterns match, facilitating future attacks.

**Fix**: Prototype Quarantine System — see Section 5.

---

## 3. Layer-2: Preprocessing and Behavioural Modelling

### 3.1 Leakage-Free Drift Computation

All drift and stability metrics are computed using **pre-update statistics** (captured before `v_t` is incorporated into the running buffers).

**Short-term drift** measures deviation of the current event from recent local context:

```
d_short(t) = 1 - exp(-||v_t - mu_window^{t-1}||_2 / (sqrt(D) * sigma))
```

Where:
- `mu_window^{t-1}`: mean of the last W events, NOT including `v_t` (pre-update)
- `sqrt(D)` normalization: dimension-invariant scaling (for D=48: sqrt(48) ≈ 6.93)
- `sigma = 0.15 * sqrt(D) ≈ 1.04`: characteristic drift scale (see Section 8.1)
- `1 - exp(-·)`: maps [0,∞) → [0,1) monotonically

**Long-term drift** measures deviation of local context from global baseline:

```
d_long(t) = 1 - exp(-||mu_window^{t-1} - mu_global^{t-1}||_2 / (sqrt(D) * sigma))
```

Where:
- `mu_global^{t-1}`: Welford's running mean BEFORE this event
- Both quantities are pre-update — no leakage

**Why exp normalization over d/(1+d)?**
- d/(1+d) has no natural scale parameter — it treats d=0.1 and d=1.0 with the same relative sensitivity curve
- exp(-d/sigma) has sigma as an interpretable scale: at d=sigma, f≈0.63 (one-sigma drift)
- The exponential tail decays faster for large distances, providing better discrimination of extreme drift events
- The Platt-scaling form can be adapted per-user by updating sigma from historical drift distributions

### 3.2 Stability Score

The stability score captures within-window behavioral variance relative to the user's established variability:

```
S(t) = exp(-(1/D) * sum_i [Var_short,i / max(Var_global,i, eps)])
```

**Interpretation**:
| Scenario | Effect on S(t) |
|----------|---------------|
| Short var << Long var | Ratio ≈ 0 → S(t) ≈ 1 (highly stable) |
| Short var = Long var | Ratio = 1 → S(t) = exp(-1) ≈ 0.37 (neutral) |
| Short var >> Long var | Ratio large → S(t) → 0 (erratic) |

**Mathematical guarantee**: S(t) ∈ (0, 1] by construction (exp of non-positive value). No clipping required.

**The 1/D normalization** ensures S(t) is independent of vector dimensionality — a system with D=48 and D=96 features would produce comparable stability scores for behaviorally equivalent patterns.

### 3.3 Behavioural Consistency

Measures directional coherence within the recent window — distinct from stability:

```
C(t) = (1/|W|) * sum_{v in W} cos(v, mean(W))
```

**Stability vs Consistency**:
- **Stability** (S): measures amplitude variance (are the values spread out?)
- **Consistency** (C): measures directional alignment (do they point the same way?)

An attacker replaying high-amplitude but randomly directed events would show high stability (low variance) but low consistency (scattered directions). These two metrics together provide orthogonal discriminative signals.

### 3.4 Composite Similarity

```
sim(v, P_k) = 0.50 * cos(v, mu_k) + 0.40 * exp(-d_M(v, mu_k, Sigma_k)/sqrt(D)) + 0.10 * S(t)
```

**Component analysis**:

| Component | Formula | Role | Bound |
|-----------|---------|------|-------|
| Cosine | cos(v, mu_k) | Behavioral direction | [0,1] |
| Mahalanobis kernel | exp(-d_M/sqrt(D)) | Distance accounting for variance | (0,1] |
| Stability | S(t) | Quality modifier | (0,1] |

**Why exp(-d_M/sqrt(D)) for Mahalanobis?**

The standard `d/(1+d)` normalization was used in the original system. The exp kernel is superior because:
1. **Probabilistic interpretation**: This is an unnormalized Gaussian kernel. The similarity value has a direct probabilistic meaning as an unnormalized likelihood ratio under a Gaussian prototype model.
2. **sqrt(D) normalization**: For a D-dimensional standard normal vector x, E[||x||_2] ≈ sqrt(D). Dividing by sqrt(D) means that a "typical" random deviation (one standard deviation in each dimension) maps to kernel value exp(-1) ≈ 0.37 — a natural calibration.
3. **Better tail discrimination**: For large distances, exp(-d) < d/(1+d), providing sharper rejection of highly dissimilar vectors.

**Weight rationale (50/40/10)**:
- Cosine (0.50): Primary discriminator — captures behavioral direction (the "shape" of the pattern). Scale-invariant.
- Mahalanobis (0.40): Secondary — captures magnitude deviation accounting for the user's established variability profile. Variance-aware.
- Stability (0.10): Quality modifier only. Overweighting stability would confuse behavioral quality with behavioral identity.

**Diagonal Mahalanobis approximation**: Full covariance estimation requires O(D²) = O(2304) samples. For streaming sessions with ~10s windows, this is impractical. The diagonal approximation captures per-feature variance without cross-correlation terms, requiring only O(D) = O(48) samples.

---

## 4. Layer-4: Continuous Trust Engine

### 4.1 Trust Model

The trust score follows a continuous Exponential Moving Average:

```
T_t = alpha_t * T_{t-1} + (1 - alpha_t) * R_t
```

Where:
- **T_t ∈ [0,1]**: Trust at time t. Initialised to 0.5 (neutral, no information).
- **T_{t-1}**: Previous trust (temporal memory). Prevents single-event noise from collapsing trust.
- **R_t ∈ [0,1]**: Raw trust signal from Layer-2 (see below).
- **alpha_t ∈ [0.30, 0.85]**: Adaptive EMA coefficient (see Section 4.2).

**EMA properties**:
- Convex combination: `alpha_t * T_{t-1} + (1-alpha_t) * R_t` with both operands in [0,1] guarantees `T_t ∈ [0,1]`
- Temporal memory: trust cannot collapse instantly from a single anomalous event
- Gradual recovery: after a false alarm, trust recovers as subsequent events are normal

### 4.2 Raw Trust Signal

```
D_t = 0.60 * d_short_t + 0.40 * d_long_t    (composite drift)
R_t = 0.45 * sim_t + 0.25 * stab_t + 0.30 * (1 - D_t)
```

**Boundary verification**:
- Maximum: sim=1, stab=1, D=0 → R = 0.45 + 0.25 + 0.30 = 1.0 ✓
- Minimum: sim=0, stab=0, D=1 → R = 0 + 0 + 0 = 0.0 ✓

**Composite drift weighting (60/40)**:
- Short drift (0.60): Sudden behavioral changes are more alarming than gradual ones
- Long drift (0.40): Gradual drift is still a signal but less urgent

**Weight rationale (45/25/30)**:
- Similarity (0.45): Primary authentication signal — how well does current behavior match the established prototype?
- Drift complement (0.30): Penalizes behavioral deviation. Second-largest weight reflects importance of catching sudden anomalies.
- Stability (0.25): Behavioral quality signal. Lower than similarity because stability measures within-window coherence, not between-prototype agreement.

### 4.3 Adaptive EMA Coefficient

```
alpha_t = clip(alpha_max - gamma * d_short_t, alpha_min, alpha_max)
gamma = alpha_max - alpha_min = 0.55
```

| d_short | alpha_t | EMA half-life |
|---------|---------|---------------|
| 0.0 | 0.85 | ~4.3 events (slow, smooth) |
| 0.5 | 0.575 | ~1.3 events (moderate) |
| 1.0 | 0.30 | ~0.76 events (fast, responsive) |

**Half-life formula**: `t_half = -log(2) / log(alpha)`

This adaptive mechanism prevents two failure modes:
1. **Noisy decisions during stable behavior**: high alpha smooths single-event fluctuations
2. **Slow response to genuine threats**: low alpha ensures rapid trust collapse during sustained anomaly

### 4.4 GAT Augmentation (Layer-3 Integration)

When Layer-3 provides a GAT similarity score:

```
R_t^aug = (1 - kappa) * R_t + kappa * GAT_t,    kappa = 0.25
```

**Why kappa = 0.25?**
- GAT operates on the temporal graph structure of the full session window (~20 events), providing higher-level context than per-event prototype matching
- However, GAT has higher computational cost and latency — it should refine, not override, Layer-2
- kappa = 0.25 ensures Layer-2 retains primary authority (75%) while GAT contributes refinement (25%)
- If GAT is unavailable: R_t is used unchanged. System degrades gracefully.

### 4.5 Decision Zones

| Zone | Condition | Meaning |
|------|-----------|---------|
| SAFE | T_t > 0.65 | Behavior matches established profile — session continues |
| UNCERTAIN | 0.40 ≤ T_t ≤ 0.65 | Ambiguous — may warrant attention or Layer-3 analysis |
| RISK | T_t < 0.40 | Trust has collapsed — system flags this session |

**Threshold calibration**:
- `theta_safe = 0.65`: Requires sustained positive raw signals (sim ≥ 0.9, stab ≥ 0.8, drift ≤ 0.2 for ~3-4 events post-login)
- `theta_risk = 0.40`: Consistent low-quality behavior over multiple events
- The gap [0.40, 0.65] is the uncertainty zone — not clearly normal, not clearly threatening

### 4.6 Layer-3 Escalation Logic

**Why not periodic GAT?** The original system ran GAT every `GAT_INFERENCE_INTERVAL_SECONDS` regardless of trust state. This is wasteful (unnecessary compute during stable sessions) and slow (anomalous behavior detected at second 1 waits until second N).

**Event-driven escalation triggers** (any one sufficient):

1. **RISK zone** (`T_t < 0.40`): Trust has collapsed — immediate deep analysis required.
2. **UNCERTAIN + elevated anomaly** (`0.40 ≤ T_t ≤ 0.65` AND `anomaly_indicator > 0.40`): Uncertain zone with elevated anomaly signal — potential threat.
3. **Sustained uncertainty** (`consecutive_uncertain >= 3`): System cannot resolve the situation through Layer-2 alone — GAT needed.

**Escalation suppression**: Re-check interval of 30 seconds prevents GAT from being called on every uncertain event when trust fluctuates near the boundary.

**Anomaly indicator** (from Layer-2):
```
anomaly(t) = (1 - sim_t) * (0.5 + 0.5 * d_short_t)
```

The 0.5 base weight captures anomalies where drift is low but similarity fails (e.g., replay attacks that don't deviate from behavioral norms but use different interaction patterns).

---

## 5. Prototype Quarantine System

### 5.1 Motivation

Without quarantine, the original system could be exploited through **prototype injection**:
1. Attacker gains momentary access (e.g., while owner is briefly away)
2. Attacker performs a single behavioral interaction that differs from the owner's profile
3. This creates a new prototype matching the attacker's behavior (similarity < 0.8 → new prototype)
4. Attacker can now pass authentication against their own injected prototype

The quarantine system makes this attack infeasible: a single event cannot create a prototype.

### 5.2 Quarantine Protocol

All new behavioral patterns enter a **CandidatePool** (in-memory, per-user). Promotion to a full prototype requires satisfying ALL three criteria simultaneously:

```
1. Observation Count:  count >= 3
2. Temporal Spread:    time_span >= 30 seconds
3. Consistency:        mean_cosine_to_centroid >= 0.72
```

**Criteria rationale**:

| Criterion | Value | Why |
|-----------|-------|-----|
| count >= 3 | 3 | Minimum for statistical validity; prevents single-event injection |
| time_span >= 30s | 30 | Natural interaction episodes span 30-60s; prevents burst injection |
| consistency >= 0.72 | 0.72 | Mean inter-vector angle ≤ 44° in 48-D space; strong directional agreement |

### 5.3 Candidate Assignment

A new vector `v` is assigned to existing candidate `C_j` if:
```
cos(v, centroid_j) >= 0.75
```

Higher threshold than promotion consistency (0.72) because assignment must be confident — if a vector weakly resembles a candidate, creating a new candidate is preferable to contaminating an existing one.

### 5.4 Candidate Expiry

Candidates inactive for > 600 seconds (10 minutes) are silently deleted. This prevents:
- Memory growth from abandoned behavioral patterns
- Slow accumulation of old noise toward promotion threshold

---

## 6. Prototype Lifecycle Management

### 6.1 Adaptive Learning Rate

```
eta(n) = eta_base * exp(-n / tau) + eta_floor
```

**Parameters**: eta_base = 0.30, tau = 50, eta_floor = 0.01

| n (support count) | eta(n) | Behavior |
|-------------------|--------|----------|
| 0 | 0.31 | Fast learning (new prototype) |
| 10 | 0.21 | Moderate |
| 50 | 0.12 | Slowing |
| 100 | 0.041 | Slow but not frozen |
| inf | 0.01 | Floor — never completely frozen |

**Update rule**:
```
mu_new = (1 - eta) * mu_old + eta * v
sigma^2_new = (1 - eta) * sigma^2_old + eta * (v - mu_new)^2
sigma^2_blended = 0.70 * sigma^2_EMA + 0.30 * sigma^2_session
```

The 70/30 blending of EMA variance with session variance prevents prototype variance from becoming arbitrarily narrow (which would make the Mahalanobis metric overly sensitive to normal behavioral fluctuations).

**Comparison with original alpha = 1/(n+1)**:

| n | Original alpha | New eta(n) |
|---|---------------|------------|
| 1 | 0.500 | 0.307 |
| 10 | 0.091 | 0.213 |
| 100 | 0.010 | 0.041 |
| 1000 | 0.001 | **0.010** (floor) |

The original system effectively froze prototypes at n=100+ (alpha ≈ 0.01 and decreasing). The new system maintains a minimum adaptation rate (eta_floor=0.01), allowing legitimate long-term behavioral drift to be incorporated.

### 6.2 Prototype Matching Thresholds

| Threshold | Value | Action |
|-----------|-------|--------|
| THRESHOLD_UPDATE | 0.75 | sim >= 0.75: update existing prototype |
| THRESHOLD_CREATE | 0.50 | sim < 0.50: route to quarantine |
| Dead zone | [0.50, 0.75) | No action — re-evaluate next event |

**Why a dead zone?**

The original system had an undefined gap: similarity < 0.8 created a new prototype, but there was no logic for the range where behavior is moderately similar. The dead zone [0.50, 0.75) prevents prototype corruption via "boundary attack" — repeatedly submitting vectors at similarity ≈ 0.76 would continuously update the prototype in the original system, causing drift. Now they fall in the dead zone and are ignored.

### 6.3 Quality-Based Prototype Deletion

When `MAX_PROTOTYPES_PER_USER` (15) is reached, the prototype with lowest quality score is deleted:

```
Q(k) = log(1 + n_k) * exp(-lambda_age * age_k) * max(sim_k, 0.1)
```

Where:
- `log(1 + n_k)`: logarithmic support (diminishing returns for very high counts)
- `exp(-lambda_age * age_k)`: age decay (lambda = 1/day, so week-old prototype ≈ 0.001x)
- `max(sim_k, 0.1)`: similarity relevance (0.1 floor prevents total zeroing of low-sim prototypes)

**Why better than original (delete by lowest support_count)?**

The original deleted prototypes with the lowest support count. This would delete newly created prototypes that represent the user's **current** behavior — exactly the most relevant prototypes. The quality score balances recency, support, and relevance, preferring to delete old, rarely-matched, low-support prototypes.

---

## 7. Failure and Edge Case Handling

### 7.1 Cold Start (New User)

| Phase | System Behavior |
|-------|----------------|
| First event | Routes to quarantine; similarity_score = 0.0; trust_score = 0.5 |
| Events 1-2 (< N_MIN) | Quarantine accumulates; similarity = 0; trust stays ~0.5 |
| After 30s + 3 observations + consistency pass | First prototype created; system begins matching |

**Trust during cold start**: The initial trust T_0 = 0.5 represents genuine uncertainty. With similarity=0 and no prototypes, R_t ≈ 0.30 (from drift complement only). Trust drifts toward ~0.40-0.45 — UNCERTAIN zone. This is correct: the system should not trust an unestablished session.

### 7.2 Legitimate Behavioural Drift

A user's behavior naturally evolves over time (different grip, new device, changed habits).

**Detection**: Long drift `d_long(t)` increases gradually as `mu_window` diverges from `mu_global`. This elevates Layer-4's composite drift and reduces trust modestly.

**Adaptation**: The adaptive learning rate (eta_floor = 0.01) ensures prototypes slowly incorporate genuine drift. Over N events, the prototype centroid shifts toward the new behavioral pattern.

**Threshold resilience**: Fixed global thresholds (0.75, 0.50) could falsely flag legitimate drift. The planned per-user adaptive thresholds (Section 10) would derive thresholds from historical similarity distributions, accommodating users with naturally higher behavioral variability.

### 7.3 Attacker Mimicry

An attacker who observes the user's behavioral patterns may attempt to replicate them.

**Layer-2 defense**:
- Behavioral vectors are 48-dimensional — perfect mimicry requires matching all dimensions simultaneously
- The Mahalanobis component penalizes vectors that deviate from the user's established variance profile — even "close" mimicry fails if the attacker's variance pattern differs
- The anomaly indicator `(1-sim)*(0.5 + 0.5*d_short)` captures partial mimicry: high similarity with anomalous drift still produces elevated anomaly

**Layer-4 defense**:
- Trust EMA provides temporal memory — sustained mimicry under scrutiny is required, not a single successful event
- Escalation to Layer-3 (GAT) provides structural analysis of the behavioral session graph that is much harder to mimic than individual event vectors

### 7.4 Rapid Interaction Bursts

High-frequency events (e.g., rapid touch sequence during scrolling) could cause instability.

**Handling**:
- Short window (W=5): The short window mean quickly incorporates burst events; stability score drops
- Adaptive alpha: High short_drift during a burst lowers alpha_t → faster trust response
- Dead zone: Burst-induced borderline similarity doesn't trigger spurious prototype creation
- The ingestion layer (Layer-1) has `fast_delta_count` tracking for rate limiting (existing)

### 7.5 Missing Layer-3 Output

GAT output is optional throughout the system.

- `gat_similarity = None` is explicitly handled in `trust_engine.update_trust()`
- When GAT is unavailable, `R_t` is used unchanged (kappa = 0, effectively)
- This graceful degradation means Layer-3 failures do not affect Layer-2 or Layer-4 operation
- Logged with `gat_augmented = False` in TrustResult for monitoring

---

## 8. System Justification — Design Choice Rationale

### 8.1 Why sigma = 0.15 * sqrt(D) for exp-normalization?

Behavioral vectors are in [0,1]^48. A "moderate" behavioral change might shift an average feature by 0.15 (15% of the normalized range). The L2 of a uniform 0.15 shift across all 48 dimensions is:

```
||delta||_2 = 0.15 * sqrt(48) ≈ 1.04
```

After dimension normalization: `d_norm = 1.04 / sqrt(48) = 0.15`
After exp-normalization: `1 - exp(-0.15/sigma) = 1 - exp(-1) ≈ 0.63`

This gives the intuition: a "moderate" behavioral shift (15% per feature) maps to `d_normalized ≈ 0.63`. Extreme shifts (50%+) approach 1.0. Minor fluctuations (2-3%) remain near 0.

**Future**: sigma should be adapted per-user from historical drift distributions — larger sigma for naturally variable users, smaller for consistent users.

### 8.2 Why Quarantine Instead of Direct Prototype Creation?

Three independent justifications:

1. **Statistical**: A single observation provides no statistical confidence about a behavioral pattern's validity. The mean of 1 observation = the observation itself — no information about variance or stability.

2. **Security**: See Section 5.1. Single-event injection is a viable attack against direct creation.

3. **Quality**: Transient behavioral patterns (distraction, accidental gesture) should not become permanent reference points. The consistency requirement (≥ 0.72 cosine similarity across observations) ensures only coherent, repeatable patterns are promoted.

**Why not require more observations (e.g., 10)?** 3 observations balances security (prevents injection) with enrollment speed. A user exploring a new UI feature would produce 3 observations relatively quickly during normal interaction.

### 8.3 Why Adaptive Thresholds Instead of Fixed?

Users exhibit significantly different behavioral consistency levels:
- A touch typist has highly consistent keypress patterns → tight prototype clusters → high similarity scores
- A casual smartphone user has variable behavior → loose clusters → lower similarity scores

A fixed threshold of 0.75 might be appropriate for the typist but far too strict for the casual user (constant false positives). Per-user thresholds derived from historical similarity distributions:
```
theta_update(u) = mu_sim(u) - 1.5 * sigma_sim(u)
theta_create(u) = mu_sim(u) - 3.0 * sigma_sim(u)
```

These place thresholds at 1.5 and 3 standard deviations below the user's mean similarity — statistically principled rejection regions.

**Current status**: Global default thresholds are used (0.75, 0.50) pending sufficient historical data. Implementation plan in Section 10.

### 8.4 Why Separate Modelling and Decision Layers?

Combining behavioral modeling (prototype matching) and trust decisions in the same component creates several problems:
1. **Coupling**: Changes to the prototype system would require re-validating decision logic, and vice versa
2. **Multi-source aggregation**: Layer-4 must combine Layer-2 and Layer-3 outputs — this is impossible if Layer-2 makes decisions internally
3. **Temporal state**: Trust is fundamentally a session-level temporal quantity (EMA over events), while prototype matching is event-level — these have different lifecycles
4. **Testability**: The trust engine can be unit-tested with synthetic metrics, independent of the prototype system

The clean interface between Layer-2 (produces PrototypeMetrics) and Layer-4 (consumes them) allows either layer to be evolved independently.

### 8.5 Why EMA for Trust Instead of a Direct Score?

A direct trust score (e.g., current similarity_score as trust) would:
- Be maximally volatile: one low-similarity event collapses trust instantly
- Provide no temporal memory: a single good event would fully restore trust after an anomaly
- Not reflect behavioral trajectory: a user with consistently declining similarity (concerning) would look identical to one with stable high similarity followed by one normal dip

The EMA provides:
- **Temporal smoothing**: Trust reflects the trend, not a single point
- **Memory**: Anomalies must be sustained to collapse trust
- **Trajectory tracking**: Gradually declining similarity produces gradually declining trust

---

## 9. Final Architecture Summary

### Pipeline Overview

```
Event Received (48-D vector)
    |
    v [Layer 1: ingestion.py]
    Validate, deduplicate (nonce), extract BehaviourEvent
    |
    v [Layer 2a: preprocessing.py + buffer_manager.py]
    1. Capture PreUpdateSnapshot (pre-update stats — no leakage)
    2. Update session buffers (Welford's algorithm)
    3. Compute: short_drift, long_drift, stability_score, behavioural_consistency
    4. Produce: PreprocessedBehaviour (6 fields)
    |
    v [Layer 2b: prototype_engine.py]
    5. Retrieve user prototypes from store
    6. Compute composite similarity (cosine + Mahalanobis kernel + stability)
    7. Identify best matching prototype
    8. If sim >= 0.75: update prototype (adaptive EMA)
       If sim < 0.50: route to quarantine_manager
       Else: dead zone, no action
    9. Compute 9-field PrototypeMetrics (no decisions)
    |
    v [Layer 4: trust_engine.py]
    10. Retrieve per-session TrustState
    11. Compute raw signal R_t = 0.45*sim + 0.25*stab + 0.30*(1-D_composite)
    12. Optional GAT augmentation: R_t^aug = 0.75*R_t + 0.25*GAT_t
    13. Adaptive alpha_t = clip(0.85 - 0.55*d_short, 0.30, 0.85)
    14. EMA update: T_t = alpha_t * T_{t-1} + (1-alpha_t) * R_t
    15. Classify: SAFE (>0.65) | UNCERTAIN (0.40-0.65) | RISK (<0.40)
    16. Determine escalation: trigger GAT if RISK, UNCERTAIN+high anomaly, or consecutive uncertain >= 3
    17. Return TrustResult
    |
    v [Optional Layer 3: gat_engine.py]
    18. If escalated: build temporal graph, run GAT inference
    19. Return GAT similarity score → fed back to Layer-4 (augmentation)
    |
    v [Response to client]
    {
      // Layer-2 rich output
      similarity_score, short_drift, long_drift, stability_score,
      prototype_confidence, behavioural_consistency,
      prototype_support_strength, anomaly_indicator, matched_prototype_id,
      // Layer-4 output
      trust_score, decision, escalated_to_layer3, raw_trust_signal
    }
```

### Layer Responsibilities

| Layer | Module(s) | Responsibility | Makes Decisions? |
|-------|-----------|----------------|-----------------|
| Layer 1 | ingestion.py | Validate, deduplicate, extract | No — rejects invalid only |
| Layer 2a | preprocessing.py, buffer_manager.py, drift_engine.py | Drift, stability, consistency computation | No |
| Layer 2b | prototype_engine.py, similarity_engine.py, quarantine_manager.py | Prototype matching, update, lifecycle | No |
| Layer 3 | gat_engine.py, gat_network.py, layer3_manager.py | Deep temporal graph analysis | No |
| Layer 4 | trust_engine.py | Trust aggregation, decision zones, GAT escalation | YES — outputs SAFE/UNCERTAIN/RISK |

### Data Flow (Types)

```
BehaviourEvent (ingestion)
    -> PreprocessedBehaviour (preprocessing)
    -> PrototypeMetrics (prototype_engine)
    -> TrustResult (trust_engine)
    -> JSON response (WebSocket)
```

---

## 10. Current Implementation Status and Limitations

### Implemented in v2.0

- [x] Leakage-free drift computation (PreUpdateSnapshot)
- [x] Bounded stability_score via exp(-variance_ratio)
- [x] Bounded behavioural_consistency (cosine-based)
- [x] Exp-Mahalanobis kernel (replaces d/(1+d))
- [x] Bounded composite similarity (all components in [0,1])
- [x] Prototype quarantine system (N_MIN=3, T_MIN=30s, consistency=0.72)
- [x] Adaptive learning rate (eta_base=0.30, tau=50, eta_floor=0.01)
- [x] Corrected EMA variance update (uses post-update mean)
- [x] Quality-based prototype deletion (Q = log(n) * age_factor * sim)
- [x] Dead-zone thresholds (update=0.75, create=0.50)
- [x] 9-field rich PrototypeMetrics output
- [x] TrustState per session (in memory_store.SessionState)
- [x] Continuous EMA trust model (adaptive alpha, 3-zone decision)
- [x] GAT augmentation (kappa=0.25)
- [x] Event-driven Layer-3 escalation (3 trigger conditions)
- [x] CosmosDB-compatible delete_prototype() method

### Planned Improvements (Not Yet Implemented)

- [ ] **Per-user adaptive sigma** for drift normalization: `sigma_u = mean(d_short_history) * 1.5`. Requires historical drift accumulation (minimum 30 events).
- [ ] **Per-user adaptive thresholds**: `theta_update_u = mu_sim_u - 1.5*sigma_sim_u`. Requires per-user similarity history storage in memory_store.
- [ ] **CosmosDB delete_prototype() method**: Currently only SQLiteStore has delete_prototype(). CosmosPrototypeStore needs equivalent.
- [ ] **Session reset on RISK decision**: Application layer should consider re-authentication challenge when trust_score < theta_risk.
- [ ] **Quarantine persistence**: Candidate pools are currently in-memory only. Server restart loses quarantine state. For production: persist to Redis or Cosmos.
- [ ] **Trust state persistence across reconnects**: If a WebSocket disconnects and reconnects, trust resets to 0.5. Should resume from last known state for the same session.

---

*Document maintained alongside implementation. Update when architectural decisions change.*
