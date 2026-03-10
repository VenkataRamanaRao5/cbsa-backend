# CBSA Backend — Current Implementation Overview

This document explains the **current running behavior** of the backend in plain language (no code), including what input it expects, what it does at each stage, and what it returns.

---

## 1) Main Purpose

The backend receives continuous behavioural events over WebSocket, processes them incrementally, and returns behavioural metrics.

It currently supports:
- Username-based identity
- Session-aware streaming validation
- Incremental drift + stability calculations
- Warm-up initialization for new users
- SQLite-persistent prototypes
- SQLite-persistent behavioural logs
- Export/import merge compatibility

It does **not** include:
- Trust decisioning
- Re-authentication logic
- Policy/GAT/allow-restrict actions

---

## 2) Endpoints in Use

### Behaviour ingestion WebSocket
- **Endpoint:** `/ws/behaviour`
- **Role:** Primary streaming endpoint used by mobile app
- **Input:** JSON event per message
- **Output:** Warm-up response OR behavioural metrics response

### Monitor WebSocket
- **Endpoint:** `/ws/monitor`
- **Role:** Feeds the monitoring web page with incoming event stream metadata

### Monitor UI page
- **Endpoint:** `/`
- **Role:** Serves graph monitor UI

### Event-flow map
- **Endpoint:** `/event-flow-map`
- **Role:** Serves static event graph map JSON for monitor visualization

### Health
- **Endpoint:** `/health`
- **Role:** Server health and active connection counts

---

## 3) Behaviour Event Input — Required Data Format

Each message on `/ws/behaviour` is expected as:

- `username` (required, non-empty string)
- `session_id` (required string)
- `timestamp` (required float/int)
- `event_type` (required string)
- `event_data` (required object)
  - `nonce` (required string)
  - `vector` (required list)
  - `deviceInfo` (optional object, forwarded to monitor)
- `signature` (optional in payload, signature verification is currently stubbed)

### Vector requirements
- Must exist
- Must have exactly **48 values**
- Every value must be numeric and in **[0, 1]**

---

## 4) Processing Pipeline (What Happens Per Event)

For each incoming WebSocket event:

### Step A — Secure ingestion validation
Checks performed:
1. JSON object shape check
2. Required fields check (`username`, `session_id`, `timestamp`, `event_type`, `event_data`, `nonce`, `vector`)
3. Vector checks (exists, len=48, numeric, value range [0,1])
4. Nonce uniqueness per session
5. Monotonic timestamp per session
6. Burst-rate guard: reject if delta `< 40ms` for more than 5 consecutive events
7. Signature verification stub currently returns true

If any check fails:
- Event is rejected
- WebSocket responds with an error payload

### Step B — Session buffer update (incremental)
Maintains per-session state in memory:
- Short window (`maxlen=5`)
- Running mean (48-dim)
- Running variance via Welford (48-dim)
- Event history
- Last timestamp / nonce tracking

### Step C — Preprocessing metrics
Computes:
- `short_window_mean`
- `long_term_mean` (running mean)
- `variance_vector`
- `short_drift` = normalized L2(current vector vs short-window mean)
- `long_drift` = normalized L2(short-window mean vs long-term mean)
- `stability_score` = mean L2 distance across recent consecutive events

### Step D — Behaviour log persistence (always)
Before prototype update, one row is inserted into `behaviour_logs` with:
- username
- session_id
- event_timestamp
- event_type
- window vector (JSON)
- short_drift
- long_drift
- stability_score
- created_at

This happens for **every processed event**.

### Step E — Warm-up handling for new users
User state is checked in DB (`users.initialized`).

If user is not initialized:
- Collect window vectors in warm-up buffer
- Warm-up target = **20 windows**
- Continue logging during warm-up
- On the 20th window:
  - Compute mean + variance from warm-up windows
  - Create first prototype in DB
  - Set `initialized=1`
  - Clear warm-up buffer

Warm-up response is returned while in this phase.

### Step F — Prototype intelligence (after warm-up)
For initialized users:
1. Load all prototypes from SQLite
2. For each prototype compute:
   - Cosine similarity
   - Mahalanobis distance (variance-aware)
   - Composite score
3. Select best match
4. Controlled adaptive update if:
   - similarity > 0.8 and short_drift < 0.3
5. Else create new prototype
6. Keep max 15 prototypes per user (remove lowest support when over limit)

---

## 5) WebSocket Output Format

## A) Warm-up response
Returned for users still in warm-up collection:

- `status`: `"WARMUP"`
- `collected_windows`: current warm-up window count

## B) Metrics response
Returned after warm-up is complete:

- `similarity_score` (float)
- `short_drift` (float)
- `long_drift` (float)
- `stability_score` (float)
- `matched_prototype_id` (integer or null)

No trust labels, no policy actions, no decision fields are returned.

---

## 6) SQLite Persistence — Current Tables

Database file: `cbsa.db`

### users
Tracks user identity and initialization state.

### prototypes
Stores persistent adaptive prototypes per username.

### behaviour_logs
Stores full behavioural history window-by-window for analytics and research.

---

## 7) Merge Compatibility (Import/Export)

Implemented in storage merge utilities.

### Export
Returns a structured object with:
- username
- user metadata
- prototypes
- behaviour logs

### Import
- Inserts user if missing
- Merges prototypes while avoiding exact duplicate vectors
- Appends behaviour logs
- Preserves earliest created timestamp semantics for user metadata

---

## 8) Runtime Behavior Summary

In one line:

**Event arrives → validate → incremental preprocess → log to DB → warm-up or prototype match/update → metrics reply.**

This is the current production behavior of the backend.
