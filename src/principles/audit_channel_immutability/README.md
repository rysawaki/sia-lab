
# Audit Channel Immutability

## What this is

This document defines a **design principle for audit logs and incident records**.

It is **not**:
- an algorithm
- a learning method
- an optimization procedure
- a causal inference engine

This principle constrains **how attribution is recorded**, not how it is inferred.

---

## Core Principle

### 1. Three causal channels

Every event attribution MUST belong to exactly one of the following channels:

- `world`  : external environment, sensors, physical processes
- `self`   : internal state, policy, threshold, dynamics of the system itself
- `other`  : external agents, users, operators, organizations

These channels are **defined once** at genesis.

---

### 2. Initial attribution is immutable

The causal channel assigned at the time of event creation is **immutable**.

- It MUST NOT be overwritten
- It MUST NOT be corrected in place
- It MUST remain as a record of the judgment made at that time

Correctness is irrelevant.
Only **historical judgment** matters.

---

### 3. Reclassification is a first-class event

If attribution changes later, it MUST be recorded as a separate event:

- `Reclassify`
- linked to the original event
- never modifying the original record

Reclassification represents a **change in belief**, not a correction of history.

---

### 4. Append-only log

All records MUST be append-only.

- No deletion
- No overwrite
- No retroactive modification

Time order is preserved as a chain.

---

## Minimal Example

### Genesis

```json
{"type":"Genesis",
 "schema":"audit_channel_immut/v1",
 "channels":["world","self","other"],
 "note":"Append-only. Initial channel attribution is immutable. Any change MUST be recorded as Reclassify and MUST NOT overwrite original channel.",
 "prev_hash":"0000000000000000000000000000000000000000000000000000000000000000"}
````

---

### Event

```json
{"type":"Event",
 "id":"evt_c3990a7a51c7",
 "channel":"world",
 "payload":"sensor:temp=36.8"}
```

This means:

> At that time, this event was judged to be `world`-origin.

Nothing more.

---

### Reclassify

```json
{"type":"Reclassify",
 "id":"rc_7a1b9c2e4d11",
 "target":"evt_c3990a7a51c7",
 "from":"world",
 "to":"self",
 "reason":"Delayed internal threshold crossing observed without external stimulus",
 "actor":"audit_daemon@v1"}
```

This does **not** change the original event.

It only records that:

> At a later time, attribution belief shifted.

---

## Non-goals

This principle does **not** aim to:

* determine the correct cause
* perform automatic attribution
* infer responsibility or blame
* enforce ethical or legal judgments
* replace human decision-making

It only guarantees that **attribution history cannot be rewritten**.

---

## Rationale

In post-incident analysis, attribution tends to drift due to:

* hindsight bias
* organizational pressure
* responsibility avoidance
* model revision

Without immutability, audit logs become narratives instead of records.

This principle prevents **blame laundering by design**.

---

## Status

This is a **fixed design principle**.

Extensions may exist.
Implementations may vary.
The core rule MUST NOT be relaxed.

```
