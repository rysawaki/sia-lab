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
