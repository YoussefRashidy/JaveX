# JavaLin / JaveX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Java](https://img.shields.io/badge/Java-21%2B-blue.svg)](https://openjdk.org/)
[![Build](https://img.shields.io/badge/build-Gradle-brightgreen.svg)](https://gradle.org/)
[![Status](https://img.shields.io/badge/status-design--phase-orange.svg)]()

A Java N-dimensional array library built for **performance evolution** — designed from the ground up so that the same user-facing API can be backed by scalar, SIMD, or native/GPU computation without changing call sites.

> **Status:** This README describes the intended architecture. The project is currently **design-first and pre-implementation**. Several modules are placeholders, and full implementation may not happen on a near-term timeline.

---

## Table of Contents

- [Motivation](#motivation)
- [Core Memory Model](#core-memory-model)
- [Architecture Overview](#architecture-overview)
  - [Core Layer](#1-core-layer-core)
  - [Array Layer](#2-array-layer-array)
  - [Operators Layer](#3-operators-layer-operators)
  - [Expression Layer](#4-expression-layer-expression)
- [Key Design Decisions](#key-design-decisions)
- [Dependency Rules](#dependency-rules-intended)
- [Performance Roadmap](#performance-roadmap)
- [Getting Started](#getting-started)
- [Source Layout](#source-layout)
- [Current Scope](#current-scope-and-expectations)
- [Contributing](#contributing)
- [License](#license)

---

## Motivation

Most Java numeric libraries either box values into `Object[]` arrays (costing cache locality and memory), or commit early to a fixed backend that can't be upgraded without rewriting call sites. JavaLin/JaveX is designed around a two-axis model:

- **dtype axis** — a typed array family (`F64Array`, `F32Array`, `I32Array`, `I64Array`) backed by primitive storage, with a single generic `NDArray<T>` surface.
- **backend axis** — per-dtype operator implementations selected at startup, making it possible to swap scalar → SIMD → native backends without touching application code.

The goal is a NumPy-like ergonomics in Java, with a clear upgrade path toward hardware-level performance.

---

## Core Memory Model

Every array is a **flat primitive array** plus index-mapping metadata:
```
flatIndex = offset + Σ(i_k × stride_k)
```

| Field | Purpose |
|---|---|
| `shape` | Logical dimensions (e.g. `[3, 4]` for a 3×4 matrix) |
| `strides` | Physical memory step per dimension |
| `offset` | Start position in the backing array |

This layout enables **zero-copy views**: transpose, slice, and reverse are just metadata edits — no data is ever copied. Contiguous views can take a fast path; strided views fall back gracefully.

**Example — a 3×4 row-major array:**
```
shape   = [3, 4]
strides = [4, 1]   // row stride = 4, column stride = 1
offset  = 0
```

**Example — a transposed view of the same array:**
```
shape   = [4, 3]
strides = [1, 4]   // strides swapped, no data moved
offset  = 0
```

---

## Architecture Overview

![Project architecture](Docx/javalin_project_structure.svg)

Design notes and rationale are in `Docx/ndarray_design_v5.docx`.

The design follows a strict layered dependency structure:
```
Expression  ──▶  Operators  ──▶  Array  ──▶  Core
```

Each layer may only depend on layers to its right. `Core` has no internal dependencies.

---

### 1) Core Layer (`Core`)

Foundation contracts and shared semantics. Everything else builds on this.

| Component | Role |
|---|---|
| `NdArray<T>` | Abstract base — owns shape/stride/offset logic, index arithmetic |
| `DType` | Dtype token model; used as a type discriminant at call sites |
| `OpsProvider` | SPI for backend/operator resolution at startup |

**Design rule:** `Core` must not import from any higher layer.

---

### 2) Array Layer (`Array`)

Concrete typed arrays that own primitive storage and delegate computation to operators.

| Class | Backing store |
|---|---|
| `F64Array` | `double[]` |
| `F32Array` | `float[]` |
| `I32Array` | `int[]` |
| `I64Array` | `long[]` |
| `NdArrays` | Factory helpers (creation, zeros, ones, arange, etc.) |

Each concrete type inherits index/stride logic from `NdArray<T>` and dispatches math operations to the operator layer.

---

### 3) Operators Layer (`Operators`)

Per-dtype computation contracts and their implementations.

**Interfaces:**
- `F64Operator`, `F32Operator`, `I32Operator`, `I64Operator`

**Implementation subpackages** (one per dtype, supports multiple backends):
```
Operators/
  f64/
    ScalarF64Operator.java     ← baseline
    SimdF64Operator.java       ← future
  f32/
    ScalarF32Operator.java
  ...
```

**Intended backend progression:**
1. Scalar baseline (correct, portable)
2. Cache-aware kernels (matmul tiling)
3. SIMD via Java Vector API (`jdk.incubator.vector`)
4. Parallel execution for large workloads
5. Optional JNI/Panama/native BLAS or CUDA

The operator implementation is selected **once at startup** by `OpsProvider`, keeping hot paths monomorphic and JIT-friendly.

---

### 4) Expression Layer (`Expression`)

A fluent expression graph surface over the lower layers.

| Component | Role |
|---|---|
| `ExpressionBuilder` | Entry point for building lazy expression graphs |
| `Expression/Nodes/` | AST nodes (binary ops, reductions, reshapes, etc.) |

This layer is intentionally kept separate so it can be evolved (e.g. adding fusion, compilation) without touching the compute kernel layer.

---

## Key Design Decisions

**Primitive backing arrays, not `Object[]`**
Keeps values packed in cache-friendly contiguous memory. Avoids boxing overhead on every element access.

**Generic API, primitive internals**
The public surface (`NDArray<T>`) is generic for ergonomics. The actual compute paths work with primitives directly via sealed dispatch.

**One shared shape/stride implementation**
All index arithmetic lives in `NdArray<T>`. Typed subclasses inherit it — no duplication across dtypes.

**Contiguity-aware compute paths**
Operations detect whether a view is contiguous and take a fast direct-loop path. Strided views fall back to index-mapped access.

**Startup-time backend selection**
`OpsProvider` resolves the operator implementation once, at startup. No per-call dispatch overhead; the JIT sees a monomorphic call site.

**Zero-copy views**
Transpose, slice, broadcast, and reverse all work by editing shape/stride/offset metadata. Useful for building linear algebra pipelines without hidden allocation.

---

## Dependency Rules (Intended)
```
Expression  →  Operators  →  Array  →  Core
```

- `Core` imports nothing from higher layers.
- `NdArray` must not import from `Expression` packages.
- `OpsProvider` contract lives in `Core`; implementations live in `Operators`.
- Tests at any layer may import lower layers freely.

Violations of these rules should be treated as bugs.

---

## Performance Roadmap

| Stage | Focus | Expected gain |
|---|---|---|
| 1 — Scalar baseline | Correctness, all dtypes | Baseline |
| 2 — Cache-aware matmul | Tiling for L1/L2 reuse | Significant for large matrices |
| 3 — Java Vector API | SIMD element-wise ops | 4–8× on supported hardware |
| 4 — Multithreading | Parallel map/reduce | Near-linear scaling with cores |
| 5 — Native backends | JNI/Panama → BLAS/CUDA | Close to hardware ceiling |

For element-wise operations, **memory bandwidth** is typically the practical ceiling. For matrix multiply, **cache tiling and compute reuse** dominate. The operator abstraction exists precisely to let different kernel strategies be swapped in per-dtype without API changes.

---

## Getting Started

### Requirements

- JDK 21+ (recommended; Vector API paths may need incubator flags on earlier versions)
- Gradle wrapper included (`gradlew.bat` / `gradlew`)

If you plan to enable Vector API code paths, you may need incubator flags at compile/runtime depending on your JDK and toolchain:
```
--add-modules jdk.incubator.vector
```

### Build

**Windows PowerShell:**
```powershell
.\gradlew.bat clean build
```

**Linux / macOS:**
```bash
./gradlew clean build
```

### Run tests
```bash
./gradlew test
```

### Run application entry point
```bash
./gradlew run
```

If `run` is not configured in your current Gradle setup, execute `Main` directly from your IDE, or add the `application` plugin with a `mainClass` in `build.gradle.kts`:
```kotlin
application {
    mainClass.set("io.github.youssefrashidy.Main")
}
```

---

## Source Layout
```
src/main/java/io/github/youssefrashidy/
├── Core/           # NdArray<T>, DType, OpsProvider — no upward dependencies
├── Array/          # F64Array, F32Array, I32Array, I64Array, NdArrays factory
├── Operators/      # Per-dtype operator interfaces + backend implementations
│   ├── f64/
│   ├── f32/
│   ├── i32/
│   └── i64/
└── Expression/     # ExpressionBuilder + Nodes/
    └── Nodes/

Docx/
├── ndarray_design_v5.docx        # Architecture rationale and design notes
└── javalin_project_structure.svg # Architecture diagram
```

---

## Current Scope and Expectations

This repository is **design-first and pre-implementation.**

- Treat all architecture notes as draft targets — APIs and class names may change while the design is settling.
- Several modules are currently placeholders with no implementation.
- Do not assume production-ready behavior.
- Expect long gaps between major implementation milestones.

The architecture is intentionally designed to be implemented **incrementally**: a correct scalar baseline over a subset of dtypes and operations is a meaningful milestone, independent of SIMD or native backends.

---

## Contributing

Contributions are welcome, especially in:

- **Correctness tests** for shape/stride/view semantics (slice, transpose, broadcast edge cases)
- **Scalar operator implementations** — the baseline that all higher backends must match
- **Matmul kernels and tiling strategies** — cache-aware baselines before SIMD
- **SIMD backend parity tests** against the scalar baseline
- **Expression planning/execution** integration
- **Benchmark harnesses** for tracking regression across backend implementations

Please treat the dependency rules above as hard constraints in any contribution.

---

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) in the repository root for details.
