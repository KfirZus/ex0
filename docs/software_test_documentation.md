# Software Test Documentation (STD)
**System:** 3D Volume Plane Projection & Visualization System  
**Version:** 1.1.0  
**Reference Document:** "Software Requirements Specification (SRS) - 3D Volume Projection System.pdf"

---

## 1. INTRODUCTION

### 1.1 Purpose
This Software Test Documentation (STD) defines the comprehensive testing strategy, test environments, and detailed test protocols for the 3D Volume Plane Projection & Visualization System. It ensures the CLI-based rendering executable adheres to all functional, non-functional, and algorithmic requirements specified in the SRS.

### 1.2 Scope
The testing scope encompasses:
*   **Smoke Testing:** Basic CLI verification and critical path execution.
*   **Unit & Logic Testing:** Granular verification of math, geometry, and data ingestion using `pytest`.
*   **End-to-End (E2E) Testing:** Complete pipeline execution from dataset ingestion to MP4 synthesis.
*   **Regression Testing:** Automated suite to prevent regressions on core rendering logic.
*   **Performance, Load, & Stress Testing:** Validation of throughput (FPS), system behavior under sustained maximum capacity, and graceful degradation under extreme limits.

---

## 2. TEST ENVIRONMENT & PREREQUISITES

*   **Operating Systems:** Windows 11, Linux (Ubuntu 22.04 LTS), macOS.
*   **Core Testing Framework:** `pytest`
    *   *Plugins:* `pytest-cov` (for code coverage), `pytest-benchmark` (for FPS throughput tracking), `pytest-xdist` (for parallel test execution).
*   **Test Data Sets (Managed via `pytest` fixtures):**
    *   `fixture_vol_8bit.nrrd`, `fixture_vol_16bit.nrrd`, `fixture_vol_32bit.nrrd` (Sizes: $500^3$ and $1023^3$ voxels).
    *   `fixture_traj_triangle.csv`, `fixture_traj_trapezoid.csv`
    *   `fixture_malformed_traj.csv` (contains missing coordinates, strings, null bytes).
*   **Monitoring Tools:**
    *   `memory_profiler` / `Valgrind` (for leak detection).
    *   FFmpeg (used in automated assertions to verify MP4 video integrity, codec, and frame counts).

---

## 3. TEST CASES

### 3.1 Smoke Testing (`@pytest.mark.smoke`)
**Objective:** Verify that the application builds, the CLI accepts arguments, and the critical path executes without fatal errors.

| Test ID | Description | Pre-conditions | Steps | Expected Result |
| :--- | :--- | :--- | :--- | :--- |
| **ST-01** | Basic CLI Invocation | Valid NRRD and CSV exist | Run `python Ex0.py valid.nrrd valid.csv out.mp4` | Script returns exit code `0` and `out.mp4` is created on disk. |
| **ST-02** | Missing Arguments | None | Run `python Ex0.py valid.nrrd` | Script exits with code `1` (or >0) and prints standard CLI help/error message to `stderr`. |

### 3.2 Unit & Logic Testing (`@pytest.mark.unit`)
**Objective:** Address REQ-TEST-1 and REQ-TEST-2 via isolated `pytest` functions using parameterized inputs.

#### 3.2.1 Data Parsing (REQ-DATA-1, REQ-TEST-2)
| Test ID | Description | Steps | Expected Result |
| :--- | :--- | :--- | :--- |
| **UT-Parse-01** | NRRD Matrix Types | `@pytest.mark.parametrize` over $500^3$ datasets of `uint8`, `uint16`, and `uint32`. | Matrix $M$ is loaded into memory array with correct bit-depth and shape properties. |
| **UT-Parse-02** | CSV Triangle Parse | Inject CSV with 3 point coordinates $(a, b, c)$ per line. | Parser yields valid 3-point spatial coordinate structures. |
| **UT-Parse-03** | CSV Trapezoid Parse | Inject CSV with 4 point coordinates $(a', b, a'', c)$ per line. | Parser yields valid 4-point spatial coordinate structures. |
| **UT-Parse-04** | Malformed CSV Handling | Inject CSV with missing commas, strings, and blank lines. | `ValueError` is caught internally; execution skips the malformed line and logs a warning without crashing. |

#### 3.2.2 Algorithmic Logic & Sampling (REQ-ALG-1, REQ-TEST-1)
| Test ID | Description | Steps | Expected Result |
| :--- | :--- | :--- | :--- |
| **UT-Logic-01** | Vector Normalization | Pass known 3D boundary points to the directional vector calculator. | Output vectors are exactly length 1.0 (using `pytest.approx`). |
| **UT-Logic-02** | Matrix E Dimensions | Request cross-section bounded by spatial distances $D_{ab}$ and $D_{bc}$. | Output 2D matrix $E$ shape strictly matches $D_{ab} \times D_{bc}$. |
| **UT-Logic-03** | Out-of-Bounds Padding | Project a plane where coordinates extend beyond the $1023^3$ spatial boundary. | No `IndexError` thrown. Extrapolated regions in Matrix $E$ are padded with exact value $0$ ($\upsilon=0$). |

### 3.3 End-to-End (E2E) Testing (`@pytest.mark.e2e`)
**Objective:** Address REQ-VID-1 and REQ-TEST-4, verifying the full CLI workflow.

| Test ID | Description | Steps | Expected Result |
| :--- | :--- | :--- | :--- |
| **E2E-01** | Full Pipeline (Triangle) | Execute CLI using an 8-bit NRRD and a CSV with 100 triangle definitions. | `out.mp4` generated. Automated FFmpeg probe confirms exactly 100 frames and playability. |
| **E2E-02** | Full Pipeline (Trapezoid) | Execute CLI using a 32-bit NRRD and a CSV with 50 trapezoid definitions. | `out.mp4` generated. Automated FFmpeg probe confirms exactly 50 frames. |

### 3.4 Regression Testing (`@pytest.mark.regression`)
**Objective:** Ensure subsequent code updates do not break established core functionalities or reintroduce known bugs. 

| Test ID | Description | Steps | Expected Result |
| :--- | :--- | :--- | :--- |
| **REG-01** | Output Checksum Verification | Render standard reference CSV and $500^3$ volume. Compare MD5/SHA256 hash of the output video frames against a known gold-standard hash. | Hashes match exactly, proving no silent algorithmic shifts in rendering logic. |
| **REG-02** | Zero-Padding Edge Case | Re-test a previously known edge case where planes intersecting the absolute corner $(0,0,0)$ caused offset errors. | Corner bounding calculates perfectly without off-by-one index shifts. |

### 3.5 Performance, Load, and Stress Testing (`@pytest.mark.performance`)
**Objective:** Validate REQ-PERF-1 (30+ FPS) and REQ-QUAL-1 (Memory Stability) under normal, peak, and extreme conditions.

#### 3.5.1 Baseline & Benchmarking
| Test ID | Description | Steps | Expected Result |
| :--- | :--- | :--- | :--- |
| **PF-01** | Baseline Throughput | Use `pytest-benchmark` on a $500^3$ reference volume with a 300-row CSV. | Benchmark asserts average processing speed is strictly $\ge 30$ FPS. |
| **PF-02** | Optimized Hardware Target | Execute benchmark suite on target optimized hardware (e.g., multi-core CPU with AVX support). | Benchmark asserts average processing speed $\ge 100$ FPS. |

#### 3.5.2 Load Testing (Sustained Maximum Capacity)
| Test ID | Description | Steps | Expected Result |
| :--- | :--- | :--- | :--- |
| **LD-01** | Sustained Max Volume Load | Execute CLI utilizing maximum spec $1023^3$ 32-bit volume and a heavy CSV ($5,000+$ frames). | Application completes entire sequence maintaining $\ge 30$ FPS without thermal throttling or memory fragmentation. |
| **LD-02** | Memory Leak Monitor | Wrap LD-01 execution in a memory profiler tracking RSS (Resident Set Size). | RAM utilization graphs show a flat line during the frame-generation loop (no steady upward growth). |

#### 3.5.3 Stress Testing (Pushing Beyond Limits)
| Test ID | Description | Steps | Expected Result |
| :--- | :--- | :--- | :--- |
| **STR-01** | Oversized Volume | Attempt to load a $2048^3$ 32-bit volume (exceeds $1023^3$ spec). | System either handles it successfully (if RAM permits) OR fails gracefully with an "Insufficient Memory / Unsupported Size" error, without hard-crashing the OS. |
| **STR-02** | Massive CSV I/O Exhaustion | Provide a CSV containing $500,000$ planar coordinates. | System reads the file efficiently using generators/streaming (avoiding `MemoryError` from loading the entire CSV into RAM at once) and begins rendering. |
| **STR-03** | Rapid Sequential Execution | Execute the CLI 50 times in rapid succession via a shell loop pointing to the same output file. | System correctly handles file locking/overwriting without corrupting the final MP4 or leaving orphaned background processes. |

---

## 4. REQUIREMENTS TRACEABILITY MATRIX (RTM)

| Requirement ID | Requirement Description | Covered By Test Case(s) |
| :--- | :--- | :--- |
| **REQ-CLI-1** | CLI Syntax & Argument Passing | ST-01, ST-02, STR-03 |
| **REQ-DATA-1** | Volumetric Matrix & Plane Parsing | UT-Parse-01, UT-Parse-02, UT-Parse-03, STR-02 |
| **REQ-ALG-1** | Directional Vector Construction & Sampling | UT-Logic-01, UT-Logic-02, UT-Logic-03, REG-02 |
| **REQ-VID-1** | Frame Sequence Generation | E2E-01, E2E-02, REG-01 |
| **REQ-PERF-1** | Processing Throughput (30+ FPS) | PF-01, PF-02, LD-01 |
| **REQ-QUAL-1** | Software Stability & Quality (No leaks) | LD-02, STR-01, STR-02 |
| **REQ-TEST-1** | Unit Math & Sampling | UT-Logic-01 to UT-Logic-03 |
| **REQ-TEST-2** | Data Parsing (including malformed) | UT-Parse-01 to UT-Parse-04 |
| **REQ-TEST-3** | Performance Benchmark | PF-01, PF-02 |
| **REQ-TEST-4** | End-to-End Pipeline | E2E-01, E2E-02, REG-01 |