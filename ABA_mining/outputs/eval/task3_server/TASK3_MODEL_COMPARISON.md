# Task 3 — cross-model comparison

Source: `outputs\eval\task3_server\TASK3_VERIFY_MASTER_SUMMARY_ALL_TOPICS.xlsx` · category `Contrary(P)Body(N)` · micro-averaged, mean ± std over runs.

## FAIR set — check-in, check-out, price (all 4 models)

Gold 'Yes' rate in this set: **69.8%**. Trivial baselines to beat:

- answering **Yes** to everything → F1 = **0.822**
- answering **Yes** (the majority class) to everything → Accuracy = **0.698**

A model is only doing real work if it clears these, so read every score below against them.

| Model | Version | Runs | N_valid/run | F1 | ΔF1 vs all-Yes | Precision | Recall | Accuracy | ΔAcc vs majority |
|---|---|---|---|---|---|---|---|---|---|
| gemma4_31b | zero_shot | 3 | 715 | 0.619 ± 0.000 | -0.203 | 0.943 | 0.461 | 0.604 | -0.094 |
| gemma4_31b | one_shot | 3 | 715 | 0.316 ± 0.000 | -0.506 | 0.773 | 0.198 | 0.400 | -0.298 |
| gemma4_31b | contrary_v1 | 3 | 715 | 0.869 ± 0.002 | +0.047 | 0.926 | 0.819 | 0.828 | +0.130 |
| gemma4_31b | contrary_v2 | 3 | 715 | 0.852 ± 0.000 | +0.030 | 0.894 | 0.814 | 0.803 | +0.105 |
| gemma4_31b | contrary_v3 | 3 | 715 | 0.857 ± 0.000 | +0.035 | 0.873 | 0.842 | 0.804 | +0.106 |
| gemma4_31b | contrary_v4 | 3 | 715 | 0.885 ± 0.000 | +0.063 | 0.837 | 0.938 | 0.829 | +0.131 |
| gemma4_31b | contrary_v5 | 3 | 715 | 0.874 ± 0.000 | +0.051 | 0.930 | 0.824 | 0.834 | +0.136 |
| gemma4_31b | contrary_v6 | 3 | 715 | 0.842 ± 0.000 | +0.020 | 0.900 | 0.792 | 0.793 | +0.095 |
| llama3.2 | zero_shot | 3 | 715 | 0.797 ± 0.001 | -0.025 | 0.759 | 0.838 | 0.702 | +0.004 |
| llama3.2 | one_shot | 3 | 715 | 0.329 ± 0.000 | -0.493 | 0.971 | 0.198 | 0.436 | -0.262 |
| llama3.2 | contrary_v1 | 3 | 715 | 0.833 ± 0.000 | +0.010 | 0.782 | 0.890 | 0.750 | +0.052 |
| llama3.2 | contrary_v2 | 3 | 715 | 0.823 ± 0.000 | +0.001 | 0.723 | 0.956 | 0.713 | +0.015 |
| llama3.2 | contrary_v3 | 3 | 715 | 0.692 ± 0.000 | -0.131 | 0.793 | 0.613 | 0.618 | -0.080 |
| llama3.2 | contrary_v4 | 3 | 715 | 0.000 ± 0.000 | -0.822 | 0.000 | 0.000 | 0.302 | -0.396 |
| llama3.2 | contrary_v5 | 3 | 715 | 0.826 ± 0.000 | +0.004 | 0.748 | 0.923 | 0.729 | +0.031 |
| llama3.2 | contrary_v6 | 3 | 715 | 0.739 ± 0.001 | -0.083 | 0.800 | 0.686 | 0.661 | -0.037 |
| llama4_scout | zero_shot | 1 | 715 | 0.522 | -0.300 | 0.814 | 0.385 | 0.509 | -0.189 |
| llama4_scout | one_shot | 3 | 715 | 0.830 ± 0.000 | +0.008 | 0.808 | 0.854 | 0.757 | +0.059 |
| llama4_scout | contrary_v1 | 2 | 715 | 0.715 ± 0.000 | -0.107 | 0.929 | 0.581 | 0.677 | -0.021 |
| qwen3.8_27b | zero_shot | 3 | 715 | 0.076 ± 0.000 | -0.746 | 0.667 | 0.040 | 0.316 | -0.382 |
| qwen3.8_27b | one_shot | 3 | 715 | 0.053 ± 0.000 | -0.769 | 0.538 | 0.028 | 0.305 | -0.393 |
| qwen3.8_27b | contrary_v1 | 3 | 715 | 0.486 ± 0.000 | -0.336 | 0.988 | 0.323 | 0.524 | -0.173 |
| qwen3.8_27b | contrary_v2 | 3 | 715 | 0.798 ± 0.000 | -0.024 | 0.955 | 0.685 | 0.758 | +0.060 |
| qwen3.8_27b | contrary_v3 | 3 | 715 | 0.776 ± 0.000 | -0.046 | 0.923 | 0.669 | 0.730 | +0.032 |
| qwen3.8_27b | contrary_v4 | 3 | 715 | 0.855 ± 0.000 | +0.033 | 0.851 | 0.860 | 0.797 | +0.099 |
| qwen3.8_27b | contrary_v5 | 3 | 715 | 0.279 ± 0.000 | -0.543 | 1.000 | 0.162 | 0.415 | -0.283 |
| qwen3.8_27b | contrary_v6 | 3 | 715 | 0.871 ± 0.000 | +0.049 | 0.905 | 0.840 | 0.827 | +0.129 |

**Best version per model:**

| Model | Version | Runs | F1 | Precision | Recall |
|---|---|---|---|---|---|
| gemma4_31b | contrary_v4 | 3 | 0.885 ± 0.000 | 0.837 | 0.938 |
| qwen3.8_27b | contrary_v6 | 3 | 0.871 ± 0.000 | 0.905 | 0.840 |
| llama3.2 | contrary_v1 | 3 | 0.833 ± 0.000 | 0.782 | 0.890 |
| llama4_scout | one_shot | 3 | 0.830 ± 0.000 | 0.808 | 0.854 |

<details><summary>Runs excluded from this set (1)</summary>

| Model | Shot | Test | Reason |
|---|---|---|---|
| llama4_scout | contrary_v1 | Test 3 | no result for price |

</details>

## FULL set — all 5 topics

Gold 'Yes' rate in this set: **20.4%**. Trivial baselines to beat:

- answering **Yes** to everything → F1 = **0.338**
- answering **No** (the majority class) to everything → Accuracy = **0.796**

A model is only doing real work if it clears these, so read every score below against them.

| Model | Version | Runs | N_valid/run | F1 | ΔF1 vs all-Yes | Precision | Recall | Accuracy | ΔAcc vs majority |
|---|---|---|---|---|---|---|---|---|---|
| gemma4_31b | zero_shot | 3 | 18657 | 0.304 ± 0.000 | -0.034 | 0.801 | 0.188 | 0.825 | +0.029 |
| gemma4_31b | one_shot | 3 | 18657 | 0.181 ± 0.000 | -0.157 | 0.628 | 0.106 | 0.805 | +0.009 |
| gemma4_31b | contrary_v1 | 3 | 18657 | 0.632 ± 0.000 | +0.293 | 0.607 | 0.658 | 0.844 | +0.047 |
| gemma4_31b | contrary_v2 | 3 | 18657 | 0.662 ± 0.000 | +0.324 | 0.574 | 0.782 | 0.837 | +0.041 |
| gemma4_31b | contrary_v3 | 3 | 18657 | 0.669 ± 0.000 | +0.331 | 0.615 | 0.735 | 0.852 | +0.056 |
| gemma4_31b | contrary_v4 | 3 | 18657 | 0.573 ± 0.000 | +0.235 | 0.429 | 0.863 | 0.739 | -0.058 |
| gemma4_31b | contrary_v5 | 3 | 18657 | 0.619 ± 0.000 | +0.280 | 0.653 | 0.588 | 0.852 | +0.056 |
| gemma4_31b | contrary_v6 | 3 | 18657 | 0.659 ± 0.000 | +0.320 | 0.659 | 0.659 | 0.861 | +0.065 |
| llama4_scout | one_shot | 3 | 18657 | 0.497 ± 0.000 | +0.159 | 0.417 | 0.616 | 0.746 | -0.050 |
| llama4_scout | contrary_v1 | 2 | 18657 | 0.442 ± 0.000 | +0.104 | 0.693 | 0.325 | 0.833 | +0.037 |
| qwen3.8_27b | zero_shot | 3 | 18657 | 0.019 ± 0.000 | -0.320 | 0.409 | 0.009 | 0.796 | -0.001 |
| qwen3.8_27b | one_shot | 3 | 18657 | 0.013 ± 0.000 | -0.325 | 0.510 | 0.007 | 0.796 | +0.000 |
| qwen3.8_27b | contrary_v1 | 3 | 18657 | 0.192 ± 0.000 | -0.146 | 0.864 | 0.108 | 0.815 | +0.019 |
| qwen3.8_27b | contrary_v2 | 3 | 18657 | 0.586 ± 0.000 | +0.247 | 0.778 | 0.470 | 0.865 | +0.068 |
| qwen3.8_27b | contrary_v3 | 3 | 18657 | 0.551 ± 0.000 | +0.212 | 0.674 | 0.466 | 0.845 | +0.049 |
| qwen3.8_27b | contrary_v4 | 3 | 18657 | 0.626 ± 0.000 | +0.287 | 0.633 | 0.619 | 0.849 | +0.053 |
| qwen3.8_27b | contrary_v5 | 3 | 18657 | 0.057 ± 0.000 | -0.282 | 0.991 | 0.029 | 0.802 | +0.006 |
| qwen3.8_27b | contrary_v6 | 3 | 18655 | 0.611 ± 0.000 | +0.273 | 0.594 | 0.629 | 0.837 | +0.041 |

**Best version per model:**

| Model | Version | Runs | F1 | Precision | Recall |
|---|---|---|---|---|---|
| gemma4_31b | contrary_v3 | 3 | 0.669 ± 0.000 | 0.615 | 0.735 |
| qwen3.8_27b | contrary_v4 | 3 | 0.626 ± 0.000 | 0.633 | 0.619 |
| llama4_scout | one_shot | 3 | 0.497 ± 0.000 | 0.417 | 0.616 |

<details><summary>Runs excluded from this set (26)</summary>

| Model | Shot | Test | Reason |
|---|---|---|---|
| llama3.2 | contrary_v1 | Test 1 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | contrary_v1 | Test 2 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | contrary_v1 | Test 3 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | contrary_v2 | Test 1 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | contrary_v2 | Test 2 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | contrary_v2 | Test 3 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | contrary_v3 | Test 1 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | contrary_v3 | Test 2 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | contrary_v3 | Test 3 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | contrary_v4 | Test 1 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | contrary_v4 | Test 2 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | contrary_v4 | Test 3 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | contrary_v5 | Test 1 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | contrary_v5 | Test 2 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | contrary_v5 | Test 3 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | contrary_v6 | Test 1 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | contrary_v6 | Test 2 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | contrary_v6 | Test 3 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | one_shot | Test 1 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | one_shot | Test 2 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | one_shot | Test 3 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | zero_shot | Test 1 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | zero_shot | Test 2 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama3.2 | zero_shot | Test 3 | ran on a subset only (facility 500/12060; staff 500/6256) |
| llama4_scout | contrary_v1 | Test 3 | no result for price, staff |
| llama4_scout | zero_shot | Test 1 | ran on a subset only (staff 2596/6256) |

</details>
