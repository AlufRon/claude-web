# Moshi Finetuning Project Goals

This document outlines the primary goals of the Moshi Finetuning project, as inferred from the available evaluation scripts and logs.

## 1. Evaluate the Moshi Model

The main objective is to assess the performance of the "Moshi" model, which appears to be a model for speech processing tasks.

## 2. Compare Baseline vs. Test-Time Training (TTT)

A core goal is to compare two versions of the Moshi model:

*   **BASELINE:** The standard, pre-trained Moshi model.
*   **TTT:** A version of the model enhanced with "Test-Time Training," a technique to adapt the model to new data at inference time.

This comparison aims to quantify the benefits of TTT for the Moshi model.

## 3. Standardized Evaluation with "Paper Metrics"

The project uses a standardized set of "paper metrics" to evaluate the performance of the different model versions. These metrics are likely established benchmarks in the speech processing community, ensuring that the results are comparable to other research in the field.

## 4. Diagnose and Understand TTT

Beyond just measuring performance, the project aims to understand *how* Test-Time Training affects the model's behavior. The "Figure 5 Diagnostic" jobs are designed to provide insights into the TTT learning process.

## 5. Finetuning and Domain Adaptation

The project involves finetuning the Moshi model on various datasets, such as "LibriLight" and "dailytalk." This indicates a goal of adapting the general-purpose Moshi model to specific speech processing tasks or domains.
