# An Empirical Study on Token Efficient Code for Enhancing LLM-based Automated Program Repair

This repository contains the source code, and experiment results of the paper "An Empirical Study on Token Efficient Code for Enhancing LLM-based Automated Program Repair".

## Overview
In LLM-based Automated Program Repair (APR), bugs exceeding the LLM's input token limit often cause failures in correctly fixing them. This study addresses this issue by applying a **method lightweight** and incorporates similar methods for additional context. **Empirical studies** demonstrate its effectiveness, and the research is conducted in two main phases: **learning and generation**.<br>

- **Learning**  -  The LLM learns how to repair buggy code into fixed code and also understands lightweight method structures.
- **Generation** - The fine-tuned LLM generates candidate patches, followed by reconstruction, patch optimization and patch validation.
<br>

## Experiment Results

### Defects4J Benchmark with CodeBERT
- **44 out of 318 bugs correctly fixed (13.8%)**
- Average bug length **726 tokens (CodeBERT limit: 512)**
- **8 multi-chunk bugs** successfully fixed

### Defects4J Benchmark with CodeLlama
- **23 out of 124 bugs fixed (18.5%)**, including **3 multi-chunk bugs**
- Average bug length: **1,481 tokens (CodeLlama limit: 1,024)**
- Lightweight method is applicable to various LLMs

## 📁File Structure
```
Lightweight
|--src/          # Source code for this study
|--results/      # Experiment results
|--figures/      # Figures and tables from the paper
|--README.md
```

## Core Component

### Method Lightweight
- Relevance score derived from distance & similarity metrics
- Line-level Lightweight based on relevance
- Example: **100 lines, 917 tokens -> 29 lines, 298 tokens**
<br><br>

|Before Lightweight|After Lightweight|
|----|-----|
|<img width="402" height="211" alt="image" src="https://github.com/user-attachments/assets/cebd7151-bfdc-4a84-9938-22dbd6d23b37" />|<img width="406" height="213" alt="image" src="https://github.com/user-attachments/assets/95346580-85c5-4927-a584-06fb123f96ce" />|
<br><br>

### Retrieve Context Method
- Retrieves up to **5 methods** similar to the buggy method within the file
- Identifies similarity using **embedding vectors** and **cosine similarity**
- Context is lightweighted based on lines similar to buggy lines

### Method Reconstruction
- Reconstructs **lightweight-generated methods** to **original method with fixed code**
- Based on characteristics of diverse method structures




