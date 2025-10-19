# An Empirical Study on Token Efficient Code for Enhancing LLM-based Automated Program Repair

This repository contains the code, model, and results of the paper "An Empirical Study on Token Efficient Code for Enhancing LLM-based Automated Program Repair".

## 1. Overall Architecture
<img width="812" height="283" alt="image" src="https://github.com/user-attachments/assets/6e8bd231-0b0e-4baf-bf74-ceef39dfdcae" />


#### Two Main Phases
- **Learning**  -  The LLM learns how to repair buggy code into fixed code and also understands lightweight method structures.
- **Generation** - The fine-tuned LLM generates candidate patches, followed by patch optimization and patch validation.
<br>

## 2. Patch Lightweight

The buggy method is lightweight based on the relevance score, which is calculated from the distance and similarity scores for each line of the code. Below is an example of the code before and after applying the lightweight algorithm.<br>
**Before: 100 lines, 917 tokens -> After: 29 lines, 298 tokens**
<br><br>

|Before Lightweight|After Lightweight|
|----|-----|
|<img width="402" height="211" alt="image" src="https://github.com/user-attachments/assets/cebd7151-bfdc-4a84-9938-22dbd6d23b37" />|<img width="406" height="213" alt="image" src="https://github.com/user-attachments/assets/95346580-85c5-4927-a584-06fb123f96ce" />|
<br><br>


## 3. Experiment Results
More detailed information about the ***Correctly fixed Code*** is available [here](./results).

### 3.1 [Table1] Repair Results on Defects4J and baselines

<img width="700" height="250" alt="image" src="https://github.com/user-attachments/assets/c004417c-23b1-4b5a-893d-dfde7c3b2dc4" />
<br><br>

### 3.2 [Table2] Comparison of repair results with and without applying methodology within the token limit

<img width="550" height="124" alt="image" src="https://github.com/user-attachments/assets/7840d9d5-4057-4543-bb9b-55972d37292f" />
<br><br>

### 3.3 [Table3] Comparison of repair results with and without lightweighting within the token limit

<img width="500" height="223" alt="image" src="https://github.com/user-attachments/assets/a354445f-8db0-4e3c-95c3-1f0a219a601c" />
<br><br>

### 3.4 [Table4] Bug Fixing Result under Different Lightweighting Criteria

<img width="700" height="175" alt="image" src="https://github.com/user-attachments/assets/62f5d9b1-ba0c-46a2-95fa-2606cc696379" />
<br><br>

## 📁File Structure
```
java
|--OverallProcess1.java
|----Patch Lightweight.py
python
|-
```
- java: preprocessing java files
- python
- results
<br><br>

## 📄 Related Publication
|Year|Title|Venue|File|
|-----|----------------------------------------------------------------------|---------|----------------------|
|TBA| An Empirical Study on Token Efficient Methods for Enhancing LLM-based Automated Program Repair (Eng)|JIPS| |
|2025.08|Enhancing Automated Program Repair using Patch Lightweighting and Context Information<br>(패치 경량화와 문맥 정보를 활용한 프로그램 자동 정정 개선) (Kor)|Journal of KIISE|[PDF](papers/APR2.pdf)| 
|2025.07|Leveraging Patch Lightweighting and Context under Constraint of Input Size of LLM<br>(LLM의 입력 길이 제한 조건 하에서 패치 경량화와 문맥 활용 기법) (Kor)|KCC2025|[PDF](papers/APR1.pdf)|    
|2024.12|🥇Utilizing Patch Lightweighting and Reconstruction to Handle Token Length Issues in LLM based Automatic Program Repair<br>(LLM 기반 프로그램 자동 정정에서 토큰 길이 문제 처리를 위한 패치 경량화와 복원 활용) (Kor)|KSC2024|[PDF](papers/APR3.pdf)| 
|2024.06|Towards Effectively Resolving Token Length Limits of LLM Models for Automatic Program Repair<br>(프로그램 자동 정정을 위한 LLM 모델의 효과적인 토큰 길이 제한 해결 기법) (Eng)|KCC2024|[PDF](papers/APR4.pdf)|
