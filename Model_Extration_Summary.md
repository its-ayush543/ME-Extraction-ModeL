# Model Extraction Attack – Summary (Tramèr et al., 2016)

## 🔍 What is a Model Extraction Attack?

Model Extraction is a type of adversarial attack where the attacker tries to replicate a machine learning model by only using **black-box query access** to its API.  
The attacker sends inputs and observes the outputs (predicted class or probabilities), then trains a **surrogate model** that mimics the original.

The goal is to **steal the model’s functionality** without access to its internal structure, data, or parameters.

---

## 🎯 Threat Model

- **Black-box setting**: No access to training data or model internals.
- **Attacker’s capability**: Only input/output (I/O) query access to the model.
- **Victim**: Typically an MLaaS platform like AWS, Google ML, BigML.
- **Assumption**: API returns class probabilities (not just labels), increasing vulnerability.

---

## ⚙️ Attack Strategies

### 1. **Equation Solving (for Linear Models)**
- Use output probabilities to reconstruct weights directly using regression or algebra.

### 2. **Path Tracing (for Decision Trees)**
- Explore input space adaptively to infer internal decision thresholds.

### 3. **Distillation (for Neural Networks)**
- Train a local neural network using I/O pairs.
- Mimics the softmax output (probability distribution).

### 4. **Random or Adaptive Querying**
- Random: Uniformly sample inputs.
- Adaptive: Use prior outputs to guide next queries (e.g., binary search in feature space).

---

## 🧪 Evaluation Results

| Model Type | Platform | Accuracy of Extracted Model |
|------------|----------|------------------------------|
| Decision Tree | BigML | 99% (almost identical) |
| Logistic Regression | Custom | High — weights reconstructed |
| Neural Networks | Custom | 90–97% match in output behavior |

- Number of queries needed varies from hundreds (trees) to thousands (DNNs).

---

## 🔓 Impact of Output Information

| Output Returned | Risk |
|------------------|------|
| Class Label only | Lower |
| Label + Confidence | Higher |
| Full Probabilities | Very High |

- APIs that reveal detailed prediction scores make extraction attacks far easier and more accurate.

---

## 🛡️ Proposed Defenses

1. **Rounding or suppressing output probabilities**
2. **Adding random noise to predictions**
3. **Query rate limiting**
4. **Detecting abnormal query patterns**
5. **Access control / user monitoring**

---

## ✅ Key Takeaways

- Model extraction is **practical and dangerous**, especially for MLaaS platforms.
- Attackers can create high-fidelity clones of models by querying only.
- Returning **detailed prediction scores** greatly increases vulnerability.
- Even complex models like DNNs are not immune if enough data is collected.
- Defenses exist, but most are not foolproof or widely adopted.

---

## 🌐 Real-world Relevance

- Companies providing facial recognition, NLP, and ML-as-a-Service are at **risk**.
- Extracted models can be used for **evasion**, **impersonation**, or **piracy**.
- Face recognition APIs returning embeddings are particularly vulnerable to MEA.

