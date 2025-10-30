# Policy Optimization for Financial Decision-Making

This repository contains the complete implementation and report for the project **"Policy Optimization for Financial Decision-Making"**, focused on leveraging **Offline Reinforcement Learning (RL)** to optimize loan approval strategies for maximum financial return.

---

## 🚀 Overview

The project aims to transition loan decision-making from conventional risk-based heuristics to **data-driven profit maximization** using Reinforcement Learning (RL). We designed an **Offline RL framework** to learn from historical loan data and compare its performance against a **Supervised Deep Learning (DL)** classifier.

---

## 🧠 Problem Statement

Financial institutions often rely on classification-based credit models that optimize for accuracy (e.g., F1 score), which may fail to maximize profits. This project reframes the problem as a **policy optimization task**, where the goal is to learn an approval policy that maximizes **expected profit** rather than minimizing misclassification.

---

## 📊 Dataset & Preprocessing

* **Dataset:** LendingClub accepted loans (2007–2018)
* **Filtered:** Only completed loans (≈91,681 records)
* **Target Mapping:**

  * `0`: Fully Paid
  * `1`: Defaulted/Charged Off
* **Split:** Chronological (70% Train, 15% Validation, 15% Test)
* **Features:** 14 financial and credit features (loan amount, interest rate, annual income, DTI, etc.)
* **Scaling:** StandardScaler applied for uniformity

---

## ⚙️ Model Architecture

### 1. Supervised Model (Baseline)

* **Architecture:** 3-layer MLP (256 → 128 → 1)
* **Loss Function:** `BCEWithLogitsLoss` with `pos_weight`
* **Metric Optimized:** F1 Score

### 2. Offline Reinforcement Learning Models

* **Algorithms Used:**

  * **Behavior Cloning (BC)** – Imitates historical decisions
  * **Conservative Q-Learning (CQL)** – Maximizes long-term expected profit
* **State (s):** 77-dimensional feature vector
* **Action (a):** {0: Deny Loan, 1: Approve Loan}
* **Reward (r):**

  ```math
  R(s, a) =
  { +Profit_Total, if Approve & Paid
  -Principal, if Approve & Defaulted
  0, if Deny }
  ```

---

## 📈 Results Summary

| Policy               | Objective          | F1 Score | Est. Policy Value (FQE) | % Gain vs Baseline |
| :------------------- | :----------------- | :------: | :---------------------: | :----------------: |
| **CQL (Profit Max)** | Maximize E[Profit] |    N/A   |      **$2,776.13**      |     **+44.0%**     |
| **BC (Imitation)**   | Emulate πβ         |    N/A   |        $1,928.09        |        0.0%        |
| **MLP (Supervised)** | Maximize F1        |   0.52   |        $1,399.88        |       −27.4%       |

### Key Insights

* **Accuracy ≠ Profitability:** Models that optimized F1 underperformed in profit.
* **CQL Outperformed:** The RL policy learned optimal risk-reward trade-offs.
* **Expected Gain:** +44% increase in average profit compared to baseline.

---

## 🧾 Project Structure

```
📂 policy-optimization-finance/
├── notebookforml.ipynb          # Jupyter Notebook (end-to-end pipeline)
├── report.md                    # Full final project report
├── README.md                    # Repository overview
├── requirements.txt             # Required dependencies
├── models/                      # Trained model weights
│   ├── cql_policy.onnx
│   ├── bc_policy.onnx
│   └── mlp_supervised.pt
├── data/                        # Processed datasets
│   └── lendingclub_processed.csv
└── utils/                       # Helper functions
    └── evaluation.py
```

---

## 🧩 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/<username>/policy-optimization-finance.git
cd policy-optimization-finance

# Create environment
python -m venv env
source env/bin/activate  # (Windows: env\Scripts\activate)

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook notebookforml.ipynb
```

---

## 📚 Technologies Used

* **Language:** Python 3.10+
* **Libraries:** d3rlpy, PyTorch, scikit-learn, NumPy, pandas
* **Evaluation:** Fitted Q Evaluation (FQE), Off-Policy Evaluation (OPE)

---

## 🧮 Key Findings

* **Conservative Q-Learning** outperforms traditional supervised models in financial optimization tasks.
* Aligning reward design with business profit objectives significantly improves long-term outcomes.
* Reinforcement Learning can be safely deployed in financial systems through **offline (batch) training** and **OPE validation**.

---

## ⚖️ Ethical Considerations

* Ensure fairness in credit approvals (check Disparate Impact Ratio across demographic proxies).
* Implement transparency in decision logic for compliance and auditability.

---

## 🚧 Future Work

* Incorporate **Fairness-Constrained RL** for equitable lending.
* Extend evaluation using **Doubly Robust OPE** methods.
* Deploy model via a **Flask or FastAPI microservice** for real-time inference.

---

## 🏁 Conclusion

This project demonstrates the power of **Offline Reinforcement Learning** for data-driven decision-making in finance. By optimizing policies directly for **expected profit**, rather than surrogate accuracy metrics, we achieved a **44% improvement in financial return**. The trained CQL policy is ready for deployment and further validation in live environments.

---

**Author:** Piyush Kumar
**Institution:** Thapar Institute of Engineering and Technology
**Year:** 2025
