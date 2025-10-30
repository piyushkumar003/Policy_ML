# 🧠 Policy Optimization for Financial Decision-Making

This repository contains the full implementation and analysis for **“Policy Optimization for Financial Decision-Making”**, a project leveraging **Offline Reinforcement Learning (RL)** to optimize loan approval strategies for maximum profitability.

---

## 🚀 Overview

Traditional loan approval systems rely on classification-based credit scoring focused on minimizing default risk. This project reframes the task as a **policy optimization problem** — where the objective is to **maximize expected profit** rather than accuracy metrics like F1 score.

We compare three paradigms:

1. **Supervised MLP (Deep Learning)** — baseline classifier.
2. **Behavior Cloning (BC)** — imitation of existing approval policy.
3. **Conservative Q-Learning (CQL)** — profit-maximizing RL policy.

---

## 📊 Dataset & Preprocessing

* **Dataset:** LendingClub accepted loans (2007–2018)
* **Records Used:** ~91,681 completed loans
* **Label Mapping:**

  * `0`: Fully Paid
  * `1`: Defaulted / Charged Off
* **Split:** Chronological (70% Train / 15% Validation / 15% Test)
* **Features:** 14 key credit & financial attributes (loan amount, interest rate, income, DTI, etc.)
* **Normalization:** StandardScaler applied for uniform scaling

---

## ⚙️ Model Architecture

### 🔹 1. Supervised Model (Baseline)

* **Type:** 3-layer MLP (256 → 128 → 1)
* **Loss Function:** `BCEWithLogitsLoss` with `pos_weight`
* **Objective:** Maximize **F1 Score**

### 🔹 2. Offline RL Models

* **Algorithms:**

  * **Behavior Cloning (BC)** — imitates historical loan approval behavior.
  * **Conservative Q-Learning (CQL)** — maximizes expected cumulative profit.
* **State (s):** 77-dimensional feature vector
* **Action (a):** {0: Deny Loan, 1: Approve Loan}
* **Reward Function:**

  
R(s, a) = {
+Profit_{Total}, \text{ if Approve & Paid } \\
-Principal, \text{ if Approve & Defaulted } \\
0, \text{ if Deny }
}

---

## 📈 Results Summary

| **Policy**           | **Objective**      | **F1 Score** | **Est. Policy Value (FQE)** | **% Gain vs Baseline** |
| :------------------- | :----------------- | :----------: | :-------------------------: | :--------------------: |
| **CQL (Profit Max)** | Maximize E[Profit] |      N/A     |        **$2,776.13**        |       **+44.0%**       |
| **BC (Imitation)**   | Emulate πβ         |      N/A     |          $1,928.09          |          0.0%          |
| **MLP (Supervised)** | Maximize F1        |     0.52     |          $1,399.88          |         −27.4%         |

---

## 🧩 Policy Comparison Table

| **Metric**                       | **Supervised MLP (DL)** | **BC (Imitation)** | **CQL (Profit Max)** |
| :------------------------------- | :---------------------: | :----------------: | :------------------: |
| **Est. Policy Value (FQE)**      |        $1,399.88        |      $1,928.09     |     **$2,776.13**    |
| **Improvement over Baseline**    |          −27.4%         |        0.0%        |      **+44.0%**      |
| **AUC-ROC (Test Set)**           |          0.6922         |       0.5291       |        0.5049        |
| **F1 Score (Test Set)**          |        **0.5248**       |       0.4475       |        0.4206        |
| **Precision (Approval Quality)** |          0.4035         |       0.3127       |        0.2783        |
| **Recall (Default Avoidance)**   |          0.7506         |       0.7861       |      **0.8614**      |

### 🔍 Insights

* **CQL** achieves the **highest financial value (+44%)**, demonstrating RL’s edge in profit optimization.
* **Supervised MLP** excels in **F1**, but fails to maximize profit.
* **BC** mirrors historical policy, serving as the neutral baseline.

---

## 🧾 Project Structure

```
📂 policy-optimization-finance/
├── notebookforml.ipynb          # Main Jupyter Notebook (end-to-end)
├── report.md                    # Final project report
├── README.md                    # Repository overview
├── requirements.txt             # Dependencies
├── models/                      # Trained weights
│   ├── cql_policy.onnx
│   ├── bc_policy.onnx
│   └── mlp_supervised.pt
├── data/                        # Processed datasets
│   └── lendingclub_processed.csv
└── utils/                       # Helper scripts
    └── evaluation.py
```

---

## ⚙️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/<username>/policy-optimization-finance.git
cd policy-optimization-finance

# Setup environment
python -m venv env
source env/bin/activate  # (Windows: env\Scripts\activate)

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook notebookforml.ipynb
```

---

## 📚 Technologies Used

* **Language:** Python 3.10+
* **Frameworks:** PyTorch, d3rlpy, scikit-learn
* **Tools:** NumPy, pandas, Matplotlib
* **Evaluation:** Fitted Q Evaluation (FQE), Off-Policy Evaluation (OPE)

---

## 🧮 Key Findings

* Offline RL delivers **profit-driven** optimization, outperforming accuracy-based models.
* Reward shaping aligned with financial goals boosts performance.
* **CQL** enables safe, efficient learning without online experimentation.

---

## ⚖️ Ethical Considerations

* Validate fairness in loan approvals (monitor Disparate Impact Ratio).
* Ensure explainability and compliance with credit regulations.
* Promote responsible deployment in financial ecosystems.

---

## 🚧 Future Work

* Introduce **Fairness-Constrained RL** for equitable lending.
* Evaluate using **Doubly Robust OPE** for improved reliability.
* Deploy via **FastAPI** or **Flask microservice** for real-time inference.

---

## 🏁 Conclusion

This project highlights the capability of **Offline Reinforcement Learning** to transform financial decision-making. By focusing on **expected profit**, we achieved a **44% improvement** over traditional models. The trained **CQL policy** is production-ready for safe and scalable deployment.

---

**Author:** Piyush Kumar
**Institution:** Thapar Institute of Engineering and Technology
**Year:** 2025
