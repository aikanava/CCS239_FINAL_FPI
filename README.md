# 🔁 Fixed Point Iteration Solver

An interactive **Streamlit web app** for solving equations of the form **x = g(x)** using the **Fixed Point Iteration method**.

---

## ✨ Features

* 📌 Fixed Point Iteration with custom initial guess & tolerance
* 📋 Automatic **Fixed Point Theorem validation** on an interval [a, b]
* 📈 Interactive **Cobweb Plot** (with y = x reference)
* 🔢 Iteration table with downloadable CSV
* 🧮 Symbolic derivative computation using SymPy
* 🧠 Smart divergence & error handling

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/aikanava/CCS239_FINAL_FPI.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open automatically in your browser.

---

## ✍️ How to Use

1. Enter a function **g(x)** (example: `np.cos(x)`)
2. Choose an **initial guess** (x_0) and **tolerance** (\varepsilon)
3. Define an interval **[a, b]** for theorem validation
4. Click **Compute Root** 🚀


You’ll get:

* Theorem validation results ✅❌
* Approximate root
* Iteration count & final error
* Cobweb convergence visualization

---

## 📐 Fixed Point Theorem Checks

The app verifies:

1. Continuity of g(x) on [a, b]
2. Self-mapping: (a \le g(x) \le b)
3. Derivative (g'(x))
4. Continuity of (g'(x))
5. Convergence condition: (|g'(x)| < 1)

If the condition fails, the app still runs — but gives you a heads-up ⚠️.

---

## 🧪 Example Functions

```text
np.cos(x)
np.exp(-x)
(x + 2/x) / 2
```

Use `np.` for math functions and `**` for powers.
