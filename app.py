import streamlit as st
import numpy as np
import sympy as sp
from scipy.fftpack import dct

st.set_page_config("GenAI Math Solver", layout="wide")
st.title("🧠 Gen-AI Mathematical Solver")
st.caption("Solves HARD mathematical problems with step-by-step explanations")

x = sp.symbols('x')
t, s = sp.symbols('t s')

# ---------------- SIDEBAR ----------------
option = st.sidebar.selectbox(
    "Select Mathematical Model",
    [
        "1) Basic Arithmetic",
        "2) Linear Algebra & Trigonometry",
        "3) Calculus, Statistics & Probability",
        "4) FFT",
        "5) Laplace Transform",
        "6) DCT",
        "7) Matrices, Vectors & Arrays",
        "8) Derivative & Integration",
        "9) Log & Exponential",
        "🔥 Hard AI Math Solver"
    ]
)

# =====================================================
# 1. BASIC ARITHMETIC
# =====================================================
if option.startswith("1"):
    st.header("➕ Basic Arithmetic")

    a = st.number_input("Enter a")
    b = st.number_input("Enter b")

    op = st.selectbox("Operation", ["+", "-", "×", "÷", "^"])

    if st.button("Solve"):
        st.subheader("Step-by-Step Solution")

        if op == "+":
            st.latex(f"{a} + {b} = {a+b}")

        elif op == "-":
            st.latex(f"{a} - {b} = {a-b}")

        elif op == "×":
            st.latex(f"{a} \\times {b} = {a*b}")

        elif op == "÷":
            if b == 0:
                st.error("Division by zero not allowed")
            else:
                st.latex(f"\\frac{{{a}}}{{{b}}} = {a/b}")

        elif op == "^":
            if abs(b) > 100:
                st.warning("Large exponent → approximation")
                st.latex(f"\\log({a}^{{{b}}}) = {b*np.log(abs(a))}")
            else:
                st.latex(f"{a}^{{{b}}} = {a**b}")

# =====================================================
# 2. LINEAR ALGEBRA & TRIGONOMETRY
# =====================================================
elif option.startswith("2"):
    st.header("📐 Linear Algebra & Trigonometry")

    A = sp.Matrix([
        [st.number_input("A11"), st.number_input("A12")],
        [st.number_input("A21"), st.number_input("A22")]
    ])

    task = st.selectbox("Task", ["Determinant", "Inverse", "Transpose", "Trigonometry"])

    if task != "Trigonometry" and st.button("Solve"):
        st.subheader("Matrix A")
        st.latex(sp.latex(A))

        if task == "Determinant":
            st.markdown("**Step:** |A| = ad − bc")
            st.latex(sp.latex(A.det()))

        elif task == "Inverse":
            if A.det() == 0:
                st.error("Inverse does not exist")
            else:
                st.markdown("**Step:** A⁻¹ = 1/|A| × adj(A)")
                st.latex(sp.latex(A.inv()))

        elif task == "Transpose":
            st.markdown("**Step:** Swap rows and columns")
            st.latex(sp.latex(A.T))

    if task == "Trigonometry":
        angle = st.number_input("Angle (degrees)")
        rad = np.deg2rad(angle)
        st.latex(r"\sin = " + sp.latex(np.sin(rad)))
        st.latex(r"\cos = " + sp.latex(np.cos(rad)))
        st.latex(r"\tan = " + sp.latex(np.tan(rad)))

# =====================================================
# 3. CALCULUS, STATISTICS & PROBABILITY
# =====================================================
elif option.startswith("3"):
    st.header("📘 Calculus, Statistics & Probability")

    task = st.selectbox("Task", ["Derivative", "Integral", "Limit", "Statistics", "Probability"])

    if task in ["Derivative", "Integral", "Limit"]:
        expr_input = st.text_input("Enter function in x")

        if expr_input and st.button("Solve"):
            f = sp.sympify(expr_input)

            st.markdown("**Given function:**")
            st.latex(sp.latex(f))

            if task == "Derivative":
                st.markdown("**Applying differentiation rules**")
                st.latex(sp.latex(sp.diff(f, x)))

            elif task == "Integral":
                st.markdown("**Applying integration rules**")
                st.latex(r"\int " + sp.latex(f) + r"\,dx = " + sp.latex(sp.integrate(f, x)))

            elif task == "Limit":
                point = st.number_input("Limit point")
                st.latex(sp.latex(sp.limit(f, x, point)))

    elif task == "Statistics":
        data = st.text_input("Enter data")
        if st.button("Analyze"):
            nums = np.array(list(map(float, data.split())))
            st.latex(r"\text{Mean} = " + sp.latex(np.mean(nums)))
            st.latex(r"\text{Variance} = " + sp.latex(np.var(nums)))
            st.latex(r"\text{Std Dev} = " + sp.latex(np.std(nums)))

    elif task == "Probability":
        fav = st.number_input("Favorable")
        total = st.number_input("Total", min_value=1)
        if st.button("Solve"):
            st.latex(r"P = \frac{" + str(fav) + "}{" + str(total) + "} = " + str(fav/total))

# =====================================================
# 4. FFT
# =====================================================
elif option.startswith("4"):
    st.header("🌊 Fast Fourier Transform")

    values = st.text_input("Enter signal")
    if st.button("Apply FFT"):
        arr = np.array(list(map(float, values.split())))
        st.markdown("**Time → Frequency domain**")
        st.latex(sp.latex(np.fft.fft(arr)))

# =====================================================
# 5. LAPLACE TRANSFORM
# =====================================================
elif option.startswith("5"):
    st.header("🔄 Laplace Transform")

    func = st.text_input("Enter function in t")
    if st.button("Solve"):
        f = sp.sympify(func)
        st.markdown("**Applying Laplace definition**")
        st.latex(sp.latex(sp.laplace_transform(f, t, s)))

# =====================================================
# 6. DCT
# =====================================================
elif option.startswith("6"):
    st.header("📉 Discrete Cosine Transform")

    values = st.text_input("Enter values")
    if st.button("Apply DCT"):
        arr = np.array(list(map(float, values.split())))
        st.latex(sp.latex(dct(arr, norm="ortho")))

# =====================================================
# 7. MATRICES, VECTORS & ARRAYS
# =====================================================
elif option.startswith("7"):
    st.header("🧩 Matrices, Vectors & Arrays")

    vec = st.text_input("Enter vector")
    if st.button("Create"):
        v = sp.Matrix(list(map(float, vec.split())))
        st.latex(sp.latex(v))

# =====================================================
# 8. DERIVATIVE & INTEGRATION (MAIN)
# =====================================================
elif option.startswith("8"):
    st.header("📐 Derivative & Integration")

    func = st.text_input("Enter function in x")
    if func and st.button("Solve Calculus"):
        f = sp.sympify(func)

        st.subheader("Derivative")
        st.latex(sp.latex(sp.diff(f, x)))

        st.subheader("Integral")
        st.latex(r"\int " + sp.latex(f) + r"\,dx = " + sp.latex(sp.integrate(f, x)))

# =====================================================
# 9. LOG & EXPONENTIAL
# =====================================================
elif option.startswith("9"):
    st.header("📈 Logarithmic & Exponential")

    func = st.text_input("Enter expression")
    if func and st.button("Solve"):
        f = sp.sympify(func)
        st.latex(sp.latex(f))
        st.latex(r"\log = " + sp.latex(sp.log(f)))
        st.latex(r"e^x = " + sp.latex(sp.exp(f)))

# =====================================================
# HARD AI SOLVER
# =====================================================
elif "Hard" in option:
    st.header("🔥 Hard AI Mathematical Solver")

    problem = st.text_area("Enter HARD mathematical expression")
    if st.button("Solve using AI"):
        expr = sp.sympify(problem)
        st.markdown("**AI reasoning applied symbolically**")
        st.latex(sp.latex(expr))
