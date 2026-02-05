import streamlit as st
import numpy as np
import sympy as sp
from scipy.fftpack import dct

st.set_page_config(page_title="GenAI Math Solver", layout="wide")
st.title("🧠 Gen-AI Mathematical Solver")
st.caption("Solves EASY ➜ HARD Mathematical Problems with Step-by-Step Explanation")

x = sp.symbols('x')
t, s = sp.symbols('t s')

# ---------------- SIDEBAR ----------------
option = st.sidebar.selectbox(
    "Select Mathematical Model",
    [
        "1. Basic Arithmetic",
        "2. Linear Algebra & Trigonometry",
        "3. Calculus, Statistics & Probability",
        "4. Fast Fourier Transform (FFT)",
        "5. Laplace Transform",
        "6. Discrete Cosine Transform (DCT)",
        "7. Matrices, Vectors & Arrays",
        "8. Derivatives & Integration",
        "9. Logarithmic & Exponential",
        "🔥 Hard AI Math Solver"
    ]
)

# =====================================================
# 1. BASIC ARITHMETIC
# =====================================================
if option == "1. Basic Arithmetic":
    st.header("➕ Basic Arithmetic Mathematical Model")

    a = st.number_input("Enter first number")
    b = st.number_input("Enter second number")

    operation = st.selectbox(
        "Select Operation",
        ["Addition", "Subtraction", "Multiplication", "Division", "Power"]
    )

    if st.button("Solve"):
        st.subheader("Step-by-Step Solution")
        if operation == "Addition":
            st.write(f"Step 1: a + b")
            st.write(f"Step 2: {a} + {b}")
            st.success(a + b)

        elif operation == "Subtraction":
            st.write(f"Step 1: a − b")
            st.write(f"Step 2: {a} − {b}")
            st.success(a - b)

        elif operation == "Multiplication":
            st.write(f"Step 1: a × b")
            st.write(f"Step 2: {a} × {b}")
            st.success(a * b)

        elif operation == "Division":
            if b == 0:
                st.error("Division by zero not allowed")
            else:
                st.write(f"Step 1: a ÷ b")
                st.write(f"Step 2: {a} ÷ {b}")
                st.success(a / b)

        elif operation == "Power":
            st.write(f"Step 1: a^b")
            st.write(f"Step 2: {a}^{b}")
            if abs(b) > 100:
                st.warning("Exponent too large → AI approximation used")
                st.write("log(result) =", b * np.log(abs(a)))
            else:
                st.success(a ** b)

# =====================================================
# 2. LINEAR ALGEBRA & TRIGONOMETRY
# =====================================================
elif option == "2. Linear Algebra & Trigonometry":
    st.header("📐 Linear Algebra & Trigonometry")

    A = np.array([
        [st.number_input("A[0][0]"), st.number_input("A[0][1]")],
        [st.number_input("A[1][0]"), st.number_input("A[1][1]")]
    ])

    task = st.selectbox("Select Task", ["Determinant", "Inverse", "Transpose", "Trigonometry"])

    if task != "Trigonometry":
        st.write("Matrix A")
        st.dataframe(A)

        if st.button("Solve"):
            det = np.linalg.det(A)

            if task == "Determinant":
                st.write("Step 1: |A| = ad − bc")
                st.success(det)

            elif task == "Inverse":
                if abs(det) < 1e-6:
                    st.error("Inverse does not exist")
                else:
                    st.write("Step 1: A⁻¹ = (1/|A|) × adj(A)")
                    st.dataframe(np.linalg.inv(A))

            elif task == "Transpose":
                st.write("Step 1: Swap rows and columns")
                st.dataframe(A.T)

    else:
        angle = st.number_input("Enter angle (degrees)")
        rad = np.deg2rad(angle)
        st.write("sin =", np.sin(rad))
        st.write("cos =", np.cos(rad))
        st.write("tan =", np.tan(rad))

# =====================================================
# 3. CALCULUS, STATISTICS & PROBABILITY
# =====================================================
elif option == "3. Calculus, Statistics & Probability":
    st.header("📘 Calculus, Statistics & Probability")

    task = st.selectbox("Select Task", ["Derivative", "Integral", "Limit", "Statistics", "Probability"])

    if task in ["Derivative", "Integral", "Limit"]:
        expr_input = st.text_input("Enter Function (example: x**3 + sin(x) + log(x))")

        if expr_input:
            expr = sp.sympify(expr_input)

            if task == "Derivative" and st.button("Solve"):
                st.write("Step 1: Identify function")
                st.write("Step 2: Apply differentiation rules")
                st.success(sp.diff(expr, x))

            elif task == "Integral" and st.button("Solve"):
                st.write("Step 1: Identify integrand")
                st.write("Step 2: Apply integration rules")
                st.success(sp.integrate(expr, x))

            elif task == "Limit":
                point = st.number_input("Limit at x →")
                if st.button("Solve"):
                    st.write("Step 1: Substitute limit value")
                    st.success(sp.limit(expr, x, point))

    elif task == "Statistics":
        data = st.text_input("Enter data (space separated)")
        if st.button("Analyze"):
            nums = np.array(list(map(float, data.split())))
            st.write("Mean =", np.mean(nums))
            st.write("Variance =", np.var(nums))
            st.write("Standard Deviation =", np.std(nums))

    elif task == "Probability":
        fav = st.number_input("Favorable outcomes", min_value=0)
        total = st.number_input("Total outcomes", min_value=1)
        if st.button("Solve"):
            st.write("Probability = favorable / total")
            st.success(fav / total)

# =====================================================
# 4. FFT
# =====================================================
elif option == "4. Fast Fourier Transform (FFT)":
    st.header("🌊 Fast Fourier Transform")

    signal = st.text_input("Enter signal values")
    if st.button("Apply FFT"):
        arr = np.array(list(map(float, signal.split())))
        st.write("Step 1: Convert time domain to frequency domain")
        st.success(np.fft.fft(arr))

# =====================================================
# 5. LAPLACE TRANSFORM
# =====================================================
elif option == "5. Laplace Transform":
    st.header("🔄 Laplace Transform")

    func = st.text_input("Enter function of t (example: exp(-2*t))")
    if st.button("Solve"):
        f = sp.sympify(func)
        st.write("Step 1: Apply Laplace definition")
        st.success(sp.laplace_transform(f, t, s))

# =====================================================
# 6. DCT
# =====================================================
elif option == "6. Discrete Cosine Transform (DCT)":
    st.header("📉 Discrete Cosine Transform")

    values = st.text_input("Enter values")
    if st.button("Apply DCT"):
        arr = np.array(list(map(float, values.split())))
        st.write("Step 1: Convert signal to cosine frequency domain")
        st.success(dct(arr, norm="ortho"))

# =====================================================
# 7. MATRICES, VECTORS & ARRAYS
# =====================================================
elif option == "7. Matrices, Vectors & Arrays":
    st.header("🧩 Matrices, Vectors & Arrays")

    vec = st.text_input("Enter vector")
    if st.button("Create"):
        st.write("Vector =", np.array(list(map(float, vec.split()))))

# =====================================================
# 8. DERIVATIVES & INTEGRATION (NUMERICAL)
# =====================================================
elif option == "8. Derivatives & Integration":
    st.header("📐 Numerical Derivatives & Integration")

    start = st.number_input("Start")
    end = st.number_input("End")
    points = st.number_input("Points", min_value=10)

    if st.button("Solve"):
        x_vals = np.linspace(start, end, int(points))
        y_vals = x_vals**2
        st.write("Numerical Derivative =", np.gradient(y_vals, x_vals))
        st.write("Numerical Integration =", np.trapz(y_vals, x_vals))

# =====================================================
# 9. LOG & EXPONENTIAL
# =====================================================
elif option == "9. Logarithmic & Exponential":
    st.header("📈 Logarithmic & Exponential")

    vals = st.text_input("Enter values")
    if st.button("Solve"):
        arr = np.array(list(map(float, vals.split())))
        st.write("log =", np.log(arr))
        st.write("exp =", np.exp(arr))

# =====================================================
# HARD AI SOLVER
# =====================================================
elif option == "🔥 Hard AI Math Solver":
    st.header("🔥 HARD Mathematical Question Solver")

    problem = st.text_area(
        "Enter HARD mathematical problem",
        placeholder="Example: integrate(x*sin(x), x)"
    )

    if st.button("Solve using AI"):
        try:
            st.write("Step-by-Step AI Reasoning:")
            result = sp.sympify(problem)
            st.success(result)
        except:
            st.error("Invalid or unsupported expression")
