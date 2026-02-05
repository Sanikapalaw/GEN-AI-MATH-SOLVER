import streamlit as st
import numpy as np
import sympy as sp
from scipy.fftpack import dct

st.set_page_config(page_title="GenAI Math Solver", layout="wide")

st.title("🧠 GenAI Mathematical Solver")
st.subheader("Solve Easy ➜ HARD Mathematical Problems using AI Logic")

# Sidebar Menu
option = st.sidebar.selectbox(
    "Choose Mathematical Module",
    [
        "Arithmetic",
        "Linear Algebra",
        "Calculus",
        "Statistics & Probability",
        "FFT",
        "Laplace Transform",
        "DCT",
        "Derivatives & Integration",
        "Log & Exponential",
        "Hard AI Math Solver"
    ]
)

# ----------------------------------------
# Arithmetic
# ----------------------------------------
if option == "Arithmetic":
    st.header("➕ Arithmetic Operations")

    a = st.number_input("Enter first number")
    b = st.number_input("Enter second number")

    if st.button("Calculate"):
        st.write("Addition:", a + b)
        st.write("Subtraction:", a - b)
        st.write("Multiplication:", a * b)
        if b != 0:
            st.write("Division:", a / b)
        st.write("Power:", a ** b)

# ----------------------------------------
# Linear Algebra
# ----------------------------------------
elif option == "Linear Algebra":
    st.header("📐 Linear Algebra")

    A = np.array([
        [st.number_input("A[0][0]"), st.number_input("A[0][1]")],
        [st.number_input("A[1][0]"), st.number_input("A[1][1]")]
    ])

    if st.button("Solve Matrix"):
        st.write("Matrix A:", A)
        st.write("Determinant:", np.linalg.det(A))
        if np.linalg.det(A) != 0:
            st.write("Inverse:", np.linalg.inv(A))

# ----------------------------------------
# Calculus
# ----------------------------------------
elif option == "Calculus":
    st.header("📘 Calculus")

    x = sp.symbols('x')
    expr = st.text_input("Enter function (example: x**3 + sin(x))")

    if st.button("Solve Calculus"):
        f = sp.sympify(expr)
        st.write("Derivative:", sp.diff(f, x))
        st.write("Integral:", sp.integrate(f, x))

# ----------------------------------------
# Statistics & Probability
# ----------------------------------------
elif option == "Statistics & Probability":
    st.header("📊 Statistics & Probability")

    data = st.text_input("Enter data (space separated):")

    if st.button("Analyze"):
        nums = np.array(list(map(float, data.split())))
        st.write("Mean:", np.mean(nums))
        st.write("Variance:", np.var(nums))
        st.write("Standard Deviation:", np.std(nums))

# ----------------------------------------
# FFT
# ----------------------------------------
elif option == "FFT":
    st.header("🌊 Fast Fourier Transform")

    signal = st.text_input("Enter signal values:")

    if st.button("Apply FFT"):
        sig = np.array(list(map(float, signal.split())))
        st.write("FFT Output:", np.fft.fft(sig))

# ----------------------------------------
# Laplace Transform
# ----------------------------------------
elif option == "Laplace Transform":
    st.header("🔄 Laplace Transform")

    t, s = sp.symbols('t s')
    func = st.text_input("Enter function of t (example: exp(-2*t))")

    if st.button("Find Laplace"):
        f = sp.sympify(func)
        st.write("Laplace Transform:", sp.laplace_transform(f, t, s))

# ----------------------------------------
# DCT
# ----------------------------------------
elif option == "DCT":
    st.header("📉 Discrete Cosine Transform")

    values = st.text_input("Enter values:")

    if st.button("Apply DCT"):
        arr = np.array(list(map(float, values.split())))
        st.write("DCT Output:", dct(arr, norm="ortho"))

# ----------------------------------------
# Derivatives & Integration (Numerical)
# ----------------------------------------
elif option == "Derivatives & Integration":
    st.header("📐 Numerical Methods")

    start = st.number_input("Start Value")
    end = st.number_input("End Value")
    points = st.number_input("Number of points", min_value=10, step=10)

    if st.button("Compute"):
        x_vals = np.linspace(start, end, int(points))
        y_vals = x_vals**2
        st.write("Derivative:", np.gradient(y_vals, x_vals))
        st.write("Integration:", np.trapz(y_vals, x_vals))

# ----------------------------------------
# Log & Exponential
# ----------------------------------------
elif option == "Log & Exponential":
    st.header("📈 Logarithmic & Exponential")

    values = st.text_input("Enter values:")

    if st.button("Solve"):
        v = np.array(list(map(float, values.split())))
        st.write("Log:", np.log(v))
        st.write("Exponential:", np.exp(v))

# ----------------------------------------
# HARD AI MATH SOLVER
# ----------------------------------------
elif option == "Hard AI Math Solver":
    st.header("🔥 HARD Mathematical Question Solver (GenAI Style)")

    st.write("Supports:")
    st.write("- Differential equations")
    st.write("- Integrals & limits")
    st.write("- Matrix equations")
    st.write("- Symbolic simplification")

    problem = st.text_area(
        "Enter HARD math problem (example: integrate(x*sin(x), x) or dsolve(y''+y=0))"
    )

    if st.button("Solve using AI"):
        try:
            result = sp.sympify(problem)
            st.success("Solution:")
            st.write(result)
        except:
            st.error("Unable to solve. Please check syntax.")
