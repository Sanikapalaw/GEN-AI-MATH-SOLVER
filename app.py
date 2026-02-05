import streamlit as st
import numpy as np
import sympy as sp
from scipy.fft import fft, dct

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Math Logic Solver", layout="wide")
st.title("🧮 9-Domain Mathematical Logic System")

x, t, s = sp.symbols('x t s')

# --------------------------------------------------
# MENU
# --------------------------------------------------
menu = [
    "1) Basic Arithmetic Mathematical Model",
    "2) Linear Algebra and Trigonometry",
    "3) Calculus, Statistics, and Probability",
    "4) Fast Fourier Transform (FFT)",
    "5) Laplace Transform",
    "6) DCT (Discrete Cosine Transform)",
    "7) Matrices, Vectors, and Arrays",
    "8) Derivatives and Integration",
    "9) Log and Exponential"
]
choice = st.sidebar.radio("Select Module", menu)

# --------------------------------------------------
# 1) BASIC ARITHMETIC
# --------------------------------------------------
if choice.startswith("1"):
    st.header("1) Basic Arithmetic Mathematical Model")

    a = st.number_input("First Number", value=10.0)
    b = st.number_input("Second Number", value=5.0)

    op = st.selectbox("Operation", ["Addition", "Subtraction", "Multiplication", "Division", "Power"])

    if st.button("Solve"):
        st.markdown("### Step-by-Step Solution")

        if op == "Addition":
            st.latex(f"{a} + {b} = {a+b}")

        elif op == "Subtraction":
            st.latex(f"{a} - {b} = {a-b}")

        elif op == "Multiplication":
            st.latex(f"{a} \\times {b} = {a*b}")

        elif op == "Division":
            if b == 0:
                st.error("Division by zero is not allowed")
            else:
                st.latex(rf"\frac{{{a}}}{{{b}}} = {a/b}")

        elif op == "Power":
            if abs(b) > 100:
                st.warning("Exponent too large — showing logarithmic approximation")
                st.latex(rf"\log({a}^{b}) = {b*np.log(abs(a))}")
            else:
                st.latex(rf"{a}^{{{b}}} = {a**b}")

# --------------------------------------------------
# 2) LINEAR ALGEBRA & TRIGONOMETRY
# --------------------------------------------------
elif choice.startswith("2"):
    st.header("2) Linear Algebra and Trigonometry")

    task = st.selectbox("Task", ["Trigonometry", "Matrix Determinant", "Matrix Inverse"])

    if task == "Trigonometry":
        angle = st.number_input("Angle (degrees)", value=45.0)
        rad = np.radians(angle)

        st.latex(rf"\sin({angle}^\circ) = {np.sin(rad)}")
        st.latex(rf"\cos({angle}^\circ) = {np.cos(rad)}")
        st.latex(rf"\tan({angle}^\circ) = {np.tan(rad)}")

    else:
        A = sp.Matrix([
            [st.number_input("A11", value=1.0), st.number_input("A12", value=2.0)],
            [st.number_input("A21", value=3.0), st.number_input("A22", value=4.0)]
        ])

        st.latex(sp.latex(A))

        if task == "Matrix Determinant":
            st.latex(r"\det(A) = " + sp.latex(A.det()))

        elif task == "Matrix Inverse":
            if A.det() == 0:
                st.error("Inverse does not exist (determinant = 0)")
            else:
                st.latex(sp.latex(A.inv()))

# --------------------------------------------------
# 3) CALCULUS, STATISTICS & PROBABILITY
# --------------------------------------------------
elif choice.startswith("3"):
    st.header("3) Calculus, Statistics, and Probability")

    task = st.selectbox("Task", ["Derivative", "Integral", "Statistics", "Probability"])

    if task in ["Derivative", "Integral"]:
        expr_input = st.text_input("Enter function in x", "log(x)*sqrt(x) + exp(x**2)")

        if st.button("Solve"):
            try:
                f = sp.sympify(expr_input)
                st.latex(sp.latex(f))

                if task == "Derivative":
                    st.markdown("**Applying differentiation rules**")
                    st.latex(sp.latex(sp.diff(f, x)))

                else:
                    st.markdown("**Applying integration rules**")
                    st.latex(r"\int " + sp.latex(f) + r"\,dx = " + sp.latex(sp.integrate(f, x)))

            except:
                st.error("Invalid mathematical expression")

    elif task == "Statistics":
        data_str = st.text_input("Enter data (space separated)", "10 20 30 40")

        if st.button("Analyze"):
            try:
                data = list(map(float, data_str.split()))
                st.latex(r"\text{Data} = " + sp.latex(data))
                st.latex(r"\text{Mean} = " + str(np.mean(data)))

                if len(data) < 2:
                    st.warning("Variance and Std Dev require at least 2 values")
                    st.stop()

                st.latex(r"\text{Variance} = " + str(np.var(data)))
                st.latex(r"\text{Std Dev} = " + str(np.std(data)))

            except:
                st.error("Invalid data input")

    elif task == "Probability":
        fav = st.number_input("Favorable Outcomes", min_value=0)
        total = st.number_input("Total Outcomes", min_value=1)

        st.latex(rf"P = \frac{{{fav}}}{{{total}}} = {fav/total}")

# --------------------------------------------------
# 4) FFT
# --------------------------------------------------
elif choice.startswith("4"):
    st.header("4) Fast Fourier Transform (FFT)")
    sig_str = st.text_input("Signal values (space separated)", "1 0 1 0")

    if st.button("Apply FFT"):
        sig = list(map(float, sig_str.split()))
        if len(sig) < 2:
            st.error("FFT requires at least 2 values")
        else:
            st.latex(sp.latex(fft(sig)))

# --------------------------------------------------
# 5) LAPLACE TRANSFORM
# --------------------------------------------------
elif choice.startswith("5"):
    st.header("5) Laplace Transform")

    f_t = st.text_input("Function f(t)", "exp(-t)")

    try:
        expr = sp.sympify(f_t)
        L = sp.laplace_transform(expr, t, s)[0]
        st.latex(r"\mathcal{L}\{" + sp.latex(expr) + r"\} = " + sp.latex(L))
    except:
        st.error("Invalid function")

# --------------------------------------------------
# 6) DCT
# --------------------------------------------------
elif choice.startswith("6"):
    st.header("6) Discrete Cosine Transform (DCT)")
    sig = st.text_input("Enter values", "1 2 3 4")

    if st.button("Apply DCT"):
        data = list(map(float, sig.split()))
        if len(data) < 2:
            st.error("DCT requires at least 2 values")
        else:
            st.latex(sp.latex(dct(data, norm="ortho")))

# --------------------------------------------------
# 7) MATRICES, VECTORS & ARRAYS
# --------------------------------------------------
elif choice.startswith("7"):
    st.header("7) Matrices, Vectors, and Arrays")

    vec = st.text_input("Enter vector", "1 2 3")
    v = sp.Matrix(list(map(float, vec.split())))
    st.latex(sp.latex(v))

# --------------------------------------------------
# 8) DERIVATIVES & INTEGRATION
# --------------------------------------------------
elif choice.startswith("8"):
    st.header("8) Derivatives and Integration")

    f_x = st.text_input("Function f(x)", "sin(x)**2")

    try:
        expr = sp.sympify(f_x)
        st.markdown("**Derivative:**")
        st.latex(sp.latex(sp.diff(expr, x)))
        st.markdown("**Integral:**")
        st.latex(r"\int " + sp.latex(expr) + r"\,dx = " + sp.latex(sp.integrate(expr, x)))
    except:
        st.error("Invalid expression")

# --------------------------------------------------
# 9) LOG & EXPONENTIAL
# --------------------------------------------------
elif choice.startswith("9"):
    st.header("9) Log and Exponential")

    val = st.number_input("Value", value=2.0)

    if val <= 0:
        st.error("Logarithm defined only for positive values")
    else:
        st.latex(rf"\ln({val}) = {np.log(val)}")
        st.latex(rf"e^{{{val}}} = {np.exp(val)}")
