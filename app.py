import streamlit as st
import numpy as np
import sympy as sp
from scipy import fft, stats

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Math Logic Solver", layout="wide")
st.title("🧮 9-Domain Step-by-Step Solver")

# Global Symbols for Symbolic Math
x, t, s = sp.symbols('x t s')

# --- 2. THE 9-NUMBERED MENU ---
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

# --- 3. DOMAIN LOGIC ---

# 1) Basic Arithmetic
if choice.startswith("1)"):
    st.header("1) Basic Arithmetic Mathematical Model")
    a = st.number_input("First Number", value=10.0)
    b = st.number_input("Second Number", value=5.0)
    st.write(f"Sum: {a + b} | Difference: {a - b} | Product: {a * b} | Division: {a / b if b != 0 else 'Error'}")

# 2) Linear Algebra & Trig
elif choice.startswith("2)"):
    st.header("2) Linear Algebra and Trigonometry")
    angle = st.number_input("Angle (Degrees)", value=45.0)
    rad = np.radians(angle)
    st.write(f"$\sin({angle}^\circ) = {np.sin(rad)}$")
    st.write(f"$\cos({angle}^\circ) = {np.cos(rad)}$")
    

[Image of the unit circle showing sine and cosine values]


# 3) Calculus, Stats & Probability
elif choice.startswith("3)"):
    st.header("3) Calculus, Statistics and Probability")
    data_str = st.text_input("Data Points (comma separated)", "10, 20, 30, 40")
    data = np.fromstring(data_str, sep=',')
    if len(data) > 0:
        st.write(f"Mean: {np.mean(data)} | Std Dev: {np.std(data)}")
    

[Image of a normal distribution curve with mean and standard deviation]


# 4) FFT
elif choice.startswith("4)"):
    st.header("4) Fast Fourier Transform (FFT)")
    sig = np.fromstring(st.text_input("Signal Array", "1, 0, 1, 0"), sep=',')
    if len(sig) > 0:
        st.write("FFT Result:", fft.fft(sig))
    

# 5) Laplace
elif choice.startswith("5)"):
    st.header("5) Laplace Transform")
    f_t = st.text_input("Function f(t)", "exp(-t)")
    expr = sp.sympify(f_t)
    st.latex(rf"\mathcal{{L}}\{{{sp.latex(expr)}\}} = {sp.latex(sp.laplace_transform(expr, t, s)[0])}")
    

# 6) DCT
elif choice.startswith("6)"):
    st.header("6) DCT (Discrete Cosine Transform)")
    sig_dct = np.fromstring(st.text_input("Data to Compress", "1, 2, 3, 4"), sep=',')
    if len(sig_dct) > 0:
        st.write("DCT Result:", fft.dct(sig_dct))

# 7) Matrices, Vectors, Arrays
elif choice.startswith("7)"):
    st.header("7) Matrices, Vectors and Arrays")
    mat = np.array([[347, 769], [847, 567]]) # Using your matrix values
    st.write("Matrix A:", mat)
    st.write("Determinant:", np.linalg.det(mat))
    

# 8) Derivatives and Integration
elif choice.startswith("8)"):
    st.header("8) Derivatives and Integration")
    f_x = st.text_input("Function f(x)", "sin(x)**2")
    expr = sp.sympify(f_x)
    st.write("Derivative:", sp.diff(expr, x))
    st.write("Indefinite Integral:", sp.integrate(expr, x))
    

# 9) Log and Exponential
elif choice.startswith("9)"):
    st.header("9) Log and Exponential")
    val = st.number_input("Value", value=2.0)
    st.write(f"$\ln({val}) = {np.log(val)}$")
    st.write(f"$e^{{{val}}} = {np.exp(val)}$")
