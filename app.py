import streamlit as st
import numpy as np
import scipy.fft as fft
from scipy import integrate, stats, signal
import sympy as sp

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="Universal Math Engine", layout="wide")
st.title("🧮 Universal Mathematical & Physical System")
st.markdown("Professional logic for calculus, transforms, and linear algebra.")

# --- 2. THE 9-DOMAIN NAVIGATOR ---
menu = [
    "1. Arithmetic, Logs & Exponentials",
    "2. Trig & Linear Algebra",
    "3. Calculus (Derivatives/Integrals)",
    "4. Statistics & Probability",
    "5. Signal Transforms (FFT/DCT/Laplace)",
    "6. Matrix & Vector Operations"
]
choice = st.sidebar.selectbox("Select Mathematical Domain", menu)

# --- 3. MODULE LOGIC ---

# Domain 1 & 9: Arithmetic, Logs, Exponentials
if "1." in choice:
    st.header("Arithmetic, Logs & Exponentials")
    num = st.number_input("Input Value", value=10.0)
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Common Log ($\log_{{10}}$): {np.log10(num) if num > 0 else 'Undefined'}")
        st.write(f"Natural Log ($\ln$): {np.log(num) if num > 0 else 'Undefined'}")
    with col2:
        st.write(f"Exponential ($e^num$): {np.exp(num)}")
        st.write(f"Square Root: {np.sqrt(num) if num >= 0 else 'Complex'}")

# Domain 2: Trig & Algebra
elif "2." in choice:
    st.header("Trigonometry & Basic Algebra")
    angle = st.number_input("Angle in Degrees", value=30.0)
    rad = np.radians(angle)
    st.write(f"$\sin({angle}^\circ) = {np.sin(rad)}$")
    st.write(f"$\cos({angle}^\circ) = {np.cos(rad)}$")

# Domain 3 & 8: Calculus (Physical Questions)
elif "3." in choice:
    st.header("Calculus: Derivatives & Integration")
    x = sp.Symbol('x')
    expr_input = st.text_input("Enter Function (e.g., sin(x)**2)", "sin(x)**2")
    
    try:
        expr = sp.sympify(expr_input)
        st.latex(rf"f(x) = {sp.latex(expr)}")
        
        # Domain 8: Derivatives
        st.write("#### Symbolic Derivative:")
        st.latex(sp.latex(sp.diff(expr, x)))
        
        # Domain 8: Integration (Example: Limit is 4)
        st.write("#### Definite Integration:")
        upper = st.number_input("Upper Limit (e.g., 4)", value=4.0)
        res = sp.integrate(expr, (x, 0, upper))
        st.success(f"Result (0 to {upper}): {res.evalf():.4f}")
    except Exception as e:
        st.error(f"Mathematical Error: {e}")

# Domain 3: Stats & Probability
elif "4." in choice:
    st.header("Statistics & Probability")
    data_str = st.text_input("Enter Dataset (commas)", "10, 12, 23, 23, 16, 23, 21, 16")
    data = np.fromstring(data_str, sep=',')
    if len(data) > 0:
        st.write(f"Mean: {np.mean(data)}")
        st.write(f"Median: {np.median(data)}")
        st.write(f"Mode: {stats.mode(data, keepdims=True).mode[0]}")
        st.write(f"Standard Deviation: {np.std(data)}")

# Domain 4, 5 & 6: FFT, DCT, Laplace
elif "5." in choice:
    st.header("Signal Transforms")
    sig_str = st.text_input("Enter Signal Data", "1, 2, 1, 0, 1, 2, 1")
    sig = np.fromstring(sig_str, sep=',')
    
    t_fft, t_dct, t_laplace = st.tabs(["FFT", "DCT", "Laplace"])
    with t_fft:
        st.write("Fast Fourier Transform (FFT):", fft.fft(sig))
        
    with t_dct:
        st.write("Discrete Cosine Transform (DCT):", fft.dct(sig))
    with t_laplace:
        t, s = sp.symbols('t s')
        st.write("Symbolic Laplace Transform Example ($e^{-t}$):")
        st.latex(sp.latex(sp.laplace_transform(sp.exp(-t), t, s)[0]))
        

# Domain 7: Matrix & Vectors
elif "6." in choice:
    st.header("Matrix, Vector & Array Operations")
    col1, col2 = st.columns(2)
    with col1:
        a00 = st.number_input("A[0,0]", 1.0)
        a01 = st.number_input("A[0,1]", 0.0)
    with col2:
        a10 = st.number_input("A[1,0]", 0.0)
        a11 = st.number_input("A[1,1]", 1.0)
    
    m = np.array([[a00, a01], [a10, a11]])
    if st.button("Analyze Matrix System"):
        st.write("Determinant:", np.linalg.det(m))
        st.write("Eigenvalues:", np.linalg.eigvals(m))
