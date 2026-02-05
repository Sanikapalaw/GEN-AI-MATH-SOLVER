import streamlit as st
import numpy as np
import scipy.fftpack as fft
import scipy.fft as dct
from scipy import integrate, stats
import sympy as sp
import google.generativeai as genai

# --- CONFIGURATION ---
# Replace with your actual API key
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="Advanced AI Math Suite", layout="wide")
st.title("📐 Advanced Mathematical AI Solver")
st.markdown("Solving everything from basic arithmetic to Laplace & Fourier Transforms.")

# --- SIDEBAR: NAVIGATION ---
menu = [
    "Arithmetic & Logs", 
    "Linear Algebra & Matrices", 
    "Calculus & Derivatives", 
    "Statistics & Probability", 
    "Signal Transforms (FFT/DCT/Laplace)",
    "AI Hard Problem Solver"
]
choice = st.sidebar.selectbox("Select Mathematical Domain", menu)

# --- HELPER: AI LOGIC ---
def get_ai_explanation(problem):
    prompt = f"Explain the step-by-step logic to solve this mathematical problem: {problem}"
    response = model.generate_content(prompt)
    return response.text

# --- 1. ARITHMETIC & LOGS ---
if choice == "Arithmetic & Logs":
    st.header("Basic Arithmetic & Exponential Functions")
    val = st.number_input("Enter a value", value=1.0)
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Natural Log (ln): {np.log(val)}")
        st.write(f"Common Log (log10): {np.log10(val)}")
    with col2:
        st.write(f"Exponential (e^x): {np.exp(val)}")
        st.write(f"Square Root: {np.sqrt(val)}")

# --- 2. LINEAR ALGEBRA & MATRICES ---
elif choice == "Linear Algebra & Matrices":
    st.header("Matrix Operations & Linear Algebra")
    st.write("Enter values for a 2x2 Matrix:")
    r1c1 = st.number_input("A[0,0]", value=1)
    r1c2 = st.number_input("A[0,1]", value=2)
    r2c1 = st.number_input("A[1,0]", value=3)
    r2c2 = st.number_input("A[1,1]", value=4)
    
    A = np.array([[r1c1, r1c2], [r2c1, r2c2]])
    st.write("Current Matrix:", A)
    
    if st.button("Calculate Properties"):
        st.write(f"Determinant: {np.linalg.det(A)}")
        st.write(f"Trace: {np.trace(A)}")
        st.write("Eigenvalues:", np.linalg.eigvals(A))

# --- 3. CALCULUS & DERIVATIVES ---
elif choice == "Calculus & Derivatives":
    st.header("Calculus: Derivatives & Integration")
    expr_input = st.text_input("Enter a function of x (e.g., x**2 + sin(x))", "x**2")
    x = sp.Symbol('x')
    expr = sp.sympify(expr_input)
    
    st.latex(f"f(x) = {sp.latex(expr)}")
    st.write("Derivative:", sp.diff(expr, x))
    st.write("Indefinite Integral:", sp.integrate(expr, x))

# --- 4. SIGNAL TRANSFORMS ---
elif choice == "Signal Transforms (FFT/DCT/Laplace)":
    st.header("FFT, DCT, and Laplace Transforms")
    data_input = st.text_input("Enter signal values separated by commas", "1, 2, 3, 4, 3, 2, 1")
    data = np.fromstring(data_input, sep=',')
    
    tab1, tab2 = st.tabs(["FFT", "DCT"])
    with tab1:
        st.write("Fast Fourier Transform (FFT):", fft.fft(data))
    with tab2:
        st.write("Discrete Cosine Transform (DCT):", dct.dct(data))

# --- 5. AI HARD PROBLEM SOLVER ---
elif choice == "AI Hard Problem Solver":
    st.header("🤖 AI Expert: Step-by-Step Logic")
    problem = st.text_area("Paste your 'Hard Problem' here (e.g., complex word problems, proofs, or Laplace derivations)")
    if st.button("Solve with Gen AI"):
        with st.spinner("Analyzing problem logic..."):
            explanation = get_ai_explanation(problem)
            st.markdown(explanation)
