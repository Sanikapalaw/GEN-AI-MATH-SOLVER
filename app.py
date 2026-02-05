import streamlit as st
import numpy as np
import sympy as sp
from scipy import integrate, fft, stats
import google.generativeai as genai

# --- 1. SYSTEM CONFIGURATION ---
# Use your Google AI Studio Key here
genai.configure(api_key="YOUR_API_KEY")
ai_model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="GenAI Math System", layout="wide")
st.title("🏛️ Advanced Mathematical & Physical System")
st.markdown("Solving Hard Problems with AI Logic and Symbolic Calculus.")

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.header("Mathematical Domains")
app_mode = st.sidebar.selectbox("Choose a Module", [
    "Physical Integration & Calculus",
    "Linear Algebra (Matrices/Vectors)",
    "Signal Processing (FFT/DCT/Laplace)",
    "Statistics & Probability",
    "GenAI Hard Problem Solver"
])

# --- 3. MODULE: PHYSICAL INTEGRATION & CALCULUS ---
if app_mode == "Physical Integration & Calculus":
    st.header("📐 Physical System Solver: Calculus")
    st.info("Example: Integration of sine square from 0 to 4")
    
    col1, col2 = st.columns(2)
    with col1:
        func_str = st.text_input("Enter Function f(x)", "sin(x)**2")
        lower_limit = st.number_input("Lower Limit (a)", value=0.0)
        upper_limit = st.number_input("Upper Limit (b)", value=4.0)
    
    # Symbolic Computation using SymPy
    x = sp.Symbol('x')
    try:
        f_x = sp.sympify(func_str)
        indefinite = sp.integrate(f_x, x)
        definite = sp.integrate(f_x, (x, lower_limit, upper_limit))
        
        with col2:
            st.write("### Symbolic Result")
            st.latex(rf"\int_{{{lower_limit}}}^{{{upper_limit}}} {sp.latex(f_x)} \, dx")
            st.success(f"Numerical Result: {float(definite.evalf()):.4f}")
            
        # AI Step-by-Step Logic
        if st.button("Explain the Physical Steps"):
            prompt = f"Explain the physical logic and step-by-step derivation for integrating {func_str} between {lower_limit} and {upper_limit}."
            response = ai_model.generate_content(prompt)
            st.markdown(response.text)
            
    except Exception as e:
        st.error(f"Error in expression: {e}")

# --- 4. MODULE: SIGNAL PROCESSING ---
elif app_mode == "Signal Processing (FFT/DCT/Laplace)":
    st.header("📡 Signal Transforms")
    signal_input = st.text_input("Enter Signal Array (comma separated)", "0, 1, 0, -1, 0, 1")
    data = np.fromstring(signal_input, sep=',')
    
    t1, t2 = st.tabs(["Fast Fourier (FFT)", "Discrete Cosine (DCT)"])
    with t1:
        st.write("Frequency Domain representation:")
        st.write(fft.fft(data))
    with t2:
        st.write("Compressed Signal (DCT):")
        st.write(fft.dct(data))

# --- 5. MODULE: LINEAR ALGEBRA ---
elif app_mode == "Linear Algebra (Matrices/Vectors)":
    st.header("📋 Matrix System")
    st.write("Define a 2x2 Matrix A")
    a00 = st.number_input("A[0,0]", value=1)
    a01 = st.number_input("A[0,1]", value=0)
    a10 = st.number_input("A[1,0]", value=0)
    a11 = st.number_input("A[1,1]", value=1)
    
    matrix = np.array([[a00, a01], [a10, a11]])
    st.write("Matrix A:", matrix)
    
    if st.button("Analyze Matrix"):
        det = np.linalg.det(matrix)
        st.write(f"Determinant: {det}")
        if det != 0:
            st.write("Inverse:", np.linalg.inv(matrix))
        else:
            st.warning("Matrix is singular (no inverse).")

# --- 6. MODULE: GENAI HARD PROBLEM SOLVER ---
elif app_mode == "GenAI Hard Problem Solver":
    st.header("🧠 Logic & Proof Solver")
    user_query = st.text_area("Paste your hard problem, word problem, or proof here:")
    if st.button("Solve with AI Logic"):
        with st.spinner("AI is thinking..."):
            res = ai_model.generate_content(user_query)
            st.markdown(res.text)
