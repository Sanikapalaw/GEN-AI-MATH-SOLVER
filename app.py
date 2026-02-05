import streamlit as st
import numpy as np
import scipy.fft as fft
from scipy import integrate, stats, signal
import sympy as sp
import google.generativeai as genai

# --- 1. SETUP ---
# Replace with your Gemini API Key
genai.configure(api_key="YOUR_GEMINI_API_KEY")
ai_model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="Universal Math AI", layout="wide")
st.title("🧮 GenAI Mathematical & Physical System")

# --- 2. THE 9-DOMAIN NAVIGATOR ---
menu = [
    "1. Basic Arithmetic & Logs",
    "2. Trig & Linear Algebra",
    "3. Calculus & Stats",
    "4. Signal Transforms (FFT/DCT/Laplace)",
    "5. Matrix & Vector Lab"
]
choice = st.sidebar.selectbox("Select Mathematical Module", menu)

# --- 3. MODULE LOGIC ---

if "1." in choice:
    st.header("Arithmetic, Logs & Exponentials")
    num = st.number_input("Input Value", value=10.0)
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Common Log ($\log_{{10}}$): {np.log10(num)}")
        st.write(f"Natural Log ($\ln$): {np.log(num)}")
    with col2:
        st.write(f"Exponential ($e^x$): {np.exp(num)}")
        st.write(f"Square Root: {np.sqrt(num)}")

elif "2." in choice:
    st.header("Trigonometry & Basic Algebra")
    angle = st.number_input("Angle in Degrees", value=30.0)
    rad = np.radians(angle)
    st.write(f"$\sin({angle}^\circ) = {np.sin(rad)}$")
    st.write(f"$\cos({angle}^\circ) = {np.cos(rad)}$")
    

[Image of unit circle with sine and cosine values]


elif "3." in choice:
    st.header("Calculus, Statistics & Probability")
    x = sp.Symbol('x')
    expr_input = st.text_input("Enter Function (e.g., x**2 + sin(x))", "sin(x)**2")
    
    # Symbolic Calculus
    expr = sp.sympify(expr_input)
    st.latex(rf"f(x) = {sp.latex(expr)}")
    st.write("Derivative:", sp.diff(expr, x))
    
    # Physical Question Integration (Example: Limit is 4)
    upper = st.number_input("Integration Upper Limit", value=4.0)
    res = sp.integrate(expr, (x, 0, upper))
    st.success(f"Definite Integral (0 to {upper}): {res.evalf():.4f}")
    

[Image of definite integral area under a curve]


elif "4." in choice:
    st.header("Advanced Transforms (FFT, DCT, Laplace)")
    sig_str = st.text_input("Enter Signal Data (commas)", "1, 2, 1, 0, 1, 2, 1")
    sig = np.fromstring(sig_str, sep=',')
    
    tab1, tab2, tab3 = st.tabs(["FFT", "DCT", "Laplace"])
    with tab1:
        st.write("Fast Fourier Transform:", fft.fft(sig))
        
    with tab2:
        st.write("Discrete Cosine Transform:", fft.dct(sig))
    with tab3:
        # Laplace is symbolic
        t, s = sp.symbols('t s')
        f_t = sp.exp(-t) * sp.sin(t)
        st.write("Example Laplace Transform of $e^{-t}\sin(t)$:")
        st.latex(sp.latex(sp.laplace_transform(f_t, t, s)[0]))

elif "5." in choice:
    st.header("Matrix, Vector & Array Operations")
    st.write("Enter a 2x2 Matrix")
    m = np.array([[st.number_input("A[0,0]", 1), st.number_input("A[0,1]", 0)],
                  [st.number_input("A[1,0]", 0), st.number_input("A[1,1]", 1)]])
    
    if st.button("Calculate Matrix Properties"):
        st.write("Determinant:", np.linalg.det(m))
        st.write("Eigenvalues:", np.linalg.eigvals(m))
        

# --- 4. THE 'HARD PROBLEM' AI SOLVER ---
st.divider()
st.header("🤖 GenAI Hard Problem Solver")
problem = st.text_area("Paste a complex physical or mathematical proof here:")
if st.button("Solve with AI Logic"):
    with st.spinner("Analyzing..."):
        response = ai_model.generate_content(f"Solve this step-by-step with mathematical rigor: {problem}")
        st.markdown(response.text)
