import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import sympy as sp  # Required for symbolic math

# ---------- PAGE CONFIG & STYLING ----------
st.set_page_config(
    page_title="Fixed Point Iteration Solver", 
    page_icon="🔁", 
    layout="wide"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    h1 {color: #4B0082;}
    .stAlert {margin-top: 1rem;}
    </style>
""", unsafe_allow_html=True)

# ---------- LOGIC HELPER FUNCTIONS ----------
SAFE_ENV = {
    "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "sqrt": np.sqrt, "log": np.log, "log10": np.log10,
    "abs": np.abs, "pi": np.pi, "e": np.e
}

def sanitize_func_string(s: str) -> str:
    """Removes 'g(x)=' or 'x=' prefixes and whitespace."""
    if not isinstance(s, str): return ""
    s = re.sub(r'^\s*(g\s*\(\s*x\s*\)\s*=|x\s*=)\s*', '', s, flags=re.IGNORECASE)
    return s.strip()

def check_fixed_point_theorem(g_str, a, b):
    """
    Validates the Fixed Point Theorem criteria:
    1. g(x) is continuous on [a, b]
    2. g(x) maps [a, b] to [a, b] (a <= g(x) <= b)
    3. Calculates g'(x)
    4. g'(x) is continuous on (a, b)
    5. |g'(x)| < 1 for all x in (a, b)
    """
    results = {
        "g_cont": False,
        "self_map": False,
        "deriv_expr": "",
        "dg_cont": False,
        "converges": False,
        "max_deriv": 0.0
    }
    
    x = sp.symbols('x')
    
    # Remove 'np.' for SymPy parsing
    clean_str = g_str.replace("np.", "")
    
    try:
        # Create symbolic expression
        expr = sp.sympify(clean_str)
        # Create numerical function for range checks
        g_func = sp.lambdify(x, expr, modules=['numpy'])
        
        # Calculate Derivative symbolically
        deriv = sp.diff(expr, x)
        results['deriv_expr'] = sp.latex(deriv)
        deriv_func = sp.lambdify(x, deriv, modules=['numpy'])
        
        # Create dense grid for numerical checking
        # (Using numerical check is more robust than symbolic solving for general apps)
        x_vals = np.linspace(a, b, 1000)
        
        # CHECK 1: Continuity of g(x)
        y_vals = g_func(x_vals)
        if np.all(np.isfinite(y_vals)):
            results['g_cont'] = True
            
        # CHECK 2: Self-mapping (Range check)
        if results['g_cont']:
            y_min, y_max = np.min(y_vals), np.max(y_vals)
            if y_min >= a and y_max <= b:
                results['self_map'] = True
        
        # CHECK 4: Continuity of g'(x)
        dy_vals = deriv_func(x_vals)
        if np.all(np.isfinite(dy_vals)):
            results['dg_cont'] = True
            
        # CHECK 5: Convergence condition |g'(x)| < 1
        if results['dg_cont']:
            max_abs_deriv = np.max(np.abs(dy_vals))
            results['max_deriv'] = max_abs_deriv
            if max_abs_deriv < 1:
                results['converges'] = True
                
    except Exception as e:
        return None, str(e)

    return results, None

def fixed_point_method(g_str, x0, tol):
    iter_data = []
    x_old = x0
    i = 1
    
    while True:
        try:
            # Calculate next step
            x_new = eval(g_str, {"__builtins__": None}, {**SAFE_ENV, "x": x_old})
        except Exception as e:
            return None, None, f"Math Error during iteration: {e}"

        if not np.isfinite(x_new):
            return x_new, iter_data, "Divergence detected (infinite value)."

        error = abs(x_new - x_old)
        
        iter_data.append({
            "Iteration": i,
            "x_n": x_old,
            "g(x_n)": x_new,
            "Error": error
        })

        if error < tol:
            return x_new, iter_data, "Converged"

        if abs(x_new) > 1e12 or i > 1000:  # Safety breakout
            status = "Diverged (Value too large)" if abs(x_new) > 1e12 else "Max iterations reached"
            return x_new, iter_data, status

        x_old = x_new
        i += 1

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.info("💡 **Syntax Tips**\n\n- Use `np.cos(x)`, `np.exp(x)` etc.\n- Use `x**2` for exponents.")

    # Function Input
    with st.container():
        raw_func_input = st.text_input("Enter g(x):", value="np.cos(x)") # Default example
        
        # Live LaTeX Preview
        if raw_func_input:
            latex_preview = raw_func_input.replace("np.", "").replace("**", "^").replace("*", "")
            st.caption("Math Preview:")
            st.latex(f"g(x) = {latex_preview}")

        x0 = st.number_input("Initial Guess ($x_0$):", value=0.5, step=0.1)
        tolerance = st.number_input("Tolerance ($\epsilon$):", value=1e-6, format="%.8f")

    st.divider()
    
    # Theorem Validation Inputs
    st.markdown("### 📋 Theorem Interval Check")
    st.caption("Define interval $[a, b]$ to check convergence criteria.")
    col_a, col_b = st.columns(2)
    with col_a:
        val_a = st.number_input("a:", value=0.0, step=0.1)
    with col_b:
        val_b = st.number_input("b:", value=1.0, step=0.1)

    st.markdown("---")
    compute_btn = st.button("🚀 Compute Root", type="primary", use_container_width=True)

# ---------- HEADER ----------
st.title("🔁 Fixed Point Iteration Solver")
st.markdown("Find the root of $x = g(x)$ and validate using the Fixed Point Theorem.")

# ---------- MAIN EXECUTION ----------
if compute_btn:
    func_input = sanitize_func_string(raw_func_input)
    
    # ---------------------------------------------------------
    # 1. FIXED POINT THEOREM VALIDATION (SymPy + Check)
    # ---------------------------------------------------------
    st.subheader("1. Fixed Point Theorem Analysis")
    
    if val_a >= val_b:
        st.error("Interval Error: 'a' must be less than 'b'.")
    else:
        results, error_msg = check_fixed_point_theorem(func_input, val_a, val_b)

        if error_msg:
            st.error(f"Could not validate theorem: {error_msg}")
        else:
            # Helper for displaying status rows with icons
            def status_row(label, passed, detail=""):
                icon = "✅" if passed else "❌"
                color = "green" if passed else "red"
                st.markdown(f"**{icon} {label}** : <span style='color:{color}'>{str(passed).upper()}</span> {detail}", unsafe_allow_html=True)

            with st.expander("Show Theorem Criteria Details", expanded=True):
                # Criterion 1
                status_row(f"1. $g(x)$ is continuous on [{val_a}, {val_b}]", results['g_cont'])
                
                # Criterion 2
                status_row(f"2. Range $a \le g(x) \le b$", results['self_map'])
                
                # Criterion 3 (Calculation Display)
                st.markdown(f"**3. Calculated Derivative $g'(x)$:**")
                st.latex(results['deriv_expr'])
                
                # Criterion 4
                status_row(f"4. $g'(x)$ is continuous on ({val_a}, {val_b})", results['dg_cont'])
                
                # Criterion 5
                detail_msg = f"(Max $|g'(x)| \\approx {results['max_deriv']:.4f}$)"
                status_row(f"5. Convergence Check ($|g'(x)| < 1$)", results['converges'], detail_msg)

            # Soft Warning if convergence not guaranteed
            if not results['converges']:
                st.warning("⚠️ Warning: The function does not satisfy $|g'(x)| < 1$ on this interval. Iteration may diverge.")
            else:
                st.success("✨ Theory guarantees convergence on this interval!")

    # ---------------------------------------------------------
    # 2. NUMERICAL COMPUTATION
    # ---------------------------------------------------------
    st.subheader("2. Numerical Iteration")
    
    # Validate Python Syntax before running
    try:
        eval(func_input, {"__builtins__": None}, {**SAFE_ENV, "x": x0})
    except Exception as e:
        st.error(f"❌ Python Syntax Error: {e}")
        st.stop()

    # Run Method
    root, data, status = fixed_point_method(func_input, x0, tolerance)
    
    if data is None: # Critical math error
        st.error(status)
        st.stop()

    df = pd.DataFrame(data)
    
    # Top Level Metrics
    m1, m2, m3, m4 = st.columns(4)
    is_success = status == "Converged"
    color = "normal" if is_success else "off"
    
    m1.metric("Status", status, delta="Success" if is_success else "Check", delta_color=color)
    m2.metric("Approximate Root", f"{root:.6f}")
    m3.metric("Iterations", len(df))
    m4.metric("Final Error", f"{df.iloc[-1]['Error']:.2e}" if not df.empty else "N/A")

    # 3. Visualization & Data Tabs
    tab_plot, tab_data, tab_summary = st.tabs(["📈 Cobweb Plot", "🔢 Iteration Table", "📝 Summary"])

    with tab_plot:
        # Prepare Plot Data
        if not df.empty:
            xs_path = [row['x_n'] for row in data] + [root]
        else:
            xs_path = [x0]
            
        # Dynamic range for plotting
        margin = 1.0
        plot_min = min(xs_path + [val_a]) - margin
        plot_max = max(xs_path + [val_b]) + margin
        x_range = np.linspace(plot_min, plot_max, 200)
        
        try:
            y_func = [eval(func_input, {"__builtins__": None}, {**SAFE_ENV, "x": v}) for v in x_range]
        except:
            st.warning("Could not render full function curve due to domain errors.")
            y_func = x_range # Fallback

        fig = go.Figure()

        # A. g(x) curve
        fig.add_trace(go.Scatter(x=x_range, y=y_func, mode='lines', name=f'g(x)', line=dict(color='blue')))
        
        # B. y = x line
        fig.add_trace(go.Scatter(x=x_range, y=x_range, mode='lines', name='y = x', line=dict(color='gray', dash='dash')))

        # C. Cobweb Path
        if not df.empty:
            cobweb_x = []
            cobweb_y = []
            curr_x = x0
            cobweb_x.append(curr_x)
            cobweb_y.append(0) 
            
            for row in data:
                # Vertical to curve
                cobweb_x.append(curr_x)
                cobweb_y.append(row['g(x_n)'])
                # Horizontal to line
                cobweb_x.append(row['g(x_n)'])
                cobweb_y.append(row['g(x_n)'])
                curr_x = row['g(x_n)']

            fig.add_trace(go.Scatter(
                x=cobweb_x, y=cobweb_y, 
                mode='lines+markers', 
                name='Iteration Path',
                line=dict(color='red', width=1),
                marker=dict(size=4)
            ))

        # D. Highlight Root
        fig.add_trace(go.Scatter(
            x=[root], y=[root],
            mode='markers',
            marker=dict(color='green', size=12, symbol='star'),
            name='Final Point'
        ))

        fig.update_layout(
            title="Cobweb Diagram",
            xaxis_title="x",
            yaxis_title="y",
            height=600,
            hovermode="closest"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with tab_data:
        st.dataframe(
            df.style.format({
                "x_n": "{:.8f}",
                "g(x_n)": "{:.8f}",
                "Error": "{:.2e}"
            }),
            use_container_width=True,
            height=400
        )
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "iteration_data.csv", "text/csv")

    with tab_summary:
        st.markdown(f"""
        ### Result Summary
        The method executed **{len(df)}** iterations starting from $x_0 = {x0}$.
        
        - **Function:** $g(x) = {raw_func_input}$
        - **Found Root:** {root:.8f}
        - **Convergence Status:** {status}
        
        The graph in the first tab shows the "Cobweb" path. If the path spirals inward, it is converging. If it spirals outward, it is diverging.
        """)

else:
    # Landing visual
    st.markdown("""
    <div style='text-align: center; margin-top: 50px; color: gray;'>
        <h3>👈 Enter parameters in the sidebar to start</h3>
        <p>This tool helps you visualize Fixed Point Iteration and validates the Convergence Theorem.</p>
    </div>
    """, unsafe_allow_html=True)