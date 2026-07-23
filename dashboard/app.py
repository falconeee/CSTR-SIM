import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the project root directory to the path to import src.CSTR
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.CSTR import CSTR, Fault

st.set_page_config(
    page_title="CSTR Interactive Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for premium aesthetic
st.markdown("""
<style>
    /* Make the header transparent to remove the white band at the top */
    header[data-testid="stHeader"], .stAppHeader {
        background: transparent !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #ffffff !important;
    }
    /* Native Streamlit components now use dark theme from config.toml */
    .stSidebar {
        background-color: rgba(15, 32, 39, 0.95) !important;
        backdrop-filter: blur(10px);
    }
    .stSlider > div > div > div > div {
        background-color: #00d2d3 !important;
    }
    .stSlider label {
        font-weight: 600 !important;
        color: #ffffff !important;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(4px);
    }
    div[data-testid="stMetricValue"] {
        color: #00d2d3 !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #ffffff !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #00d2d3 !important;
    }
    .g-gtitle {
        fill: #ffffff !important;
    }
    /* Style the download button */
    .stDownloadButton > button {
        background-color: #00d2d3 !important;
        color: #0f2027 !important;
        font-weight: bold !important;
    }
    /* Style the expanders to prevent white background */
    [data-testid="stExpander"] {
        background-color: rgba(15, 32, 39, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {
        color: #00d2d3 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ CSTR Real-Time Interactive Dashboard")
st.markdown("Explore the dynamics of the Continuous Stirred-Tank Reactor. Adjust the process parameters or inject predefined faults to observe how the system responds in the charts below.")

# Sidebar Controls
with st.sidebar:
    st.header("🎛️ Configuration")
    
    # Simulation Config
    st.subheader("Simulation Config")
    timehoriz = st.slider("Time Horizon (minutes)", min_value=100, max_value=2000, value=500, step=100)
    
    # Mode Selection
    mode = st.radio("Simulation Approach:", 
                    ["Interactive Sliders (Step Changes)", "Predefined Faults"])
    
    st.markdown("---")
    
    faults_to_inject = []
    delay_val = None
    
    if mode == "Interactive Sliders (Step Changes)":
        st.subheader("Process Inputs")
        feed_conc = st.slider("Feed Concentration (CA0)", min_value=10.0, max_value=30.0, value=20.0, step=0.5)
        feed_temp = st.slider("Feed Temperature (T0) [°C]", min_value=10.0, max_value=50.0, value=30.0, step=1.0)
        cw_temp = st.slider("Cooling Water Temp (T2) [°C]", min_value=10.0, max_value=30.0, value=20.0, step=1.0)
        
        st.subheader("Controller Setpoints")
        level_sp = st.slider("Level Setpoint (SP_Level) [m]", min_value=1.5, max_value=2.5, value=2.0, step=0.05)
        temp_sp = st.slider("Temperature Setpoint (SP_Temp) [°C]", min_value=70.0, max_value=90.0, value=80.0, step=0.5)
        
        delay_time = timehoriz * 0.3
        delay_val = delay_time
        
        if feed_conc != 20.0:
            faults_to_inject.append(Fault(id=14, is_sensor_fault=False, EXTENT0=feed_conc, DELAY=delay_time, TC=100.0))
        if feed_temp != 30.0:
            faults_to_inject.append(Fault(id=13, is_sensor_fault=False, EXTENT0=feed_temp, DELAY=delay_time, TC=100.0))
        if cw_temp != 20.0:
            faults_to_inject.append(Fault(id=15, is_sensor_fault=False, EXTENT0=cw_temp, DELAY=delay_time, TC=100.0))
        if level_sp != 2.0:
            faults_to_inject.append(Fault(id=19, is_sensor_fault=False, EXTENT0=level_sp, DELAY=delay_time, TC=100.0))
        if temp_sp != 80.0:
            faults_to_inject.append(Fault(id=20, is_sensor_fault=False, EXTENT0=temp_sp, DELAY=delay_time, TC=100.0))
            
    else:
        st.subheader("Predefined Faults")
        predefined_faults = {
            1: {"name": "1: NO FAULT", "params": None},
            2: {"name": "2: BLOCKAGE AT TANK OUTLET", "params": {"EXTENT0": 200, "DELAY": 2000, "TC": 0.1}},
            3: {"name": "3: BLOCKAGE IN JACKET", "params": {"EXTENT0": 200, "DELAY": 2000, "TC": 0.1}},
            4: {"name": "4: JACKET LEAK TO ENV.", "params": {"EXTENT0": 100, "DELAY": 2000, "TC": 0.01}},
            5: {"name": "5: JACKET LEAK TO TANK", "params": {"EXTENT0": 10, "DELAY": 2000, "TC": 0.001}},
            6: {"name": "6: LEAK FROM PUMP", "params": {"EXTENT0": 10, "DELAY": 2000, "TC": 0.001}},
            7: {"name": "7: LOSS OF PUMP PRESSURE", "params": {"EXTENT0": 50, "DELAY": 2000, "TC": 0.0001}},
            8: {"name": "8: SURFACE FOULING", "params": {"EXTENT0": 50, "DELAY": 2000, "TC": 0.0001}},
            9: {"name": "9: HEAT SOURCE (SINK)", "params": {"EXTENT0": 200, "DELAY": 2000, "TC": 0.01}},
            10: {"name": "10: REACTION ACT. ENERGY", "params": {"EXTENT0": 50, "DELAY": 2000, "TC": 0.0001}},
            11: {"name": "11: REACTION ACT. ENERGY", "params": {"EXTENT0": 50, "DELAY": 2000, "TC": 0.0001}},
            12: {"name": "12: ABNORMAL FEED FLOW", "params": {"EXTENT0": 1, "DELAY": 2000, "TC": 0.00001}},
            13: {"name": "13: ABNORMAL FEED TEMP", "params": {"EXTENT0": 1, "DELAY": 2000, "TC": 0.0001}},
            14: {"name": "14: ABNORMAL FEED CONC", "params": {"EXTENT0": 1, "DELAY": 2000, "TC": 0.0001}},
            15: {"name": "15: ABNORMAL CW TEMP", "params": {"EXTENT0": 10, "DELAY": 2000, "TC": 0.0001}},
            16: {"name": "16: ABNORMAL CW PRESS", "params": {"EXTENT0": 1, "DELAY": 2000, "TC": 0.0001}},
            17: {"name": "17: ABNORMAL J-EFF PRESS", "params": {"EXTENT0": 1000, "DELAY": 2000, "TC": 0.1}},
            18: {"name": "18: ABNORMAL R-EFF PRESS", "params": {"EXTENT0": 2000, "DELAY": 2000, "TC": 1}},
            19: {"name": "19: LEVEL CTRL SP", "params": {"EXTENT0": 1, "DELAY": 2000, "TC": 0.00001}},
            20: {"name": "20: TEMP CTRL SP", "params": {"EXTENT0": 0.1, "DELAY": 2000, "TC": 0.000001}},
            21: {"name": "21: CV-1 STUCK", "params": {"EXTENT0": 1, "DELAY": 2000, "TC": 0.01}},
            22: {"name": "22: CV-2 STUCK", "params": {"EXTENT0": 0.000001, "DELAY": 2000, "TC": 0.0000001}},
        }
        
        fault_options = {k: v["name"] for k, v in predefined_faults.items()}
        selected_fault_id = st.selectbox("Select Process Fault", list(fault_options.keys()), format_func=lambda x: fault_options[x])
        
        delay_time = timehoriz * 0.3
        if selected_fault_id != 1:
            params = predefined_faults[selected_fault_id]["params"]
            
            st.markdown("##### Adjust Fault Parameters")
            extent = st.number_input("Extent (Extent0)", value=float(params["EXTENT0"]), format="%f")
            # Enforce 30% rule for delay
            delay = st.number_input("Delay (Fixed at 30%)", value=float(delay_time), disabled=True, format="%f")
            tc = st.number_input("Time Constant (TC)", value=float(params["TC"]), format="%f")
            
            faults_to_inject.append(Fault(id=selected_fault_id, is_sensor_fault=False, EXTENT0=extent, DELAY=delay_time, TC=tc))
            delay_val = delay_time
        else:
            delay_val = None
            
        run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

def run_simulation(time_horizon, _faults):
    os.makedirs('data', exist_ok=True)
    
    cstr = CSTR(id=None, timehoriz=time_horizon, faults=_faults)
    
    cstr.open()
    cstr.run()
    cstr.close()
    
    data_path = './data/X_py.parquet'
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        return df
    return pd.DataFrame()

# State management
if 'sim_df' not in st.session_state:
    st.session_state.sim_df = pd.DataFrame()
if 'delay_val' not in st.session_state:
    st.session_state.delay_val = None

# Run the simulation based on mode
if mode == "Interactive Sliders (Step Changes)":
    with st.spinner("Simulating dynamics..."):
        st.session_state.sim_df = run_simulation(timehoriz, faults_to_inject)
        st.session_state.delay_val = delay_val
else:
    if run_btn:
        with st.spinner("Simulating dynamics..."):
            st.session_state.sim_df = run_simulation(timehoriz, faults_to_inject)
            st.session_state.delay_val = delay_val

df = st.session_state.sim_df
delay_val = st.session_state.delay_val

if not df.empty:
    df['Time'] = df.index
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Top metrics
    with col1:
        st.markdown("### 🌡️ Reactor Temperature")
        current_temp = df['T_1'].iloc[-1]
        st.metric("Final Temp (T_1)", f"{current_temp:.2f} °C")
    
    with col2:
        st.markdown("### 📏 Reactor Level")
        current_level = df['L'].iloc[-1]
        st.metric("Final Level (L)", f"{current_level:.2f} m")
        
    with col3:
        st.markdown("### 💾 Export Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='cstr_simulation.csv',
            mime='text/csv',
        )

    st.markdown("---")
    
    # CSTR Diagram Expander
    with st.expander("View CSTR Process Diagram"):
        img_path = os.path.join(os.path.dirname(__file__), 'cstr_diagram.png')
        if os.path.exists(img_path):
            st.image(img_path, caption="CSTR Process Diagram", use_container_width=True)
        else:
            st.warning("Diagram image not found at 'dashboard/cstr_diagram.png'")
            
    # Fault Descriptions Expander
    with st.expander("View Faults Description"):
        st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: flex-start;">

<div style="width: 48%;">

### Process Faults
| Fault | Var | Nominal | Description |
| :--- | :--- | :--- | :--- |
| 1 | - | - | NO FAULT |
| 2 | R1 | 100 | BLOCKAGE AT TANK OUTLET |
| 3 | R9 | 0 | BLOCKAGE IN JACKET |
| 4 | R8 | 1M | JACKET LEAK TO ENV. |
| 5 | R7 | 1M | JACKET LEAK TO TANK |
| 6 | R2 | 1M | LEAK FROM PUMP |
| 7 | PP | 48k | LOSS OF PUMP PRESSURE |
| 8 | UA | 1901 | SURFACE FOULING |
| 9 | QEXT | 0 | HEAT SOURCE (SINK) |
| 10 | B1 | 25k | REACTION ACT. ENERGY |
| 11 | B2 | 45k | REACTION ACT. ENERGY |
| 12 | F1 | 0.25 | ABNORMAL FEED FLOW |
| 13 | T1 | 30 | ABNORMAL FEED TEMP |
| 14 | CA0 | 20 | ABNORMAL FEED CONC |
| 15 | T3 | 20 | ABNORMAL CW TEMP |
| 16 | PCW | 56k | ABNORMAL CW PRESS |
| 17 | JEP | 0 | ABNORMAL J-EFF PRESS |
| 18 | REP | 0 | ABNORMAL R-EFF PRESS |
| 19 | SP1 | 2 | LEVEL CTRL SP |
| 20 | SP2 | 80 | TEMP CTRL SP |
| 21 | V1 | 25.3 | CV-1 STUCK |
| 22 | V2 | 40.7 | CV-2 STUCK |

</div>

<div style="width: 48%;">

### Sensor Faults
| Fault | Variable | Nominal | Description |
| :--- | :--- | :--- | :--- |
| 1 | MEAS1 | 20 | FEED CONCENTRATION |
| 2 | MEAS2 | 0.25 | FEED FLOWRATE |
| 3 | MEAS3 | 30 | FEED TEMPERATURE |
| 4 | MEAS4 | 2 | REACTOR LEVEL |
| 5 | MEAS5 | 2.85 | CONC A |
| 6 | MEAS6 | 17.11 | CONC B |
| 7 | MEAS7 | 80 | REACTOR TEMP |
| 8 | MEAS8 | 0.9 | CW FLOWRATE |
| 9 | MEAS9 | 0.25 | PRODUCT FLOWRATE |
| 10 | MEAS10 | 20 | CW TEMP |
| 11 | MEAS11 | 56250 | CW PRESSURE |
| 12 | MEAS12 | 25.3 | LEVEL CTRL OUTPUT |
| 13 | MEAS13 | 40.7 | CW FLOW CTRL OUT |
| 14 | MEAS14 | 0.9 | CW SETPOINT |

</div>

</div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive Plots
    st.subheader("📈 Dynamics Timeline")
    if delay_val is not None:
        st.markdown(f"*(Changes active at t = {delay_val} min)*")
    
    available_vars = [col for col in df.columns if col not in ['CLASS', 'Time'] and not col.startswith('Unnamed')]
    num_plots = len(available_vars)
    
    fig = make_subplots(
        rows=num_plots, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.005
    )
    
    for i, col in enumerate(available_vars):
        row = i + 1
        fig.add_trace(go.Scatter(
            x=df['Time'], 
            y=df[col], 
            mode='lines', 
            name=col, 
            line=dict(width=2)
        ), row=row, col=1)
        
        # Add shaded region for the step change
        if delay_val is not None and delay_val < df.index.max():
            fig.add_vrect(
                x0=delay_val, x1=df.index.max(),
                fillcolor="red", opacity=0.1,
                layer="below", line_width=0,
                row=row, col=1
            )
        
        # Configure y-axes
        fig.update_yaxes(
            title_text=col, 
            row=row, 
            col=1, 
            showgrid=True, 
            gridcolor="rgba(255,255,255,0.1)",
            title_font=dict(color="#f1f2f6", size=10),
            tickfont=dict(color="#a4b0be", size=10)
        )
        
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.2)",
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=150 * num_plots
    )
    
    # Configure x-axis on the bottom plot only
    fig.update_xaxes(
        title_text="Time (min)", 
        row=num_plots, 
        col=1, 
        showgrid=True, 
        gridcolor="rgba(255,255,255,0.1)",
        title_font=dict(color="#f1f2f6"),
        tickfont=dict(color="#a4b0be")
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Failed to generate simulation data.")
