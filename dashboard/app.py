import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the project root directory to the path to import src.CSTR
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.CSTR import CSTR, Fault

st.set_page_config(page_title="CSTR Simulation Dashboard", layout="wide")

st.title("CSTR-SIM")

# Sidebar - General Settings
st.sidebar.header("Parameters")
timehoriz = st.sidebar.slider("Time Horizon (timehoriz)", min_value=1000, max_value=5000, value=5000, step=100)
theta = st.sidebar.slider("EWMA Parameter (theta)", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
randseed = 42

# Sidebar - Faults
st.sidebar.header("Fault Injection")

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
selected_fault_id = st.sidebar.radio("Select Process Fault", list(fault_options.keys()), format_func=lambda x: fault_options[x])

faults = []
fault_delay_value = None

if selected_fault_id != 1:
    params = predefined_faults[selected_fault_id]["params"]
    
    st.sidebar.markdown("### Adjust Fault Parameters")
    st.sidebar.info("Default values")
    
    # We use the fault id in 'key' so Streamlit reloads the default value when switching faults
    extent = st.sidebar.number_input("Extent (Extent0)", value=float(params["EXTENT0"]), format="%f", key=f"ext_{selected_fault_id}")
    delay = st.sidebar.number_input("Delay", value=float(params["DELAY"]), format="%f", key=f"del_{selected_fault_id}")
    tc = st.sidebar.number_input("Time Constant (TC)", value=float(params["TC"]), format="%f", key=f"tc_{selected_fault_id}")
    
    fault = Fault(id=selected_fault_id, is_sensor_fault=False, EXTENT0=extent, DELAY=delay, TC=tc)
    faults.append(fault)
    fault_delay_value = delay

# Function to run cached simulation
@st.cache_data(show_spinner=False)
def run_simulation(timehoriz, theta, randseed, _faults):
    # Ensure data/ folder exists
    os.makedirs('data', exist_ok=True)
    
    data_path = './data/X_py.csv'
    
    # Run CSTR simulation
    cstr = CSTR(id=None, theta=theta, randseed=randseed, faults=_faults, timehoriz=timehoriz)
    cstr.open()
    cstr.run()
    cstr.close()
    
    # Read generated CSV data
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, sep=';')
        return df
    return pd.DataFrame()

# Initialize dataframe in session_state if it doesn't exist
if 'sim_df' not in st.session_state:
    st.session_state.sim_df = pd.DataFrame()
    st.session_state.fault_delay = None

# Add run button in the sidebar
if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner("Running simulation, please wait..."):
        # Clear cache to force re-execution
        run_simulation.clear()
        st.session_state.sim_df = run_simulation(timehoriz, theta, randseed, faults)
        st.session_state.fault_delay = fault_delay_value

df = st.session_state.sim_df

# Render Results
if not df.empty:
    st.success("Simulation completed!")
    
    # Create time axis (1 row = 1 minute in X_py.csv since it's sampled at smppermin)
    df['Time'] = df.index
    
    # Available variables for plotting, ignoring Time and CLASS
    available_vars = [col for col in df.columns if col not in ['CLASS', 'Time'] and not col.startswith('Unnamed')]
    
    num_variaveis = len(available_vars)
    fig = make_subplots(
        rows=num_variaveis, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.005
    )

    for i, col in enumerate(available_vars):
        fig.add_trace(go.Scatter(
            x=df['Time'], 
            y=df[col], 
            mode='lines', 
            name=col
        ), row=i+1, col=1)
        
        fig.update_yaxes(title_text=col, row=i+1, col=1)
        
        # Add shaded region if a fault was active during the simulation run
        if st.session_state.fault_delay is not None:
            delay_val = st.session_state.fault_delay
            max_time = df.index.max()
            if delay_val < max_time:
                fig.add_vrect(
                    x0=delay_val, x1=max_time,
                    fillcolor="red", opacity=0.15,
                    layer="below", line_width=0,
                    row=i+1, col=1
                )

    fig.update_layout(
        height=120 * num_variaveis, 
        title='CSTR Variables Dynamics',
        hovermode='x unified',
        showlegend=False,
        template="plotly_dark"
    )
    fig.update_xaxes(title_text="Time (min)", row=num_variaveis, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
        
    with st.expander("View Raw Data Table"):
        st.dataframe(df)

    with st.expander("View CSTR Process Diagram"):
        import os
        img_path = os.path.join(os.path.dirname(__file__), 'cstr_diagram.png')
        if os.path.exists(img_path):
            st.image(img_path, caption="CSTR Process Diagram", use_container_width=True)
        else:
            st.warning("Please save the image as 'cstr_diagram.png' inside the 'dashboard' folder to view it here.")

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

else:
    st.error("Simulation error. No data generated.")
