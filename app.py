import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
import pandas as pd
import threading
import plotly.io as pio
import streamlit.components.v1 as components
from src.simulation_core import SimulationCore
from src.ai_agent import AIAgent
from src.visualizer import get_bioreactor_html

st.set_page_config(layout="wide", page_title="BioTwin CAD")

st.markdown("""
<style>
    .element-container { margin-bottom: 0rem; }
    div.block-container { padding-top: 1rem; padding-bottom: 0rem; }
</style>
""", unsafe_allow_html=True)

class SimManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.sim = SimulationCore(dt=0.01)
        self.ai = AIAgent()
        self.history = {'time': [], 'biomass': [], 'do': [], 'glucose': [], 'rpm': [], 'feed': []}
        self.running = False
        self.stop_thread = False
        self.thread = None
        self.sim_speed = 5
        self.controls = {
            'rpm': 200, 'air': 1.0, 'feed': 0.0, 
            'ai_active': False, 'ai_mode': 'maximize', 'target': 0.0
        }
        self.last_real_time = time.time()
        self.accumulator = 0.0

    def start_thread(self):
        if self.thread is None or not self.thread.is_alive():
            self.stop_thread = False
            self.last_real_time = time.time()
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _loop(self):
        while not self.stop_thread:
            now = time.time()
            real_dt = now - self.last_real_time
            self.last_real_time = now
            
            if self.running:
                with self.lock:
                    speed_val = self.sim_speed
                    ai_active = self.controls['ai_active']
                    ai_mode = self.controls['ai_mode']
                    target = self.controls['target']
                    man_rpm = self.controls['rpm']
                    man_air = self.controls['air']
                    man_feed = self.controls['feed']

                target_sim_rate = speed_val * 0.05 
                self.accumulator += real_dt * target_sim_rate
                steps_taken = 0
                max_steps_per_frame = 50
                
                while self.accumulator >= 0.01 and steps_taken < max_steps_per_frame:
                    if ai_active:
                        try:
                            act = self.ai.suggest_optimization(self.sim.state, mode=ai_mode, target_biomass=target)
                            self.sim.state.rpm = act['rpm']
                            self.sim.state.airflow = act['airflow']
                            self.sim.state.feed_rate = act['feed_rate']
                        except: pass
                    else:
                        self.sim.state.rpm = man_rpm
                        self.sim.state.airflow = man_air
                        self.sim.state.feed_rate = man_feed
                    
                    self.sim.step()
                    self.accumulator -= 0.01
                    steps_taken += 1
                
                if steps_taken >= max_steps_per_frame:
                    self.accumulator = 0.0
                
                if steps_taken > 0:
                    with self.lock:
                        h = self.history
                        h['time'].append(self.sim.state.time)
                        h['biomass'].append(self.sim.state.biomass)
                        h['do'].append(self.sim.state.do)
                        h['glucose'].append(self.sim.state.glucose)
                        h['rpm'].append(self.sim.state.rpm)
                        h['feed'].append(self.sim.state.feed_rate)
                        if len(h['time']) > 200: 
                            for k in h: h[k].pop(0)
            
            time.sleep(0.01)

@st.cache_resource
def get_manager():
    m = SimManager()
    m.start_thread()
    return m

manager = get_manager()

st.sidebar.title("Control Panel")
ai_active = st.sidebar.checkbox("Enable AI Agent", value=False)

if ai_active:
    st.sidebar.markdown("### AI Objective")
    goal = st.sidebar.radio("Goal", ["Maximize Biomass", "Target Setpoint"])
    ai_mode = 'maximize'
    target = 0.0
    if goal == "Target Setpoint":
        ai_mode = 'maintain'
        target = st.sidebar.number_input("Target (g/L)", 0.0, 20.0, 5.0)
    rpm_in, air_in, feed_in = 200, 1.0, 0.0
else:
    st.sidebar.markdown("### Manual Controls")
    rpm_in = st.sidebar.slider("RPM", 0, 1000, 200)
    air_in = st.sidebar.slider("Airflow (vvm)", 0.0, 5.0, 1.0)
    feed_in = st.sidebar.slider("Feed Rate (L/h)", 0.0, 0.5, 0.0)
    ai_mode = 'maximize'; target = 0.0

st.sidebar.markdown("---")
sim_speed = st.sidebar.slider("Simulation Speed", 1, 20, 5)

if st.sidebar.button("Start / Pause", type="primary"):
    with manager.lock:
        manager.running = not manager.running

if st.sidebar.button("Reset Simulation"):
    with manager.lock:
        manager.sim = SimulationCore()
        manager.history = {'time': [], 'biomass': [], 'do': [], 'glucose': [], 'rpm': [], 'feed': []}
        manager.running = False
        st.session_state.last_rpm = -1

with manager.lock:
    manager.sim_speed = sim_speed
    manager.controls.update({
        'rpm': rpm_in, 'air': air_in, 'feed': feed_in,
        'ai_active': ai_active, 'ai_mode': ai_mode, 'target': target
    })

st.title("Digital Culture")

m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with manager.lock:
    curr_time = manager.sim.state.time
    curr_rpm = manager.sim.state.rpm
    curr_air = manager.sim.state.airflow
    curr_feed = manager.sim.state.feed_rate
    curr_gluc = manager.sim.state.glucose
    curr_spatial = manager.sim.get_spatial_field()
    is_running = manager.running
    if len(manager.history['time']) > 0:
        df = pd.DataFrame({
            'time': manager.history['time'][-50:],
            'biomass': manager.history['biomass'][-50:],
            'do': manager.history['do'][-50:],
            'glucose': manager.history['glucose'][-50:],
            'feed': manager.history['feed'][-50:]
        })
    else:
        df = pd.DataFrame(columns=['time', 'biomass', 'do', 'glucose', 'feed'])

m_col1.metric("Status", "Running" if is_running else "Paused")
m_col2.metric("Time", f"{curr_time:.2f} h")
m_col3.metric("RPM", f"{curr_rpm:.0f}")
m_col4.metric("Feed", f"{curr_feed:.2f} L/h")

row1_1, row1_2 = st.columns(2)

with row1_1:
    fig_bio = go.Figure()
    if not df.empty:
        fig_bio.add_trace(go.Scatter(x=df['time'], y=df['biomass'], name='Biomass', line=dict(color='#00ff00')))
        fig_bio.add_trace(go.Scatter(x=df['time'], y=df['glucose'], name='Glucose', line=dict(color='orange', dash='dot')))
        fig_bio.add_trace(go.Scatter(x=df['time'], y=df['feed'] * 100, name='Feed Rate (x100)', line=dict(color='white', width=1)))
    fig_bio.update_layout(
        title="Biomass & Glucose Profile",
        height=280, margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark",
        xaxis_title="Time (h)", yaxis_title="Conc. (g/L)"
    )
    st.plotly_chart(fig_bio, width="stretch", key="p1")

with row1_2:
    fig_do = go.Figure()
    if not df.empty:
        fig_do.add_trace(go.Scatter(x=df['time'], y=df['do'], name='DO', line=dict(color='cyan'), fill='tozeroy'))
    fig_do.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="Critical Limit")
    fig_do.update_layout(
        title="Dissolved Oxygen (DO)",
        height=280, margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark",
        xaxis_title="Time (h)", yaxis_title="mg/L"
    )
    st.plotly_chart(fig_do, width="stretch", key="p2")

row2_1, row2_2 = st.columns(2)

def get_vol_html(spatial_field, g_conc):
    try:
        X, Y, Z = np.mgrid[0:4:5j, 0:4:5j, 0:4:5j]
        V = spatial_field
        
        v_min = V.min()
        v_max = V.max()
        
        if v_max - v_min < 1e-6:
            v_max = v_min + 0.1
            
        fig = go.Figure(data=go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=V.flatten(), 
            isomin=v_min, 
            isomax=v_max,
            opacity=0.2, 
            surface_count=10, 
            colorscale='Viridis'
        ))
        fig.update_layout(
            title=f"Glucose Gradients (Spatial) [{v_min:.2f} - {v_max:.2f}]",
            margin=dict(l=0, r=0, t=30, b=0), height=300,
            scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        )
        return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    except:
        return "<div>Error rendering Volume</div>"

with row2_1:
    if 'last_vol_time' not in st.session_state: st.session_state.last_vol_time = 0
    if 'vol_html_cache' not in st.session_state: st.session_state.vol_html_cache = get_vol_html(curr_spatial, curr_gluc)
    if time.time() - st.session_state.last_vol_time > 5:
        st.session_state.vol_html_cache = get_vol_html(curr_spatial, curr_gluc)
        st.session_state.last_vol_time = time.time()
    components.html(st.session_state.vol_html_cache, height=310)

with row2_2:
    st.markdown("### Insilico fermentor")
    if 'last_rpm' not in st.session_state: st.session_state.last_rpm = -1
    if abs(st.session_state.last_rpm - curr_rpm) > 1.0 or st.session_state.last_rpm == -1:
        st.session_state.bio_html_cache = get_bioreactor_html(curr_rpm)
        st.session_state.last_rpm = curr_rpm
    components.html(st.session_state.bio_html_cache, height=310)

if is_running:
    time.sleep(0.1) 
    st.rerun()