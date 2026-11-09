"""
app.py - Interactive SVPWM dashboard (three-phase two-level inverter, 180-degree conduction).

Run with:   streamlit run app.py
"""
import dataclasses

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import losses as ls
import svpwm as sv

st.set_page_config(page_title="SVPWM Inverter Lab", page_icon="⚡", layout="wide")

# Guard against mixing old and new files (Python keeps an already-imported svpwm.py until the server restarts).
import inspect  # noqa: E402

if "scheme" not in inspect.signature(sv.simulate).parameters:
    st.error("svpwm.py is out of date. Replace svpwm.py with the latest version (same folder as app.py), "
             "then stop the server with Ctrl+C and start it again: python -m streamlit run app.py")
    st.stop()

CA, CB, CC = "#d62728", "#2ca02c", "#1f77b4"
PH_COL = [CA, CB, CC]


def show(fig):
    st.plotly_chart(fig, width="stretch")


# ------------------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------------------
st.sidebar.title("⚡ Inverter settings")
mode = st.sidebar.radio(
    "Modulation / conduction mode",
    ["SVPWM (180° complementary gating)", "Six-step (180° conduction)"],
)
sixstep = mode.startswith("Six")

vdc = st.sidebar.number_input("DC-link voltage Vdc [V]", 50.0, 2000.0, 400.0, 10.0)
f1 = st.sidebar.number_input("Fundamental frequency f₁ [Hz]", 1.0, 400.0, 50.0, 1.0)
fsw = st.sidebar.number_input("Switching frequency fsw [Hz]", 300.0, 50000.0, 3000.0, 100.0,
                              disabled=sixstep,
                              help="Snapped to a multiple of 6·f₁ so the waveform is exactly periodic.")
m = st.sidebar.slider("Modulation index m", 0.0, float(sv.M_HEX_VERTEX), 0.90, 0.01, disabled=sixstep,
                      help="m = |Vref| / (Vdc/√3).  m ≤ 1: linear.  1 < m ≤ 1.1547: overmodulation.")
if sixstep:
    st.sidebar.caption("Six-step: m is fixed by the hardware (m = 2√3/π ≈ 1.103).")

st.sidebar.subheader("Load (balanced star R-L)")
R = st.sidebar.number_input("R [Ω]", 0.1, 500.0, 10.0, 0.5)
L = st.sidebar.number_input("L [mH]", 0.1, 1000.0, 30.0, 1.0) * 1e-3

st.sidebar.subheader("Display / analysis")
cycles = st.sidebar.slider("Cycles to display", 1, 4, 2)
npc = st.sidebar.select_slider("Samples per switching period", [20, 40, 60, 100, 200], value=40)
hmax = st.sidebar.slider("Spectrum / THD limit (harmonic order)", 20, 400, 100, 10)
theta_deg = st.sidebar.slider("Inspect reference angle θ [deg]", 0.0, 359.0, 20.0, 1.0)


@st.cache_data(show_spinner=False)
def run(sixstep, vdc, f1, fsw, m, npc, R, L, cycles):
    return sv.simulate(sixstep, vdc, f1, fsw, m, npc, R, L, cycles)


@st.cache_data(show_spinner=False)
def sweep(vdc, f1, fsw, hmax):
    ms = np.linspace(0.02, sv.M_HEX_VERTEX, 30)
    v1, thd_v = [], []
    for mm in ms:
        r = sv.simulate(False, vdc, f1, fsw, mm, npc=20, cycles=1, load=False)
        x = r["vll"][0][-r["spc"]:]
        orders, mag = sv.harmonic_spectrum(x, hmax)
        v1.append(mag[1] / vdc)
        thd_v.append(100 * sv.thd(mag))
    r = sv.simulate(True, vdc, f1, fsw, 1.0, npc=20, cycles=1, load=False)
    _, magx = sv.harmonic_spectrum(r["vll"][0][-r["spc"]:], hmax)
    return ms, np.array(v1), np.array(thd_v), magx[1] / vdc, 100 * sv.thd(magx)


res = run(sixstep, vdc, f1, fsw, m, npc, R, L, cycles)
spc, mf, npc_eff, K = res["spc"], res["mf"], res["npc"], res["K"]
t_ms = res["t"] * 1e3
hmax = min(hmax, spc // 2 - 1)

# ------------------------------------------------------------------------------------------
# Analysis of the last cycle
# ------------------------------------------------------------------------------------------
vll_c = res["vll"][0][-spc:]
vph_c = res["vph"][0][-spc:]
ia_c = res["i"][0][-spc:]

c1_ll = sv.fundamental_phasor(vll_c)
c1_ph = sv.fundamental_phasor(vph_c)
c1_i = sv.fundamental_phasor(ia_c)
ord_v, mag_v = sv.harmonic_spectrum(vll_c, hmax)
ord_i, mag_i = sv.harmonic_spectrum(ia_c, hmax)
thd_v, thd_i = 100 * sv.thd(mag_v), 100 * sv.thd(mag_i)
m_eff = abs(c1_ph) / (vdc / sv.SQRT3)


def fund_wave(c1):
    return np.real(c1 * np.exp(1j * 2 * np.pi * f1 * res["t"]))


# ------------------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------------------
st.title("Space Vector PWM — three-phase two-level inverter (180° conduction)")
st.caption(
    "Ideal switches, no dead time. Each leg is driven complementarily (S4 = S̄1, S6 = S̄3, S2 = S̄5), so every "
    "switch conducts 180° on average per cycle. Six-step is the limiting case where each switch conducts exactly 180°."
)

c = st.columns(6)
c[0].metric("Effective fsw", f"{res['fsw_eff']:.0f} Hz", f"mf = {mf}" if not sixstep else "no PWM")
c[1].metric("m (measured)", f"{m_eff:.3f}", "linear" if m_eff <= 1.0 + 1e-3 else "overmodulation")
c[2].metric("V_LL fundamental (rms)", f"{abs(c1_ll)/np.sqrt(2):.1f} V", f"{abs(c1_ll)/vdc:.3f} Vdc peak")
c[3].metric("V_phase fundamental (peak)", f"{abs(c1_ph):.1f} V")
c[4].metric("THD  V_LL", f"{thd_v:.1f} %", f"up to h={hmax}", delta_color="off")
c[5].metric("THD  i_a", f"{thd_i:.2f} %", f"I₁ = {abs(c1_i):.2f} A pk", delta_color="off")

tabs = st.tabs(["Space vector", "Gating & switching", "Waveforms", "Spectrum", "Linearity sweep", "Losses", "Theory"])

# ==========================================================================================
# TAB 1 - space vector diagram
# ==========================================================================================
with tabs[0]:
    left, right = st.columns([3, 2])
    R0 = 2 * vdc / 3
    th = np.deg2rad(theta_deg)

    fig = go.Figure()
    ang = np.deg2rad(np.arange(0, 361, 60))
    hx, hy = R0 * np.cos(ang), R0 * np.sin(ang)

    if sixstep:
        n_act = int(sv.six_step_active_vector(th))
        sec, t1, t2, t0 = n_act, 1.0, 0.0, 0.0
    else:
        sec_a, thp, t1a, t2a, t0a = sv.dwell_times(m, th)
        sec, t1, t2, t0 = int(sec_a), float(t1a), float(t2a), float(t0a)
    n1, n2 = sec, sec % 6 + 1
    vn1 = R0 * np.exp(1j * np.deg2rad(60 * (n1 - 1)))
    vn2 = R0 * np.exp(1j * np.deg2rad(60 * (n2 - 1)))

    if not sixstep:  # highlight sector
        fig.add_trace(go.Scatter(x=[0, vn1.real, vn2.real, 0], y=[0, vn1.imag, vn2.imag, 0], fill="toself",
                                 fillcolor="rgba(255,200,0,0.18)", mode="lines", line=dict(width=0),
                                 name=f"Sector {sec}", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines", line=dict(color="#555", width=2), name="Hexagon",
                             hoverinfo="skip"))
    for k in range(6):
        fig.add_trace(go.Scatter(x=[0, hx[k]], y=[0, hy[k]], mode="lines", line=dict(color="#bbb", dash="dot"),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=1.18 * hx[:6], y=1.18 * hy[:6], mode="text",
        text=[f"V{k+1}<br>({sv.state_str(k+1)})" for k in range(6)], showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hx[:6], y=hy[:6], mode="markers", marker=dict(size=9, color="#555"),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", text=["V0 / V7"], textposition="bottom center",
                             marker=dict(size=9, color="black", symbol="x"), showlegend=False, hoverinfo="skip"))
    # inscribed circle + reference circle
    cc = np.linspace(0, 2 * np.pi, 361)
    fig.add_trace(go.Scatter(x=(vdc / sv.SQRT3) * np.cos(cc), y=(vdc / sv.SQRT3) * np.sin(cc), mode="lines",
                             line=dict(color="#999", dash="dash"), name="Linear limit  Vdc/√3"))
    if not sixstep:
        fig.add_trace(go.Scatter(x=(m * vdc / sv.SQRT3) * np.cos(cc), y=(m * vdc / sv.SQRT3) * np.sin(cc),
                                 mode="lines", line=dict(color="#1f77b4", dash="dot"), name="|Vref| circle"))
    fig.add_trace(go.Scatter(x=res["v_avg"][0], y=res["v_avg"][1], mode="markers",
                             marker=dict(size=3, color="rgba(214,39,40,0.5)"),
                             name="Averaged output (per Ts)"))
    # components of Vref
    c1v, c2v = t1 * vn1, t2 * vn2
    vref = c1v + c2v
    if not sixstep:
        fig.add_trace(go.Scatter(x=[0, c1v.real, vref.real], y=[0, c1v.imag, vref.imag], mode="lines+markers",
                                 line=dict(color="#2ca02c", width=2), marker=dict(size=5),
                                 name=f"T1·V{n1} + T2·V{n2}"))
    fig.add_trace(go.Scatter(x=[0, vref.real], y=[0, vref.imag], mode="lines+markers",
                             line=dict(color="#d62728", width=4), marker=dict(size=[1, 12], symbol="arrow",
                                                                               angleref="previous"),
                             name="Vref (selected θ)" if not sixstep else "Active vector"))
    fig.update_layout(height=640, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(scaleanchor="y", scaleratio=1, zeroline=True, title="Vα [V]"),
                      yaxis=dict(zeroline=True, title="Vβ [V]"),
                      legend=dict(orientation="h", y=-0.08))
    with left:
        show(fig)

    with right:
        st.subheader("Dwell times at selected θ")
        if sixstep:
            st.write(f"θ = **{theta_deg:.0f}°** → active vector **V{n1}** ({sv.state_str(n1)}), held for 60° (T₁/6) "
                     "per step. The zero vectors are never used.")
            st.table({"Quantity": ["Active vector", "Hold time"], "Value": [f"V{n1} ({sv.state_str(n1)})", f"{1e3/(6*f1):.3f} ms"]})
        else:
            Ts = res["Ts"]
            st.write(f"θ = **{theta_deg:.0f}°**, sector **{sec}**, θ′ = {theta_deg - 60*(sec-1):.1f}°"
                     + ("  ·  ⚠ overmodulation (T₀ = 0)" if t0 < 1e-9 else ""))
            st.table({
                "Time": ["T1 (V%d)" % n1, "T2 (V%d)" % n2, "T0 (V0+V7)"],
                "Fraction of Ts": [f"{t1:.4f}", f"{t2:.4f}", f"{t0:.4f}"],
                "Microseconds": [f"{t1*Ts*1e6:.1f}", f"{t2*Ts*1e6:.1f}", f"{t0*Ts*1e6:.1f}"],
            })
            seq = sv.switching_sequence(sec, t1, t2, t0)
            st.markdown("**Seven-segment sequence (one Ts):**")
            st.code("  →  ".join(f"{name}[{s}]" for s, name, _ in seq), language="text")
            st.table({"Segment": [f"{nm} [{s}]" for s, nm, _ in seq],
                      "Time [µs]": [f"{tt*Ts*1e6:.1f}" for _, _, tt in seq]})
            st.latex(r"T_1=m\,T_s\sin(60^\circ-\theta'),\;\; T_2=m\,T_s\sin\theta',\;\; T_0=T_s-T_1-T_2")

# ==========================================================================================
# TAB 2 - gating
# ==========================================================================================
with tabs[1]:
    if sixstep:
        st.write("Six-step: each switch conducts 180°, legs are shifted by 120°, and every 60° one switch changes state.")
        i0, i1 = 0, spc
    else:
        win = st.slider("Switching periods shown", 2, 8, 4)
        kc = int(theta_deg / 360 * mf) % mf
        k0 = int(np.clip(kc - win // 2, 0, K - win))
        i0, i1 = k0 * npc_eff, (k0 + win) * npc_eff

    rows = 6 if sixstep else 7
    heights = [1] * 6 if sixstep else [2.2] + [1] * 6
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.025, row_heights=heights)
    r = 1
    tt = t_ms[i0:i1]
    if not sixstep:
        fig.add_trace(go.Scatter(x=tt, y=res["carrier"][i0:i1], name="carrier", line=dict(color="#888", width=1.5)),
                      row=1, col=1)
        for j, nm in enumerate("ABC"):
            fig.add_trace(go.Scatter(x=tt, y=res["duty_rep"][j][i0:i1], name=f"d{nm}", line=dict(color=PH_COL[j], width=2)),
                          row=1, col=1)
        fig.update_yaxes(title_text="duty / carrier", row=1, col=1)
        r = 2
    S = res["S"]
    sig = [("S1 (A+)", S[0], CA), ("S4 (A−)", 1 - S[0], CA),
           ("S3 (B+)", S[1], CB), ("S6 (B−)", 1 - S[1], CB),
           ("S5 (C+)", S[2], CC), ("S2 (C−)", 1 - S[2], CC)]
    for j, (nm, y, col) in enumerate(sig):
        fig.add_trace(go.Scatter(x=tt, y=y[i0:i1], name=nm, line=dict(color=col, width=1.8, shape="hv"),
                                 showlegend=False), row=r + j, col=1)
        fig.update_yaxes(title_text=nm, range=[-0.2, 1.2], tickvals=[0, 1], row=r + j, col=1)
    if not sixstep:
        fig.add_vrect(x0=t_ms[kc * npc_eff], x1=t_ms[min((kc + 1) * npc_eff, len(t_ms)) - 1], fillcolor="orange",
                      opacity=0.15, line_width=0, row="all", col=1)
    fig.update_xaxes(title_text="time [ms]", row=rows, col=1)
    fig.update_layout(height=820, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.04))
    show(fig)
    if not sixstep:
        st.caption("Shaded: switching period containing the angle selected in the sidebar. A leg's upper switch is on "
                   "while its duty exceeds the symmetric triangular carrier, which reproduces the 0-Vn-Vn+1-7-Vn+1-Vn-0 pattern.")

        fig = go.Figure()
        t_one = t_ms[:spc]
        step = npc_eff
        tk = (np.arange(mf) + 0.5) * res["Ts"] * 1e3
        for j, nm in enumerate("ABC"):
            fig.add_trace(go.Scatter(x=tk, y=res["duty_k"][j][:mf], name=f"d{nm}", mode="lines",
                                     line=dict(color=PH_COL[j])))
        fig.update_layout(height=320, title="Reference (duty) signals over one cycle — note the min/max common-mode injection",
                          xaxis_title="time [ms]", yaxis_title="duty", margin=dict(l=10, r=10, t=40, b=10))
        show(fig)

# ==========================================================================================
# TAB 3 - waveforms
# ==========================================================================================
with tabs[2]:
    # Order requested: v_an, v_bn, v_cn, v_ab  (then load currents and pole voltages as extras)
    c1_vb = sv.fundamental_phasor(res["vph"][1][-spc:])
    c1_vc = sv.fundamental_phasor(res["vph"][2][-spc:])
    panels = [
        ("Phase voltage v_an (+ fundamental)", res["vph"][0], c1_ph, PH_COL[0], "v_an"),
        ("Phase voltage v_bn (+ fundamental)", res["vph"][1], c1_vb, PH_COL[1], "v_bn"),
        ("Phase voltage v_cn (+ fundamental)", res["vph"][2], c1_vc, PH_COL[2], "v_cn"),
        ("Line voltage v_ab (+ fundamental)", res["vll"][0], c1_ll, "#9467bd", "v_ab"),
    ]
    titles = [p[0] for p in panels] + ["Load currents i_a, i_b, i_c",
                                       "Pole voltages (leg to negative DC rail)"]
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.035, subplot_titles=titles)
    for r_, (_, y, c1x, col, nm) in enumerate(panels, start=1):
        fig.add_trace(go.Scatter(x=t_ms, y=y, name=nm, line=dict(color=col, width=1, shape="hv")), row=r_, col=1)
        fig.add_trace(go.Scatter(x=t_ms, y=fund_wave(c1x), name=f"{nm} fund.", line=dict(color="orange", width=3)),
                      row=r_, col=1)
        fig.update_yaxes(title_text="V", row=r_, col=1)
    for j, nm in enumerate("abc"):
        fig.add_trace(go.Scatter(x=t_ms, y=res["i"][j], name=f"i_{nm}", line=dict(color=PH_COL[j], width=2)), row=5, col=1)
    fig.update_yaxes(title_text="A", row=5, col=1)
    for j, nm in enumerate("ABC"):
        fig.add_trace(go.Scatter(x=t_ms, y=res["vpole"][j], name=f"v_{nm}0",
                                 line=dict(color=PH_COL[j], width=1, shape="hv")), row=6, col=1)
    fig.update_yaxes(title_text="V", row=6, col=1)
    fig.update_xaxes(title_text="time [ms]", row=6, col=1)
    fig.update_layout(height=1350, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.03))
    show(fig)

# ==========================================================================================
# TAB 4 - spectrum
# ==========================================================================================
with tabs[3]:
    cols = st.columns(2)
    rel = st.toggle("Show as % of fundamental (otherwise absolute)", value=True)
    for col, orders, mag, name, unit, thdv in (
        (cols[0], ord_v, mag_v, "Line voltage v_AB", "V", thd_v),
        (cols[1], ord_i, mag_i, "Phase current i_a", "A", thd_i),
    ):
        y = 100 * mag[1:] / max(mag[1], 1e-12) if rel else mag[1:]
        fig = go.Figure(go.Bar(x=orders[1:], y=y, marker_color="#1f77b4"))
        fig.update_layout(title=f"{name} spectrum — THD = {thdv:.2f} % (h ≤ {hmax})", xaxis_title="harmonic order (×f₁)",
                          yaxis_title=("% of fundamental" if rel else unit), height=430,
                          margin=dict(l=10, r=10, t=50, b=10))
        if rel:
            fig.update_yaxes(type="log", range=[-2, 2.3])
        with col:
            show(fig)
    if sixstep:
        st.info("Six-step line voltage contains only harmonics of order 6k ± 1 excluding triplens (5, 7, 11, 13 …) with "
                "amplitude 1/h of the fundamental → THD ≈ 31 % (low-order), which a load inductance must filter.")
    else:
        st.info("SVPWM energy is concentrated in sidebands around multiples of the switching frequency "
                f"(h ≈ {mf}, {2*mf} …). Low-order harmonics are absent in the linear region.")

# ==========================================================================================
# TAB 5 - linearity
# ==========================================================================================
with tabs[4]:
    ms, v1, tv, v1_six, thd_six = sweep(vdc, f1, fsw, hmax)
    # two stacked panels sharing the m axis (one y-scale per panel, no dual axis)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Fundamental line voltage / Vdc", "THD of line voltage [%]"))
    fig.add_trace(go.Scatter(x=ms, y=v1, name="SVPWM  |V_LL1|pk / Vdc", mode="lines+markers",
                             line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ms, y=np.minimum(ms, 10), name="ideal linear (= m)",
                             line=dict(color="#aaa", dash="dash")), row=1, col=1)
    fig.add_hline(y=v1_six, line_dash="dot", line_color="#d62728", row=1, col=1,
                  annotation_text=f"six-step: {v1_six:.3f} Vdc", annotation_position="bottom right")
    fig.add_trace(go.Scatter(x=ms, y=tv, name="THD V_LL [%]", mode="lines+markers", line=dict(color="#ff7f0e")),
                  row=2, col=1)
    fig.add_vline(x=1.0, line_dash="dot", annotation_text="linear limit", annotation_position="top left")
    fig.update_xaxes(title_text="modulation index m", row=2, col=1)
    fig.update_yaxes(title_text="V_LL1 / Vdc", row=1, col=1)
    fig.update_yaxes(title_text="THD [%]", row=2, col=1)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.12))
    show(fig)
    st.caption(f"Six-step reference: V_LL1 = {v1_six:.3f}·Vdc (theory 2√3/π = {sv.M_SIXSTEP:.3f}), THD = {thd_six:.1f} % "
               "(h limit as selected). SVPWM is linear up to m = 1; beyond that the fundamental saturates toward six-step.")

# ==========================================================================================
# TAB 6 - losses
# ==========================================================================================
SCH = ["SVPWM", "DPWMMAX", "DPWMMIN", "DPWM3", "Six-step"]
# Categorical slots 1-5 of the validated palette (light / dark steps); order = SCH order.
_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
_EV_LIGHT, _EV_DARK = ("#6250d6", "#e34948"), ("#9085e9", "#e66767")   # turn-on / turn-off event colours
DASH = {"SVPWM": "solid", "DPWMMAX": "dash", "DPWMMIN": "dot", "DPWM3": "dashdot", "Six-step": "longdash"}


def _dark_theme() -> bool:
    try:
        return st.context.theme.type != "light"
    except Exception:
        return True


SCH_COL = dict(zip(SCH, _DARK if _dark_theme() else _LIGHT))
EV_ON, EV_OFF = _EV_DARK if _dark_theme() else _EV_LIGHT


@st.cache_data(show_spinner=False)
def loss_run(scheme, vdc, f1, fsw, m, npc, R, L, dev_t):
    six = scheme == "Six-step"
    # Duty resolution is 2/npc; below ~100 samples per Ts small duty changes at low m are not resolved
    # (jagged loss-vs-m curves), so loss runs always use at least 100.
    r = sv.simulate(six, vdc, f1, fsw, m, max(int(npc), 100), R, L, cycles=1, scheme="SVPWM" if six else scheme)
    return ls.analyse(r, ls.DeviceParams(*dev_t), vdc, f1)


with tabs[5]:
    st.subheader("Switching & conduction losses")
    st.caption("Loss energy is computed per commutation from the simulated gate signals and the phase current at that "
               "instant (turn-on, turn-off, diode reverse recovery), plus conduction loss from the on-state drops. "
               "Everything on this tab uses the sidebar's Vdc, f₁, fsw, m, R and L.")

    with st.expander("Device & thermal parameters — replace with your datasheet values", expanded=False):
        st.caption("Defaults are representative of a 1200 V / 50 A class IGBT module, not a specific part number.")
        a1, a2, a3, a4 = st.columns(4)
        v_ref = a1.number_input("Datasheet test voltage [V]", 50.0, 3000.0, 600.0, 50.0)
        i_ref = a2.number_input("Datasheet test current [A]", 1.0, 2000.0, 50.0, 5.0)
        k_i = a3.number_input("Current exponent k_i", 0.3, 2.0, 1.0, 0.05,
                              help="Switching energy ∝ (i/I_ref)^k_i. 1.0 = linear.")
        k_v = a4.number_input("Voltage exponent k_v", 0.5, 2.0, 1.0, 0.05,
                              help="Switching energy ∝ (Vdc/V_ref)^k_v. 1.0–1.4 is typical.")
        b1, b2, b3 = st.columns(3)
        e_on = b1.number_input("IGBT turn-on Eon [mJ]", 0.0, 100.0, 3.5, 0.1)
        e_off = b2.number_input("IGBT turn-off Eoff [mJ]", 0.0, 100.0, 2.8, 0.1)
        e_rr = b3.number_input("Diode recovery Err [mJ]", 0.0, 100.0, 1.8, 0.1)
        d1, d2, d3, d4 = st.columns(4)
        vce0 = d1.number_input("IGBT Vce0 [V]", 0.0, 5.0, 1.0, 0.1)
        rce = d2.number_input("IGBT r_ce [mΩ]", 0.0, 500.0, 25.0, 1.0) * 1e-3
        vf0 = d3.number_input("Diode Vf0 [V]", 0.0, 5.0, 1.0, 0.1)
        rd = d4.number_input("Diode r_d [mΩ]", 0.0, 500.0, 20.0, 1.0) * 1e-3
        t1c, t2c, t3c, t4c = st.columns(4)
        rth_i = t1c.number_input("Rth(j-c) IGBT [K/W]", 0.01, 10.0, 0.60, 0.05)
        rth_d = t2c.number_input("Rth(j-c) diode [K/W]", 0.01, 10.0, 0.90, 0.05)
        rth_ca = t3c.number_input("Rth(c-a) module+heatsink [K/W]", 0.01, 10.0, 0.30, 0.05)
        t_amb = t4c.number_input("Ambient [°C]", -20.0, 100.0, 40.0, 5.0)

    dev = ls.DeviceParams(v_ref, i_ref, e_on * 1e-3, e_off * 1e-3, e_rr * 1e-3, k_i, k_v,
                          vce0, rce, vf0, rd, rth_i, rth_d, rth_ca, t_amb)
    dev_t = dataclasses.astuple(dev)

    sel = st.radio("Scheme for the detailed view", SCH, index=4 if sixstep else 0, horizontal=True,
                   help="SVPWM = centred zero vectors. DPWMMAX / DPWMMIN / DPWM3 = discontinuous PWM: one leg is held "
                        "on a rail for 120° of every cycle, so it does not switch there. Six-step = 180° conduction.")
    if m > 1.0 and sel != "Six-step":
        st.warning("m > 1: the discontinuous schemes are only defined up to m = 1 here (duties are clipped).")

    allr = {s: loss_run(s, vdc, f1, fsw, m, npc, R, L, dev_t) for s in SCH}
    a = allr[sel]
    base = allr["SVPWM"]["p_loss"]

    k = st.columns(6)
    k[0].metric("Total loss", f"{a['p_loss']:.1f} W",
                None if sel == "SVPWM" else f"{100 * (a['p_loss'] / base - 1):+.1f} % vs SVPWM", delta_color="inverse")
    k[1].metric("Switching loss", f"{a['p_sw']:.1f} W", f"{100 * a['p_sw'] / max(a['p_loss'], 1e-9):.0f} % of total",
                delta_color="off")
    k[2].metric("Conduction loss", f"{a['p_cond']:.1f} W", f"{100 * a['p_cond'] / max(a['p_loss'], 1e-9):.0f} % of total",
                delta_color="off")
    k[3].metric("Efficiency", f"{100 * a['eff']:.2f} %", f"P_out = {a['p_out'] / 1e3:.2f} kW", delta_color="off")
    k[4].metric("Tj IGBT (est.)", f"{a['tj_igbt']:.1f} °C", f"{a['p_igbt_dev']:.1f} W / device", delta_color="off")
    k[5].metric("Tj diode (est.)", f"{a['tj_diode']:.1f} °C", f"{a['p_diode_dev']:.1f} W / device", delta_color="off")

    # ---- 1. Breakdown by component, one bar per scheme ----------------------------------------
    comps = ["IGBT switching", "Diode recovery", "IGBT conduction", "Diode conduction"]
    keys = ["p_sw_igbt", "p_sw_diode", "p_cond_igbt", "p_cond_diode"]
    fig = go.Figure()
    for s in SCH:
        fig.add_trace(go.Bar(x=comps, y=[allr[s][kk] for kk in keys], name=s, marker_color=SCH_COL[s],
                             marker_line=dict(width=0), opacity=1.0 if s == sel else 0.7,
                             hovertemplate="%{x}<br>" + s + ": %{y:.2f} W<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25, height=380, title="Loss breakdown by mechanism (all six devices)",
                      yaxis_title="average power [W]", margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.15))
    show(fig)

    rows = []
    for s in SCH:
        r_ = allr[s]
        rows.append({"Scheme": s, "Switching [W]": round(r_["p_sw"], 2), "Conduction [W]": round(r_["p_cond"], 2),
                     "Total [W]": round(r_["p_loss"], 2), "Efficiency [%]": round(100 * r_["eff"], 2),
                     "P_out [W]": round(r_["p_out"], 0), "Transitions / leg / cycle": r_["n_trans"],
                     "Energy / cycle [mJ]": round(1e3 * r_["e_cycle"], 2),
                     "Tj IGBT [°C]": round(r_["tj_igbt"], 1), "Tj diode [°C]": round(r_["tj_diode"], 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Six-step has a fixed, larger output voltage (m ≈ 1.10), so its loss is not at the same output power as "
               "the PWM schemes; compare it through the loss-per-kW ratio rather than absolute watts.")

    # ---- 2. Where in the cycle the switching energy is spent -----------------------------------
    fig = go.Figure()
    for s in SCH[:4]:
        e_p = allr[s]["e_period"] * 1e3
        ang = (np.arange(len(e_p)) + 0.5) * 360.0 / len(e_p)
        fig.add_trace(go.Scatter(x=ang, y=e_p, name=s, mode="lines", line=dict(color=SCH_COL[s], dash=DASH[s],
                                                                                width=3 if s == sel else 1.8)))
    fig.update_layout(height=380, title="Switching energy per switching period around the fundamental cycle",
                      xaxis_title="electrical angle θ [deg]", yaxis_title="energy per Ts, all legs [mJ]",
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(range=[0, 360], tickvals=list(range(0, 361, 60)))
    show(fig)
    st.caption("SVPWM switches every leg in every period, so its energy follows |i|. The discontinuous schemes stop "
               "switching one leg for 120° per cycle; where that clamp falls relative to the current peak decides how "
               "much loss is saved, so the saving depends on the load power factor.")

    # ---- 3. Commutation events on leg A -------------------------------------------------------
    ia, on_a, off_a = a["i_a"], a["on_a"], a["off_a"]
    ang = (np.arange(len(ia)) + 0.5) * 360.0 / len(ia)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ang, y=ia, name="i_a", mode="lines", line=dict(color="#8a8a85", width=2)))
    for mask, nm, col, sym in ((on_a, "turn-on (+ diode recovery)", EV_ON, "triangle-up"),
                               (off_a, "turn-off", EV_OFF, "triangle-down")):
        idx = np.where(mask)[0]
        fig.add_trace(go.Scatter(x=ang[idx], y=ia[idx], name=nm, mode="markers",
                                 marker=dict(color=col, size=8, symbol=sym, line=dict(width=0))))
    fig.update_layout(height=380, title=f"Leg A commutations — {sel}", xaxis_title="electrical angle θ [deg]",
                      yaxis_title="phase current i_a [A]", margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(range=[0, 360], tickvals=list(range(0, 361, 60)))
    show(fig)
    st.caption("Each marker sits at the current that is being switched. Turn-on and turn-off at large |i| cost the most; "
               "the gaps in the markers are the clamped intervals.")

    # ---- 4. Sweeps ----------------------------------------------------------------------------
    st.markdown("#### Sweeps (SVPWM vs discontinuous schemes)")
    fsw_list = [1000, 2000, 3000, 5000, 8000, 12000, 16000, 20000]
    m_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sw_f = {s: [loss_run(s, vdc, f1, f_, m, 20, R, L, dev_t)["p_sw"] for f_ in fsw_list] for s in SCH[:4]}
    tot_m = {s: [loss_run(s, vdc, f1, fsw, m_, 20, R, L, dev_t)["p_loss"] for m_ in m_list] for s in SCH[:4]}
    six_sw = allr["Six-step"]["p_sw"]

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                        subplot_titles=("Switching loss vs switching frequency", "Total loss vs modulation index"))
    for s in SCH[:4]:
        fig.add_trace(go.Scatter(x=np.array(fsw_list) / 1e3, y=sw_f[s], name=s, mode="lines+markers",
                                 line=dict(color=SCH_COL[s], dash=DASH[s], width=2), marker=dict(size=8),
                                 legendgroup=s), row=1, col=1)
        fig.add_trace(go.Scatter(x=m_list, y=tot_m[s], name=s, mode="lines+markers", showlegend=False,
                                 line=dict(color=SCH_COL[s], dash=DASH[s], width=2), marker=dict(size=8),
                                 legendgroup=s), row=1, col=2)
    fig.add_hline(y=six_sw, line_dash=DASH["Six-step"], line_color=SCH_COL["Six-step"], row=1, col=1,
                  annotation_text="six-step (no PWM)", annotation_position="top right")
    fig.update_xaxes(title_text="fsw [kHz]", row=1, col=1)
    fig.update_xaxes(title_text="modulation index m", row=1, col=2)
    fig.update_yaxes(title_text="switching loss [W]", row=1, col=1)
    fig.update_yaxes(title_text="switching + conduction loss [W]", row=1, col=2)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.22))
    show(fig)
    st.caption(f"Left: m = {m:.2f}, load as set in the sidebar. Right: fsw = {fsw:.0f} Hz. Switching loss rises linearly "
               "with fsw while conduction loss does not, which is why the discontinuous schemes pay off most at "
               "high switching frequency.")

    with st.expander("Model assumptions"):
        st.markdown(
            "- Ideal gating with **no dead time**; no stray-inductance overshoot or snubber loss.\n"
            "- Energies scale as E = E_ref·(|i|/I_ref)^k_i·(Vdc/V_ref)^k_v; **no temperature dependence**.\n"
            "- On-state drop V = V₀ + r·|i| for IGBT and diode; in each leg exactly one device conducts at any instant.\n"
            "- The six IGBTs (and six diodes) are assumed identical and share the loss equally.\n"
            "- Tj = T_amb + P_total·Rth(c-a) + P_device·Rth(j-c): steady state only, no thermal impedance or cross-heating.\n"
            "- Load is a balanced star R-L with isolated neutral, so current ripple depends on L and fsw as simulated.")

# ==========================================================================================
# TAB 7 - theory
# ==========================================================================================
with tabs[6]:
    st.markdown(r"""
**Switching states.** Eight states $(S_A,S_B,S_C)$ → six active vectors of magnitude $\tfrac{2}{3}V_{dc}$ spaced 60° apart plus two zero vectors.

**Clarke transform of the pole voltages**

$$V_\alpha+jV_\beta=\tfrac{2}{3}\left(v_{A0}+v_{B0}e^{j2\pi/3}+v_{C0}e^{j4\pi/3}\right)$$

**Dwell times** (sector $n$, angle $\theta'$ from $V_n$, modulation index $m=\sqrt{3}|V_{ref}|/V_{dc}$)

$$T_1=mT_s\sin(60^\circ-\theta'),\quad T_2=mT_s\sin\theta',\quad T_0=T_s-T_1-T_2$$

**Linear range.** $|V_{ref}|\le V_{dc}/\sqrt3$ → phase peak $0.577V_{dc}$, line peak $V_{dc}$ (15.5 % more than sine-PWM).

**Six-step (180° conduction).** Each switch conducts 180°, legs lag by 120°; $v_{an}$ is a six-step waveform whose fundamental peak is $\tfrac{2}{\pi}V_{dc}$ and
$v_{LL}$ has harmonics $6k\pm1$ with amplitude $1/h$.

**Overmodulation (implemented).** If $T_1+T_2>T_s$ the times are scaled so $T_1+T_2=T_s$, $T_0=0$ (reference clipped to the hexagon along its own angle).
This is a simple hexagon-limiting scheme; the dedicated Modes I/II of Holtz/Bolognani are not implemented.

**Model assumptions.** Ideal switches, no dead time or device drops, constant $V_{dc}$, balanced star R-L load with isolated neutral.
""")
