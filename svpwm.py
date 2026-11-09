"""
svpwm.py - Space Vector PWM engine for a three-phase, two-level voltage source inverter.

Gating convention (180-degree conduction, complementary legs)
-------------------------------------------------------------
Each leg has an upper and a lower switch that are driven in complementary fashion
(S4 = not S1, S6 = not S3, S2 = not S5), so every leg is always connected to either
+Vdc/2 or -Vdc/2 and each switch conducts, on average, 180 electrical degrees per
fundamental cycle.  Two operating modes are provided:

  * SVPWM      - seven-segment symmetric sequence (0-Vn-Vn+1-7-Vn+1-Vn-0),
                 with hexagon-boundary overmodulation (angle preserving).
  * Six-step   - classic 180-degree conduction: each switch on for exactly 180 deg,
                 phases displaced by 120 deg, only the six active vectors are used.

Modulation index
----------------
    m = |Vref| / (Vdc / sqrt(3))
    m = 1.0         -> edge of the linear region (circle inscribed in the hexagon)
    m = 2*sqrt(3)/pi = 1.1027 -> fundamental produced by six-step operation
    m = 2/sqrt(3)  = 1.1547 -> reference circle touches the hexagon vertices

Vector numbering (switch states are (SA SB SC), 1 = upper switch on)
    V1=100 (0 deg)  V2=110 (60)  V3=010 (120)  V4=011 (180)  V5=001 (240)  V6=101 (300)
    V0=000 and V7=111 are the zero vectors.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

SQRT3 = np.sqrt(3.0)
M_LINEAR = 1.0
M_SIXSTEP = 2.0 * SQRT3 / np.pi          # ~1.1027
M_HEX_VERTEX = 2.0 / SQRT3               # ~1.1547

VECTOR_STATES = {
    1: (1, 0, 0),
    2: (1, 1, 0),
    3: (0, 1, 0),
    4: (0, 1, 1),
    5: (0, 0, 1),
    6: (1, 0, 1),
}


def state_str(n: int) -> str:
    return "".join(str(b) for b in VECTOR_STATES[n])


# --------------------------------------------------------------------------------------
# Core SVPWM maths
# --------------------------------------------------------------------------------------
def sector_and_angle(theta):
    """Return sector (1..6) and the angle inside the sector (0..pi/3) for angle theta [rad]."""
    th = np.mod(np.asarray(theta, dtype=float), 2.0 * np.pi)
    sec = np.clip(np.floor(th / (np.pi / 3.0)).astype(int) + 1, 1, 6)
    thp = th - (sec - 1) * np.pi / 3.0
    return sec, thp


def dwell_times(m, theta):
    """
    Normalised dwell times (fractions of the switching period Ts) of the two adjacent
    active vectors (t1 -> V_n, t2 -> V_n+1) and of the zero vectors (t0).

    If t1 + t2 > 1 (reference outside the hexagon) the times are scaled so that
    t1 + t2 = 1 and t0 = 0; the angle of the reference is preserved.
    """
    sec, thp = sector_and_angle(theta)
    t1 = m * np.sin(np.pi / 3.0 - thp)
    t2 = m * np.sin(thp)
    tsum = t1 + t2
    scale = np.where(tsum > 1.0, 1.0 / np.maximum(tsum, 1e-12), 1.0)
    t1 = t1 * scale
    t2 = t2 * scale
    t0 = 1.0 - t1 - t2
    return sec, thp, t1, t2, t0


def duty_cycles(sec, t1, t2, t0):
    """On-time (fraction of Ts) of the upper switch of legs A, B, C for a symmetric sequence."""
    h = t0 / 2.0
    conds = [sec == k for k in range(1, 7)]
    da = np.select(conds, [t1 + t2 + h, t1 + h, h, h, t2 + h, t1 + t2 + h])
    db = np.select(conds, [t2 + h, t1 + t2 + h, t1 + t2 + h, t1 + h, h, h])
    dc = np.select(conds, [h, h, t2 + h, t1 + t2 + h, t1 + t2 + h, t1 + h])
    return da, db, dc


def switching_sequence(sec: int, t1: float, t2: float, t0: float):
    """Human readable seven-segment sequence for one switching period: [(state, name, time/Ts)]."""
    n1 = int(sec)
    n2 = n1 % 6 + 1
    s1, s2 = state_str(n1), state_str(n2)
    return [
        ("000", "V0", t0 / 4),
        (s1, f"V{n1}", t1 / 2),
        (s2, f"V{n2}", t2 / 2),
        ("111", "V7", t0 / 2),
        (s2, f"V{n2}", t2 / 2),
        (s1, f"V{n1}", t1 / 2),
        ("000", "V0", t0 / 4),
    ]


def six_step_active_vector(theta):
    """Index (1..6) of the active vector in six-step mode (V1 is centred on theta = 0)."""
    th = np.mod(np.asarray(theta, dtype=float) + np.pi / 6.0, 2.0 * np.pi)
    return (np.floor(th / (np.pi / 3.0)).astype(int) % 6) + 1


# --------------------------------------------------------------------------------------
# Spectrum helpers
# --------------------------------------------------------------------------------------
def harmonic_spectrum(x_cycle, hmax):
    """Single-sided amplitude spectrum of exactly one fundamental cycle. Returns (orders, amplitude)."""
    x_cycle = np.asarray(x_cycle, dtype=float)
    M = len(x_cycle)
    X = np.fft.rfft(x_cycle) / M
    mag = 2.0 * np.abs(X)
    mag[0] = np.abs(X[0])
    hmax = int(min(hmax, len(mag) - 1))
    return np.arange(hmax + 1), mag[: hmax + 1]


def thd(mag):
    """THD (ratio, not %) from an amplitude spectrum (index = harmonic order)."""
    if len(mag) < 3 or mag[1] < 1e-9:
        return 0.0
    return float(np.sqrt(np.sum(mag[2:] ** 2)) / mag[1])


def fundamental_phasor(x_cycle):
    """Complex amplitude (peak) of the fundamental of exactly one cycle of x."""
    M = len(x_cycle)
    return 2.0 * np.fft.rfft(np.asarray(x_cycle, dtype=float))[1] / M


# --------------------------------------------------------------------------------------
# Inverter simulation
# --------------------------------------------------------------------------------------
SCHEMES = ["SVPWM", "DPWMMAX", "DPWMMIN", "DPWM3"]


def reference_duties(scheme: str, m, thk):
    """
    Leg duty cycles (3, K) for carrier-based schemes, built as 'sine + zero-sequence injection'.

        u_x = M cos(theta - phase_x),  M = 2m/sqrt(3)   (normalised to Vdc/2)
        d_x = 0.5 * (1 + u_x + u0)

    SVPWM    : u0 = -(max+min)/2          (centres the zero-vector time; identical to dwell-time SVPWM)
    DPWMMAX  : u0 = 1 - max(u)            (highest phase clamped to + rail, 120 deg per cycle)
    DPWMMIN  : u0 = -1 - min(u)           (lowest phase clamped to - rail, 120 deg per cycle)
    DPWM3    : clamp the phase with the largest |u| to the rail it is nearest
               (60 deg clamp centred on each phase peak, both polarities)
    Valid for m <= 1; duties are clipped to [0, 1] beyond that.
    """
    M = 2.0 * m / SQRT3
    th = np.asarray(thk, dtype=float)
    u = np.stack([M * np.cos(th), M * np.cos(th - 2 * np.pi / 3), M * np.cos(th + 2 * np.pi / 3)])
    if scheme == "SVPWM":
        u0 = -(u.max(axis=0) + u.min(axis=0)) / 2.0
    elif scheme == "DPWMMAX":
        u0 = 1.0 - u.max(axis=0)
    elif scheme == "DPWMMIN":
        u0 = -1.0 - u.min(axis=0)
    elif scheme == "DPWM3":
        k = np.argmax(np.abs(u), axis=0)
        uk = np.take_along_axis(u, k[None, :], axis=0)[0]
        u0 = np.sign(uk) - uk
    else:
        raise ValueError(f"unknown scheme {scheme}")
    return np.clip(0.5 * (1.0 + u + u0), 0.0, 1.0)


def simulate(sixstep: bool, vdc: float, f1: float, fsw: float, m: float,
             npc: int = 40, R: float = 10.0, L: float = 20e-3,
             cycles: int = 2, load: bool = True, scheme: str = "SVPWM") -> dict:
    """
    Switch-level simulation of an ideal two-level VSI (no dead time, ideal switches)
    feeding a balanced star-connected R-L load with isolated neutral.

    scheme: "SVPWM" (dwell-time implementation, with overmodulation) or one of the
    discontinuous schemes in `reference_duties` (ignored when sixstep=True).

    The switching frequency is snapped to a multiple of 6*f1 so that the waveform is
    exactly periodic with six-fold symmetry (clean FFT, no leakage).

    Returned arrays cover `cycles` fundamental cycles of steady state.
    """
    mf = max(6, 6 * int(round(fsw / (6.0 * f1))))
    fsw_eff = mf * f1
    Ts = 1.0 / fsw_eff
    npc = max(8, int(npc) // 2 * 2)
    dt = Ts / npc
    spc = mf * npc                                    # samples per fundamental cycle

    warm = int(np.clip(np.ceil(5.0 * L / max(R, 1e-6) * f1), 1, 40)) if load else 0
    total = warm + int(cycles)
    K = mf * total                                    # number of switching periods
    n = K * npc
    t = (np.arange(n) + 0.5) * dt

    if sixstep:
        theta = 2.0 * np.pi * f1 * t
        S = np.stack([(np.mod(theta - k * 2.0 * np.pi / 3.0 + np.pi / 2.0, 2.0 * np.pi) < np.pi)
                      for k in range(3)]).astype(float)
        duty_k = None
        duty_rep = None
        carrier = None
    else:
        k = np.arange(K)
        thk = 2.0 * np.pi * f1 * (k + 0.5) * Ts       # reference angle sampled mid-period
        if scheme == "SVPWM":
            sec, thp, t1, t2, t0 = dwell_times(m, thk)
            duty_k = np.stack(duty_cycles(sec, t1, t2, t0))             # (3, K)
        else:
            duty_k = reference_duties(scheme, m, thk)
        duty_rep = np.repeat(duty_k, npc, axis=1)                       # (3, n)
        tau = (np.arange(npc) + 0.5) / npc
        carrier = np.tile(np.abs(2.0 * tau - 1.0), K)                   # 1 at ends, 0 at centre
        S = (duty_rep > carrier).astype(float)                          # upper-switch gates

    # --- electrical quantities ---------------------------------------------------------
    vpole = S * vdc                                                     # leg voltage wrt negative rail
    vA, vB, vC = vpole
    vph = np.stack([(2 * vA - vB - vC) / 3.0, (2 * vB - vA - vC) / 3.0, (2 * vC - vA - vB) / 3.0])
    vll = np.stack([vA - vB, vB - vC, vC - vA])
    v_alpha = (2.0 / 3.0) * (vA - 0.5 * vB - 0.5 * vC)
    v_beta = (1.0 / SQRT3) * (vB - vC)
    v_avg = np.stack([v_alpha.reshape(K, npc).mean(axis=1), v_beta.reshape(K, npc).mean(axis=1)])

    if load:
        a = np.exp(-R * dt / L)
        b = (1.0 - a) / R
        cur = np.stack([lfilter([b], [1.0, -a], vph[j]) for j in range(3)])
    else:
        cur = None

    s0 = warm * spc                                   # drop the warm-up cycles
    k0 = warm * mf
    n_out = n - s0
    out = dict(
        t=(np.arange(n_out) + 0.5) * dt,
        S=S[:, s0:],
        vpole=vpole[:, s0:],
        vll=vll[:, s0:],
        vph=vph[:, s0:],
        i=None if cur is None else cur[:, s0:],
        v_avg=v_avg[:, k0:],
        duty_k=None if duty_k is None else duty_k[:, k0:],
        duty_rep=None if duty_rep is None else duty_rep[:, s0:],
        carrier=None if carrier is None else carrier[s0:],
        mf=mf, fsw_eff=fsw_eff, npc=npc, spc=spc, Ts=Ts, dt=dt, K=K - k0,
    )
    return out
