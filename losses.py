"""
losses.py - event-based loss model for a two-level, three-leg SiC MOSFET inverter.

Method (per fundamental cycle, from the simulated gate signals and phase currents)
-----------------------------------------------------------------------------------
Sign convention: i > 0 flows from the leg mid-point into the load.

Hard-switched commutations (the device that turns on does so against the conducting body
diode of its partner):

    S rises (0->1) and i > 0 : upper MOSFET turns ON  (Eon), lower body diode recovers
    S falls (1->0) and i < 0 : lower MOSFET turns ON  (Eon), upper body diode recovers
    S falls (1->0) and i > 0 : upper MOSFET turns OFF (Eoff)
    S rises (0->1) and i < 0 : lower MOSFET turns OFF (Eoff)

Switching energies scale from the datasheet point:

    E(i, Vdc) = E_ref * (|i| / I_ref)^k_i * (Vdc / V_ref)^k_v

Dead time (loss only - the simulated waveforms stay ideal): at every commutation the current
flows through a body diode for t_dead, E_dt = Vsd * |i| * t_dead.

Conduction (complementary gating, no dead time in the waveforms): in each leg one device carries
the current.  It conducts "forward" (drain->source through the channel) when (S=1, i>0) or
(S=0, i<0); otherwise the current is "reverse".
    * synchronous rectification ON : reverse current also uses the channel  -> P = Rds(on) * i^2
    * synchronous rectification OFF: reverse current uses the body diode    -> P = Vsd*|i| + r_sd*i^2

Rds(on) rises with junction temperature, Rds(Tj) = Rds25 * (1 + k_t * (Tj - 25)).  Because the
temperature depends on the loss, the model iterates the electro-thermal loop to a fixed point,
separately for the upper and the lower device (they are not equally stressed, especially in the
discontinuous schemes).  Tj = T_amb + P_total*Rth(c-a) + P_device*Rth(j-c).

Assumptions: balanced three-phase operation (the three upper devices are identical, as are the
three lower ones), parameters otherwise independent of temperature/current/Vgs, no stray-inductance
overshoot, no Coss/Qoss-specific loss beyond what the datasheet Eon/Eoff contain.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeviceParams:
    v_ref: float = 800.0        # V   - test voltage of the datasheet switching energies
    i_ref: float = 20.0         # A   - test current of the datasheet switching energies
    e_on: float = 0.40e-3       # J   - turn-on energy (datasheet value normally includes body-diode recovery)
    e_off: float = 0.12e-3      # J   - turn-off energy
    e_rr: float = 0.0           # J   - EXTRA body-diode recovery energy (leave 0 if already inside Eon)
    k_i: float = 1.0            # current exponent of the switching energies
    k_v: float = 1.0            # voltage exponent of the switching energies
    rds25: float = 0.075        # ohm - Rds(on) at 25 degC
    k_t: float = 0.004          # 1/K - linear temperature coefficient (0.004 -> x1.5 at 150 degC)
    sync: bool = True           # synchronous rectification (reverse current through the channel)
    vsd: float = 4.0            # V   - body-diode drop (threshold)
    rsd: float = 0.060          # ohm - body-diode slope resistance
    t_dead: float = 200e-9      # s   - dead time per commutation
    rth_jc: float = 0.80        # K/W per device, junction to case
    rth_ca: float = 0.40        # K/W whole module/heatsink, case to ambient (all 6 devices)
    t_amb: float = 40.0         # degC
    tj_max: float = 175.0       # degC - used only to raise a warning


def _fund(x: np.ndarray) -> complex:
    return 2.0 * np.fft.rfft(np.asarray(x, dtype=float))[1] / len(x)


def analyse(res: dict, p: DeviceParams, vdc: float, f1: float) -> dict:
    """Loss analysis of the last fundamental cycle of a `svpwm.simulate(...)` result (load=True)."""
    spc, npc, mf = res["spc"], res["npc"], res["mf"]
    S = res["S"][:, -spc:]
    i = res["i"][:, -spc:]
    vph = res["vph"][:, -spc:]

    # --- commutation events (circular: the cycle is periodic) ----------------------------
    S_prev = np.roll(S, 1, axis=1)
    i_prev = np.roll(i, 1, axis=1)                    # current at the instant of the transition
    rising = (S - S_prev) > 0.5
    falling = (S - S_prev) < -0.5
    trans = rising | falling

    on_u = rising & (i_prev > 0)                      # upper turns on (hard)
    off_u = falling & (i_prev > 0)                    # upper turns off
    on_l = falling & (i_prev < 0)                     # lower turns on (hard)
    off_l = rising & (i_prev <= 0)                    # lower turns off
    turn_on = on_u | on_l

    scale = (vdc / p.v_ref) ** p.k_v * (np.abs(i_prev) / p.i_ref) ** p.k_i
    eon, eoff, err = p.e_on * scale, p.e_off * scale, p.e_rr * scale
    edt = p.vsd * np.abs(i_prev) * p.t_dead * trans   # dead-time body-diode conduction, J

    # per-device energies per cycle (average over the three identical legs), J
    dt_u = trans & (i_prev < 0)                       # i<0: upper body diode carries the current
    dt_l = trans & (i_prev >= 0)
    e_sw_u = float((eon * on_u + eoff * off_u).sum() / 3)
    e_sw_l = float((eon * on_l + eoff * off_l).sum() / 3)
    e_rr_u = float((err * on_l).sum() / 3)            # upper body diode recovers when lower turns on
    e_rr_l = float((err * on_u).sum() / 3)
    e_dt_u = float((edt * dt_u).sum() / 3)
    e_dt_l = float((edt * dt_l).sum() / 3)

    # --- conduction ----------------------------------------------------------------------------
    up = S > 0.5
    lo = ~up
    fwd_u, rev_u = up & (i > 0), up & (i <= 0)
    fwd_l, rev_l = lo & (i < 0), lo & (i >= 0)
    i2, ai = i ** 2, np.abs(i)
    if p.sync:
        i2ch_u = float((i2 * up).mean())              # whole conduction interval is channel conduction
        i2ch_l = float((i2 * lo).mean())
        p_dio_u = p_dio_l = 0.0
    else:
        i2ch_u = float((i2 * fwd_u).mean())
        i2ch_l = float((i2 * fwd_l).mean())
        p_dio_u = float(p.vsd * (ai * rev_u).mean() + p.rsd * (i2 * rev_u).mean())
        p_dio_l = float(p.vsd * (ai * rev_l).mean() + p.rsd * (i2 * rev_l).mean())

    # per-device temperature-independent power (W)
    fx_u = f1 * (e_sw_u + e_rr_u + e_dt_u) + p_dio_u
    fx_l = f1 * (e_sw_l + e_rr_l + e_dt_l) + p_dio_l

    # --- electro-thermal fixed point ----------------------------------------------------------------
    tj_u = tj_l = p.t_amb
    runaway = False
    for _ in range(200):
        r_u = p.rds25 * (1 + p.k_t * (tj_u - 25.0))
        r_l = p.rds25 * (1 + p.k_t * (tj_l - 25.0))
        pu, pl = fx_u + r_u * i2ch_u, fx_l + r_l * i2ch_l
        p_tot = 3.0 * (pu + pl)
        tc = p.t_amb + p_tot * p.rth_ca
        n_u, n_l = tc + pu * p.rth_jc, tc + pl * p.rth_jc
        if max(n_u, n_l) > 600.0:
            runaway = True
            tj_u, tj_l = n_u, n_l
            break
        done = abs(n_u - tj_u) < 1e-4 and abs(n_l - tj_l) < 1e-4
        tj_u, tj_l = n_u, n_l
        if done:
            break
    r_u = p.rds25 * (1 + p.k_t * (tj_u - 25.0))
    r_l = p.rds25 * (1 + p.k_t * (tj_l - 25.0))
    pu, pl = fx_u + r_u * i2ch_u, fx_l + r_l * i2ch_l

    # --- totals over the six devices (W) ---------------------------------------------------------------
    p_sw_fet = 3.0 * f1 * (e_sw_u + e_sw_l)
    p_sw_diode = 3.0 * f1 * (e_rr_u + e_rr_l)
    p_dt = 3.0 * f1 * (e_dt_u + e_dt_l)
    p_cond_fet = 3.0 * (r_u * i2ch_u + r_l * i2ch_l)
    p_cond_diode = 3.0 * (p_dio_u + p_dio_l)
    p_sw = p_sw_fet + p_sw_diode
    p_cond = p_cond_fet + p_cond_diode
    p_loss = p_sw + p_dt + p_cond

    # --- output power, efficiency -------------------------------------------------------------------------
    p_out = float(np.mean(np.sum(vph * i, axis=0)))
    eff = p_out / (p_out + p_loss) if p_out > 0 else 0.0
    v1, i1 = _fund(vph[0]), _fund(i[0])
    pf1 = float(np.cos(np.angle(v1) - np.angle(i1)))

    hot_is_upper = tj_u >= tj_l

    # --- distribution over the fundamental cycle ---------------------------------------------------------
    e_total = eon * turn_on + eoff * (off_u | off_l) + err * turn_on + edt
    e_period = e_total.reshape(3, mf, npc).sum(axis=(0, 2))          # J per switching period (all legs)

    return dict(
        p_sw_fet=p_sw_fet, p_sw_diode=p_sw_diode, p_dt=p_dt,
        p_cond_fet=p_cond_fet, p_cond_diode=p_cond_diode,
        p_sw=p_sw, p_cond=p_cond, p_loss=p_loss,
        p_out=p_out, eff=eff, pf1=pf1, i1_pk=float(abs(i1)), i_rms=float(np.sqrt(np.mean(i[0] ** 2))),
        p_dev_u=pu, p_dev_l=pl, tj_u=tj_u, tj_l=tj_l,
        tj=max(tj_u, tj_l), hot="upper" if hot_is_upper else "lower",
        p_hot=pu if hot_is_upper else pl, rds_hot=r_u if hot_is_upper else r_l,
        t_case=p.t_amb + p_loss * p.rth_ca, runaway=runaway,
        n_trans=int(trans[0].sum()),
        e_period=e_period, mf=mf, spc=spc,
        i_a=i[0], on_a=turn_on[0], off_a=(off_u | off_l)[0],
        e_cycle=float(e_total.sum()),
    )