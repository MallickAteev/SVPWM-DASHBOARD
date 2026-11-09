"""
losses.py - event-based switching + conduction loss model for a two-level, three-leg IGBT inverter.

Method (per fundamental cycle, from the simulated gate signals and load currents)
---------------------------------------------------------------------------------
Every commutation of a leg is classified by the sign of the phase current at that instant
(i > 0 flows from the leg into the load):

    S rises (0->1)   and i > 0  : upper IGBT turns ON,  lower diode recovers      -> Eon + Err
    S falls (1->0)   and i < 0  : lower IGBT turns ON,  upper diode recovers      -> Eon + Err
    S falls (1->0)   and i > 0  : upper IGBT turns OFF                            -> Eoff
    S rises (0->1)   and i < 0  : lower IGBT turns OFF                            -> Eoff

(the other device of the pair simply picks up the current and has no loss).  Each energy is
scaled from its datasheet point:

    E(i, Vdc) = E_ref * (|i| / I_ref)^k_i * (Vdc / V_ref)^k_v

Conduction: with complementary gating and no dead time, in every leg exactly one device carries
the current at any instant - the IGBT when (S=1, i>0) or (S=0, i<0), otherwise the anti-parallel
diode.  Voltage drop model: V = V0 + r*|i|.

Assumptions: ideal (zero) dead time, parameters independent of temperature, no stray-inductance
overshoot, and the six devices of each type are identical (losses are split evenly by symmetry).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeviceParams:
    v_ref: float = 600.0        # V   - test voltage of the datasheet energies
    i_ref: float = 50.0         # A   - test current of the datasheet energies
    e_on: float = 3.5e-3        # J   - IGBT turn-on energy at (v_ref, i_ref)
    e_off: float = 2.8e-3       # J   - IGBT turn-off energy
    e_rr: float = 1.8e-3        # J   - diode reverse-recovery energy
    k_i: float = 1.0            # current exponent of the switching energies
    k_v: float = 1.0            # voltage exponent of the switching energies
    vce0: float = 1.0           # V   - IGBT on-state threshold
    rce: float = 0.025          # ohm - IGBT on-state slope resistance
    vf0: float = 1.0            # V   - diode threshold
    rd: float = 0.020           # ohm - diode slope resistance
    rth_jc_igbt: float = 0.60   # K/W per IGBT, junction to case
    rth_jc_diode: float = 0.90  # K/W per diode, junction to case
    rth_ca: float = 0.30        # K/W whole module (case to ambient, through the heatsink)
    t_amb: float = 40.0         # degC


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
    turn_on = (rising & (i_prev > 0)) | (falling & (i_prev < 0))
    turn_off = (rising & (i_prev <= 0)) | (falling & (i_prev > 0))

    scale = (vdc / p.v_ref) ** p.k_v * (np.abs(i_prev) / p.i_ref) ** p.k_i
    e_on = p.e_on * scale * turn_on                   # J, (3, spc)
    e_off = p.e_off * scale * turn_off
    e_rr = p.e_rr * scale * turn_on

    p_sw_igbt = float((e_on + e_off).sum() * f1)
    p_sw_diode = float(e_rr.sum() * f1)

    # --- conduction ------------------------------------------------------------------------
    igbt_conducts = ((S > 0.5) & (i > 0)) | ((S < 0.5) & (i < 0))
    ai = np.abs(i)
    p_cond_igbt = float(np.where(igbt_conducts, p.vce0 * ai + p.rce * ai ** 2, 0.0).sum(axis=0).mean())
    p_cond_diode = float(np.where(~igbt_conducts, p.vf0 * ai + p.rd * ai ** 2, 0.0).sum(axis=0).mean())

    p_sw = p_sw_igbt + p_sw_diode
    p_cond = p_cond_igbt + p_cond_diode
    p_loss = p_sw + p_cond

    # --- output power, efficiency, thermal --------------------------------------------------
    p_out = float(np.mean(np.sum(vph * i, axis=0)))
    eff = p_out / (p_out + p_loss) if p_out > 0 else 0.0
    v1, i1 = _fund(vph[0]), _fund(i[0])
    pf1 = float(np.cos(np.angle(v1) - np.angle(i1)))

    p_igbt_dev = (p_sw_igbt + p_cond_igbt) / 6.0
    p_diode_dev = (p_sw_diode + p_cond_diode) / 6.0
    t_case = p.t_amb + p_loss * p.rth_ca
    tj_igbt = t_case + p_igbt_dev * p.rth_jc_igbt
    tj_diode = t_case + p_diode_dev * p.rth_jc_diode

    # --- distribution over the fundamental cycle ---------------------------------------------
    e_total = e_on + e_off + e_rr
    e_period = e_total.reshape(3, mf, npc).sum(axis=(0, 2))          # J per switching period (all legs)

    return dict(
        p_sw_igbt=p_sw_igbt, p_sw_diode=p_sw_diode,
        p_cond_igbt=p_cond_igbt, p_cond_diode=p_cond_diode,
        p_sw=p_sw, p_cond=p_cond, p_loss=p_loss,
        p_out=p_out, eff=eff, pf1=pf1, i1_pk=float(abs(i1)), i_rms=float(np.sqrt(np.mean(i[0] ** 2))),
        p_igbt_dev=p_igbt_dev, p_diode_dev=p_diode_dev,
        t_case=t_case, tj_igbt=tj_igbt, tj_diode=tj_diode,
        n_trans=int((rising | falling)[0].sum()),
        e_period=e_period, mf=mf, spc=spc,
        i_a=i[0], on_a=turn_on[0], off_a=turn_off[0],
        e_cycle=float((e_on + e_off + e_rr).sum()),
    )
