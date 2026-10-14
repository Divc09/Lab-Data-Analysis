"""Parameter symbols, units and descriptions (copied from nvfit.gui_app so the
v2 server does not import the Qt application)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParamMeta:
    symbol: str
    units: str
    description: str


PARAM_METADATA: dict[str, ParamMeta] = {
    "tau_s": ParamMeta("Decay time", "s", "Descriptive exponential drift/bleaching timescale in seconds."),
    "background": ParamMeta("Background", "arb.", "Fitted baseline in the selected observable's recorded units."),
    "amplitude": ParamMeta("Amplitude", "arb.", "Exponential component in the selected observable's recorded units."),
    "y0": ParamMeta("y_0", "arb.", "Vertical offset / baseline level."),
    "m": ParamMeta("m", "arb./ns", "Linear drift slope across scan window."),
    "A": ParamMeta("A", "arb.", "Oscillation/contrast amplitude."),
    "A1": ParamMeta("A_1", "arb.", "Primary oscillation amplitude."),
    "A2": ParamMeta("A_2", "arb.", "Secondary oscillation amplitude."),
    "C": ParamMeta("C", "arb.", "ODMR resonance amplitude."),
    "f": ParamMeta("f", "GHz", "Oscillation frequency (detuning or Rabi frequency)."),
    "f1": ParamMeta("f_1", "GHz", "First oscillation frequency component."),
    "f2": ParamMeta("f_2", "GHz", "Second oscillation frequency component."),
    "phi": ParamMeta("phi", "rad", "Initial oscillation phase."),
    "phi1": ParamMeta("phi_1", "rad", "First oscillation phase."),
    "phi2": ParamMeta("phi_2", "rad", "Second oscillation phase."),
    "tau": ParamMeta("tau", "ns", "Decay time constant."),
    "T": ParamMeta("T", "ns", "Envelope decay timescale."),
    "t0": ParamMeta("t_0", "ns", "Effective oscillation start time."),
    "tau_ramp": ParamMeta(r"\tau_r", "ns", "Microwave startup / pulse-area buildup timescale."),
    "beta": ParamMeta(r"\beta", "-", "Envelope stretch exponent."),
    "alpha": ParamMeta(r"\alpha", "1/ns^2", "Linear chirp term."),
    # Adaptive pulse-area Rabi (nvfit.models.rabi_adaptive_pulse_area_model): cycles = f0 u + chirp1 u^2/2 + chirp2 u^3/3, u in ns.
    "f0": ParamMeta("f_0", "GHz", "Effective Rabi frequency at the start of the pulse (cycles/ns)."),
    "chirp1": ParamMeta("c_1", "GHz/ns", "Linear change of the effective Rabi frequency with pulse duration (pulse-area calibration, not a carrier chirp)."),
    "chirp2": ParamMeta("c_2", "GHz/ns^2", "Quadratic change of the effective Rabi frequency with pulse duration."),
    "delay": ParamMeta("delay", "ns", "Fitted timing delay before the pulse area starts accumulating."),
    "T2": ParamMeta("T_2", "ns", "Spin-echo coherence time."),
    "T2_star": ParamMeta("T_2*", "ns", "Ramsey dephasing time."),
    "n": ParamMeta("n", "-", "Stretch exponent (1=exp, 2=Gaussian-like)."),
    "w": ParamMeta("gamma", "axis units", "Half width at half maximum."),
    "x0": ParamMeta("x_0", "axis units", "ODMR resonance center."),
}




def param_meta(name: str) -> ParamMeta:
    if name.startswith("center_"):
        return ParamMeta(name, "axis units", "Gaussian peak center.")
    if name.startswith("sigma_"):
        return ParamMeta(name, "axis units", "Gaussian standard deviation; FWHM = 2.35482 sigma.")
    if name.startswith("amplitude_"):
        return ParamMeta(name, "arb.", "Gaussian peak amplitude.")
    if name.startswith("x0_"):
        return ParamMeta(name, "axis units", "Resonance center.")
    return PARAM_METADATA.get(name, ParamMeta(name, "", "Model parameter."))
