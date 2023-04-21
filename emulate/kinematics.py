import numpy as np
from .constants import hbar_c


def compute_reduced_mass(m1, m2):
    return (m1 * m2) / (m1 + m2)


def t_lab_to_t_cm(t_lab, mass_beam, mass_target):
    return t_lab / ((mass_beam + mass_target) / mass_target)


def t_cm_to_t_lab(t_cm, mass_beam, mass_target):
    return t_cm * ((mass_beam + mass_target) / mass_target)


def t_lab_to_q_cm_beam_and_target(t_lab, mass_beam, mass_target):
    n = mass_target**2 * t_lab * (t_lab + 2 * mass_beam)
    d = (mass_target + mass_beam) ** 2 + 2 * t_lab * mass_target
    return np.sqrt(n / d) / hbar_c


def t_cm_to_q_cm(t_cm, mass_beam, mass_target):
    t_lab = t_cm_to_t_lab(t_cm, mass_beam, mass_target)
    return t_lab_to_q_cm_beam_and_target(t_lab, mass_beam, mass_target)
