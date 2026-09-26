"""kygwow fork: opt-in Li inverse-rule factorization for 1-D gratings (add_layer(..., factorization=))."""

import math

import pytest
import torch

import torcwa

WAVELENGTH = 470e-9
PERIOD = 400e-9
EPS_SI = (4.497 + 0.062j) ** 2
CODES = ["pp", "sp", "ps", "ss"]


def _eps_lines(axis, n=256, eps_line=EPS_SI):
    """Lines of eps_line in air, duty 0.5; axis='x' -> eps varies along x (lines along y)."""
    x = torch.linspace(-0.5, 0.5, n + 1, dtype=torch.float64)[:-1]
    fill = (torch.abs(x) < 0.25).to(torch.complex128)
    profile = 1.0 + fill * (eps_line - 1.0)
    return profile[:, None].repeat(1, n) if axis == "x" else profile[None, :].repeat(n, 1)


def _solve(eps, order, theta, azimuth, factorization="laurent", substrate=EPS_SI, **kwargs):
    sim = torcwa.rcwa(freq=1 / WAVELENGTH, order=order, L=[PERIOD, PERIOD], dtype=torch.complex128, device=torch.device("cpu"))
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=substrate)
    sim.set_incident_angle(theta, azimuth)
    sim.add_layer(200e-9, eps=eps, factorization=factorization, **kwargs)
    sim.solve_global_smatrix()
    return sim


def _r00(sim, code, port="reflection"):
    return sim.S_parameters([[0, 0]], direction="forward", port=port, polarization=code, ref_order=[0, 0], power_norm=True)[0]


def test_default_is_laurent_bit_identical():
    eps = _eps_lines("x")
    a = _solve(eps, [10, 0], math.radians(10), 0.0)
    sim = torcwa.rcwa(freq=1 / WAVELENGTH, order=[10, 0], L=[PERIOD, PERIOD], dtype=torch.complex128, device=torch.device("cpu"))
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=EPS_SI)
    sim.set_incident_angle(math.radians(10), 0.0)
    sim.add_layer(200e-9, eps=eps)  # upstream call signature
    sim.solve_global_smatrix()
    for code in CODES:
        assert torch.equal(_r00(a, code), _r00(sim, code))


def test_homogeneous_layer_is_unaffected():
    eps = torch.tensor(EPS_SI, dtype=torch.complex128)
    for code in CODES:
        a = _r00(_solve(eps, [5, 0], math.radians(12), 0.3), code)
        b = _r00(_solve(eps, [5, 0], math.radians(12), 0.3, factorization="li_x"), code)
        assert torch.equal(a, b)


def test_te_is_unchanged_in_planar_incidence():
    # lines along y, incidence plane xz: s (E_y, parallel to the walls) does not see the E_x block.
    eps = _eps_lines("x")
    a = _r00(_solve(eps, [15, 0], math.radians(10), 0.0), "ss")
    b = _r00(_solve(eps, [15, 0], math.radians(10), 0.0, factorization="li_x"), "ss")
    assert abs(a - b) < 1e-10


@pytest.mark.parametrize("theta_deg,azimuth_deg", [(0.0, 0.0), (15.0, 0.0), (15.0, 35.0), (20.0, 90.0)])
def test_tm_converges_faster_with_li(theta_deg, azimuth_deg):
    # Reference: Li at a high order. Li at a low order must sit much closer to it than Laurent does.
    eps = _eps_lines("x")
    th, az = math.radians(theta_deg), math.radians(azimuth_deg)
    ref = _solve(eps, [60, 0], th, az, factorization="li_x")
    li = _solve(eps, [15, 0], th, az, factorization="li_x")
    la = _solve(eps, [15, 0], th, az)
    code = "pp" if azimuth_deg < 45 else "ss"  # the component with E across the walls
    err_li = abs(_r00(li, code) - _r00(ref, code))
    err_la = abs(_r00(la, code) - _r00(ref, code))
    assert err_li < err_la / 5


@pytest.mark.parametrize("theta_deg,azimuth_deg", [(0.0, 0.0), (15.0, 0.0), (15.0, 35.0)])
def test_lossless_energy_is_conserved(theta_deg, azimuth_deg):
    # Real-index grating on a real-index substrate: R + T = 1 for both input polarizations, including conical.
    n_order = 12
    eps = _eps_lines("x", eps_line=2.25 ** 2 + 0j)
    sim = _solve(eps, [n_order, 0], math.radians(theta_deg), math.radians(azimuth_deg), factorization="li_x", substrate=1.5 ** 2 + 0j)
    orders = [[m, 0] for m in range(-n_order, n_order + 1)]
    for inc in ("p", "s"):
        total = 0.0
        for port in ("reflection", "transmission"):
            for out in ("p", "s"):
                s = sim.S_parameters(orders, direction="forward", port=port, polarization=out + inc, ref_order=[0, 0], power_norm=True)
                s = torch.nan_to_num(s)
                total += float(torch.sum(torch.abs(s) ** 2))
        assert abs(total - 1.0) < 1e-8


@pytest.mark.parametrize("code_x,code_y", [("pp", "pp"), ("ss", "ss"), ("ps", "ps")])
def test_li_y_is_li_x_rotated(code_x, code_y):
    # Rotating the structure by 90 degrees and the incidence azimuth with it must give the same p/s response.
    th, az = math.radians(12), math.radians(20)
    a = _r00(_solve(_eps_lines("x"), [15, 0], th, az, factorization="li_x"), code_x)
    b = _r00(_solve(_eps_lines("y"), [0, 15], th, az + math.pi / 2, factorization="li_y"), code_y)
    assert abs(abs(a) - abs(b)) < 1e-9


def test_two_dimensional_pattern_is_rejected():
    eps = _eps_lines("x") * 0 + 1.0
    eps[:64, :64] = EPS_SI
    with pytest.raises(ValueError, match="normal-vector"):
        _solve(eps, [5, 5], 0.0, 0.0, factorization="li_x")


def test_unknown_factorization_is_rejected():
    with pytest.raises(ValueError, match="factorization"):
        _solve(_eps_lines("x"), [5, 0], 0.0, 0.0, factorization="li")
