"""kygwow fork regressions: exact-normal azimuth and non-finite warning."""

import math
import warnings

import pytest
import torch

import torcwa


def _line_grating(theta, azimuth, period=200e-9, wavelength=470e-9):
    sim = torcwa.rcwa(freq=1 / wavelength, order=[5, 0], L=[period, period], dtype=torch.complex128, device=torch.device("cpu"))
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=(4.497 + 0.1j) ** 2)
    sim.set_incident_angle(theta, azimuth)
    x = torch.linspace(-0.5, 0.5, 64, dtype=torch.float64)
    fill = (torch.abs(x) < 0.25).to(torch.complex128)[:, None].repeat(1, 64)  # lines along y
    eps_si = (4.497 + 0.1j) ** 2
    sim.add_layer(100e-9, eps=1.0 + fill * (eps_si - 1.0))
    sim.solve_global_smatrix()
    return sim


@pytest.mark.parametrize("azimuth_deg", [0.0, 40.0, 90.0, 135.0])
@pytest.mark.parametrize("code", ["pp", "sp", "ps", "ss"])
def test_exact_normal_matches_the_oblique_limit(azimuth_deg, code):
    azimuth = math.radians(azimuth_deg)
    orders = [[m, 0] for m in range(-5, 6)]
    normal = _line_grating(0.0, azimuth).S_parameters(orders, direction="forward", port="reflection", polarization=code, ref_order=[0, 0], power_norm=False)
    near = _line_grating(1e-7, azimuth).S_parameters(orders, direction="forward", port="reflection", polarization=code, ref_order=[0, 0], power_norm=False)
    assert torch.allclose(normal, near, atol=1e-6)


def test_nonfinite_replacement_warns():
    # pitch = wavelength at normal incidence puts orders +-1 on the air Rayleigh anomaly.
    sim = _line_grating(0.0, 0.0, period=470e-9)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.S_parameters([[1, 0]], direction="forward", port="reflection", polarization="pp", ref_order=[0, 0], power_norm=True)
    assert any("non-finite" in str(item.message) for item in caught)
