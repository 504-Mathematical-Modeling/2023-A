"""问题一核心计算冒烟测试（CI 用，单时刻小规模，确保不超过 1 分钟）。"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "01_代码"))

from q1_functions import (
    MIRROR_AREA,
    build_neighbors,
    collector_trunc_efficiency,
    dni,
    effective_points,
    field_power_mw,
    load_mirrors,
    mirror_attitude,
    mirror_optical_efficiency,
    precompute_position_terms,
    sun_geometry,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTACH = os.path.join(ROOT, "00_题目与数据", "附件.xlsx")


@pytest.fixture(scope="module")
def field():
    _, mc = load_mirrors(ATTACH)
    area = np.full(len(mc), MIRROR_AREA)
    pt = precompute_position_terms(mc)
    return mc, area, pt


def test_mirror_count(field):
    mc, _, _ = field
    assert len(mc) == 1745


def test_sun_geometry_sane():
    _, _, _, _, s_summer = sun_geometry(6, 12.0)
    _, _, _, _, s_winter = sun_geometry(12, 12.0)
    assert s_summer[2] > s_winter[2] > 0
    assert s_summer[0] ** 2 + s_summer[1] ** 2 + s_summer[2] ** 2 == pytest.approx(1.0)


def test_single_moment_eta_bounds(field):
    mc, area, pt = field
    nb = build_neighbors(mc, 20.0)
    _, _, _, _, s = sun_geometry(6, 12.0)
    _, n, ec, _, ea = mirror_attitude(mc, s, pt)
    P, valid = effective_points(mc, s, n, nb)
    esb = valid.mean(axis=1)
    et = collector_trunc_efficiency(P, s, n, valid)
    ei = mirror_optical_efficiency(ec, ea, eta_sb=esb, eta_trunc=et)
    eta = float(np.sum(area * ei) / np.sum(area))
    p = field_power_mw(dni(s[2]), area, ei)
    assert 0.0 < eta < 1.0
    assert 0.0 < p < 100.0
    assert 0.0 < esb.mean() < 1.0
    assert 0.0 < et.mean() < 1.0


def test_annual_reproduces_paper(field):
    """60 时刻年平均应与论文数值一致（允许微小舍入误差）。"""
    mc, area, pt = field
    nb = build_neighbors(mc, 20.0)
    e_sum = p_sum = 0.0
    n = 0
    for m in range(1, 13):
        for h in [9.0, 10.5, 12.0, 13.5, 15.0]:
            _, _, _, _, s = sun_geometry(m, h)
            _, nv, ec, _, ea = mirror_attitude(mc, s, pt)
            P, v = effective_points(mc, s, nv, nb)
            esb = v.mean(axis=1)
            et = collector_trunc_efficiency(P, s, nv, v)
            ei = mirror_optical_efficiency(ec, ea, eta_sb=esb, eta_trunc=et)
            e_sum += float(np.sum(area * ei) / np.sum(area))
            p_sum += field_power_mw(dni(s[2]), area, ei)
            n += 1
    eta_annual = e_sum / n
    p_annual = p_sum / n
    assert eta_annual == pytest.approx(0.5777, abs=5e-4)
    assert p_annual == pytest.approx(35.32, abs=0.05)