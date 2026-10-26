"""Adjoint of the PISO step against central finite differences, and the live state a
backward leaves behind.

A small 2D duct with a passive scalar, a buoyancy-like velocity source and an
advective outflow, in float64 with tight solver tolerances: over a few steps the
finite differences are then exact to ``eps**2``, so the adjoint has to match them to
round-off.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

import phipict  # noqa: E402
from phipict.core.hooks import Hook, Hooks  # noqa: E402
from phipict.core.piso_simulation import update_advective_boundaries  # noqa: E402

DEV = torch.device("cuda")
DTYPE = torch.float64
NX, NY = 24, 10
TIGHT = {
    "advection_tol": phipict.SolverTolerance(rtol=1e-12, atol=1e-14),
    "pressure_tol": phipict.SolverTolerance(rtol=1e-12, atol=1e-14),
}
SCHEMES = [phipict.AdvectionScheme.CENTRAL, phipict.AdvectionScheme.LINEAR_UPWIND]


def make_duct(scheme: phipict.AdvectionScheme) -> phipict.Domain:
    """Duct with a parabolic inflow at -x, an advective outflow at +x, walls at +-y.

    The scalar is 0 at the inflow and 1 at the -y wall.
    """
    domain = phipict.Domain(
        2, torch.tensor([0.02], dtype=DTYPE), name="duct", device=DEV, dtype=DTYPE
    )
    vel = torch.zeros(1, 2, NY, NX, dtype=DTYPE, device=DEV)
    block = domain.CreateBlock(velocity=vel, name="duct")
    y = (torch.arange(NY, dtype=DTYPE, device=DEV) + 0.5) / NY
    inflow = torch.zeros(1, 2, NY, 1, dtype=DTYPE, device=DEV)
    inflow[0, 0, :, 0] = 6 * y * (1 - y)
    block.CloseBoundary("-x", inflow)
    block.CloseBoundary("+x", inflow.clone())
    outflow = block.getBoundary("+x")
    outflow.makeVelocityVarying()
    outflow.setPassiveScalar(torch.zeros(1, 1, NY, 1, dtype=DTYPE, device=DEV))
    block.CloseBoundary("-y")
    block.CloseBoundary("+y")
    block.getBoundary("-y").setPassiveScalar(torch.ones(1, 1, dtype=DTYPE, device=DEV))
    domain.setAdvectionScheme(scheme)
    domain.PrepareSolve()
    return domain


def initial_state() -> tuple[torch.Tensor, torch.Tensor]:
    """A perturbed inflow profile, and a scalar that is exactly 0 near the inlet.

    The uniform region has flat faces, where the forward skips the deferred
    correction of LINEAR_UPWIND.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    y = (torch.arange(NY, dtype=DTYPE) + 0.5) / NY
    u = torch.zeros(1, 2, NY, NX, dtype=DTYPE)
    u[0, 0] = 6 * y[:, None] * (1 - y[:, None])
    u = u + 0.2 * torch.randn(u.shape, generator=g, dtype=DTYPE)
    T = torch.rand(1, 1, NY, NX, generator=g, dtype=DTYPE)
    T[..., : NX // 4] = 0.0
    return u.to(DEV), T.to(DEV)


def run(domain, u0, T0, steps, differentiable, backward_after=None, loss_fn=None):
    """Advance ``steps`` steps from ``u0``, ``T0``; optionally back-propagate
    ``loss_fn`` of the state after step ``backward_after`` and continue."""
    block = domain.getBlocks()[0]
    outflow = block.getBoundary("+x")
    block.setVelocity(u0.clone())
    block.setPassiveScalar(T0.clone())
    domain.UpdateDomainData()
    char_vel = torch.tensor([[1.0, 0.0]], dtype=DTYPE, device=DEV)

    def buoyancy_fn(domain, time_step, **kwargs):
        T = block.passiveScalar
        block.setVelocitySource(torch.cat([0.5 * (T - 0.5), torch.zeros_like(T)], 1))
        domain.UpdateDomainData()

    def outflow_fn(domain, time_step, **kwargs):
        update_advective_boundaries(
            domain,
            [outflow],
            char_vel,
            time_step.to(DEV),
            differentiable=differentiable,
        )

    hooks = (
        Hooks()
        .append(Hook.PRE_VELOCITY_SETUP, buoyancy_fn)
        .append(Hook.PRE, outflow_fn)
    )
    sim = phipict.Simulation(
        domain=domain,
        dt=0.05,
        substeps=1,
        non_orthogonal=False,
        hooks=hooks,
        differentiable=differentiable,
        **TIGHT,
    )
    for step in range(steps):
        sim.single_step()
        if step == backward_after:
            loss_fn(block).backward()
            domain.Detach()
    return block


@pytest.mark.parametrize("scheme", SCHEMES, ids=lambda s: s.name)
def test_adjoint_matches_finite_differences(scheme):
    steps, eps = 3, 1e-5
    u0, T0 = initial_state()
    g = torch.Generator(device="cpu").manual_seed(1)

    def randn(like):
        return torch.randn(like.shape, generator=g, dtype=DTYPE).to(DEV)

    w_u, w_T = randn(u0), randn(T0)

    def loss(block):
        return (w_u * block.velocity).sum() + (w_T * block.passiveScalar).sum()

    u = u0.clone().requires_grad_(True)
    T = T0.clone().requires_grad_(True)
    block = run(make_duct(scheme), u, T, steps, differentiable=True)
    grad_u, grad_T = torch.autograd.grad(loss(block), [u, T])

    inlet = torch.zeros(1, 1, NY, NX, dtype=torch.bool, device=DEV)
    inlet[..., : NX // 4] = True
    directions = {
        "u": (randn(u0), torch.zeros_like(T0)),
        "T": (torch.zeros_like(u0), randn(T0) * ~inlet),
        "T_uniform_region": (torch.zeros_like(u0), randn(T0) * inlet),
    }
    for name, (du, dT) in directions.items():
        with torch.no_grad():
            j_plus = loss(
                run(make_duct(scheme), u0 + eps * du, T0 + eps * dT, steps, True)
            )
            j_minus = loss(
                run(make_duct(scheme), u0 - eps * du, T0 - eps * dT, steps, True)
            )
        fd = float(j_plus - j_minus) / (2 * eps)
        adjoint = float((grad_u * du).sum() + (grad_T * dT).sum())
        assert adjoint == pytest.approx(fd, rel=1e-6), name


@pytest.mark.parametrize("scheme", SCHEMES, ids=lambda s: s.name)
def test_backward_keeps_live_state(scheme):
    """A forward that continues after a backward is the uninterrupted forward."""
    u0, T0 = initial_state()

    def loss(block):
        return (block.velocity**2).sum() + block.passiveScalar.sum()

    reference = run(make_duct(scheme), u0, T0, 3, differentiable=True)
    u_ref = reference.velocity.detach().clone()
    T_ref = reference.passiveScalar.detach().clone()
    outflow_ref = reference.getBoundary("+x").velocity.detach().clone()

    u = u0.clone().requires_grad_(True)
    block = run(make_duct(scheme), u, T0, 3, True, backward_after=1, loss_fn=loss)
    assert u.grad is not None
    assert torch.equal(block.velocity.detach(), u_ref)
    assert torch.equal(block.passiveScalar.detach(), T_ref)
    assert torch.equal(block.getBoundary("+x").velocity.detach(), outflow_ref)
