# Copyright 2026 Jannis Becktepe
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""3D magnetoconvection: Rayleigh-Bénard convection in a horizontal magnetic field.

A closed box of liquid metal (Pr = 0.025) is heated from below and cooled from
above. Buoyancy drives convection rolls; a horizontal magnetic field along x
damps all motion across it, so the rolls align with the field::

        y (up)
        ^      cold plate, T = 0 (y = +H/2)
        |   +---------------------------+
        |   |      ^         |          |
        |   |   (  |  roll   v  )       |  --> field B along x
        |   |      |                    |
        |   +---------------------------+
        |      hot plate, T = 1 (y = -H/2)
        +------------------------------------> x

Geometry and boundary conditions (lengths in units of the height H = 1):

- Box ``[0, Lx] x [-H/2, H/2] x [0, Lz]`` with Lx = Lz = 5 H by default.
- Bottom plate (y = -H/2): no-slip, T = 1. Top plate (y = +H/2): no-slip, T = 0.
- Side walls (x and z): no-slip, adiabatic.
- All walls are electrically insulating.

The equations use the free-fall scaling: the viscosity is sqrt(Pr / Ra), the
thermal diffusivity 1 / sqrt(Ra Pr), and the buoyancy force is T - 1/2. The
field strength enters through the Chandrasekhar number Q = Ha^2, as the
interaction parameter N = Q sqrt(Pr / Ra).

The walls at x = 0 and x = Lx are normal to the field and carry thin Hartmann
layers of thickness H / Ha, so the mesh is refined there until each layer holds
a few cells.

Usage:
    python examples/mhd/rbc_mhd.py
    python examples/mhd/rbc_mhd.py --ra 1e5 --q 0     # no magnetic field
"""

import argparse
import math

import torch

import phipict
import phipict.meshing as pm
from phipict import Hook, Hooks, bc
from phipict.io import export
from phipict.meshing.quality import cell_volumes

T_HOT, T_COLD = 1.0, 0.0
T_REF = 0.5 * (T_HOT + T_COLD)


def hartmann_grading(
    cells: int, length: float, layer: float, min_layer_cells: int = 3
) -> pm.Grading:
    """The weakest tanh grading with ``min_layer_cells`` cells in each wall layer.

    Parameters
    ----------
    cells : int
        Number of cells between the two walls.
    length : float
        Distance between the walls.
    layer : float
        Thickness of the boundary layer at each wall.
    min_layer_cells : int, optional
        Cell centres required inside each layer. Default is 3.
    """

    def resolves(beta: float) -> bool:
        w = pm.Tanh(beta).weights(cells)
        centers = 0.5 * (w[1:] + w[:-1])
        return int((centers < layer / length).sum()) >= min_layer_cells

    lo, hi = 0.0, 10.0  # stronger stretching puts more cells into the layer
    if resolves(lo):
        return pm.Uniform()
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if resolves(mid) else (mid, hi)
    return pm.Tanh(hi)


def make_mesh(
    lx: float, lz: float, ny: int, ha: float, dtype: torch.dtype, device: torch.device
) -> pm.Mesh:
    """The box as a single block: hot bottom, cold top, adiabatic sides."""
    height = 1.0
    nx, nz = round(lx / height * ny), round(lz / height * ny)

    def temperature(value: float) -> torch.Tensor:
        return torch.tensor([[value]], dtype=dtype, device=device)

    # No-slip walls everywhere; they differ only in their temperature condition
    hot = bc.Scalar.Dirichlet(temperature(T_HOT))
    cold = bc.Scalar.Dirichlet(temperature(T_COLD))
    adiabatic = bc.Scalar.Neumann(temperature(0.0))
    hot_plate = pm.Patch("hot_plate", pm.Wall(extra=(hot,)))
    cold_plate = pm.Patch("cold_plate", pm.Wall(extra=(cold,)))
    side_walls = pm.Patch("side_walls", pm.Wall(extra=(adiabatic,)))

    # Thin Hartmann layers at the walls normal to the field (x); the plates and
    # the z walls only need a mild refinement for the thermal boundary layers
    layer = height / ha if ha > 0 else math.inf
    block = pm.make_box(
        (0.0, -height / 2, 0.0),
        (lx, height / 2, lz),
        cells=(nx, ny, nz),
        grading=(hartmann_grading(nx, lx, layer), pm.Tanh(2.0), pm.Tanh(2.0)),
        patches=pm.FacePatches(
            x_minus=side_walls,
            x_plus=side_walls,
            y_minus=hot_plate,
            y_plus=cold_plate,
            z_minus=side_walls,
            z_plus=side_walls,
        ),
        name="Box",
    )
    return pm.Mesh([block])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ra", type=float, default=5e4, help="Rayleigh number")
    parser.add_argument("--q", type=float, default=1e3, help="Chandrasekhar number")
    parser.add_argument("--pr", type=float, default=0.025, help="Prandtl number")
    parser.add_argument("--lx", type=float, default=5.0, help="extent along B")
    parser.add_argument("--lz", type=float, default=5.0, help="extent across B")
    parser.add_argument("--ny", type=int, default=32, help="cells over the height")
    parser.add_argument("--time", type=float, default=100.0, help="free-fall times")
    parser.add_argument("--out", default=None, help="optional VTK output path")
    args = parser.parse_args()

    dtype, device = torch.float32, torch.device("cuda")
    ha = math.sqrt(args.q)
    mesh = make_mesh(args.lx, args.lz, args.ny, ha, dtype, device)
    print(mesh)

    # Free-fall scaling
    viscosity = math.sqrt(args.pr / args.ra)
    diffusivity = 1.0 / math.sqrt(args.ra * args.pr)
    domain = mesh.get_domain(
        viscosity=viscosity,
        scalar_viscosity=torch.tensor([diffusivity], dtype=dtype, device=device),
        dtype=dtype,
        device=device,
        name="RBC-MHD",
    )
    block = domain.getBlocks()[0]

    # At rest, with the linear conduction profile plus 1% noise to start the rolls
    y = mesh.blocks[0].cell_centers()[1].to(device, dtype)  # [nz, ny, nx]
    T = T_HOT + (T_COLD - T_HOT) * (y + 0.5)
    T = T + 0.01 * (T_HOT - T_COLD) * torch.randn_like(T)
    block.setPassiveScalar(T[None, None].contiguous())

    # Buoyancy along +y, (T - T_ref) in the free-fall scaling. MHDSimulation adds
    # the Lorentz force on top of the velocity source, so it is reset every step.
    def apply_buoyancy(domain: phipict.Domain, **kwargs: object) -> None:
        block = domain.getBlocks()[0]
        T = block.passiveScalar
        assert T is not None
        zero = torch.zeros_like(T)
        block.setVelocitySource(torch.cat([zero, T - T_REF, zero], dim=1))
        domain.UpdateDomainData()

    hooks = Hooks().append(Hook.PRE_VELOCITY_SETUP, apply_buoyancy)
    series = None
    if args.out is not None:
        series = export.VTKSeries(args.out, mesh=mesh)
        hooks.append(Hook.POST, series.hook(every=10))

    domain.setAdvectionScheme(phipict.AdvectionScheme.LINEAR_UPWIND)
    sim = phipict.MHDSimulation(
        domain=domain,
        dt=0.05,
        stuart_number=torch.tensor(args.q * math.sqrt(args.pr / args.ra)),
        e_b=torch.tensor([1.0, 0.0, 0.0], dtype=dtype),  # field along x
        substeps="ADAPTIVE",
        adaptive_CFL=0.8,
        corrector_steps=2,
        non_orthogonal=False,
        hooks=hooks,
        pressure_tol=phipict.SolverTolerance(atol=1e-6),
        potential_tol=phipict.SolverTolerance(rtol=1e-4, atol=1e-14),
        potential_use_preconditioner=True,  # AMG-preconditioned CG, needs pyamg
    )
    sim.make_divergence_free()

    # Volume-weighted Nusselt number Nu = 1 + sqrt(Ra Pr) <u_y (T - T_ref)>
    volume = cell_volumes(mesh.blocks[0].coords).to(device, dtype)  # [nz, ny, nx]

    def nusselt() -> float:
        T = block.passiveScalar
        assert T is not None
        flux = block.velocity[0, 1] * (T[0, 0] - T_REF)
        mean = (flux * volume).sum() / volume.sum()
        return 1.0 + math.sqrt(args.ra * args.pr) * float(mean)

    steps = round(args.time / 0.05)
    for step in range(1, steps + 1):
        sim.single_step()
        if step % 100 == 0 or step == steps:
            print(f"t = {step * 0.05:6.1f}: Nu = {nusselt():.3f}")
    if series is not None:
        print(f"wrote {series.path}")


if __name__ == "__main__":
    main()
