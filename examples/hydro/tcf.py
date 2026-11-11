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

"""Turbulent channel flow at a friction Reynolds number Re_tau = 180.

Flow between two parallel plates, driven by a pressure gradient. The channel
is periodic along the flow (x) and across it (z), so it models a section of an
infinitely long and wide channel (a direct numerical simulation, as in Kim, Moin
& Moser, 1987)::

        y
        ^      top wall (no-slip, y = +1)
        |   +--------------------------------+
        |   |  --> --->  ---->  --->  -->    |   periodic in x and z
        |   +--------------------------------+
        |      bottom wall (no-slip, y = -1)
        +-----------------------------------------> x

Geometry and boundary conditions (lengths in units of the half height):

- Channel ``[-L/2, L/2] x [-1, 1] x [-D/2, D/2]`` with L = pi and D = pi/2
  (``--large``: L = 2 pi and D = pi).
- Walls at y = +-1: no-slip. The cells are refined towards them by a growth of
  ~9.5% per cell, which puts the first cell at y+ < 1.
- The flow is driven by a uniform force that balances the current wall shear
  stress, so the mean flow neither accelerates nor decays.

The flow starts from the mean turbulent profile (Reichardt's law of the wall)
plus divergence-free noise, which becomes turbulent after a while.

Usage:
    python examples/hydro/tcf.py
    python examples/hydro/tcf.py --large
"""

import argparse
import math

import torch

import phipict
import phipict.meshing as pm
from phipict import Hook, Hooks
from phipict import _noise as noise
from phipict.analysis import tcf_tools
from phipict.io import export


def make_mesh(length: float, depth: float, nx: int, ny: int, nz: int) -> pm.Mesh:
    """The channel as a single block, periodic in x and z."""
    walls = pm.Patch("walls", pm.Wall())
    block = pm.make_box(
        (-length / 2, -1.0, -depth / 2),
        (length / 2, 1.0, depth / 2),
        cells=(nx, ny, nz),
        # cells grow by a factor sqrt(1.2) per cell away from both walls
        grading=(None, pm.Geometric(math.sqrt(1.2), pm.Cluster.BOTH), None),
        patches=pm.FacePatches(y_minus=walls, y_plus=walls),
        name="Channel",
    )
    mesh = pm.Mesh([block])
    mesh.make_periodic("x")
    mesh.make_periodic("z")
    return mesh


def reichardt(y_plus: torch.Tensor) -> torch.Tensor:
    """Mean velocity in wall units over the wall distance in wall units."""
    kappa = 0.41
    return torch.log(1 + kappa * y_plus) / kappa + 7.8 * (
        1 - torch.exp(-y_plus / 11) - y_plus / 11 * torch.exp(-y_plus / 3)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--re-tau", type=float, default=180.0, help="Re_tau")
    parser.add_argument("--large", action="store_true", help="2 pi x 2 x pi channel")
    parser.add_argument("--ny", type=int, default=64, help="cells between the walls")
    parser.add_argument("--time", type=float, default=50.0, help="simulated time")
    parser.add_argument("--out", default=None, help="optional VTK output path")
    args = parser.parse_args()

    length, depth = (2 * math.pi, math.pi) if args.large else (math.pi, math.pi / 2)
    n = 128 if args.large else 64  # cells along x and z
    mesh = make_mesh(length, depth, n, args.ny, n)
    print(mesh)

    # The centreline Reynolds number Re_c = u_c delta / nu is set to 1 / nu with
    # the half height delta = 1; the friction velocity follows from Re_tau
    re_center = tcf_tools.Re_wall_to_cl(args.re_tau)
    viscosity = 1.0 / re_center
    u_tau = args.re_tau / re_center

    dtype, device = torch.float32, torch.device("cuda")
    domain = mesh.get_domain(
        viscosity=viscosity,
        dtype=dtype,
        device=device,
        passive_scalar_channels=0,
        name="Channel",
    )
    block = domain.getBlocks()[0]

    # Initial state: the mean turbulent profile plus curl noise of half its size
    y = mesh.blocks[0].cell_centers()[1].to(device, dtype)  # [nz, ny, nx]
    u = u_tau * reichardt((1 - y.abs()) * u_tau / viscosity)
    velocity = torch.stack([u, torch.zeros_like(u), torch.zeros_like(u)])[None]
    curl_noise = noise.GenerateSimplexNoiseVariation(
        [n, args.ny, n],
        device,
        [2 / n, 2 / args.ny, 2 / n],  # noise scale per axis
        [0] * 3,  # offset
        noise.NoiseVariation.CURL,
    ).to(dtype)
    curl_noise *= 0.5 * velocity[:, :1] / curl_noise.norm(dim=1).max()
    block.setVelocity((velocity + curl_noise).contiguous())

    # Wall distance of the first cell centres, for the wall shear stress
    y_centers = y[0, :, 0]
    first_cells = (1 + y_centers[0], 1 - y_centers[-1])

    # Drive the flow with the mean wall shear stress of both walls: a uniform
    # force along x, the pressure gradient that balances the wall friction
    def apply_forcing(domain: phipict.Domain, **kwargs: object) -> None:
        block = domain.getBlocks()[0]
        mean_u = block.velocity[0, 0].mean(dim=(0, 2))  # profile over y
        tau_bottom = viscosity * mean_u[0] / first_cells[0]
        tau_top = viscosity * mean_u[-1] / first_cells[1]
        force = 0.5 * (tau_bottom + tau_top)
        block.setVelocitySource(
            torch.stack([force, 0 * force, 0 * force]).reshape(1, 3).contiguous()
        )

    hooks = Hooks().append(Hook.PRE, apply_forcing)
    series = None
    if args.out is not None:
        series = export.VTKSeries(args.out, mesh=mesh)
        hooks.append(Hook.POST, series.hook(every=50))

    # Time step: 0.06 viscous time units nu / u_tau^2
    dt = 0.06 * tcf_tools.t_star(viscosity, u_tau)
    sim = phipict.Simulation(
        domain=domain,
        dt=dt,
        substeps="ADAPTIVE",
        adaptive_CFL=0.1,
        corrector_steps=2,
        advection_use_BiCG=True,
        advection_tol=1e-6,
        pressure_tol=1e-6,
        pressure_return_best_result=True,
        velocity_corrector="FD",
        non_orthogonal=True,
        hooks=hooks,
    )
    sim.make_divergence_free()

    steps = round(args.time / dt)
    for step in range(1, steps + 1):
        sim.single_step()
        if step % 100 == 0 or step == steps:
            # Friction Reynolds number from the current wall shear stress
            mean_u = block.velocity[0, 0].mean(dim=(0, 2))
            tau = (
                0.5
                * viscosity
                * (mean_u[0] / first_cells[0] + mean_u[-1] / first_cells[1])
            )
            re_tau = float(tau.sqrt() / viscosity)
            print(f"t = {step * dt:6.2f}: Re_tau = {re_tau:.1f}")
    if series is not None:
        print(f"wrote {series.path}")


if __name__ == "__main__":
    main()
