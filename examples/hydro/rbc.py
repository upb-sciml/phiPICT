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

"""Rayleigh-Bénard convection: a fluid layer heated from below.

A layer of height 1 between a hot bottom plate and a cold top plate. Above a
critical Rayleigh number, buoyancy drives convection rolls (2D) or cells (3D)::

        y (up)
        ^         cold plate, T = 0 (y = +1/2)
        |   ----------------------------------
        |     ^   |    ^   |    ^   |    ^
        |   ( |   v )( |   v )( |   v )( |  v )   periodic in x (and z)
        |   ----------------------------------
        |         hot plate, T = 1 (y = -1/2)
        +------------------------------------------> x

Geometry and boundary conditions:

- Layer ``[0, pi] x [-1/2, 1/2]`` (x ``[0, pi]`` in 3D), periodic in x (and z).
- Bottom plate: no-slip, T = 1. Top plate: no-slip, T = 0.
- The cells are refined towards both plates for the thermal boundary layers.

The equations use the free-fall scaling: the viscosity is sqrt(Pr / Ra), the
thermal diffusivity 1 / sqrt(Ra Pr), and the buoyancy force is T along +y.

Usage:
    python examples/hydro/rbc.py              # 2D, Ra = 8e4
    python examples/hydro/rbc.py --three-d    # 3D, Ra = 2e3
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
LENGTH = math.pi  # width of the layer in x (and z), the height is 1


def make_mesh(
    nx: int, three_d: bool, dtype: torch.dtype, device: torch.device
) -> pm.Mesh:
    """The layer as a single block, with twice as many cells per unit length in y."""
    ny = round(2.0 * nx / LENGTH)

    def temperature(value: float) -> torch.Tensor:
        return torch.tensor([[value]], dtype=dtype, device=device)

    hot = pm.Patch(
        "hot_plate", pm.Wall(extra=(bc.Scalar.Dirichlet(temperature(T_HOT)),))
    )
    cold = pm.Patch(
        "cold_plate", pm.Wall(extra=(bc.Scalar.Dirichlet(temperature(T_COLD)),))
    )

    block = pm.make_box(
        (0.0, -0.5),
        (LENGTH, 0.5),
        cells=(nx, ny),
        # cells grow by 2% per cell away from each plate
        grading=(None, pm.Geometric(1.02, pm.Cluster.BOTH)),
        patches=pm.FacePatches(y_minus=hot, y_plus=cold),
        name="Layer",
    )
    mesh = pm.Mesh([block])
    mesh.make_periodic("x")
    if three_d:
        mesh = mesh.extrude((0.0, LENGTH), cells=nx, periodic=True)
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--three-d", action="store_true", help="3D layer")
    parser.add_argument("--ra", type=float, default=None, help="Rayleigh number")
    parser.add_argument("--pr", type=float, default=0.7, help="Prandtl number")
    parser.add_argument("--nx", type=int, default=None, help="cells along x (and z)")
    parser.add_argument("--time", type=float, default=100.0, help="free-fall times")
    parser.add_argument("--out", default=None, help="optional VTK output path")
    args = parser.parse_args()
    # Defaults: 96 cells (2D) and 64 cells (3D) along x
    ra = args.ra or (2e3 if args.three_d else 8e4)
    nx = args.nx or (64 if args.three_d else 96)

    dtype, device = torch.float32, torch.device("cuda")
    mesh = make_mesh(nx, args.three_d, dtype, device)
    print(mesh)

    # Free-fall scaling
    viscosity = math.sqrt(args.pr / ra)
    diffusivity = 1.0 / math.sqrt(ra * args.pr)
    domain = mesh.get_domain(
        viscosity=viscosity,
        scalar_viscosity=torch.tensor([diffusivity], dtype=dtype, device=device),
        dtype=dtype,
        device=device,
        name="RBC",
    )
    block = domain.getBlocks()[0]

    # Linear conduction profile plus noise in temperature and velocity
    y = mesh.blocks[0].cell_centers()[1].to(device, dtype)  # [(nz,) ny, nx]
    T = T_HOT + (T_COLD - T_HOT) * (y + 0.5)
    T = T + 0.1 * (T_HOT - T_COLD) * torch.randn_like(T)
    block.setPassiveScalar(T.clamp(T_COLD, T_HOT)[None, None].contiguous())
    block.setVelocity((0.05 * torch.randn_like(block.velocity)).contiguous())

    # Buoyancy along +y, T in the free-fall scaling
    def apply_buoyancy(domain: phipict.Domain, **kwargs: object) -> None:
        block = domain.getBlocks()[0]
        T = block.passiveScalar
        assert T is not None
        zero = torch.zeros_like(T)
        source = [zero, T, zero] if args.three_d else [zero, T]
        block.setVelocitySource(torch.cat(source, dim=1))
        domain.UpdateDomainData()

    hooks = Hooks().append(Hook.PRE_VELOCITY_SETUP, apply_buoyancy)
    series = None
    if args.out is not None:
        series = export.VTKSeries(args.out, mesh=mesh)
        hooks.append(Hook.POST, series.hook(every=20))

    sim = phipict.Simulation(
        domain=domain,
        dt=0.05,
        substeps="ADAPTIVE",
        adaptive_CFL=0.8,
        corrector_steps=2,
        pressure_tol=1e-5,
        pressure_return_best_result=True,
        velocity_corrector="FD",
        non_orthogonal=False,  # the cells are rectangular
        hooks=hooks,
    )
    sim.make_divergence_free()

    # Volume-weighted Nusselt number Nu = 1 + sqrt(Ra Pr) <u_y T>
    volume = cell_volumes(mesh.blocks[0].coords).to(device, dtype)

    def nusselt() -> float:
        T = block.passiveScalar
        assert T is not None
        flux = block.velocity[0, 1] * T[0, 0]
        mean = (flux * volume).sum() / volume.sum()
        return 1.0 + math.sqrt(ra * args.pr) * float(mean)

    steps = round(args.time / 0.05)
    for step in range(1, steps + 1):
        sim.single_step()
        if step % 100 == 0 or step == steps:
            print(f"t = {step * 0.05:6.1f}: Nu = {nusselt():.3f}")
    if series is not None:
        print(f"wrote {series.path}")


if __name__ == "__main__":
    main()
