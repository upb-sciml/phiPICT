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

"""Cooling duct of a fusion reactor blanket: heated liquid metal in a magnetic field.

A rectangular duct of length ``L`` carries a liquid metal (Pr = 0.025) from an
inlet at x = 0 to an outlet at x = L. One side wall is heated, and the strong
magnetic field of the reactor points across the duct::

            y (field B)
            ^
            |   +-------------------------------+
            |  /     Hartmann wall (y = +Ly/2) /|
            | +-------------------------------+ |
    inlet   | |                               | |   outlet
    (x = 0) | |  --> flow along x             | +   (x = L)
            | |                               |/
            | +-------------------------------+
            +------------------------------------> x
           /    heated wall at z = -Lz/2, cooled by the flow
          z

Geometry and boundary conditions (lengths in units of the half-gap Lz / 2):

- Duct ``[0, L] x [-Ly/2, Ly/2] x [-Lz/2, Lz/2]`` with Ly = 7 and Lz = 2.
- Inlet (x = 0): parabolic velocity profile with bulk velocity 1, T = 0.
- Outlet (x = L): advective outflow, open for the electric current.
- Shercliff walls (z = +-Lz/2, parallel to B): no-slip, electrically insulating.
  The -z wall is heated with a constant heat flux that ramps up over the first
  7.5 units; the +z wall is adiabatic.
- Hartmann walls (y = +-Ly/2, normal to B): no-slip, electrically insulating,
  adiabatic.
- Magnetic field along y, ramped up smoothly over the first unit of the duct.
- Gravity along +x, so the heated fluid near the hot wall is decelerated by
  buoyancy (Boussinesq approximation).

Usage:
    python examples/mhd/blanket_cooling.py                 # L = 75
    python examples/mhd/blanket_cooling.py --length 18.75  # shorter duct
"""

import argparse

import torch

import phipict
import phipict.meshing as pm
from phipict import Hook, Hooks, bc
from phipict import _noise as noise
from phipict.io import export

# Cross-section of the duct, in units of the half-gap Lz / 2
LY = 7.0  # along the field (Hartmann direction)
LZ = 2.0  # between the Shercliff walls, one of them heated

# Streamwise cell size: 1920 cells over the default length L = 75
DX = 75.0 / 1920

T_IN = 0.0  # inlet temperature
T_REF = 0.5  # reference temperature of the buoyancy force
HEAT_FLUX = 2.0  # temperature gradient at the heated wall
HEATING_RAMP = 7.5  # length over which the wall heat flux ramps up
FIELD_RAMP = 1.0  # length over which the magnetic field ramps up


def smoothstep(x: torch.Tensor, length: float) -> torch.Tensor:
    """Rises smoothly from 0 at x = 0 to 1 at x = ``length``, then stays at 1."""
    t = torch.clamp(x / length, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def parabolic_inlet(points: torch.Tensor) -> torch.Tensor:
    """Parabolic inflow with bulk velocity 1, zero at all four walls.

    ``points`` are the inlet face centres ``[n, 3]``; returns velocities ``[n, 3]``.
    The product of two parabolas has mean 4/9, hence the factor 9/4.
    """
    y, z = points[:, 1], points[:, 2]
    u = 2.25 * (1.0 - (2.0 * y / LY) ** 2) * (1.0 - (2.0 * z / LZ) ** 2)
    return torch.stack([u, torch.zeros_like(u), torch.zeros_like(u)], dim=1)


def make_mesh(
    length: float, ny: int, nz: int, dtype: torch.dtype, device: torch.device
) -> pm.Mesh:
    """The duct as a single block, with one patch per boundary condition."""
    nx = round(length / DX)

    # The boundary conditions are given per patch; temperature values must live
    # on the device of the domain
    def temperature(value: float) -> torch.Tensor:
        return torch.tensor([[value]], dtype=dtype, device=device)

    # Wall heat flux at the face centres of the heated wall, ramped up along x
    # (the cells are uniform along x)
    x = (torch.arange(nx, dtype=dtype) + 0.5) * length / nx
    flux = HEAT_FLUX * smoothstep(x, HEATING_RAMP)
    flux = flux.reshape(1, 1, 1, 1, nx).expand(1, 1, 1, ny, nx)  # [1, 1, 1, ny, nx]

    inlet = pm.Patch(
        "inlet",
        pm.Inflow(
            parabolic_inlet,
            extra=(bc.Potential.Open(), bc.Scalar.Dirichlet(temperature(T_IN))),
        ),
    )
    outlet = pm.Patch(
        "outlet",
        pm.Outflow(
            extra=(
                bc.Potential.Open(),
                # varying, so that the outflow hook can advect the temperature
                bc.Scalar.Dirichlet(
                    torch.full((1, 1, nz, ny, 1), T_IN, dtype=dtype, device=device)
                ),
            )
        ),
    )
    adiabatic = bc.Scalar.Neumann(temperature(0.0))
    heated_wall = pm.Patch(
        "heated_wall",
        pm.Wall(extra=(bc.Scalar.Neumann(flux.to(device, dtype).contiguous()),)),
    )
    shercliff_wall = pm.Patch("shercliff_wall", pm.Wall(extra=(adiabatic,)))
    hartmann_walls = pm.Patch("hartmann_walls", pm.Wall(extra=(adiabatic,)))

    block = pm.make_box(
        (0.0, -LY / 2, -LZ / 2),
        (length, LY / 2, LZ / 2),
        cells=(nx, ny, nz),
        # uniform along the flow, refined towards the thin Hartmann layers (y)
        # and, more mildly, towards the side layers (z)
        grading=(None, pm.Tanh(3.3), pm.ChebyshevBlend(0.96)),
        patches=pm.FacePatches(
            x_minus=inlet,
            x_plus=outlet,
            y_minus=hartmann_walls,
            y_plus=hartmann_walls,
            z_minus=heated_wall,
            z_plus=shercliff_wall,
        ),
        name="Duct",
    )
    return pm.Mesh([block])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=float, default=75.0, help="duct length L")
    parser.add_argument("--ny", type=int, default=164, help="cells along the field")
    parser.add_argument("--nz", type=int, default=64, help="cells across the gap")
    parser.add_argument("--re", type=float, default=5000.0, help="Reynolds number")
    parser.add_argument("--ha", type=float, default=200.0, help="Hartmann number")
    parser.add_argument("--pr", type=float, default=0.025, help="Prandtl number")
    parser.add_argument("--gr", type=float, default=1.25e6, help="Grashof number")
    parser.add_argument("--time", type=float, default=10.0, help="simulated time")
    parser.add_argument("--out", default=None, help="optional VTK output path")
    args = parser.parse_args()

    dtype, device = torch.float64, torch.device("cuda")
    mesh = make_mesh(args.length, args.ny, args.nz, dtype, device)
    print(mesh)

    viscosity = 1.0 / args.re  # bulk velocity 1 and half-gap 1
    domain = mesh.get_domain(
        viscosity=viscosity,
        scalar_viscosity=torch.tensor(
            [viscosity / args.pr], dtype=dtype, device=device
        ),
        dtype=dtype,
        device=device,
        name="Blanket",
    )
    block = domain.getBlocks()[0]

    # Start from the bulk velocity plus divergence-free (curl) noise, which
    # triggers the buoyant instability sooner
    nz, ny, nx = block.velocity.shape[2:]
    velocity = torch.zeros_like(block.velocity)
    velocity[0, 0] = 1.0
    curl_noise = noise.GenerateSimplexNoiseVariation(
        (nx, ny, nz),
        device,
        [2 / s for s in (nz, ny, nx)],  # noise scale per axis
        [0] * 3,  # offset
        noise.NoiseVariation.CURL,
    )
    block.setVelocity((velocity + 0.1 * curl_noise.to(dtype)).contiguous())

    # Magnetic field along y, ramped up over the first FIELD_RAMP of the duct.
    # A varying field is given per cell, here as [3, 1, 1, nx] (broadcast over
    # z and y)
    x = mesh.blocks[0].cell_centers()[0, 0, 0]
    e_b = torch.zeros(3, 1, 1, nx, dtype=dtype)
    e_b[1, 0, 0] = smoothstep(x, FIELD_RAMP)

    # Buoyancy for gravity along +x: f_x = -Gr / Re^2 (T - T_ref). The hook sets
    # the whole velocity source before every step, and MHDSimulation adds the
    # Lorentz force on top of it, so it has to be registered before the solver
    # runs (PRE_VELOCITY_SETUP).
    buoyancy = args.gr / args.re**2

    def apply_buoyancy(domain: phipict.Domain, **kwargs: object) -> None:
        block = domain.getBlocks()[0]
        T = block.passiveScalar
        assert T is not None
        zero = torch.zeros_like(T)
        block.setVelocitySource(torch.cat([-buoyancy * (T - T_REF), zero, zero], 1))
        domain.UpdateDomainData()

    hooks = (
        Hooks()
        # advect the velocity and temperature out of the outlet
        .append(Hook.PRE, mesh.outflow_hook(domain, velocity=1.0))
        .append(Hook.PRE_VELOCITY_SETUP, apply_buoyancy)
    )
    series = None
    if args.out is not None:
        series = export.VTKSeries(args.out, mesh=mesh)
        hooks.append(Hook.POST, series.hook(every=20))

    domain.setAdvectionScheme(phipict.AdvectionScheme.LINEAR_UPWIND)
    sim = phipict.MHDSimulation(
        domain=domain,
        dt=5e-3,
        stuart_number=torch.tensor(args.ha**2 / args.re),  # N = Ha^2 / Re
        e_b=e_b,
        substeps="ADAPTIVE",
        adaptive_CFL=0.8,
        corrector_steps=2,
        non_orthogonal=False,
        hooks=hooks,
        pressure_tol=phipict.SolverTolerance(rtol=1e-2, atol=1e-14),
        potential_tol=phipict.SolverTolerance(rtol=1e-4, atol=1e-14),
        potential_use_preconditioner=True,  # AMG-preconditioned CG, needs pyamg
    )
    sim.make_divergence_free()

    # Each step advances the time by dt = 0.005, split into adaptive substeps
    steps = round(args.time / 5e-3)
    for step in range(1, steps + 1):
        sim.single_step()
        if step % 100 == 0 or step == steps:
            T = block.passiveScalar[0, 0]
            print(
                f"t = {step * 5e-3:6.2f}: max. wall temperature "
                f"{T[0].max():.4f}, mean outlet temperature {T[..., -1].mean():.4f}"
            )
    if series is not None:
        print(f"wrote {series.path}")


if __name__ == "__main__":
    main()
