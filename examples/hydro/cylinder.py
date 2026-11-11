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

"""Flow past a cylinder in a channel: the von Kármán vortex street.

A cylinder of diameter 1 sits slightly below the centre of a 22 x 4.1 channel
(the benchmark of Schäfer & Turek, 1996). At Re = 100 it sheds vortices
periodically::

        +y wall (no-slip)
    +----+------------+------------------------------------------+
    |    | quad  top  |                                          |
    | in +----+--+----+                                          | out
    | -> |left|()|rght|          wake ("vortex street")          | ->
    |    +----+--+----+                                          |
    |    | quad bottom|                                          |
    +----+------------+------------------------------------------+
        -y wall (no-slip)

The cylinder is surrounded by an O-grid of four blocks: each is a quarter ring
around the cylinder joined with a quad that reaches the square around it. A fifth
block holds the wake. Blocks are connected wherever their faces coincide.

Boundary conditions:

- Inlet (x = -2): parabolic velocity profile with mean velocity 1.
- Outlet (x = 20): advective outflow.
- Channel walls and cylinder: no-slip.

Usage:
    python examples/hydro/cylinder.py
    python examples/hydro/cylinder.py --three-d   # extruded, periodic in z
"""

import argparse
import math

import torch

import phipict
import phipict.meshing as pm
from phipict import Hook, Hooks
from phipict.io import export

RADIUS = 0.5  # cylinder diameter 1
RING = 0.5  # thickness of the ring blocks around the cylinder
QUAD = 1.0  # thickness of the quad blocks around the rings (x)
OFFSET = 0.05  # the cylinder sits this far below the channel centre
LENGTH = 22.0  # channel length; the inlet is at x = -2


def parabola(y_min: float, y_max: float) -> pm.boundary.VelocityProfile:
    """Parabolic inflow between the walls at ``y_min`` and ``y_max``, mean 1."""

    def profile(points: torch.Tensor) -> torch.Tensor:
        t = (points[:, 1] - y_min) / (y_max - y_min)
        u = 6.0 * t * (1.0 - t)
        zero = torch.zeros_like(u)
        return torch.stack([u] + [zero] * (points.shape[1] - 1), dim=1)

    return profile


def make_mesh(n: int, three_d: bool) -> pm.Mesh:
    """O-grid around the cylinder plus a wake block; ``n`` cells per quarter ring."""
    r_out = RADIUS + RING  # outer radius of the rings
    half_x = r_out + QUAD  # half width of the square around the cylinder
    half_y = r_out + QUAD + OFFSET  # half height (the cylinder is off-centre)
    top, bottom = half_y + OFFSET, -(half_y - OFFSET)  # channel walls
    corner = math.sin(math.radians(45)) * r_out  # corner of the rings, 45 deg

    # Boundary conditions
    cylinder = pm.Patch("cylinder", pm.Wall())
    walls = pm.Patch("walls", pm.Wall())
    inlet = pm.Patch("inlet", pm.Inflow(parabola(bottom, top)))
    outlet = pm.Patch("outlet", pm.Outflow())

    arc = pm.Arc(center=(0.0, 0.0))  # the outer edge of the rings
    wake_grading = pm.Geometric(0.95, pm.Cluster.BOTH)  # refined towards the wake

    def ring(start_deg: float) -> pm.MeshBlock:
        """Quarter ring, clockwise from ``start_deg``: x along it, y outwards."""
        return pm.make_annulus(
            (0.0, 0.0), RADIUS, r_out, start_deg, -90.0, cells=(n, None)
        )

    # Radial cells of the quads, so that they continue the rings' cell size
    n_radial = math.ceil((QUAD + OFFSET) / RING * ring(135).cells[1]) - 1

    # Each block is a quarter ring joined with a quad. The rings are re-oriented
    # so that their axes match the quads they are joined with.
    block_top = ring(135).concat(
        pm.make_quad(
            [(-corner, corner), (corner, corner), (-half_x, top), (half_x, top)],
            cells=(n, n_radial),
            edges=pm.QuadEdges(y_minus=arc),
        ),
        "y",
    )
    block_right = (
        ring(45)
        .permute("yx")
        .flip("y")
        .concat(
            pm.make_quad(
                [(corner, -corner), (half_x, bottom), (corner, corner), (half_x, top)],
                cells=(n_radial, n),
                grading=(None, wake_grading),
                edges=pm.QuadEdges(x_minus=pm.Edge(arc, pm.Uniform())),
            ),
            "x",
        )
    )
    block_bottom = pm.make_quad(
        [(-half_x, bottom), (half_x, bottom), (-corner, -corner), (corner, -corner)],
        cells=(n, n_radial),
        edges=pm.QuadEdges(y_plus=arc),
    ).concat(ring(-45).flip("x").flip("y"), "y")
    block_left = pm.make_quad(
        [(-half_x, bottom), (-corner, -corner), (-half_x, top), (-corner, corner)],
        cells=(n_radial, n),
        edges=pm.QuadEdges(x_plus=arc),
    ).concat(ring(-135).permute("yx").flip("x"), "x")
    wake = pm.make_box(
        (half_x, bottom),
        (LENGTH - half_x, top),
        cells=(int((n_radial + 1) / (QUAD + OFFSET) * 18), n),
        grading=(None, wake_grading),
    )

    block_left.patches = pm.FacePatches(x_minus=inlet, x_plus=cylinder)
    block_top.patches = pm.FacePatches(y_minus=cylinder, y_plus=walls)
    block_right.patches = pm.FacePatches(x_minus=cylinder)
    block_bottom.patches = pm.FacePatches(y_minus=walls, y_plus=cylinder)
    wake.patches = pm.FacePatches(x_plus=outlet, y_minus=walls, y_plus=walls)
    for block, name in [
        (block_left, "Left"),
        (block_top, "Top"),
        (block_right, "Right"),
        (block_bottom, "Bottom"),
        (wake, "Wake"),
    ]:
        block.name = name

    mesh = pm.Mesh([block_left, block_top, block_right, block_bottom, wake])
    if three_d:
        mesh = mesh.extrude((-2.0, 2.0), cells=n, periodic=True)
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=float, default=100.0, help="Reynolds number")
    parser.add_argument("--n", type=int, default=24, help="cells per quarter ring")
    parser.add_argument("--three-d", action="store_true", help="extrude along z")
    parser.add_argument("--time", type=float, default=20.0, help="simulated time")
    parser.add_argument("--out", default=None, help="optional VTK output path")
    args = parser.parse_args()

    mesh = make_mesh(args.n, args.three_d)
    print(mesh)
    print(mesh.check())

    # Mean inflow velocity 1 and cylinder diameter 1
    dtype, device = torch.float32, torch.device("cuda")
    domain = mesh.get_domain(
        viscosity=1.0 / args.re,
        dtype=dtype,
        device=device,
        passive_scalar_channels=0,
        name="Cylinder",
    )

    hooks = Hooks().append(Hook.PRE, mesh.outflow_hook(domain, velocity=1.0, tol=5e-6))
    series = None
    if args.out is not None:
        series = export.VTKSeries(args.out, mesh=mesh)
        hooks.append(Hook.POST, series.hook(every=25))

    sim = phipict.Simulation(
        domain=domain,
        dt=0.01,
        substeps="ADAPTIVE",
        adaptive_CFL=0.8,
        corrector_steps=2,
        pressure_tol=5e-7 if args.three_d else 1e-5,
        advect_non_ortho_steps=1,
        pressure_non_ortho_steps=4 if args.three_d else 1,
        pressure_return_best_result=True,
        velocity_corrector="FD",
        non_orthogonal=True,  # the O-grid cells are not orthogonal
        hooks=hooks,
    )
    sim.make_divergence_free()

    steps = round(args.time / 0.01)
    wake = domain.getBlocks()[-1]
    for step in range(1, steps + 1):
        sim.single_step()
        if step % 100 == 0 or step == steps:
            # Cross-stream velocity in the wake: oscillates once vortices shed
            v = wake.velocity[0, 1]
            print(f"t = {step * 0.01:6.2f}: max. |v| in the wake {v.abs().max():.3f}")
    if series is not None:
        print(f"wrote {series.path}")


if __name__ == "__main__":
    main()
