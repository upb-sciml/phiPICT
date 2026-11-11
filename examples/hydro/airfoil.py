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

"""Flow around a NACA 0012 airfoil at 10 degrees angle of attack.

An airfoil of chord 1 sits in a channel of height 1.4. At Re = 3000 the flow
separates on the suction side and sheds vortices. Six blocks wrap the airfoil
(a C-type grid) and continue into the wake::

        +y wall (no-slip)
    +------+-----------------------------------------------+
    |      |                 top          |   tail upper   |
    | left | front  ___-------------------+----------------| out
    |  ->  |      (_____________________)--+    (wake)      |  ->
    |      |                bottom        |   tail lower   |
    +------+-----------------------------------------------+
        -y wall (no-slip)

The blocks on the airfoil follow its surface points (``pm.Points`` edges) and
are refined towards the surface. The tail blocks start with the smallest cell
size on the surface and grow by 1% per cell towards the outlet.

Boundary conditions:

- Inlet (x = -1.5): parabolic velocity profile with mean velocity 0.3.
- Outlet (x = 4.5): advective outflow.
- Channel walls and airfoil: no-slip.

Usage:
    python examples/hydro/airfoil.py
    python examples/hydro/airfoil.py --three-d   # extruded, periodic in z
"""

import argparse
import math

import torch

import phipict
import phipict.meshing as pm
from phipict import Hook, Hooks
from phipict.io import export

HEIGHT = 1.4  # channel height; the chord is 1
LENGTH = 4.5  # end of the channel; the leading edge is at x = 0
INLET_X = -1.5  # start of the channel
FRONT_X = -0.5  # end of the inlet block, in front of the airfoil
U_MEAN = 0.3  # mean inflow velocity
NORMAL_CELLS = 95  # cells between the airfoil and the channel walls


def naca0012(angle_deg: float, n: int = 80) -> torch.Tensor:
    """Surface of a NACA 0012 airfoil with a sharp trailing edge, ``[2n, 2]``.

    The points run from the trailing edge over the upper side to the leading edge
    and back over the lower side; the first and last point are the trailing edge.
    They are clustered towards both edges and rotated by the angle of attack
    around the leading edge.
    """
    t = 1.0 - torch.arange(n, dtype=torch.float64) / n  # 1 (trailing) -> 0 (leading)
    x = t**2 * (1.0 + 1.3 * (1.0 - t))  # clustered towards both edges
    y = 0.6 * (
        0.2969 * x.sqrt() - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4
    )
    upper = torch.stack([x, y], dim=1)  # trailing -> leading edge
    lower = torch.stack([x, -y], dim=1).flip(0)  # leading -> trailing edge
    points = torch.cat([upper, lower])

    a = math.radians(-angle_deg)  # nose up
    rotation = torch.tensor(
        [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=x.dtype
    )
    return points @ rotation.T


def parabola(points: torch.Tensor) -> torch.Tensor:
    """Parabolic inflow between the channel walls with mean ``U_MEAN``."""
    t = points[:, 1] / HEIGHT + 0.5
    u = 6.0 * U_MEAN * t * (1.0 - t)
    zero = torch.zeros_like(u)
    return torch.stack([u] + [zero] * (points.shape[1] - 1), dim=1)


def make_mesh(angle_deg: float, three_d: bool, res_z: int) -> pm.Mesh:
    """C-type grid of six blocks around the airfoil and its wake."""
    h = HEIGHT / 2
    surface = naca0012(angle_deg)

    # Split the surface into the parts the blocks are attached to: the upper side
    # (left to right), the nose (bottom to top) and the lower side (left to right)
    top_points = surface[:70].flip(0)
    front_points = surface[69:91].flip(0)
    bottom_points = surface[90:]
    t0, t1 = top_points[0].tolist(), top_points[-1].tolist()  # nose, trailing edge
    b0, b1 = bottom_points[0].tolist(), bottom_points[-1].tolist()

    # Boundary conditions
    walls = pm.Patch("walls", pm.Wall())
    airfoil = pm.Patch("airfoil", pm.Wall())
    inlet = pm.Patch("inlet", pm.Inflow(parabola))
    outlet = pm.Patch("outlet", pm.Outflow())

    # Small cells at the airfoil, growing by 3% per cell away from it: at the
    # start or the end of the axis, depending on the side of the block
    airfoil_at_start = pm.Geometric(1 / 0.97, pm.Cluster.START)
    airfoil_at_end = pm.Geometric(1 / 0.97, pm.Cluster.END)
    # The wake cells grow by 1% per cell from the trailing edge; there are as
    # many as it takes to grow from the smallest surface cell to the half height
    min_size = float((surface[1:] - surface[:-1]).norm(dim=1).min())
    tail = pm.Geometric(1.01, pm.Cluster.START)
    tail_cells = math.ceil(math.log(1 + 0.01 * h / min_size) / math.log(1.01))

    left = pm.make_quad(
        [(INLET_X, -h), (FRONT_X, -h), (INLET_X, h), (FRONT_X, h)],
        cells=(int(0.75 * (NORMAL_CELLS + 1)) - 1, len(front_points) - 1),
        patches=pm.FacePatches(x_minus=inlet, y_minus=walls, y_plus=walls),
        name="Left",
    )
    front = pm.make_quad(
        [(FRONT_X, -h), b0, (FRONT_X, h), t0],
        cells=(NORMAL_CELLS, None),  # None: taken from the connected block
        grading=(airfoil_at_end, None),
        edges=pm.QuadEdges(x_plus=pm.Points(front_points)),
        patches=pm.FacePatches(x_plus=airfoil),
        name="Front",
    )
    top = pm.make_quad(
        [t0, t1, (FRONT_X, h), (t1[0], h)],
        cells=(None, NORMAL_CELLS),
        grading=(None, airfoil_at_start),
        edges=pm.QuadEdges(y_minus=pm.Points(top_points)),
        patches=pm.FacePatches(y_minus=airfoil, y_plus=walls),
        name="Top",
    )
    bottom = pm.make_quad(
        [(FRONT_X, -h), (b1[0], -h), b0, b1],
        cells=(None, NORMAL_CELLS),
        grading=(None, airfoil_at_end),
        edges=pm.QuadEdges(y_plus=pm.Points(bottom_points)),
        patches=pm.FacePatches(y_minus=walls, y_plus=airfoil),
        name="Bottom",
    )
    tail_upper = pm.make_quad(
        [t1, (LENGTH, t1[1]), (t1[0], h), (LENGTH, h)],
        cells=(tail_cells, NORMAL_CELLS),
        grading=(tail, airfoil_at_start),
        patches=pm.FacePatches(x_plus=outlet, y_plus=walls),
        name="TailUpper",
    )
    tail_lower = pm.make_quad(
        [(b1[0], -h), (LENGTH, -h), b1, (LENGTH, b1[1])],
        cells=(tail_cells, NORMAL_CELLS),
        grading=(tail, airfoil_at_end),
        patches=pm.FacePatches(x_plus=outlet, y_minus=walls),
        name="TailLower",
    )
    mesh = pm.Mesh([left, front, top, bottom, tail_upper, tail_lower])
    if three_d:
        mesh = mesh.extrude((-h, h), cells=res_z, periodic=True)
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=float, default=3000.0, help="Reynolds number")
    parser.add_argument("--angle", type=float, default=10.0, help="angle of attack")
    parser.add_argument("--three-d", action="store_true", help="extrude along z")
    parser.add_argument("--res-z", type=int, default=96, help="cells along z (3D)")
    parser.add_argument("--time", type=float, default=30.0, help="simulated time")
    parser.add_argument("--out", default=None, help="optional VTK output path")
    args = parser.parse_args()

    mesh = make_mesh(args.angle, args.three_d, args.res_z)
    print(mesh)
    print(mesh.check())

    # Reynolds number with the mean inflow velocity and the chord length 1
    dtype, device = torch.float32, torch.device("cuda")
    domain = mesh.get_domain(
        viscosity=U_MEAN / args.re,
        dtype=dtype,
        device=device,
        passive_scalar_channels=0,
        name="Airfoil",
    )

    hooks = Hooks().append(Hook.PRE, mesh.outflow_hook(domain, velocity=U_MEAN))
    series = None
    if args.out is not None:
        series = export.VTKSeries(args.out, mesh=mesh)
        hooks.append(Hook.POST, series.hook(every=10))

    sim = phipict.Simulation(
        domain=domain,
        dt=0.05,
        substeps="ADAPTIVE",
        adaptive_CFL=0.8,
        corrector_steps=2,
        advection_tol=1e-6,
        pressure_tol=1e-8 if args.three_d else 1e-7,
        advect_non_ortho_steps=2,
        pressure_non_ortho_steps=4,
        pressure_return_best_result=True,
        velocity_corrector="FD",
        non_orthogonal=True,  # the cells around the airfoil are curved
        hooks=hooks,
    )
    sim.make_divergence_free()

    steps = round(args.time / 0.05)
    for step in range(1, steps + 1):
        sim.single_step()
        if step % 50 == 0 or step == steps:
            u = torch.cat([b.velocity[0, 0].flatten() for b in domain.getBlocks()])
            print(f"t = {step * 0.05:6.2f}: max. u {u.max():.3f}")
    if series is not None:
        print(f"wrote {series.path}")


if __name__ == "__main__":
    main()
