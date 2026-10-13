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

"""The cell fields of a block, and access to them by name.

Used wherever code acts on "the velocity of every block" and the like, e.g.
:meth:`phipict.core.piso_simulation.Simulation.resample` and the VTK export, so
the field-to-block-attribute mapping lives in one place.
"""

from enum import Flag, auto

import torch

from phipict import _C

__all__ = ["Fields", "block_field", "domain_field"]


class Fields(Flag):
    """Cell fields of a block. Members combine, e.g. ``VELOCITY | PRESSURE``."""

    VELOCITY = auto()
    PRESSURE = auto()
    PASSIVE_SCALAR = auto()
    VELOCITY_SOURCE = auto()
    EPOT = auto()
    VISCOSITY = auto()
    #: Cell volume and scaled Jacobian, computed from the vertices (VTK export
    #: only; not stored on the block).
    QUALITY = auto()
    NONE = 0
    ALL = VELOCITY | PRESSURE | PASSIVE_SCALAR | VELOCITY_SOURCE | EPOT | VISCOSITY

    @property
    def attr(self) -> str:
        """Name of the block attribute holding this field.

        Returns
        -------
        str
            The attribute of :class:`phipict._C.Block`, e.g. ``"velocity"``.

        Raises
        ------
        ValueError
            If this is not a single field stored on the block.
        """
        try:
            return _BLOCK_ATTRS[self][0]
        except KeyError:
            raise ValueError(f"{self} is not a single block field.") from None


# field -> (block attribute, presence check or None if always present)
_BLOCK_ATTRS: dict[Fields, tuple[str, str | None]] = {
    Fields.VELOCITY: ("velocity", None),
    Fields.PRESSURE: ("pressure", None),
    Fields.PASSIVE_SCALAR: ("passiveScalar", "hasPassiveScalar"),
    Fields.VELOCITY_SOURCE: ("velocitySource", "hasVelocitySource"),
    Fields.EPOT: ("epot", "hasEpot"),
    Fields.VISCOSITY: ("viscosity", "hasViscosity"),
}


def block_field(block: _C.Block, field: Fields) -> torch.Tensor | None:
    """The data of one field of a block, or None if the block does not have it.

    Parameters
    ----------
    block : phipict._C.Block
        The block.
    field : Fields
        A single field stored on the block.

    Returns
    -------
    torch.Tensor or None
        The cell data ``[N, C, *spatial]``, or None for an optional field the
        block does not have.
    """
    attr = field.attr
    has = _BLOCK_ATTRS[field][1]
    if has is not None and not getattr(block, has)():
        return None
    return getattr(block, attr)


def domain_field(domain: _C.Domain, field: Fields) -> list[torch.Tensor]:
    """The data of one field of every block of a domain, in block order.

    Parameters
    ----------
    domain : phipict._C.Domain
        The domain.
    field : Fields
        A single field stored on the blocks.

    Returns
    -------
    list of torch.Tensor
        Per-block cell data.

    Raises
    ------
    ValueError
        If a block does not have the field.
    """
    data = []
    for block in domain.getBlocks():
        value = block_field(block, field)
        if value is None:
            raise ValueError(f"Block {block.name!r} has no {field.attr}.")
        data.append(value)
    return data
