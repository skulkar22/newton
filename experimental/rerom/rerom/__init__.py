"""re-ROM: adaptive re-linearization of Newton's reduced elastic (FFRF) bodies -- experimental.

Not part of the ``newton`` package and not covered by its API guarantees. See ``README.md``.

Public API
----------
Controllers     :class:`ReROMController` (one body, :class:`BodyAdapter` strategy),
                :class:`FreeBodyAdapter`, :class:`InterfaceBodyAdapter`, :class:`MultiFrameController`
Options         :class:`BasisOptions`, :class:`TriggerOptions`, :class:`UpdatePolicy`,
                :class:`NonlinearForceOptions`; enums :class:`RestRotationMode`, :class:`IndefiniteGuard`,
                :class:`UpdateSync`, :class:`UpdatePath`, :class:`TransferMode`
Triggers        :class:`RotationTrigger`, :class:`EnergyMismatchTrigger`, :class:`ResidualTrigger`,
                :class:`ValidityTrigger`, :class:`CompositeTrigger` (OR of triggers)
Assembly        :class:`TangentAssembler` (protocol), :class:`NumpyTetAssembler`, :class:`TetMesh`,
                :func:`make_beam_mesh`, :func:`make_flexure_mesh`, :func:`default_assembler`
Eigensolvers    :class:`RestrictedArpackBackend`, :class:`RestrictedSubspaceBackend`,
                :class:`ShiftedRestrictedSolver` (any ``newton.ModalEigensolverBackend`` is accepted too)
Craig-Bampton   :func:`cb_basis`, :class:`CBSegment`, :class:`CBChain`, :func:`build_cb_chain`,
                :func:`segment_from_mesh`, :func:`install_interface_endpoint_maps`
References      :class:`FullFEReference`, :class:`PrescribedMotionReference`, :class:`BaseMotion`

Submodules :mod:`rerom.numerics`, :mod:`rerom.model_io`, :mod:`rerom.hyper_reduction`, :mod:`rerom.strain`,
:mod:`rerom.update_kernels` (device-side update), :mod:`rerom.ccx` and :mod:`rerom.panel` (CalculiX; need the ``ccx`` binary) are imported on demand.
"""

from .assembly import NumpyTetAssembler, TangentAssembler, TetMesh, default_assembler, make_beam_mesh, make_flexure_mesh
from .controller import (
    BodyAdapter,
    FreeBodyAdapter,
    InterfaceBodyAdapter,
    MultiFrameController,
    ReROMController,
    UpdateSync,
)
from .craig_bampton import (
    CBChain,
    CBSegment,
    build_cb_chain,
    cb_basis,
    install_interface_endpoint_maps,
    segment_from_mesh,
)
from .eigen import RestrictedArpackBackend, RestrictedSubspaceBackend, ShiftedRestrictedSolver
from .fe_reference import BaseMotion, FullFEReference, PrescribedMotionReference
from .options import (
    BasisOptions,
    IndefiniteGuard,
    NonlinearForceOptions,
    RestRotationMode,
    TransferMode,
    TriggerOptions,
    UpdatePath,
    UpdatePolicy,
)
from .triggers import (
    CompositeTrigger,
    EnergyMismatchTrigger,
    ResidualTrigger,
    RotationTrigger,
    Trigger,
    ValidityTrigger,
)

__all__ = [
    "BaseMotion",
    "BasisOptions",
    "BodyAdapter",
    "CBChain",
    "CBSegment",
    "CompositeTrigger",
    "EnergyMismatchTrigger",
    "FreeBodyAdapter",
    "FullFEReference",
    "IndefiniteGuard",
    "InterfaceBodyAdapter",
    "MultiFrameController",
    "NonlinearForceOptions",
    "NumpyTetAssembler",
    "PrescribedMotionReference",
    "ReROMController",
    "ResidualTrigger",
    "RestRotationMode",
    "RestrictedArpackBackend",
    "RestrictedSubspaceBackend",
    "RotationTrigger",
    "ShiftedRestrictedSolver",
    "TangentAssembler",
    "TetMesh",
    "TransferMode",
    "Trigger",
    "TriggerOptions",
    "UpdatePath",
    "UpdatePolicy",
    "UpdateSync",
    "ValidityTrigger",
    "build_cb_chain",
    "cb_basis",
    "default_assembler",
    "install_interface_endpoint_maps",
    "make_beam_mesh",
    "make_flexure_mesh",
    "segment_from_mesh",
]
