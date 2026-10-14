# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import contextlib
import math
import os
import tracemalloc
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.sim.modal import _estimate_sample_psi, _reduced_inertia_from_pairs
from newton._src.solvers.vbd.reduced_elastic_kernels import (
    accumulate_elastic_frame_coupling,
    compute_elastic_mode_sums,
    create_solve_elastic_body_tiled,
    elastic_block_dense_width,
    launch_solve_elastic_body_dense,
)
from newton._src.solvers.vbd.rigid_vbd_kernels import ElasticReducedMobility, _elastic_reduced_mobility_pair
from newton._src.solvers.vbd.solver_vbd import _elastic_schur_inverse_blocks, _max_elastic_block_solve_tiled_width
from newton.examples.basic._reduced_elastic import (
    beam_render_sample_points,
    beam_torsion_linear_modal_properties,
    box_surface_mesh,
    finite_torsion_displacement,
    joint_endpoint_world,
    mesh_volume,
    quat_rotate,
)
from newton.examples.basic._reduced_elastic_contact import (
    apply_kinematic_targets,
)
from newton.examples.basic.example_basic_reduced_elastic_angular_frame_coupling import (
    Example as AngularFrameCouplingExample,
)
from newton.examples.basic.example_basic_reduced_elastic_base_excitation import Example as BaseExcitationExample
from newton.examples.basic.example_basic_reduced_elastic_base_rotation import Example as BaseRotationExample
from newton.examples.basic.example_basic_reduced_elastic_centrifugal import Example as CentrifugalExample
from newton.examples.basic.example_basic_reduced_elastic_chair_stick_slip import (
    BIN_NAME as CHAIR_BIN_NAME,
)
from newton.examples.basic.example_basic_reduced_elastic_chair_stick_slip import (
    GLTF_NAME as CHAIR_GLTF_NAME,
)
from newton.examples.basic.example_basic_reduced_elastic_chair_stick_slip import Example as ChairStickSlipExample
from newton.examples.basic.example_basic_reduced_elastic_chair_stick_slip import (
    _asset_cache_dir as chair_asset_cache_dir,
)
from newton.examples.basic.example_basic_reduced_elastic_clamp_moment import Example as ClampMomentExample
from newton.examples.basic.example_basic_reduced_elastic_coriolis import Example as CoriolisExample
from newton.examples.basic.example_basic_reduced_elastic_dipper import Example as DipperExample
from newton.examples.basic.example_basic_reduced_elastic_frame_coupling import Example as FrameCouplingExample
from newton.examples.basic.example_basic_reduced_elastic_gravity_coupling import Example as GravityCouplingExample
from newton.examples.basic.example_basic_reduced_elastic_gripper_contact import Example as GripperContactExample
from newton.examples.basic.example_basic_reduced_elastic_matrix_rom import Example as MatrixROMExample
from newton.examples.basic.example_basic_reduced_elastic_scraper_contact import Example as ScraperContactExample
from newton.examples.basic.example_basic_reduced_elastic_wall_contact import Example as WallContactExample
from newton.tests import _reduced_elastic_reference as elastic_reference
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()


class _ElasticMeshColorProbe(newton.viewer.ViewerNull):
    def __init__(self):
        super().__init__(num_frames=1)
        self.mesh_colors = {}
        self.mesh_points = {}
        self.mesh_hidden = {}

    def log_mesh(
        self,
        name,
        points,
        indices,
        normals=None,
        uvs=None,
        texture=None,
        hidden=False,
        backface_culling=True,
        color=None,
        roughness=None,
        metallic=None,
        dynamic=False,
        opacity=None,
        colors=None,
    ):
        if name.startswith("/model/elastic_shapes/"):
            self.mesh_colors[name] = None if colors is None else colors.numpy()
            self.mesh_points[name] = points.numpy()
            self.mesh_hidden[name] = hidden


def _identity_inertia():
    return wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


# Rigid VBD formulations swept by tests whose result must not depend on the joint solver:
# the deprecated legacy AVBD path and the default compliant ALM path.
_RIGID_ALM_MODES = (False, True)
# Compliant ALM reaches the implicit-Euler answer of the material law once its joint multipliers
# converge; these single-body fixtures converge to float32 precision within eight iterations.
_ALM_CONVERGED_ITERATIONS = 16


@contextlib.contextmanager
def _legacy_alm_allowed():
    """Suppress only the ``rigid_compliant_alm=False`` deprecation warning.

    Used where the deprecated legacy AVBD path is exercised deliberately: the rigid-ALM sweeps and
    examples that stay on the legacy path.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="rigid_compliant_alm=False is deprecated", category=DeprecationWarning
        )
        yield


def _make_vbd_solver(model, *, compliant: bool, **kwargs):
    """Create a VBD solver on the compliant ALM path or on the legacy AVBD path."""
    if compliant:
        return newton.solvers.SolverVBD(model, rigid_compliant_alm=True, **kwargs)
    with _legacy_alm_allowed():
        return newton.solvers.SolverVBD(model, rigid_compliant_alm=False, **kwargs)


def _vbd_fixed_joint_solver(
    model,
    *,
    compliant: bool,
    iterations: int,
    linear_k: float,
    angular_k: float,
    linear_kd: float = 0.0,
    angular_kd: float = 0.0,
    **kwargs,
):
    """Create a VBD solver whose structural joint rows use fixed material stiffness.

    Legacy AVBD pins its penalty to the material stiffness. Compliant ALM uses the same values as its
    material law and needs ``_ALM_CONVERGED_ITERATIONS`` to reproduce single-solve closed forms.
    """
    stiffness = {
        "rigid_joint_linear_ke": linear_k,
        "rigid_joint_angular_ke": angular_k,
        "rigid_joint_linear_kd": linear_kd,
        "rigid_joint_angular_kd": angular_kd,
    }
    if compliant:
        return _make_vbd_solver(model, compliant=True, iterations=iterations, **stiffness, **kwargs)
    return _make_vbd_solver(
        model,
        compliant=False,
        iterations=iterations,
        rigid_joint_linear_k_start=linear_k,
        rigid_joint_angular_k_start=angular_k,
        rigid_joint_adaptive_stiffness=False,
        **stiffness,
        **kwargs,
    )


def _first_solve_joint_k(solver, joint: int, slot: int, material_k: float, compliant: bool) -> float:
    """Return a structural row's stiffness in the first local solve of a step.

    Legacy AVBD uses the material stiffness ``K``. Compliant ALM uses ``K * rho / (K + rho)`` with the
    solver-selected ``rho``; its multiplier, zero on the first iteration, carries the remainder.
    """
    if not compliant:
        return material_k
    row = int(solver.joint_constraint_start.numpy()[joint]) + slot
    rho = float(solver.joint_rho.numpy()[row])
    return material_k * rho / (material_k + rho)


# Nominal frame mass properties for layout tests whose few-sample bases have a degenerate rigid
# inertia. These tests never step a solver, so the frame need not match the basis.
_NOMINAL_FRAME = {"mass": 1.0, "inertia": _identity_inertia(), "inertia_tolerance": None}


def _quat_from_angle_z(theta: float):
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), theta)


def _quat_angle_error(q_a: np.ndarray, q_b: np.ndarray) -> float:
    dot = abs(float(np.dot(q_a, q_b)))
    dot = min(max(dot, -1.0), 1.0)
    return 2.0 * math.acos(dot)


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    qv = np.array([x, y, z], dtype=float)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + w * v)


def _transform_point_np(xform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return xform[:3] + _quat_rotate_np(xform[3:7], point)


def _add_modal_projection_joint(builder, joint_kind: str, parent: int, child: int, parent_xform, child_xform):
    if joint_kind == "ball":
        return builder.add_joint_ball(parent=parent, child=child, parent_xform=parent_xform, child_xform=child_xform)
    if joint_kind == "fixed":
        return builder.add_joint_fixed(parent=parent, child=child, parent_xform=parent_xform, child_xform=child_xform)
    if joint_kind == "revolute":
        return builder.add_joint_revolute(
            parent=parent,
            child=child,
            axis=(1.0, 0.0, 0.0),
            parent_xform=parent_xform,
            child_xform=child_xform,
        )
    if joint_kind == "prismatic":
        return builder.add_joint_prismatic(
            parent=parent,
            child=child,
            axis=(1.0, 0.0, 0.0),
            parent_xform=parent_xform,
            child_xform=child_xform,
        )
    raise ValueError(f"Unsupported joint kind: {joint_kind}")


def _assert_elastic_modal_projection_matches_joint_force(test, device, joint_kind: str, elastic_side: str = "child"):
    phi_local = np.array([0.35, -0.45, 0.25], dtype=np.float32)
    elastic_anchor = np.array([0.12, -0.03, 0.04], dtype=float)
    elastic_pos = np.array([0.02, 0.01, -0.015], dtype=float)
    elastic_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.37)
    elastic_xform = wp.transform(wp.vec3(*elastic_pos), elastic_quat)
    elastic_quat_np = np.array([elastic_quat[0], elastic_quat[1], elastic_quat[2], elastic_quat[3]], dtype=float)
    k = 2500.0

    def shape_fn(_x):
        return np.array([phi_local], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if elastic_side == "child":
        body = builder.add_body_elastic(
            xform=elastic_xform,
            mass=1.0,
            inertia=_identity_inertia(),
            mode_count=1,
            mode_mass=[0.0],
            mode_stiffness=[0.0],
            mode_damping=[0.0],
            mode_q=[0.0],
            mode_shape_fn=shape_fn,
            is_kinematic=True,
        )
        parent_anchor = np.array([-0.08, 0.07, -0.02], dtype=float)
        child_anchor = elastic_anchor
        _add_modal_projection_joint(
            builder,
            joint_kind,
            parent=-1,
            child=body,
            parent_xform=wp.transform(wp.vec3(*parent_anchor), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(*child_anchor), wp.quat_identity()),
        )
        x_p = parent_anchor
        x_c = elastic_pos + _quat_rotate_np(elastic_quat_np, child_anchor)
        dC_dq = _quat_rotate_np(elastic_quat_np, phi_local.astype(float))
        side_force_sign = -1.0
    elif elastic_side == "parent":
        body = builder.add_body_elastic(
            xform=elastic_xform,
            mass=1.0,
            inertia=_identity_inertia(),
            mode_count=1,
            mode_mass=[0.0],
            mode_stiffness=[0.0],
            mode_damping=[0.0],
            mode_q=[0.0],
            mode_shape_fn=shape_fn,
            is_kinematic=True,
        )
        child_pos = np.array([-0.06, 0.09, 0.035], dtype=float)
        child = builder.add_body(
            xform=wp.transform(wp.vec3(*child_pos), wp.quat_identity()),
            mass=1.0,
            inertia=_identity_inertia(),
            is_kinematic=True,
        )
        parent_anchor = elastic_anchor
        child_anchor = np.array([0.03, -0.015, 0.02], dtype=float)
        _add_modal_projection_joint(
            builder,
            joint_kind,
            parent=body,
            child=child,
            parent_xform=wp.transform(wp.vec3(*parent_anchor), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(*child_anchor), wp.quat_identity()),
        )
        x_p = elastic_pos + _quat_rotate_np(elastic_quat_np, parent_anchor)
        x_c = child_pos + child_anchor
        endpoint_phi_world = _quat_rotate_np(elastic_quat_np, phi_local.astype(float))
        dC_dq = -endpoint_phi_world
        side_force_sign = 1.0
    else:
        raise ValueError(f"Unsupported elastic side: {elastic_side}")

    builder.color()
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    P = np.eye(3)
    if joint_kind == "prismatic":
        axis = np.array([1.0, 0.0, 0.0])
        P -= np.outer(axis, axis)

    C = x_c - x_p
    PC = P @ C
    PdC = P @ dC_dq
    h = k * float(np.dot(PdC, PdC))
    grad = k * float(np.dot(PC, PdC))
    test.assertGreater(h, 0.0)

    endpoint_phi_world = _quat_rotate_np(elastic_quat_np, phi_local.astype(float))
    rigid_side_force = side_force_sign * k * PC
    modal_force_from_joint = float(np.dot(endpoint_phi_world, rigid_side_force))
    modal_force_from_energy = -grad
    np.testing.assert_allclose(modal_force_from_energy, modal_force_from_joint, rtol=1.0e-6, atol=1.0e-7)

    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    q_expected = modal_force_from_joint / h
    # The massless mode makes the first solve independent of the effective row stiffness.
    for compliant in _RIGID_ALM_MODES:
        with test.subTest(rigid_compliant_alm=compliant):
            state_0 = model.state()
            state_1 = model.state()
            solver = _vbd_fixed_joint_solver(model, compliant=compliant, iterations=1, linear_k=k, angular_k=0.0)
            solver.step(state_0, state_1, control, None, 0.01)
            np.testing.assert_allclose(state_1.joint_q.numpy()[q_start + 7], q_expected, rtol=1.0e-5, atol=1.0e-6)


def test_modal_basis_add_sample(test, device):
    """Check that add_sample reuses a coincident point and stores explicit mode values for a new one."""
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]]],
        mode_mass=[2.0, 3.0],
        mode_stiffness=[4.0, 5.0],
        mode_damping=[0.1, 0.2],
    )

    test.assertEqual(basis.mode_count, 2)
    test.assertEqual(basis.sample_count, 1)
    test.assertEqual(basis.add_sample([0.0, 0.0, 0.0]), 0)
    sample = basis.add_sample([1.0, 0.0, 0.0], phi=[[0.5, 0.0, 0.0], [0.0, 0.0, 0.25]])
    test.assertEqual(sample, 1)
    np.testing.assert_allclose(basis.sample_phi[sample], [[0.5, 0.0, 0.0], [0.0, 0.0, 0.25]], atol=1.0e-7)


def test_modal_basis_add_samples_preserves_mass_invariant(test, device):
    """Check that add_samples deduplicates points, keeps existing values, and gives new samples zero mass."""
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[1.0, 0.0, 0.0]]],
        sample_mass=[2.0],
        mode_stiffness=[1.0],
    )
    indices = basis.add_samples(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        phi=[[[9.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[7.0, 0.0, 0.0]]],
    )

    np.testing.assert_array_equal(indices, [0, 1, 1])
    np.testing.assert_allclose(basis.sample_phi[:, 0], [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    np.testing.assert_allclose(basis.sample_mass, [2.0, 0.0])
    clone = basis.copy()
    np.testing.assert_allclose(clone.sample_mass, basis.sample_mass)


def test_modal_generator_beam_samples(test, device):
    """Check axial, pinned-pinned bending, and linear torsion beam mode values at an off-axis sample."""
    length = 1.0
    basis = newton.ModalGeneratorBeam(
        length=length,
        half_width_y=0.1,
        half_width_z=0.05,
        mode_specs=[
            {"type": newton.ModalGeneratorBeam.Mode.AXIAL},
            {
                "type": newton.ModalGeneratorBeam.Mode.BENDING_Y,
                "boundary": newton.ModalGeneratorBeam.Boundary.PINNED_PINNED,
            },
            {
                "type": newton.ModalGeneratorBeam.Mode.TORSION,
                "boundary": newton.ModalGeneratorBeam.Boundary.LINEAR,
            },
        ],
        sample_count=3,
    ).build()

    sample = basis.add_sample([0.0, 0.1, 0.05])
    phi = basis.sample_phi[sample]
    np.testing.assert_allclose(phi[0], [0.0, 0.0, 0.0], atol=1.0e-7)
    np.testing.assert_allclose(phi[1], [0.0, 1.0, 0.0], atol=1.0e-7)
    np.testing.assert_allclose(phi[2], [0.0, -0.025, 0.05], atol=1.0e-7)
    test.assertGreater(float(basis.mode_stiffness[0]), 0.0)


def test_modal_generator_pod_rank_one(test, device):
    """Check that one POD snapshot gives a unit-peak mode with lumped modal mass and scaled stiffness."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    displacements = np.array([[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype=np.float32)

    basis = newton.ModalGeneratorPOD(
        sample_points=points,
        displacements=displacements,
        mode_count=1,
        total_mass=2.0,
        stiffness_scale=3.0,
    ).build()

    test.assertEqual(basis.mode_count, 1)
    np.testing.assert_allclose(np.abs(basis.sample_phi[1][0]), [0.0, 1.0, 0.0], atol=1.0e-7)
    np.testing.assert_allclose(basis.mode_mass, [1.0], atol=1.0e-7)
    np.testing.assert_allclose(basis.mode_stiffness, [3.0], atol=1.0e-7)


def test_modal_generator_pod_explicit_zero_modes(test, device):
    """Check that ``mode_count=0`` gives an empty POD basis."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    displacements = np.array([[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype=np.float32)
    basis = newton.ModalGeneratorPOD(points, displacements=displacements, mode_count=0).build()

    test.assertEqual(basis.mode_count, 0)
    test.assertEqual(basis.sample_phi.shape, (2, 0, 3))


def test_modal_generator_beam_linear_torsion_properties(test, device):
    """Check the linear torsion modal stiffness formula and the rejection of order 0."""
    length = 2.0
    shear_modulus = 7.0
    polar_moment = 0.25
    order = 2
    basis = newton.ModalGeneratorBeam(
        length=length,
        half_width_y=0.1,
        half_width_z=0.1,
        shear_modulus=shear_modulus,
        polar_moment=polar_moment,
        mode_specs=[
            {
                "type": newton.ModalGeneratorBeam.Mode.TORSION,
                "boundary": newton.ModalGeneratorBeam.Boundary.LINEAR,
                "order": order,
            }
        ],
    ).build()

    expected = shear_modulus * polar_moment * order**2 / (length * (2 * order - 1))
    np.testing.assert_allclose(basis.mode_stiffness, [expected], rtol=1.0e-6)
    with test.assertRaisesRegex(ValueError, "at least 1"):
        newton.ModalGeneratorBeam(
            length=length,
            mode_specs=[{"type": newton.ModalGeneratorBeam.Mode.TORSION, "order": 0}],
        )


def test_modal_generator_fem_matrix_rom(test, device):
    """Check FEM modes, frequencies, damping, and node sampling for a diagonal two-node model."""
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    mass = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    stiffness = np.zeros((6, 6), dtype=np.float64)
    stiffness[3, 3] = 8.0
    stiffness[4, 4] = 18.0
    stiffness[5, 5] = 32.0
    damping = np.zeros((6, 6), dtype=np.float64)
    damping[3, 3] = 0.4
    damping[4, 4] = 0.6
    damping[5, 5] = 0.8
    sample_points = np.array([[1.0, 0.0, 0.2], [0.0, 0.0, -0.1]], dtype=np.float32)

    generator = newton.ModalGeneratorFEM(
        node_positions=nodes,
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        damping_matrix=damping,
        sample_points=sample_points,
        sample_node_indices=[1, 0],
        fixed_node_indices=[0],
        mode_count=3,
    )
    basis = generator.build()

    test.assertEqual(basis.mode_count, 3)
    np.testing.assert_allclose(basis.mode_mass, [1.0, 1.0, 1.0], atol=1.0e-6)
    np.testing.assert_allclose(basis.mode_stiffness, [4.0, 9.0, 16.0], atol=1.0e-6)
    np.testing.assert_allclose(basis.mode_damping, [0.2, 0.3, 0.4], atol=1.0e-6)
    np.testing.assert_allclose(
        generator.frequencies, [1.0 / math.pi, 3.0 / (2.0 * math.pi), 2.0 / math.pi], atol=1.0e-6
    )
    np.testing.assert_allclose(np.abs(basis.sample_phi[0]), np.eye(3) / math.sqrt(2.0), atol=1.0e-6)
    np.testing.assert_allclose(basis.sample_phi[1], np.zeros((3, 3)), atol=1.0e-7)


def _build_craig_bampton_test_data():
    interface_positions = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)

    def skew(value):
        x, y, z = value
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)

    rigid_map = np.zeros((12, 6), dtype=np.float64)
    for interface, position in enumerate(interface_positions):
        start = 6 * interface
        rigid_map[start : start + 3, :3] = np.eye(3)
        rigid_map[start : start + 3, 3:] = -skew(position)
        rigid_map[start + 3 : start + 6, 3:] = np.eye(3)

    coordinate_map = np.zeros((14, 14), dtype=np.float64)
    coordinate_map[:12, :6] = rigid_map
    coordinate_map[6:12, 6:12] = np.eye(6)
    coordinate_map[12:, 12:] = np.eye(2)

    spatial_mass = np.diag([4.0, 4.0, 4.0, 0.2, 0.3, 0.4])
    elastic_mass = np.diag([2.0, 2.5, 3.0, 0.5, 0.6, 0.7, 1.0, 1.1])
    elastic_mass[0, 1] = elastic_mass[1, 0] = 0.2
    elastic_mass[0, 6] = elastic_mass[6, 0] = 0.15
    elastic_mass[1, 7] = elastic_mass[7, 1] = -0.12
    elastic_stiffness = np.diag([20.0, 30.0, 40.0, 5.0, 6.0, 7.0, 100.0, 120.0])
    elastic_stiffness[1, 2] = elastic_stiffness[2, 1] = 1.5
    rigid_elastic_mass = np.zeros((6, 8), dtype=np.float64)
    rigid_elastic_mass[:, :6] = np.diag([0.10, -0.08, 0.06, 0.02, -0.015, 0.01])
    rigid_elastic_mass[0, 6] = 0.03
    rigid_elastic_mass[1, 7] = -0.025

    transformed_mass = np.zeros((14, 14), dtype=np.float64)
    transformed_mass[:6, :6] = spatial_mass
    transformed_mass[:6, 6:] = rigid_elastic_mass
    transformed_mass[6:, :6] = rigid_elastic_mass.T
    transformed_mass[6:, 6:] = elastic_mass
    transformed_stiffness = np.zeros((14, 14), dtype=np.float64)
    transformed_stiffness[6:, 6:] = elastic_stiffness
    transformed_damping = 0.01 * transformed_stiffness

    inverse_map = np.linalg.inv(coordinate_map)
    mass = inverse_map.T @ transformed_mass @ inverse_map
    stiffness = inverse_map.T @ transformed_stiffness @ inverse_map
    damping = inverse_map.T @ transformed_damping @ inverse_map

    sample_points = np.array(
        [
            [-0.5, -0.1, -0.1],
            [-0.5, 0.1, 0.1],
            [0.0, -0.1, 0.1],
            [0.0, 0.1, -0.1],
            [0.5, -0.1, -0.1],
            [0.5, 0.1, 0.1],
        ],
        dtype=np.float64,
    )
    sample_rigid_map = np.hstack(
        (
            np.tile(np.eye(3), (sample_points.shape[0], 1)),
            np.vstack([-skew(point) for point in sample_points]),
        )
    )
    sample_elastic_map = np.zeros((3 * sample_points.shape[0], 8), dtype=np.float64)
    x_weight = sample_points[:, 0] + 0.5
    for sample, weight in enumerate(x_weight):
        rows = slice(3 * sample, 3 * sample + 3)
        sample_elastic_map[rows, :3] = weight * np.eye(3)
        sample_elastic_map[rows, 3:6] = -weight * skew(sample_points[sample])
        internal_weight = 1.0 - 4.0 * sample_points[sample, 0] ** 2
        sample_elastic_map[rows, 6] = internal_weight * np.array([0.0, 0.0, 0.2])
        sample_elastic_map[rows, 7] = internal_weight * np.array([0.0, 0.15, 0.0])
    recovery = np.column_stack((sample_rigid_map, sample_elastic_map)) @ inverse_map

    return interface_positions, mass, stiffness, damping, sample_points, recovery, spatial_mass


def test_modal_generator_craig_bampton_interface_modes(test, device):
    """Check Craig-Bampton mass-normalized modes, rigid properties, and interface sample values."""
    interface_positions, mass, stiffness, damping, sample_points, recovery, spatial_mass = (
        _build_craig_bampton_test_data()
    )

    generator = newton.ModalGeneratorCraigBampton(
        interface_positions=interface_positions,
        interface_names=["left", "right"],
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        damping_matrix=damping,
        sample_points=sample_points,
        recovery_matrix=recovery,
    )
    basis = generator.build()

    test.assertEqual(generator.fixed_interface_mode_count, 2)
    test.assertEqual(generator.discarded_mode_count, 0)
    test.assertEqual(basis.mode_count, 8)
    test.assertEqual(set(generator.interface_sample_indices), {"left", "right"})
    test.assertAlmostEqual(generator.mass, 4.0, places=6)
    np.testing.assert_allclose(generator.com, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(generator.inertia, np.diag([0.2, 0.3, 0.4]), atol=1.0e-7)

    modes = generator.modal_matrix
    np.testing.assert_allclose(modes.T @ mass @ modes, np.eye(8), atol=1.0e-7)
    np.testing.assert_allclose(
        modes.T @ stiffness @ modes,
        np.diag(basis.mode_stiffness),
        atol=1.0e-5,
    )
    np.testing.assert_allclose(modes.T @ damping @ modes, np.diag(basis.mode_damping), atol=1.0e-7)
    np.testing.assert_allclose(basis.mode_coupling_linear, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(basis.mode_coupling_angular, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(generator.spatial_mass, spatial_mass, atol=1.0e-7)

    recovered_modes = recovery @ modes
    expected_phi = np.transpose(recovered_modes.reshape((sample_points.shape[0], 3, 8)), (0, 2, 1))
    np.testing.assert_allclose(basis.sample_phi[: sample_points.shape[0]], expected_phi, atol=1.0e-7)
    test.assertGreater(float(np.linalg.norm(modes[12:])), 0.0)

    left = generator.interface_sample_indices["left"]
    right = generator.interface_sample_indices["right"]
    np.testing.assert_allclose(basis.sample_phi[left], modes[:3].T, atol=1.0e-7)
    np.testing.assert_allclose(basis.sample_psi[left], modes[3:6].T, atol=1.0e-7)
    np.testing.assert_allclose(basis.sample_phi[right], modes[6:9].T, atol=1.0e-7)
    np.testing.assert_allclose(basis.sample_psi[right], modes[9:12].T, atol=1.0e-7)


def test_modal_generator_craig_bampton_single_interface(test, device):
    """Check a single-interface Craig-Bampton model with interface/internal mass coupling."""
    interface_positions = np.zeros((1, 3), dtype=np.float64)
    spatial_mass = np.diag([3.0, 3.0, 3.0, 0.2, 0.25, 0.3])
    mass = np.zeros((8, 8), dtype=np.float64)
    mass[:6, :6] = spatial_mass
    mass[6:, 6:] = np.diag([1.0, 1.2])
    mass[0, 6] = mass[6, 0] = 0.05
    mass[1, 7] = mass[7, 1] = -0.04
    stiffness = np.zeros((8, 8), dtype=np.float64)
    stiffness[6:, 6:] = np.diag([40.0, 75.0])
    damping = 0.02 * stiffness
    sample_points = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.2]], dtype=np.float64)
    sample_rigid = np.hstack(
        (
            np.tile(np.eye(3), (sample_points.shape[0], 1)),
            np.vstack(
                [
                    np.array([[0.0, point[2], -point[1]], [-point[2], 0.0, point[0]], [point[1], -point[0], 0.0]])
                    for point in sample_points
                ]
            ),
        )
    )
    sample_internal = np.zeros((3 * sample_points.shape[0], 2), dtype=np.float64)
    sample_internal[3:, 0] = np.tile([0.0, 0.0, 0.1], sample_points.shape[0] - 1)
    sample_internal[3:, 1] = np.tile([0.0, 0.08, 0.0], sample_points.shape[0] - 1)
    recovery = np.column_stack((sample_rigid, sample_internal))

    generator = newton.ModalGeneratorCraigBampton(
        interface_positions=interface_positions,
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        damping_matrix=damping,
        sample_points=sample_points,
        recovery_matrix=recovery,
    )
    basis = generator.build()

    test.assertEqual(generator.fixed_interface_mode_count, 2)
    test.assertEqual(basis.mode_count, 2)
    np.testing.assert_allclose(generator.spatial_mass, spatial_mass, atol=1.0e-7)
    np.testing.assert_allclose(generator.modal_matrix.T @ mass @ generator.modal_matrix, np.eye(2), atol=1.0e-7)


def test_modal_generator_craig_bampton_nonclassical_damping(test, device):
    """Check that non-diagonal projected Craig-Bampton damping warns and keeps its diagonal."""
    interface_positions, mass, stiffness, _, sample_points, recovery, _ = _build_craig_bampton_test_data()
    undamped_generator = newton.ModalGeneratorCraigBampton(
        interface_positions=interface_positions,
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        sample_points=sample_points,
        recovery_matrix=recovery,
    )
    undamped_basis = undamped_generator.build()

    modes = undamped_generator.modal_matrix
    modal_damping = np.diag(0.01 * undamped_basis.mode_stiffness.astype(np.float64))
    modal_damping[0, 1] = modal_damping[1, 0] = 0.05 * float(np.max(np.diag(modal_damping)))
    damping = mass @ modes @ modal_damping @ modes.T @ mass
    generator = newton.ModalGeneratorCraigBampton(
        interface_positions=interface_positions,
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        damping_matrix=damping,
        sample_points=sample_points,
        recovery_matrix=recovery,
    )

    with test.assertWarnsRegex(UserWarning, "damping is not diagonal"):
        basis = generator.build()
    test.assertGreater(generator.damping_off_diagonal_ratio, generator.damping_coupling_tolerance)
    np.testing.assert_allclose(basis.mode_damping, np.diag(modal_damping), rtol=1.0e-5)


def _cantilever_root(order):
    """Return the ``order``-th root of ``cos(x) cosh(x) = -1`` by bisection."""
    lower = (order - 1) * math.pi
    upper = order * math.pi

    def residual(x):
        return math.cos(x) + 1.0 / math.cosh(x)

    for _ in range(200):
        middle = 0.5 * (lower + upper)
        if residual(lower) * residual(middle) <= 0.0:
            upper = middle
        else:
            lower = middle
    return 0.5 * (lower + upper)


def test_modal_generator_beam_cantilever_eigenfunctions(test, device):
    """Verify cantilever-tip bending modes are clamped, orthogonal Euler-Bernoulli eigenfunctions.

    Each shape must have zero root deflection and rotation, unit tip deflection,
    mass orthogonality to the other orders, and modal mass and stiffness equal
    to their quadrature integrals with ``k / m = beta^4 E I / (rho A)``.
    """
    length = 1.25
    half_width_y = 0.055
    half_width_z = 0.038
    density = 420.0
    young_modulus = 7.5e5
    orders = (1, 2, 3, 4)
    generator = newton.ModalGeneratorBeam(
        length=length,
        half_width_y=half_width_y,
        half_width_z=half_width_z,
        mode_specs=[
            {
                "type": newton.ModalGeneratorBeam.Mode.BENDING_Z,
                "boundary": newton.ModalGeneratorBeam.Boundary.CANTILEVER_TIP,
                "order": order,
            }
            for order in orders
        ],
        density=density,
        young_modulus=young_modulus,
    )
    area = generator.area
    ei = young_modulus * generator.area_moment_y

    endpoints = np.array([[-0.5 * length, 0.0, 0.0], [0.5 * length, 0.0, 0.0]], dtype=np.float32)
    basis = generator.build(sample_points=endpoints)
    np.testing.assert_allclose(basis.sample_phi[0], 0.0, atol=1.0e-9)
    np.testing.assert_allclose(basis.sample_psi[0], 0.0, atol=1.0e-9)
    np.testing.assert_allclose(basis.sample_phi[1, :, 2], 1.0, atol=1.0e-6)

    # Uniform lumped mass at cell midpoints of the centerline.
    cell_count = 4000
    ds = length / cell_count
    s = (np.arange(cell_count) + 0.5) * ds
    points = np.column_stack((s - 0.5 * length, np.zeros(cell_count), np.zeros(cell_count)))
    phi, psi = generator._evaluate_modes(points)
    deflection = phi[:, :, 2]
    gram = density * area * ds * deflection.T @ deflection
    normalized = gram / np.sqrt(np.outer(np.diag(gram), np.diag(gram)))
    test.assertLess(float(np.max(np.abs(normalized - np.eye(len(orders))))), 1.0e-3)

    curvature = np.gradient(-psi[:, :, 1], ds, axis=0)
    for index, order in enumerate(orders):
        mode_mass, mode_stiffness = generator._mode_properties(generator.mode_specs[index])
        beta = _cantilever_root(order) / length
        np.testing.assert_allclose(mode_mass, density * area * length / 4.0, rtol=1.0e-12)
        np.testing.assert_allclose(mode_stiffness / mode_mass, beta**4 * ei / (density * area), rtol=1.0e-9)
        np.testing.assert_allclose(gram[index, index], mode_mass, rtol=1.0e-3)
        stiffness_integral = ei * float(np.sum(curvature[1:-1, index] ** 2)) * ds
        np.testing.assert_allclose(stiffness_integral, mode_stiffness, rtol=1.0e-2)
        np.testing.assert_allclose(basis.mode_mass[index], mode_mass, rtol=1.0e-6)
        np.testing.assert_allclose(basis.mode_stiffness[index], mode_stiffness, rtol=1.0e-6)


def test_modal_generator_beam_cantilever_static_compat(test, device):
    """Verify the cantilever-static shape keeps the exact static tip-load response."""
    length = 1.0
    ei = 0.24
    generator = newton.ModalGeneratorBeam(
        length=length,
        half_width_y=0.045,
        half_width_z=0.035,
        area_moment_y=ei / 1.0e6,
        young_modulus=1.0e6,
        mode_specs=[
            {
                "type": newton.ModalGeneratorBeam.Mode.BENDING_Z,
                "boundary": newton.ModalGeneratorBeam.Boundary.CANTILEVER_STATIC,
            }
        ],
    )
    s = np.linspace(0.0, length, 7)
    points = np.column_stack((s - 0.5 * length, np.zeros_like(s), np.zeros_like(s))).astype(np.float32)
    basis = generator.build(sample_points=points)

    expected = s * s * (3.0 * length - s) / (2.0 * length**3)
    np.testing.assert_allclose(basis.sample_phi[:, 0, 2], expected, atol=1.0e-6)
    np.testing.assert_allclose(basis.sample_psi[0, 0], 0.0, atol=1.0e-9)
    np.testing.assert_allclose(basis.mode_stiffness, [3.0 * ei / length**3], rtol=1.0e-6)
    np.testing.assert_allclose(
        basis.mode_mass, [generator.density * generator.area * length * 33.0 / 140.0], rtol=1.0e-6
    )
    with test.assertRaisesRegex(ValueError, "only order 1"):
        newton.ModalGeneratorBeam(
            length=length,
            mode_specs=[
                {
                    "type": newton.ModalGeneratorBeam.Mode.BENDING_Z,
                    "boundary": newton.ModalGeneratorBeam.Boundary.CANTILEVER_STATIC,
                    "order": 2,
                }
            ],
        )


def test_modal_generator_beam_axial_fixed_free_properties(test, device):
    """Verify fixed-free axial modes and reject ignored linear axial orders."""
    length = 2.0
    density = 800.0
    young_modulus = 3.0e6
    orders = (1, 2)
    generator = newton.ModalGeneratorBeam(
        length=length,
        half_width_y=0.1,
        half_width_z=0.05,
        density=density,
        young_modulus=young_modulus,
        mode_specs=[
            {
                "type": newton.ModalGeneratorBeam.Mode.AXIAL,
                "boundary": newton.ModalGeneratorBeam.Boundary.FIXED_FREE,
                "order": order,
            }
            for order in orders
        ],
    )
    area = generator.area
    endpoints = np.array([[-0.5 * length, 0.05, 0.0], [0.5 * length, 0.0, -0.02]], dtype=np.float32)
    basis = generator.build(sample_points=endpoints)
    np.testing.assert_allclose(basis.sample_phi[0], 0.0, atol=1.0e-7)
    np.testing.assert_allclose(basis.sample_phi[1], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], atol=1.0e-6)
    np.testing.assert_allclose(basis.sample_psi, 0.0, atol=1.0e-9)

    cell_count = 2000
    ds = length / cell_count
    s = (np.arange(cell_count) + 0.5) * ds
    phi, _ = generator._evaluate_modes(np.column_stack((s - 0.5 * length, np.zeros_like(s), np.zeros_like(s))))
    strain = np.gradient(phi[:, :, 0], ds, axis=0)
    for index, order in enumerate(orders):
        k = (order - 0.5) * math.pi / length
        np.testing.assert_allclose(basis.mode_mass[index], density * area * length / 2.0, rtol=1.0e-6)
        np.testing.assert_allclose(
            basis.mode_stiffness[index], young_modulus * area * k * k * length / 2.0, rtol=1.0e-6
        )
        np.testing.assert_allclose(
            density * area * ds * np.sum(phi[:, index, 0] ** 2), basis.mode_mass[index], rtol=1.0e-4
        )
        np.testing.assert_allclose(
            young_modulus * area * ds * np.sum(strain[1:-1, index] ** 2), basis.mode_stiffness[index], rtol=1.0e-2
        )

    with test.assertRaisesRegex(ValueError, "only order 1"):
        newton.ModalGeneratorBeam(
            length=length, mode_specs=[{"type": newton.ModalGeneratorBeam.Mode.AXIAL, "order": 2}]
        )


def _spring_lattice(shape, spacing, spring_stiffness, node_mass, braced=True):
    """Build a lumped-mass spring lattice that preserves rigid-body motion.

    Args:
        shape: Node counts along ``x``, ``y``, and ``z``.
        spacing: Node spacing [m].
        spring_stiffness: Axial stiffness scale [N] divided by spring length.
        node_mass: Lumped node mass [kg].
        braced: Whether to add face and body diagonal springs. Without them
            the lattice has zero-energy shear mechanisms.

    Returns:
        Node positions [m], mass matrix [kg], and stiffness matrix [N/m].
    """
    points = np.array(
        [
            [i * spacing, j * spacing, k * spacing]
            for i in range(shape[0])
            for j in range(shape[1])
            for k in range(shape[2])
        ],
        dtype=np.float64,
    )
    points -= points.mean(axis=0)
    node_count = points.shape[0]
    reach = (math.sqrt(3.0) if braced else 1.0) * spacing * 1.0001
    stiffness = np.zeros((3 * node_count, 3 * node_count), dtype=np.float64)
    for a in range(node_count):
        for b in range(a + 1, node_count):
            delta = points[b] - points[a]
            distance = float(np.linalg.norm(delta))
            if distance <= reach:
                direction = delta / distance
                block = spring_stiffness / distance * np.outer(direction, direction)
                stiffness[3 * a : 3 * a + 3, 3 * a : 3 * a + 3] += block
                stiffness[3 * b : 3 * b + 3, 3 * b : 3 * b + 3] += block
                stiffness[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] -= block
                stiffness[3 * b : 3 * b + 3, 3 * a : 3 * a + 3] -= block
    mass = node_mass * np.eye(3 * node_count)
    return points, mass, stiffness


def _generalized_eigenvalues(mass, stiffness):
    chol = np.linalg.cholesky(mass)
    standard = np.linalg.solve(chol, np.linalg.solve(chol, stiffness).T).T
    return np.sort(np.linalg.eigvalsh(0.5 * (standard + standard.T)))


def test_modal_generator_fem_free_free_rigid_modes_removed(test, device):
    """Remove all six rigid-body modes of a stiff free-free FEM model.

    At steel-like stiffness, roundoff lifts the rigid-body eigenvalues above
    any small absolute threshold; the generator must still exclude them and
    return only elastic modes.
    """
    points, mass, stiffness = _spring_lattice((4, 2, 2), 0.01, 2.0e9, 8.0e-3)
    reference = _generalized_eigenvalues(mass, stiffness)
    first_elastic = float(reference[6])
    total_mass = float(np.trace(mass)) / 3.0

    generator = newton.ModalGeneratorFEM(points, mass, stiffness, mode_count=8)
    basis = generator.build()

    test.assertEqual(basis.mode_count, 8)
    test.assertGreater(float(np.min(generator.eigenvalues)), 1.0e-6 * first_elastic)
    np.testing.assert_allclose(generator.eigenvalues, reference[6:14], rtol=1.0e-6)
    test.assertEqual(generator.rigid_mode_count, 6)
    rigid = generator.rigid_modal_matrix
    np.testing.assert_allclose(rigid.T @ mass @ rigid, np.eye(6), atol=1.0e-9)
    np.testing.assert_allclose(rigid.T @ mass @ generator.modal_matrix, 0.0, atol=1.0e-6)
    np.testing.assert_allclose(basis.mode_mass, 1.0, rtol=1.0e-6)
    test.assertLess(float(np.max(np.abs(basis.mode_coupling_linear))), 1.0e-6 * math.sqrt(total_mass))
    radius = float(np.max(np.linalg.norm(points, axis=1)))
    test.assertLess(float(np.max(np.abs(basis.mode_coupling_angular))), 1.0e-6 * math.sqrt(total_mass) * radius)


def test_modal_generator_fem_discard_skips_elastic_modes(test, device):
    """Skip the lowest elastic modes, not rigid-body modes, with ``discard_mode_count``."""
    points, mass, stiffness = _spring_lattice((4, 2, 2), 0.01, 2.0e9, 8.0e-3)
    reference = _generalized_eigenvalues(mass, stiffness)
    test.assertGreaterEqual(reference.shape[0] - 6, 16)

    generator = newton.ModalGeneratorFEM(points, mass, stiffness, mode_count=4, discard_mode_count=2)
    basis = generator.build()

    test.assertEqual(basis.mode_count, 4)
    np.testing.assert_allclose(generator.eigenvalues, reference[8:12], rtol=1.0e-6)
    test.assertEqual(generator.rigid_mode_count, 6)


def test_modal_generator_fem_rejects_mechanism(test, device):
    """Reject zero-energy mechanisms and constraints that leave rigid-body motion."""
    points, mass, stiffness = _spring_lattice((4, 2, 2), 0.1, 1.0e3, 1.0e-2, braced=False)
    with test.assertRaisesRegex(ValueError, "zero-energy mechanism"):
        newton.ModalGeneratorFEM(points, mass, stiffness, mode_count=4).build()

    points, mass, stiffness = _spring_lattice((4, 2, 2), 0.1, 1.0e3, 1.0e-2)
    with test.assertRaisesRegex(ValueError, "do not remove all rigid-body motion"):
        newton.ModalGeneratorFEM(points, mass, stiffness, mode_count=4, fixed_node_indices=[0]).build()

    broken = stiffness.copy()
    broken[0, 0] += 1.0e3
    with test.assertRaisesRegex(ValueError, "does not preserve rigid-body motion"):
        newton.ModalGeneratorFEM(points, mass, broken, mode_count=4).build()

    # The fixed DOFs admit no rigid motion here, so the bar sliding along x away from the fixed node
    # is a mechanism, not remaining rigid-body motion, although it translates every free DOF.
    line = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    loose = _truss_stiffness(line, [(1, 2)], [1.0])
    with test.assertRaisesRegex(ValueError, "zero-energy mechanism"):
        newton.ModalGeneratorFEM(line, np.eye(9), loose, fixed_dof_indices=[0, 1, 2, 4, 5, 7, 8]).build()


def _truss_stiffness(points, bars, bar_stiffness):
    """Assemble the stiffness matrix [N/m] of axial bars between ``points``."""
    points = np.asarray(points, dtype=np.float64)
    stiffness = np.zeros((3 * points.shape[0], 3 * points.shape[0]), dtype=np.float64)
    for (a, b), k in zip(bars, bar_stiffness, strict=True):
        direction = points[b] - points[a]
        direction /= np.linalg.norm(direction)
        block = k * np.outer(direction, direction)
        for i, j, sign in ((a, a, 1.0), (b, b, 1.0), (a, b, -1.0), (b, a, -1.0)):
            stiffness[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += sign * block
    return stiffness


def test_modal_generator_fem_wide_spectrum_keeps_soft_modes(test, device):
    """Keep every elastic mode of models whose spectrum spans many decades.

    A soft mount (k = 1 N/m) in series with a stiff bar (k = 1e9 N/m) has modes at about 0.11 Hz and
    7.1 kHz, an eigenvalue ratio of 4e9; a free-free tetrahedron with one soft bar has a ratio of
    about 1e10. Both are valid elastic models and must build at the default settings.
    """
    line = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    stiffness = _truss_stiffness(line, [(0, 1), (1, 2)], [1.0, 1.0e9])
    mass = np.eye(9)
    fixed = [0, 1, 2, 4, 5, 7, 8]
    free = [3, 6]
    reference = _generalized_eigenvalues(mass[np.ix_(free, free)], stiffness[np.ix_(free, free)])
    generator = newton.ModalGeneratorFEM(line, mass, stiffness, fixed_dof_indices=fixed)
    basis = generator.build()
    test.assertEqual(basis.mode_count, 2)
    test.assertEqual(generator.rigid_mode_count, 0)
    np.testing.assert_allclose(generator.eigenvalues, reference, rtol=1.0e-6)
    test.assertGreater(float(reference[1] / reference[0]), 1.0e9)

    tet = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    bars = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    stiffness = _truss_stiffness(tet, bars, [1.0e10] * 5 + [1.0])
    reference = _generalized_eigenvalues(np.eye(12), stiffness)
    generator = newton.ModalGeneratorFEM(tet, np.eye(12), stiffness)
    basis = generator.build()
    test.assertEqual(generator.rigid_mode_count, 6)
    test.assertEqual(basis.mode_count, 6)
    np.testing.assert_allclose(generator.eigenvalues, reference[6:], rtol=1.0e-6)


def test_modal_generator_fem_damping_projection(test, device):
    """Record, warn about, or reject non-diagonal projected FEM damping."""
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    mass = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    stiffness = np.diag([0.0, 0.0, 0.0, 8.0, 18.0, 32.0])
    damping = np.diag([0.0, 0.0, 0.0, 0.4, 0.6, 0.8])
    damping[3, 4] = damping[4, 3] = 0.2

    def make(**kwargs):
        return newton.ModalGeneratorFEM(
            nodes, mass, stiffness, damping_matrix=damping, fixed_node_indices=[0], mode_count=3, **kwargs
        )

    generator = make()
    with test.assertWarnsRegex(UserWarning, "damping is not diagonal"):
        basis = generator.build()
    np.testing.assert_allclose(generator.damping_off_diagonal_ratio, 0.1 / 0.4, rtol=1.0e-6)
    np.testing.assert_allclose(basis.mode_damping, [0.2, 0.3, 0.4], rtol=1.0e-6)
    with test.assertRaisesRegex(ValueError, "damping is not diagonal"):
        make(damping_projection="error").build()
    with test.assertRaisesRegex(ValueError, "damping_projection"):
        make(damping_projection="full")


def test_modal_generator_craig_bampton_damping_projection_error(test, device):
    """Reject non-diagonal Craig-Bampton modal damping when projection errors are requested."""
    interface_positions, mass, stiffness, _, sample_points, recovery, _ = _build_craig_bampton_test_data()
    undamped_generator = newton.ModalGeneratorCraigBampton(
        interface_positions=interface_positions,
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        sample_points=sample_points,
        recovery_matrix=recovery,
    )
    undamped_basis = undamped_generator.build()
    modes = undamped_generator.modal_matrix
    modal_damping = np.diag(0.01 * undamped_basis.mode_stiffness.astype(np.float64))
    modal_damping[0, 1] = modal_damping[1, 0] = 0.05 * float(np.max(np.diag(modal_damping)))
    damping = mass @ modes @ modal_damping @ modes.T @ mass

    def make(**kwargs):
        return newton.ModalGeneratorCraigBampton(
            interface_positions=interface_positions,
            mass_matrix=mass,
            stiffness_matrix=stiffness,
            damping_matrix=damping,
            sample_points=sample_points,
            recovery_matrix=recovery,
            **kwargs,
        )

    generator = make(damping_projection="error")
    with test.assertRaisesRegex(ValueError, "damping is not diagonal"):
        generator.build()
    np.testing.assert_allclose(generator.damping_off_diagonal_ratio, 0.05, rtol=1.0e-4)

    tolerant = make(damping_projection="error", damping_coupling_tolerance=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        basis = tolerant.build()
    np.testing.assert_allclose(basis.mode_damping, np.diag(modal_damping), rtol=1.0e-5)


def test_modal_generator_craig_bampton_negative_damping_warns(test, device):
    """Warn when negative projected Craig-Bampton modal damping is clamped to zero."""
    interface_positions, mass, stiffness, _, sample_points, recovery, _ = _build_craig_bampton_test_data()
    generator = newton.ModalGeneratorCraigBampton(
        interface_positions=interface_positions,
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        damping_matrix=-0.01 * stiffness,
        sample_points=sample_points,
        recovery_matrix=recovery,
    )
    with test.assertWarnsRegex(UserWarning, "negative modal diagonal"):
        basis = generator.build()
    np.testing.assert_allclose(basis.mode_damping, 0.0, atol=0.0)


def test_modal_basis_lumped_inertia_coupling(test, device):
    """Check the lumped linear and angular coupling integrals of a two-sample basis."""
    points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    sample_phi = np.array(
        [
            [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    basis = newton.ModalBasis(sample_points=points, sample_phi=sample_phi, sample_mass=[1.0, 1.0])

    np.testing.assert_allclose(basis.mode_coupling_linear, [[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]], atol=1.0e-6)
    np.testing.assert_allclose(basis.mode_coupling_angular, [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]], atol=1.0e-6)


def test_modal_basis_coupling_unavailable_without_mass(test, device):
    """Check that a basis without sample masses has no coupling integrals and unit modal mass."""
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]],
    )
    test.assertIsNone(basis.sample_mass)
    test.assertIsNone(basis.mode_coupling_linear)
    test.assertIsNone(basis.mode_coupling_angular)
    test.assertIsNotNone(basis.mode_mass)
    np.testing.assert_allclose(basis.mode_mass, [1.0], atol=1.0e-7)


def test_modal_basis_coupling_explicit_override(test, device):
    """Check that an explicit linear coupling overrides the lumped value and is shape-validated."""
    explicit = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]],
        sample_mass=[1.0, 1.0],
        mode_coupling_linear=explicit,
    )
    np.testing.assert_allclose(basis.mode_coupling_linear, explicit, atol=1.0e-7)
    test.assertIsNotNone(basis.mode_coupling_angular)
    with test.assertRaises(ValueError):
        newton.ModalBasis(
            sample_points=[[0.0, 0.0, 0.0]],
            sample_phi=[[[0.0, 0.0, 1.0]]],
            mode_coupling_linear=[[1.0, 2.0]],
        )


def test_modal_generator_fem_inertia_coupling(test, device):
    """Check FEM linear and angular coupling integrals against the lumped node masses."""
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    mass = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    stiffness = np.zeros((6, 6), dtype=np.float64)
    stiffness[3, 3] = 8.0
    stiffness[4, 4] = 18.0
    stiffness[5, 5] = 32.0
    basis = newton.ModalGeneratorFEM(
        node_positions=nodes,
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        sample_points=nodes,
        sample_node_indices=[0, 1],
        fixed_node_indices=[0],
        mode_count=3,
    ).build()

    node_mass = np.array([1.0, 2.0])
    phi_nodes = np.stack([basis.sample_phi[0], basis.sample_phi[1]]).astype(np.float64)
    expected_linear = np.einsum("j,jmc->mc", node_mass, phi_nodes)
    expected_angular = np.einsum("j,jmc->mc", node_mass, np.cross(phi_nodes, nodes[:, None, :].astype(np.float64)))
    np.testing.assert_allclose(basis.mode_coupling_linear, expected_linear, atol=1.0e-5)
    np.testing.assert_allclose(basis.mode_coupling_angular, expected_angular, atol=1.0e-5)


def test_modal_generator_pod_inertia_coupling(test, device):
    """Check that POD ``total_mass`` yields the lumped linear coupling integral."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    displacements = np.array([[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype=np.float32)
    basis = newton.ModalGeneratorPOD(
        sample_points=points,
        displacements=displacements,
        mode_count=1,
        total_mass=2.0,
    ).build()

    test.assertIsNotNone(basis.mode_coupling_linear)
    sample_mass = np.full(2, 1.0)
    expected_linear = np.einsum("s,smc->mc", sample_mass, basis.sample_phi.astype(np.float64))
    np.testing.assert_allclose(basis.mode_coupling_linear, expected_linear, atol=1.0e-6)


def test_modal_generator_pod_coupling_matches_lumped(test, device):
    """Check POD linear, angular, centrifugal, and Coriolis integrals against a uniform lumped mass."""
    n = 9
    xs = np.linspace(0.0, 1.0, n, dtype=np.float32)
    points = np.column_stack([xs, 0.3 * xs, -0.2 * xs]).astype(np.float32)
    displacements = np.zeros((2, n, 3), dtype=np.float32)
    displacements[0, :, 2] = xs
    displacements[1, :, 1] = xs * xs
    total_mass = 3.0
    basis = newton.ModalGeneratorPOD(
        sample_points=points,
        displacements=displacements,
        mode_count=2,
        total_mass=total_mass,
    ).build()

    sample_mass = np.full(n, total_mass / n, dtype=np.float64)
    phi = basis.sample_phi.astype(np.float64)
    points64 = points.astype(np.float64)

    expected_linear = np.einsum("s,smc->mc", sample_mass, phi)
    expected_angular = np.einsum("s,smc->mc", sample_mass, np.cross(phi, points64[:, None, :]))
    expected_centrifugal = np.einsum("s,smc,sd->mcd", sample_mass, phi, points64)
    cross_pairs = np.cross(phi[:, None, :, :], phi[:, :, None, :])
    expected_coriolis = np.einsum("s,sijc->ijc", sample_mass, cross_pairs)

    np.testing.assert_allclose(basis.mode_coupling_linear, expected_linear, atol=1.0e-6)
    np.testing.assert_allclose(basis.mode_coupling_angular, expected_angular, atol=1.0e-6)
    np.testing.assert_allclose(basis.mode_coupling_centrifugal, expected_centrifugal, atol=1.0e-6)
    np.testing.assert_allclose(basis.mode_coupling_coriolis, expected_coriolis, atol=1.0e-6)


def test_estimate_sample_psi_rotation_and_translation(test, device):
    """Check that psi estimation recovers a rigid rotation and ignores a rigid translation."""
    rng = np.random.default_rng(0)
    points = rng.normal(size=(40, 3)).astype(np.float32)

    omega = np.array([0.3, -0.5, 0.2], dtype=np.float64)
    rotation = np.cross(np.tile(omega, (points.shape[0], 1)), points.astype(np.float64))
    psi = _estimate_sample_psi(points, rotation[:, None, :].astype(np.float32))
    np.testing.assert_allclose(psi[:, 0, :], np.tile(omega, (points.shape[0], 1)), atol=1.0e-5)

    translation = np.tile(np.array([1.0, -2.0, 3.0], dtype=np.float32), (points.shape[0], 1))
    psi_t = _estimate_sample_psi(points, translation[:, None, :])
    np.testing.assert_allclose(psi_t, 0.0, atol=1.0e-6)


def test_estimate_sample_psi_degenerate_warns(test, device):
    """Check that coplanar samples warn and still recover the in-plane rotation."""
    xs = np.linspace(-1.0, 1.0, 6, dtype=np.float32)
    points = np.array([[x, y, 0.0] for x in xs for y in xs], dtype=np.float32)
    phi = np.cross(np.tile([0.0, 0.0, 1.0], (points.shape[0], 1)), points)[:, None, :].astype(np.float32)
    with test.assertWarns(UserWarning):
        psi = _estimate_sample_psi(points, phi)
    test.assertTrue(np.isfinite(psi).all())
    np.testing.assert_allclose(psi[:, 0, 2], 1.0, atol=1.0e-4)


def test_modal_generator_sampled_psi(test, device):
    """Check that the sampled generator passes explicit psi through and defaults it to zero."""
    n = 5
    xs = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    points = np.column_stack([xs, np.zeros(n), np.zeros(n)]).astype(np.float32)
    phi = np.zeros((n, 1, 3), dtype=np.float32)
    phi[:, 0, 1] = xs + 0.5
    explicit_psi = np.tile(np.array([0.7, 0.0, 0.0], dtype=np.float32), (n, 1))[:, None, :]

    passthrough = newton.ModalGeneratorSampled(points, phi, sample_psi=explicit_psi).build()
    np.testing.assert_allclose(passthrough.sample_psi, explicit_psi, atol=1.0e-7)

    default = newton.ModalGeneratorSampled(points, phi).build()
    np.testing.assert_allclose(default.sample_psi, 0.0, atol=1.0e-7)


def test_modal_generator_pod_derives_psi(test, device):
    """Check that POD ``derive_psi`` recovers the rotation axis and is off by default."""
    rng = np.random.default_rng(1)
    points = rng.normal(size=(48, 3)).astype(np.float32)
    omega = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    snapshot = np.cross(np.tile(omega, (points.shape[0], 1)), points.astype(np.float64))

    basis = newton.ModalGeneratorPOD(
        sample_points=points,
        displacements=snapshot[None].astype(np.float32),
        mode_count=1,
        derive_psi=True,
    ).build()
    psi = basis.sample_psi[:, 0, :].astype(np.float64)
    test.assertGreater(float(np.linalg.norm(psi, axis=1).min()), 1.0e-3)
    directions = psi / np.linalg.norm(psi, axis=1, keepdims=True)
    np.testing.assert_allclose(np.abs(directions @ omega), 1.0, atol=1.0e-3)

    uncoupled = newton.ModalGeneratorPOD(
        sample_points=points,
        displacements=snapshot[None].astype(np.float32),
        mode_count=1,
        derive_psi=False,
    ).build()
    np.testing.assert_allclose(uncoupled.sample_psi, 0.0, atol=1.0e-7)


def test_modal_generator_fem_derives_psi(test, device):
    """Check that FEM ``derive_psi`` gives finite nonzero angular samples and zeros when disabled."""
    nodes = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)], dtype=np.float32)
    dof = 3 * nodes.shape[0]
    mass = np.eye(dof)
    stiffness = np.diag(np.linspace(1.0, 5.0, dof))

    basis = newton.ModalGeneratorFEM(
        node_positions=nodes,
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        fixed_node_indices=[0],
        mode_count=3,
        derive_psi=True,
    ).build()
    test.assertEqual(basis.sample_psi.shape, (nodes.shape[0], 3, 3))
    test.assertTrue(np.isfinite(basis.sample_psi).all())
    test.assertGreater(float(np.max(np.abs(basis.sample_psi))), 0.0)

    uncoupled = newton.ModalGeneratorFEM(
        node_positions=nodes,
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        fixed_node_indices=[0],
        mode_count=3,
        derive_psi=False,
    ).build()
    np.testing.assert_allclose(uncoupled.sample_psi, 0.0, atol=1.0e-7)


def test_modal_basis_copy_preserves_coupling(test, device):
    """Check that ModalBasis.copy preserves sample masses and coupling integrals."""
    n = 5
    xs = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    sample_points = np.column_stack([xs, np.zeros(n), np.zeros(n)]).astype(np.float32)
    sample_phi = np.zeros((n, 2, 3), dtype=np.float32)
    sample_phi[:, 0, 2] = xs + 0.5
    sample_phi[:, 1, 1] = (xs + 0.5) ** 2
    basis = newton.ModalBasis(
        sample_points=sample_points,
        sample_phi=sample_phi,
        sample_mass=np.full(n, 0.2, dtype=np.float32),
        mode_stiffness=[10.0, 20.0],
        mode_damping=[0.1, 0.2],
    )
    test.assertIsNotNone(basis.sample_mass)
    test.assertIsNotNone(basis.mode_coupling_linear)
    test.assertIsNotNone(basis.mode_coupling_angular)
    test.assertIsNotNone(basis.mode_coupling_centrifugal)
    test.assertIsNotNone(basis.mode_coupling_coriolis)

    clone = basis.copy()
    np.testing.assert_allclose(clone.sample_mass, basis.sample_mass, atol=0)
    np.testing.assert_allclose(clone.mode_coupling_linear, basis.mode_coupling_linear, atol=0)
    np.testing.assert_allclose(clone.mode_coupling_angular, basis.mode_coupling_angular, atol=0)
    np.testing.assert_allclose(clone.mode_coupling_centrifugal, basis.mode_coupling_centrifugal, atol=0)
    np.testing.assert_allclose(clone.mode_coupling_coriolis, basis.mode_coupling_coriolis, atol=0)


def test_modal_basis_mode_mass_from_sample_mass(test, device):
    """Check that modal masses are integrated from the sample masses."""
    basis = newton.ModalBasis(
        sample_points=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        sample_phi=[
            [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ],
        sample_mass=[1.0, 1.0],
    )
    np.testing.assert_allclose(basis.mode_mass, [2.0, 2.0], atol=1.0e-6)


def test_modal_basis_mode_mass_explicit_wins(test, device):
    """Check that an explicit mode_mass overrides the sampled modal mass."""
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]],
        sample_mass=[1.0, 1.0],
        mode_mass=[99.0],
    )
    np.testing.assert_allclose(basis.mode_mass, [99.0], atol=1.0e-6)


def test_elastic_mode_coupling_arrays(test, device):
    """Check that finalize stores basis coupling integrals and zeros for a body without a basis."""
    basis = newton.ModalBasis(
        sample_points=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]],
        sample_mass=[1.0, 1.0],
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body_elastic(**_NOMINAL_FRAME, modal_basis=basis, label="coupled")
    builder.add_body_elastic(
        mass=1.0, inertia=_identity_inertia(), mode_count=1, mode_mass=[1.0], mode_stiffness=[1.0], label="plain"
    )
    builder.color()
    model = builder.finalize(device=device)

    np.testing.assert_allclose(
        model.elastic_mode_coupling_linear.numpy(), [[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]], atol=1.0e-6
    )
    np.testing.assert_allclose(
        model.elastic_mode_coupling_angular.numpy(), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], atol=1.0e-6
    )
    np.testing.assert_allclose(model.elastic_mode_coupling_centrifugal.numpy(), np.zeros((2, 3, 3)), atol=1.0e-6)
    np.testing.assert_allclose(model.elastic_mode_coupling_coriolis.numpy(), np.zeros((2, 3)), atol=1.0e-6)


def test_elastic_coriolis_array_padding(test, device):
    """Check that per-body Coriolis blocks are zero-padded to the largest mode count."""
    basis_a = newton.ModalBasis(
        sample_points=[[1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]]],
        sample_mass=[1.0],
    )
    basis_b = newton.ModalBasis(
        sample_points=[[1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]],
        sample_mass=[2.0],
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body_elastic(**_NOMINAL_FRAME, modal_basis=basis_a, label="a")
    builder.add_body_elastic(**_NOMINAL_FRAME, modal_basis=basis_b, label="b")
    builder.color()
    model = builder.finalize(device=device)

    expected = np.zeros((8, 3), dtype=np.float32)
    expected[5] = [2.0, 0.0, 0.0]
    expected[6] = [-2.0, 0.0, 0.0]
    np.testing.assert_allclose(model.elastic_mode_coupling_coriolis.numpy(), expected, atol=1.0e-6)


def test_elastic_mode_coupling_arrays_merge(test, device):
    """Check that add_builder copies the elastic coupling integrals of each instance."""
    basis = newton.ModalBasis(
        sample_points=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]],
        sample_mass=[1.0, 1.0],
    )
    sub = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    sub.add_body_elastic(**_NOMINAL_FRAME, modal_basis=basis, label="e")
    main = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    main.add_builder(sub)
    main.add_builder(sub)
    main.color()
    model = main.finalize(device=device)

    np.testing.assert_allclose(
        model.elastic_mode_coupling_linear.numpy(), [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]], atol=1.0e-6
    )


def test_elastic_coriolis_array_merge_different_mode_count(test, device):
    """Check the Coriolis block layout after merging builders with different mode counts."""
    basis_a = newton.ModalBasis(
        sample_points=[[1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]]],
        sample_mass=[1.0],
    )
    basis_b = newton.ModalBasis(
        sample_points=[[1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]],
        sample_mass=[2.0],
    )
    sub_a = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    sub_a.add_body_elastic(**_NOMINAL_FRAME, modal_basis=basis_a, label="a")
    sub_b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    sub_b.add_body_elastic(**_NOMINAL_FRAME, modal_basis=basis_b, label="b")
    main = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    main.add_builder(sub_a)
    main.add_builder(sub_b)
    main.color()
    model = main.finalize(device=device)

    test.assertEqual(model.elastic_body_count, 2)
    test.assertEqual(model.elastic_max_mode_count, 2)
    max_modes = int(model.elastic_max_mode_count)
    block_a = 0 * max_modes * max_modes
    block_b = 1 * max_modes * max_modes
    test.assertEqual(block_b, 4)

    coriolis = model.elastic_mode_coupling_coriolis.numpy()
    test.assertEqual(coriolis.shape, (8, 3))
    np.testing.assert_allclose(coriolis[block_a : block_a + 4], np.zeros((4, 3)), atol=1.0e-6)
    expected_b = np.zeros((4, 3), dtype=np.float32)
    expected_b[1] = [2.0, 0.0, 0.0]
    expected_b[2] = [-2.0, 0.0, 0.0]
    np.testing.assert_allclose(coriolis[block_b : block_b + 4], expected_b, atol=1.0e-6)


def test_elastic_mode_pair_arrays_padding(test, device):
    """Pad quadratic and modal mass matrix blocks per body and default to the diagonal mode mass."""
    basis_a = newton.ModalBasis(
        sample_points=[[1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]]],
        sample_mass=[1.0],
    )
    basis_b = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        sample_phi=[[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        sample_mass=[1.0, 2.0],
    )
    sub = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    sub.add_body_elastic(**_NOMINAL_FRAME, modal_basis=basis_b, label="b")
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body_elastic(**_NOMINAL_FRAME, modal_basis=basis_a, label="a")
    builder.add_builder(sub)
    builder.add_body_elastic(
        mass=1.0, inertia=_identity_inertia(), mode_count=2, mode_mass=[3.0, 4.0], mode_stiffness=[1.0, 1.0]
    )
    builder.color()
    model = builder.finalize(device=device)

    test.assertEqual(model.elastic_max_mode_count, 2)
    quadratic = model.elastic_mode_coupling_quadratic.numpy()
    mass_matrix = model.elastic_mode_mass_matrix.numpy()
    test.assertEqual(quadratic.shape, (12, 3, 3))
    test.assertEqual(mass_matrix.shape, (12,))

    expected_quadratic = np.zeros((3, 2, 2, 3, 3), dtype=np.float32)
    expected_quadratic[0, :1, :1] = basis_a.mode_coupling_quadratic
    expected_quadratic[1] = basis_b.mode_coupling_quadratic
    np.testing.assert_allclose(quadratic, expected_quadratic.reshape((12, 3, 3)), atol=1.0e-6)

    expected_mass_matrix = np.zeros((3, 2, 2), dtype=np.float32)
    expected_mass_matrix[0, 0, 0] = 1.0
    expected_mass_matrix[1] = [[1.0, 1.0], [1.0, 3.0]]
    expected_mass_matrix[2] = np.diag([3.0, 4.0])
    np.testing.assert_allclose(mass_matrix, expected_mass_matrix.reshape((12,)), atol=1.0e-6)
    np.testing.assert_allclose(model.elastic_mode_mass.numpy(), [1.0, 1.0, 3.0, 3.0, 4.0], atol=1.0e-6)


def _deformed_endpoint_world(model, state, joint: int, side: str) -> np.ndarray:
    body = int(model.joint_parent.numpy()[joint]) if side == "parent" else int(model.joint_child.numpy()[joint])
    xforms = model.joint_X_p.numpy() if side == "parent" else model.joint_X_c.numpy()
    local = np.array(xforms[joint, :3], dtype=float)
    endpoint_ids = (
        model.joint_parent_elastic_endpoint.numpy() if side == "parent" else model.joint_child_elastic_endpoint.numpy()
    )
    endpoint = int(endpoint_ids[joint])
    if endpoint >= 0:
        elastic = int(model.body_elastic_index.numpy()[body])
        owner_joint = int(model.elastic_joint.numpy()[elastic])
        q_start = int(model.joint_q_start.numpy()[owner_joint])
        mode_count = int(model.elastic_mode_count.numpy()[elastic])
        phi = model.elastic_endpoint_phi.numpy().reshape((-1, 3))
        max_modes = int(model.elastic_max_mode_count)
        q = state.joint_q.numpy()
        for i in range(mode_count):
            local += phi[endpoint * max_modes + i] * q[q_start + 7 + i]
    if body < 0:
        return local
    return _transform_point_np(state.body_q.numpy()[body], local)


def _solve_fourbar(theta2: float, a: float, b: float, c: float, d: float) -> tuple[float, float]:
    """Return open-branch coupler and rocker angles for a planar four-bar."""
    k1 = d / a
    k2 = d / c
    k3 = (a * a - b * b + c * c + d * d) / (2.0 * a * c)
    A = k1 - math.cos(theta2)
    B = -math.sin(theta2)
    C = k2 * math.cos(theta2) - k3
    denom = math.sqrt(A * A + B * B)
    theta4 = math.atan2(B, A) + math.acos(np.clip(C / denom, -1.0, 1.0))
    cx = d + c * math.cos(theta4) - a * math.cos(theta2)
    cy = c * math.sin(theta4) - a * math.sin(theta2)
    theta3 = math.atan2(cy, cx)
    return theta3, theta4


def test_elastic_link_layout(test, device):
    """Check the elastic owner joint's coordinate layout and initial modal state."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=2,
        mode_mass=[2.0, 3.0],
        mode_stiffness=[5.0, 7.0],
        mode_damping=[0.2, 0.3],
        mode_q=[0.1, -0.2],
        mode_qd=[0.4, -0.5],
        label="elastic",
    )
    builder.color()
    model = builder.finalize(device=device)

    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    qd_start = int(model.joint_qd_start.numpy()[owner_joint])

    test.assertEqual(body, 0)
    test.assertEqual(model.elastic_body_count, 1)
    test.assertEqual(model.elastic_max_mode_count, 2)
    test.assertEqual(int(model.body_elastic_index.numpy()[body]), 0)
    test.assertEqual(int(model.joint_type.numpy()[owner_joint]), int(newton.JointType.ELASTIC))
    test.assertEqual(int(model.joint_q_start.numpy()[owner_joint + 1] - model.joint_q_start.numpy()[owner_joint]), 9)
    test.assertEqual(int(model.joint_qd_start.numpy()[owner_joint + 1] - model.joint_qd_start.numpy()[owner_joint]), 8)
    np.testing.assert_allclose(model.joint_q.numpy()[q_start + 7 : q_start + 9], [0.1, -0.2], atol=1.0e-7)
    np.testing.assert_allclose(model.joint_qd.numpy()[qd_start + 6 : qd_start + 8], [0.4, -0.5], atol=1.0e-7)
    np.testing.assert_allclose(model.elastic_mode_mass.numpy(), [2.0, 3.0], atol=1.0e-7)


def test_add_joint_elastic_initializes_relative_frame_coordinates(test, device):
    """Check that add_joint_elastic stores the relative frame pose and FK restores the child pose."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent_xform_world = wp.transform(wp.vec3(0.4, -0.2, 0.1), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.35))
    child_xform_world = wp.transform(wp.vec3(-0.3, 0.6, 0.7), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.25))
    parent = builder.add_link(xform=parent_xform_world, mass=1.0, inertia=_identity_inertia())
    child = builder.add_link(xform=child_xform_world, mass=1.0, inertia=_identity_inertia())
    parent_anchor = wp.transform(wp.vec3(0.1, 0.2, -0.1), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.2))
    child_anchor = wp.transform(wp.vec3(-0.2, 0.05, 0.15), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -0.15))
    joint = builder.add_joint_elastic(
        parent=parent,
        child=child,
        mode_count=1,
        parent_xform=parent_anchor,
        child_xform=child_anchor,
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)

    expected_frame = wp.transform_inverse(parent_xform_world * parent_anchor) * child_xform_world * child_anchor
    q_start = int(model.joint_q_start.numpy()[joint])
    np.testing.assert_allclose(model.joint_q.numpy()[q_start : q_start + 7], np.asarray(expected_frame), atol=1.0e-6)

    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    np.testing.assert_allclose(state.body_q.numpy()[child], np.asarray(child_xform_world), atol=1.0e-6)


def test_elastic_fk_sync(test, device):
    """Check that eval_fk copies the elastic frame pose and twist to the body and keeps the modes."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), mode_count=1)
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    qd_start = int(model.joint_qd_start.numpy()[owner_joint])

    q = state.joint_q.numpy()
    q[q_start : q_start + 7] = [1.0, 2.0, 3.0, 0.0, 0.0, math.sin(0.25), math.cos(0.25)]
    q[q_start + 7] = 0.35
    state.joint_q.assign(q)

    qd = state.joint_qd.numpy()
    qd[qd_start : qd_start + 7] = [0.4, 0.5, 0.6, 0.1, 0.2, 0.3, -0.7]
    state.joint_qd.assign(qd)

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    np.testing.assert_allclose(state.body_q.numpy()[body], q[q_start : q_start + 7], atol=1.0e-6)
    np.testing.assert_allclose(state.body_qd.numpy()[body], qd[qd_start : qd_start + 6], atol=1.0e-6)
    np.testing.assert_allclose(state.joint_q.numpy()[q_start + 7], 0.35, atol=1.0e-7)


def test_elastic_fk_ik_com_velocity_round_trip(test, device):
    """Check that FK and IK round-trip the frame velocity of an elastic body with an offset COM."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        com=(0.2, -0.1, 0.3),
        inertia=_identity_inertia(),
        mode_count=1,
    )
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    qd_start = int(model.joint_qd_start.numpy()[owner_joint])

    q = state.joint_q.numpy()
    q[q_start : q_start + 7] = [0.3, -0.2, 0.4, 0.0, math.sin(0.2), 0.0, math.cos(0.2)]
    state.joint_q.assign(q)
    expected_qd = np.array([0.4, -0.5, 0.6, 0.7, -0.8, 0.9], dtype=np.float32)
    qd = state.joint_qd.numpy()
    qd[qd_start : qd_start + 6] = expected_qd
    state.joint_qd.assign(qd)

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    np.testing.assert_allclose(state.body_qd.numpy()[body], expected_qd, atol=1.0e-6)

    recovered_q = wp.zeros_like(state.joint_q)
    recovered_qd = wp.zeros_like(state.joint_qd)
    newton.eval_ik(model, state, recovered_q, recovered_qd)
    np.testing.assert_allclose(recovered_q.numpy()[q_start : q_start + 7], q[q_start : q_start + 7], atol=1.0e-6)
    np.testing.assert_allclose(recovered_qd.numpy()[qd_start : qd_start + 6], expected_qd, atol=1.0e-6)


def test_elastic_endpoint_shape_sampling(test, device):
    """Check that a mode_shape_fn is sampled at an elastic joint endpoint."""

    def shape_fn(x):
        return np.array([[2.0 * x[0], -x[1], 0.5]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_body(mass=1.0, inertia=_identity_inertia())
    child = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), mode_count=1, mode_shape_fn=shape_fn)
    joint = builder.add_joint_revolute(
        parent=parent,
        child=child,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.5, -0.25, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)

    test.assertEqual(int(model.joint_parent_elastic_endpoint.numpy()[joint]), -1)
    endpoint = int(model.joint_child_elastic_endpoint.numpy()[joint])
    test.assertGreaterEqual(endpoint, 0)
    phi = model.elastic_endpoint_phi.numpy().reshape((-1, 3))
    np.testing.assert_allclose(phi[endpoint], [1.0, 0.25, 0.5], atol=1.0e-7)


def test_elastic_endpoint_modal_basis_sampling(test, device):
    """Check that a joint endpoint reuses a coincident basis sample and its mode values."""
    basis = newton.ModalBasis(
        sample_points=[[0.5, -0.25, 0.0]],
        sample_phi=[[[1.0, 0.25, 0.5]]],
        mode_mass=[1.0],
    )

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_body(mass=1.0, inertia=_identity_inertia())
    child = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), modal_basis=basis)
    joint = builder.add_joint_revolute(
        parent=parent,
        child=child,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.5, -0.25, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)

    endpoint = int(model.joint_child_elastic_endpoint.numpy()[joint])
    test.assertGreaterEqual(endpoint, 0)
    test.assertEqual(int(model.elastic_endpoint_sample.numpy()[endpoint]), 0)
    test.assertEqual(model.modal_basis_count, 1)
    np.testing.assert_allclose(model.elastic_basis.numpy(), [0], atol=0)
    phi = model.elastic_endpoint_phi.numpy().reshape((-1, 3))
    np.testing.assert_allclose(phi[endpoint], [1.0, 0.25, 0.5], atol=1.0e-7)


def test_elastic_endpoint_angular_sampling(test, device):
    """Check that endpoint angular mode values match the beam basis evaluation."""
    length = 2.0
    attach_local = (-0.5 * length, 0.0, 0.0)
    gen = newton.ModalGeneratorBeam(
        length=length,
        half_width_y=0.1,
        half_width_z=0.1,
        mode_specs=[{"type": "bending_z", "boundary": "pinned-pinned"}],
        sample_count=9,
    )
    basis = gen.build()
    _phi, expected = basis.evaluate(np.array(attach_local, dtype=np.float32))
    test.assertGreater(float(np.linalg.norm(expected)), 1.0e-3)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_body(mass=1.0, inertia=_identity_inertia())
    child = builder.add_body_elastic(modal_basis=basis)
    joint = builder.add_joint_fixed(
        parent=parent,
        child=child,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(*attach_local), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)

    endpoint = int(model.joint_child_elastic_endpoint.numpy()[joint])
    test.assertGreaterEqual(endpoint, 0)
    max_modes = int(model.elastic_max_mode_count)
    psi = model.elastic_endpoint_psi.numpy().reshape((-1, max_modes, 3))
    np.testing.assert_allclose(psi[endpoint, 0], expected[0], atol=1.0e-6)


def test_elastic_endpoint_angular_zero_without_beam(test, device):
    """Check that a basis without angular samples gives zero endpoint angular values."""
    basis = newton.ModalBasis(
        sample_points=[[0.5, -0.25, 0.0]],
        sample_phi=[[[1.0, 0.25, 0.5]]],
        mode_mass=[1.0],
    )
    np.testing.assert_allclose(basis.sample_psi, np.zeros((1, 1, 3)), atol=0)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_body(mass=1.0, inertia=_identity_inertia())
    child = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), modal_basis=basis)
    joint = builder.add_joint_fixed(
        parent=parent,
        child=child,
        parent_xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.5, -0.25, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)

    endpoint = int(model.joint_child_elastic_endpoint.numpy()[joint])
    test.assertGreaterEqual(endpoint, 0)
    max_modes = int(model.elastic_max_mode_count)
    psi = model.elastic_endpoint_psi.numpy().reshape((-1, max_modes, 3))
    np.testing.assert_allclose(psi[endpoint, 0], np.zeros(3), atol=0)


def _pure_twist_basis(
    clamp_local, has_psi: bool, mode_mass: float = 0.05, mode_stiffness: float = 2.0, mode_damping: float = 0.0
) -> "newton.ModalBasis":
    psi = [1.0, 0.0, 0.0] if has_psi else [0.0, 0.0, 0.0]
    return newton.ModalBasis(
        sample_points=[list(clamp_local)],
        sample_phi=[[[0.0, 0.0, 0.0]]],
        sample_psi=[[psi]],
        mode_mass=[mode_mass],
        mode_stiffness=[mode_stiffness],
        mode_damping=[mode_damping],
    )


def _check_elastic_clamp_moment_reaction(test, device, compliant: bool):
    clamp_local = (0.5, 0.0, 0.0)
    initial_twist = 0.3

    shape_cfg = newton.ModelBuilder.ShapeConfig()
    shape_cfg.density = 0.0
    shape_cfg.has_shape_collision = False
    shape_cfg.has_particle_collision = False

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    rigid_bodies = {}
    for name, has_psi, y in (("twist", True, 0.0), ("control", False, 2.0)):
        beam = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(0.0, y, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=_identity_inertia(),
            mode_q=[initial_twist],
            modal_basis=_pure_twist_basis(clamp_local, has_psi),
            label=f"twist_beam_{name}",
        )
        builder.add_shape_box(beam, hx=0.5, hy=0.05, hz=0.05, cfg=shape_cfg)
        rigid = builder.add_body(
            xform=wp.transform(wp.vec3(0.5, y, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
            label=f"twist_rigid_{name}",
        )
        builder.add_joint_fixed(
            parent=rigid,
            child=beam,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform(wp.vec3(*clamp_local), wp.quat_identity()),
        )
        rigid_bodies[name] = rigid

    builder.color()
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    solver = _make_vbd_solver(model, compliant=compliant, iterations=24)

    identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    max_angle = dict.fromkeys(rigid_bodies, 0.0)
    dt = 1.0 / (60.0 * 8.0)
    for _ in range(120):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        body_q = state_0.body_q.numpy()
        if not np.isfinite(body_q).all():
            raise AssertionError("body transforms contain non-finite values")
        for name, body in rigid_bodies.items():
            max_angle[name] = max(max_angle[name], _quat_angle_error(body_q[body][3:7], identity_quat))

    test.assertGreater(
        max_angle["twist"],
        1.0e-2,
        f"clamp moment did not rotate the rigid body: max angle = {max_angle['twist']:.3e}",
    )
    test.assertLess(
        max_angle["control"],
        1.0e-3,
        f"rigid body rotated without angular mode shapes: max angle = {max_angle['control']:.3e}",
    )


def test_elastic_clamp_moment_reaction(test, device):
    """Transmit a pure modal twist moment through a clamp to the rigid side in both rigid formulations."""
    for compliant in _RIGID_ALM_MODES:
        with test.subTest(rigid_compliant_alm=compliant):
            _check_elastic_clamp_moment_reaction(test, device, compliant)


def _check_elastic_clamp_moment_reaction_parent_side(test, device, compliant: bool):
    clamp_local = (0.5, 0.0, 0.0)
    initial_twist = 0.3

    shape_cfg = newton.ModelBuilder.ShapeConfig()
    shape_cfg.density = 0.0
    shape_cfg.has_shape_collision = False
    shape_cfg.has_particle_collision = False

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    rigid_bodies = {}
    twist_joint = -1
    for name, has_psi, y in (("twist", True, 0.0), ("control", False, 2.0)):
        beam = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(0.0, y, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=_identity_inertia(),
            mode_q=[initial_twist],
            modal_basis=_pure_twist_basis(clamp_local, has_psi),
            label=f"twist_beam_{name}",
        )
        builder.add_shape_box(beam, hx=0.5, hy=0.05, hz=0.05, cfg=shape_cfg)
        rigid = builder.add_body(
            xform=wp.transform(wp.vec3(0.5, y, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
            label=f"twist_rigid_{name}",
        )
        joint = builder.add_joint_fixed(
            parent=beam,
            child=rigid,
            parent_xform=wp.transform(wp.vec3(*clamp_local), wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
        if name == "twist":
            twist_joint = joint
        rigid_bodies[name] = rigid

    builder.color()
    model = builder.finalize(device=device)

    test.assertGreaterEqual(int(model.joint_parent_elastic_endpoint.numpy()[twist_joint]), 0)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    solver = _make_vbd_solver(model, compliant=compliant, iterations=24)

    identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    max_angle = dict.fromkeys(rigid_bodies, 0.0)
    dt = 1.0 / (60.0 * 8.0)
    for _ in range(120):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        body_q = state_0.body_q.numpy()
        if not np.isfinite(body_q).all():
            raise AssertionError("body transforms contain non-finite values")
        for name, body in rigid_bodies.items():
            max_angle[name] = max(max_angle[name], _quat_angle_error(body_q[body][3:7], identity_quat))

    test.assertGreater(
        max_angle["twist"],
        1.0e-2,
        f"clamp moment did not rotate the rigid body: max angle = {max_angle['twist']:.3e}",
    )
    test.assertLess(
        max_angle["control"],
        1.0e-3,
        f"rigid body rotated without angular mode shapes: max angle = {max_angle['control']:.3e}",
    )


def test_elastic_clamp_moment_reaction_parent_side(test, device):
    """Transmit a modal twist moment from an elastic parent endpoint in both rigid formulations."""
    for compliant in _RIGID_ALM_MODES:
        with test.subTest(rigid_compliant_alm=compliant):
            _check_elastic_clamp_moment_reaction_parent_side(test, device, compliant)


def _check_elastic_clamp_moment_conserves_angular_momentum(test, device, compliant: bool):
    clamp_local = (0.5, 0.0, 0.0)
    initial_twist = 0.3
    inertia = wp.mat33(0.02, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05)

    shape_cfg = newton.ModelBuilder.ShapeConfig()
    shape_cfg.density = 0.0
    shape_cfg.has_shape_collision = False
    shape_cfg.has_particle_collision = False

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    beam = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=inertia,
        com=wp.vec3(0.0, 0.0, 0.0),
        mode_q=[initial_twist],
        modal_basis=_pure_twist_basis(clamp_local, True, mode_mass=1.0e6, mode_stiffness=0.0, mode_damping=0.0),
        label="momentum_beam",
    )
    builder.add_shape_box(beam, hx=0.5, hy=0.05, hz=0.05, cfg=shape_cfg)
    rigid = builder.add_body(
        xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=inertia,
        label="momentum_rigid",
    )
    builder.add_joint_fixed(
        parent=rigid,
        child=beam,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(*clamp_local), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    solver = _make_vbd_solver(model, compliant=compliant, iterations=128)

    max_spin = 0.0
    max_imbalance = 0.0
    max_off_axis = 0.0
    dt = 1.0 / (60.0 * 8.0)
    for _ in range(120):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        body_q = state_0.body_q.numpy()
        if not np.isfinite(body_q).all():
            raise AssertionError("body transforms contain non-finite values")
        spin_beam = float(body_q[beam][3])
        spin_rigid = float(body_q[rigid][3])
        max_spin = max(max_spin, abs(spin_beam), abs(spin_rigid))
        max_imbalance = max(max_imbalance, abs(spin_beam + spin_rigid))
        for body in (beam, rigid):
            max_off_axis = max(max_off_axis, abs(float(body_q[body][4])), abs(float(body_q[body][5])))

    test.assertGreater(max_spin, 1.0e-2, f"system did not rotate about the twist axis: max |qx| = {max_spin:.3e}")
    test.assertLess(max_off_axis, 1.0e-2, f"rotation left the twist axis: max off-axis |q| = {max_off_axis:.3e}")
    test.assertLess(
        max_imbalance,
        0.1 * max_spin,
        f"angular momentum not conserved at the joint: max |qx_beam + qx_rigid| = {max_imbalance:.3e} "
        f"vs max |qx| = {max_spin:.3e}",
    )


def test_elastic_clamp_moment_conserves_angular_momentum(test, device):
    """Conserve angular momentum across an elastic clamp in both rigid formulations."""
    for compliant in _RIGID_ALM_MODES:
        with test.subTest(rigid_compliant_alm=compliant):
            _check_elastic_clamp_moment_conserves_angular_momentum(test, device, compliant)


def test_modal_basis_shared_by_elastic_bodies(test, device):
    """Check that two bodies sharing a basis register it once."""
    basis = newton.ModalGeneratorBeam(
        length=1.0,
        half_width_y=0.05,
        half_width_z=0.03,
        mode_specs=[{"type": newton.ModalGeneratorBeam.Mode.AXIAL}],
        sample_count=3,
    ).build()

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        modal_basis=basis,
    )
    builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, 0.4, 0.0), wp.quat_identity()),
        modal_basis=basis,
    )
    builder.color()
    model = builder.finalize(device=device)

    test.assertEqual(model.elastic_body_count, 2)
    test.assertEqual(model.modal_basis_count, 1)
    np.testing.assert_allclose(model.elastic_basis.numpy(), [0, 0], atol=0)


def test_elastic_render_shape_sampling(test, device):
    """Check the render polyline points of an elastic box and their mode samples."""

    def shape_fn(x):
        xi = x[0] / 0.5
        return np.array([[0.0, 1.0 - xi * xi, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_shape_fn=shape_fn,
    )
    builder.add_shape_box(body, hx=0.5, hy=0.05, hz=0.03)
    builder.color()
    model = builder.finalize(device=device)

    test.assertEqual(model.elastic_render_point_total_count, 33)
    test.assertEqual(int(model.elastic_render_point_start.numpy()[0]), 0)
    test.assertEqual(int(model.elastic_render_point_count.numpy()[0]), 33)

    local = model.elastic_render_point_local.numpy()
    np.testing.assert_allclose(local[0, 0], -0.5, atol=1.0e-7)
    np.testing.assert_allclose(local[-1, 0], 0.5, atol=1.0e-7)
    test.assertGreater(local[0, 2], 0.03)

    phi = model.elastic_render_point_phi.numpy().reshape((-1, 3))
    np.testing.assert_allclose(phi[16], [0.0, 1.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(phi[0], [0.0, 0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(phi[-1], [0.0, 0.0, 0.0], atol=1.0e-6)


def test_elastic_shape_mesh_sampling(test, device):
    """Check the mode samples of the render-mesh vertices of an elastic box."""
    length = 1.0
    hy = 0.05
    hz = 0.03

    def shape_fn(x):
        s = float(x[0] + 0.5 * length)
        xi = s / length
        phi = math.sin(math.pi * xi)
        slope = (math.pi / length) * math.cos(math.pi * xi)
        return np.array([[-float(x[1]) * slope, phi, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), mode_count=1, mode_shape_fn=shape_fn)
    builder.add_shape_box(body, hx=0.5 * length, hy=hy, hz=hz)
    builder.color()
    model = builder.finalize(device=device)

    test.assertEqual(model.elastic_shape_count, 1)
    test.assertGreater(model.elastic_shape_vertex_total_count, 8)
    test.assertGreater(model.elastic_shape_index_total_count, 36)
    test.assertEqual(int(model.elastic_shape_shape.numpy()[0]), 0)

    local = model.elastic_shape_vertex_local.numpy()
    phi = model.elastic_shape_vertex_phi.numpy().reshape((-1, 3))
    mid = int(np.argmin(np.abs(local[:, 0]) + np.abs(local[:, 1] - hy) + np.abs(local[:, 2] - hz)))
    np.testing.assert_allclose(phi[mid], [0.0, 1.0, 0.0], atol=1.0e-6)

    left = np.where(np.isclose(local[:, 0], -0.5 * length))[0]
    right = np.where(np.isclose(local[:, 0], 0.5 * length))[0]
    np.testing.assert_allclose(phi[left, 1], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(phi[right, 1], 0.0, atol=1.0e-6)


def test_elastic_shape_box_winding(test, device):
    """Check that the triangles of an elastic box mesh wind outward."""
    hx = 0.5
    hy = 0.05
    hz = 0.03

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), mode_count=1)
    builder.add_shape_box(body, hx=hx, hy=hy, hz=hz)
    builder.color()
    model = builder.finalize(device=device)

    vertices = model.elastic_shape_vertex_local.numpy()
    indices = model.elastic_shape_indices.numpy()
    test.assertGreater(len(indices), 36)

    for tri in indices.reshape((-1, 3)):
        a, b, c = vertices[tri]
        normal = np.cross(b - a, c - a)
        center = (a + b + c) / 3.0
        axis = int(np.argmax(np.abs(center / np.array([hx, hy, hz], dtype=np.float32))))
        expected_sign = 1.0 if center[axis] > 0.0 else -1.0
        test.assertGreater(float(normal[axis] * expected_sign), 0.0)


def test_elastic_shape_box_exact_modal_samples(test, device):
    """Check that elastic box mesh vertices map exactly to the beam render samples."""
    length = 0.8
    hy = 0.04
    hz = 0.03
    surface_points, _ = box_surface_mesh(length, hy, hz)
    for axis in range(3):
        axis_values = np.unique(np.round(surface_points[:, axis], decimals=8))
        test.assertLessEqual(float(np.max(np.diff(axis_values))), 0.010001)
    test.assertGreater(np.unique(surface_points[:, 1]).size, 2)
    test.assertGreater(np.unique(surface_points[:, 2]).size, 2)

    basis = newton.ModalGeneratorBeam(
        length=length,
        half_width_y=hy,
        half_width_z=hz,
        mode_specs=[
            {
                "type": newton.ModalGeneratorBeam.Mode.BENDING_Z,
                "boundary": newton.ModalGeneratorBeam.Boundary.CANTILEVER_TIP,
            }
        ],
    ).build(
        sample_points=beam_render_sample_points(
            length,
            hy,
            hz,
            extra_points=((-0.5 * length, 0.0, 0.0), (0.5 * length, 0.0, 0.0)),
        )
    )

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(modal_basis=basis)
    builder.add_shape_box(body, hx=0.5 * length, hy=hy, hz=hz)
    builder.color()
    model = builder.finalize(device=device)

    samples = model.elastic_shape_vertex_sample.numpy()
    np.testing.assert_array_equal(samples, np.arange(surface_points.shape[0], dtype=np.int32))
    np.testing.assert_allclose(model.elastic_shape_vertex_local.numpy(), surface_points, atol=1.0e-7)

    phi = model.elastic_shape_vertex_phi.numpy().reshape((-1, model.elastic_max_mode_count, 3))[:, 0]
    expected_phi = np.asarray([basis.sample_phi[i][0] for i in range(surface_points.shape[0])], dtype=np.float32)
    np.testing.assert_allclose(phi, expected_phi, atol=1.0e-7)


def _build_elastic_ground_contact_model(device, z: float, q0: float = 0.0, is_kinematic: bool = True):
    def downward_shape_fn(_x):
        return np.array([[0.0, 0.0, -1.0]], dtype=np.float32)

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.ke = 1000.0
    cfg.kd = 0.0
    cfg.mu = 0.0
    cfg.margin = 0.0
    cfg.gap = 0.0

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis="Z")
    builder.add_ground_plane(cfg=cfg)
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()),
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_mass=[0.0 if is_kinematic else 1.0],
        mode_stiffness=[0.0 if is_kinematic else 100.0],
        mode_damping=[0.0],
        mode_q=[q0],
        mode_shape_fn=downward_shape_fn,
        is_kinematic=is_kinematic,
    )
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    builder.color()
    return builder.finalize(device=device), body


def test_elastic_surface_contact_generation(test, device):
    """Check that ground contacts on an elastic box carry the elastic shape's vertex sample."""
    model, body = _build_elastic_ground_contact_model(device, z=0.04)
    state = model.state()
    pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)

    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)

    samples0 = contacts.rigid_contact_elastic_sample0.numpy()[:count]
    samples1 = contacts.rigid_contact_elastic_sample1.numpy()[:count]
    shapes0 = contacts.rigid_contact_shape0.numpy()[:count]
    shapes1 = contacts.rigid_contact_shape1.numpy()[:count]
    shape_body = model.shape_body.numpy()
    active = np.logical_and(shapes0 >= 0, shapes1 >= 0)
    test.assertTrue(bool(np.all(active)))

    np.testing.assert_array_equal(samples0[active], -np.ones_like(samples0[active]))
    test.assertTrue(bool(np.any(samples1[active] >= 0)))
    test.assertTrue(bool(np.all(shape_body[shapes0[active]] == -1)))
    test.assertTrue(bool(np.all(shape_body[shapes1[active]] == body)))


def test_elastic_surface_contact_work_uses_pair_vertex_ranges(test, device):
    """Check that elastic contact work counts only the vertices of each pair's elastic shape."""
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]]],
        mode_mass=[1.0],
        mode_stiffness=[1.0],
    )
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for x in (-0.2, 0.2):
        body = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(x, 0.0, 0.2), wp.quat_identity()),
            mass=1.0,
            inertia=_identity_inertia(),
            modal_basis=basis,
        )
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")

    shape_to_count = dict(
        zip(model.elastic_shape_shape.numpy().tolist(), model.elastic_shape_vertex_count.numpy().tolist(), strict=True)
    )
    shape_body = model.shape_body.numpy()
    body_elastic_index = model.body_elastic_index.numpy()
    expected_work = 0
    for shape_a, shape_b in pipeline.elastic_shape_pairs.numpy():
        body_a = int(shape_body[shape_a])
        elastic_shape = int(shape_a) if body_a >= 0 and body_elastic_index[body_a] >= 0 else int(shape_b)
        expected_work += int(shape_to_count[elastic_shape])

    test.assertEqual(pipeline.elastic_contact_work_count, expected_work)
    test.assertLess(
        pipeline.elastic_contact_work_count,
        pipeline.elastic_shape_pairs_max * model.elastic_shape_vertex_total_count,
    )


def test_elastic_surface_contact_deterministic_sort_preserves_samples(test, device):
    """Check that deterministic contact sorting keeps each elastic sample matched to its vertex."""
    model, _body = _build_elastic_ground_contact_model(device, z=0.04)
    state = model.state()
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)

    count = min(int(contacts.rigid_contact_count.numpy()[0]), contacts.rigid_contact_max)
    test.assertGreater(count, 1)
    samples = contacts.rigid_contact_elastic_sample1.numpy()[:count]
    points = contacts.rigid_contact_point1.numpy()[:count]
    rest_points = model.elastic_shape_vertex_local.numpy()
    elastic_rows = samples >= 0
    test.assertTrue(bool(np.any(elastic_rows)))
    np.testing.assert_allclose(points[elastic_rows], rest_points[samples[elastic_rows]], atol=0.0)


def test_elastic_surface_contact_reduction_preserves_modal_samples(test, device):
    """Check that contact reduction keeps valid elastic vertex samples."""
    model, _body = _build_elastic_ground_contact_model(device, z=0.04)
    state = model.state()

    reduced_pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=True)
    reduced_contacts = reduced_pipeline.contacts()
    reduced_pipeline.collide(state, reduced_contacts)
    reduced_count = min(int(reduced_contacts.rigid_contact_count.numpy()[0]), reduced_contacts.rigid_contact_max)

    full_pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
    full_contacts = full_pipeline.contacts()
    full_pipeline.collide(state, full_contacts)
    full_count = min(int(full_contacts.rigid_contact_count.numpy()[0]), full_contacts.rigid_contact_max)

    test.assertGreater(reduced_count, 0)
    test.assertLess(reduced_count, full_count)
    samples = reduced_contacts.rigid_contact_elastic_sample1.numpy()[:reduced_count]
    test.assertTrue(bool(np.all(samples >= 0)))
    test.assertTrue(bool(np.all(samples < model.elastic_shape_vertex_total_count)))


def test_elastic_surface_contact_uses_deformed_vertex(test, device):
    """Check that a modal deflection brings a separated elastic box vertex into ground contact."""
    model, _body = _build_elastic_ground_contact_model(device, z=0.07)
    state = model.state()
    pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
    contacts = pipeline.contacts()

    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)

    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    q = state.joint_q.numpy()
    q[q_start + 7] = 0.03
    state.joint_q.assign(q)

    pipeline.collide(state, contacts)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_vbd_elastic_contact_solves_modal_penetration(test, device):
    """Solve a massless-mode penetration in one legacy hard-contact iteration.

    Legacy AVBD only: compliant ALM conditions a contact by the inverse-mass mobility of its row, which is
    zero for a massless mode, so it drops the row. The compliant counterpart with a massive mode is
    ``test_vbd_elastic_compliant_contact_solves_modal_penetration``.
    """
    q0 = 0.005
    model, _body = _build_elastic_ground_contact_model(device, z=0.05, q0=q0)
    state_0 = model.state()
    state_1 = model.state()
    pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
    contacts = pipeline.contacts()
    control = model.control()

    pipeline.collide(state_0, contacts)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)

    solver = _make_vbd_solver(
        model,
        compliant=False,
        iterations=1,
        rigid_contact_k_start=1000.0,
        rigid_avbd_contact_alpha=0.0,
        rigid_body_contact_buffer_size=128,
        elastic_body_relaxation=1.0,
        elastic_solver_metrics=True,
    )
    solver.step(state_0, state_1, control, contacts, 0.01)

    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    test.assertLess(abs(float(state_1.joint_q.numpy()[q_start + 7])), 1.0e-5)

    metrics = solver.elastic_mode_solve_metrics()
    for values in metrics.values():
        test.assertEqual(values.shape, (1,))
        test.assertTrue(bool(np.isfinite(values).all()))
    initial_residual = float(metrics["initial_residual_norm"][0])
    test.assertGreater(initial_residual, 0.0)
    test.assertLess(float(metrics["solve_residual_norm"][0]) / initial_residual, 1.0e-6)
    test.assertLess(float(metrics["applied_residual_norm"][0]) / initial_residual, 1.0e-6)
    test.assertGreater(float(metrics["update_norm"][0]), 1.0e-4)
    test.assertLess(float(metrics["update_norm"][0]), 1.0e-2)


def test_vbd_elastic_body_relaxation_is_general_and_validated(test, device):
    """Check that elastic_body_relaxation is stored, metrics stay empty by default, and invalid values raise."""
    model, _body = _build_elastic_ground_contact_model(device, z=0.2)
    solver = newton.solvers.SolverVBD(model, elastic_body_relaxation=0.37)

    contact_max, _contact_count, relaxation = solver._elastic_solve_inputs(None)
    test.assertEqual(contact_max, 0)
    test.assertAlmostEqual(relaxation, 0.37)
    for values in solver.elastic_mode_solve_metrics().values():
        test.assertEqual(values.shape, (0,))
    for invalid in (-0.1, 1.1, np.nan, np.inf):
        with test.assertRaisesRegex(ValueError, "elastic_body_relaxation"):
            newton.solvers.SolverVBD(model, elastic_body_relaxation=invalid)


def test_vbd_elastic_compliant_contact_solves_modal_penetration(test, device):
    """Check that compliant-ALM contact measures modal penetration and pushes the mode back."""
    initial_modal_q = 0.005
    model, _body = _build_elastic_ground_contact_model(device, z=0.05, q0=initial_modal_q)
    model.elastic_mode_mass.fill_(1.0)
    state_0 = model.state()
    state_1 = model.state()
    pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=4,
        rigid_compliant_alm=True,
        rigid_avbd_contact_alpha=0.0,
        rigid_body_contact_buffer_size=128,
        elastic_body_relaxation=1.0,
    )
    solver.step(state_0, state_1, model.control(), contacts, 0.01)

    count = min(int(contacts.rigid_contact_count.numpy()[0]), contacts.rigid_contact_max)
    samples = contacts.rigid_contact_elastic_sample1.numpy()[:count]
    elastic_rows = samples >= 0
    normals = contacts.rigid_contact_normal.numpy()[:count][elastic_rows]
    C0 = solver.body_body_contact_C0.numpy()[:count][elastic_rows]
    normal_C0 = np.einsum("ij,ij->i", C0, normals)
    test.assertTrue(bool(np.any(elastic_rows)))
    np.testing.assert_allclose(normal_C0, 0.005, atol=1.0e-6)

    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    modal_q = abs(float(state_1.joint_q.numpy()[q_start + 7]))
    test.assertGreater(modal_q, 0.0)
    test.assertLess(modal_q, 0.4 * initial_modal_q)


def test_vbd_elastic_compliant_contact_uses_modal_mobility_for_rho_and_forces(test, device):
    """A kinematic frame with dynamic modes conditions and reports ALM contact consistently."""
    model, _body = _build_elastic_ground_contact_model(device, z=0.05, q0=0.005)
    model.elastic_mode_mass.fill_(1.0)
    state_0 = model.state()
    state_1 = model.state()
    pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)
    body_q_prev = wp.clone(state_0.body_q)
    joint_q_prev = wp.clone(state_0.joint_q)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_avbd_contact_alpha=0.0,
        rigid_body_contact_buffer_size=128,
        elastic_body_relaxation=0.0,
    )
    solver.step(state_0, state_1, model.control(), contacts, 0.01)

    count = min(int(contacts.rigid_contact_count.numpy()[0]), contacts.rigid_contact_max)
    elastic_rows = contacts.rigid_contact_elastic_sample1.numpy()[:count] >= 0
    test.assertTrue(bool(np.any(elastic_rows)))
    normal_rho = solver.body_body_contact_normal_rho.numpy()[:count][elastic_rows]
    np.testing.assert_allclose(normal_rho, 1.0e4, rtol=2.0e-5)

    _body0, _body1, _point0, _point1, force, _count = solver.collect_rigid_contact_forces(
        state_1.body_q,
        body_q_prev,
        contacts,
        0.01,
        joint_q=state_1.joint_q,
        joint_q_prev=joint_q_prev,
    )
    force_norm = np.linalg.norm(force.numpy()[:count][elastic_rows], axis=1)
    multiplier_norm = np.linalg.norm(solver.body_body_contact_lambda.numpy()[:count][elastic_rows], axis=1)
    test.assertGreater(float(np.max(multiplier_norm)), 0.0)
    test.assertGreater(float(np.max(force_norm)), 0.0)


def _check_vbd_elastic_contact_overflow_assembles_consistent_block(test, device, compliant: bool):
    """Overflowing the per-body contact list must not truncate only the rigid frame block."""

    def assemble_block(contact_buffer_size: int):
        model, _body = _build_elastic_ground_contact_model(device, z=0.04, is_kinematic=False)
        state_0 = model.state()
        state_1 = model.state()
        pipeline = newton.CollisionPipeline(model, broad_phase="explicit", deterministic=True)
        contacts = pipeline.contacts()
        pipeline.collide(state_0, contacts)
        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        test.assertGreater(contact_count, 1)

        legacy_contact = {} if compliant else {"rigid_contact_k_start": 1000.0}
        solver = _make_vbd_solver(
            model,
            compliant=compliant,
            iterations=1,
            rigid_body_contact_buffer_size=contact_buffer_size,
            elastic_body_relaxation=1.0,
            **legacy_contact,
        )
        solver.step(state_0, state_1, model.control(), contacts, 0.01)

        matrix_upper = solver.elastic_body_block_matrix.numpy()[0]
        matrix = np.triu(matrix_upper) + np.triu(matrix_upper, 1).T
        return contact_count, matrix, solver.elastic_body_block_delta.numpy()[0]

    contact_count, overflow_matrix, overflow_delta = assemble_block(contact_buffer_size=1)
    _, complete_matrix, complete_delta = assemble_block(contact_buffer_size=contact_count)

    np.testing.assert_allclose(overflow_matrix, complete_matrix, rtol=2.0e-6, atol=1.0e-5)
    np.testing.assert_allclose(overflow_delta, complete_delta, rtol=2.0e-6, atol=1.0e-7)
    test.assertGreater(float(np.linalg.eigvalsh(overflow_matrix)[0]), 0.0)


def test_vbd_elastic_contact_overflow_assembles_consistent_block(test, device):
    """Assemble the same elastic block whether or not the per-body contact list overflows."""
    for compliant in _RIGID_ALM_MODES:
        with test.subTest(rigid_compliant_alm=compliant):
            _check_vbd_elastic_contact_overflow_assembles_consistent_block(test, device, compliant)


def test_elastic_contact_local_mat33_projection_matches_world(test, device):
    """Check that projecting a contact force and Hessian onto modes in the body frame matches the world frame."""
    del device

    quat = wp.quat_from_axis_angle(wp.vec3(0.3, -0.7, 0.2), 0.83)
    quat_np = np.array([quat[0], quat[1], quat[2], quat[3]], dtype=float)
    rot = np.column_stack(
        (
            _quat_rotate_np(quat_np, np.array([1.0, 0.0, 0.0])),
            _quat_rotate_np(quat_np, np.array([0.0, 1.0, 0.0])),
            _quat_rotate_np(quat_np, np.array([0.0, 0.0, 1.0])),
        )
    )
    phi_local = np.array(
        [
            [0.35, -0.12, 0.08],
            [-0.18, 0.24, 0.31],
            [0.07, 0.42, -0.16],
            [-0.29, -0.05, 0.23],
        ],
        dtype=float,
    )
    force_world = np.array([4.0, -7.0, 2.5], dtype=float)
    hessian_world = np.array(
        [
            [18.0, -2.0, 4.0],
            [-2.0, 11.0, 1.5],
            [4.0, 1.5, 7.0],
        ],
        dtype=float,
    )

    phi_world = phi_local @ rot.T
    grad_world = -(phi_world @ force_world)
    hessian_modal_world = phi_world @ hessian_world @ phi_world.T

    force_local = rot.T @ force_world
    hessian_local = rot.T @ hessian_world @ rot
    grad_local = -(phi_local @ force_local)
    hessian_modal_local = phi_local @ hessian_local @ phi_local.T

    np.testing.assert_allclose(grad_local, grad_world, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(hessian_modal_local, hessian_modal_world, rtol=1.0e-12, atol=1.0e-12)


def test_elastic_strain_visualization_colors(test, device):
    """Check that strain colors of an elastic mesh change with modal amplitude and stay in [0, 1]."""
    length = 1.0

    def axial_gradient_shape(x):
        s = float(x[0] + 0.5 * length)
        return np.array([[s / length, 0.0, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_shape_fn=axial_gradient_shape,
    )
    builder.add_shape_box(body, hx=0.5 * length, hy=0.04, hz=0.03)
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()

    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])

    viewer = _ElasticMeshColorProbe()
    viewer.set_model(model)
    viewer.show_elastic_strain = True

    q = state.joint_q.numpy()
    q[q_start + 7] = 0.04
    state.joint_q.assign(q)
    viewer.log_state(state)
    colors_small = viewer.mesh_colors["/model/elastic_shapes/shape_0"].copy()

    q[q_start + 7] = 0.08
    state.joint_q.assign(q)
    viewer.log_state(state)

    test.assertIn("/model/elastic_shapes/shape_0", viewer.mesh_colors)
    colors = viewer.mesh_colors["/model/elastic_shapes/shape_0"]
    test.assertIsNotNone(colors)
    test.assertEqual(colors.shape[1], 3)
    test.assertGreater(float(np.max(np.abs(colors - colors_small))), 0.2)
    test.assertGreater(float(np.ptp(colors[:, 0])), 0.1)
    test.assertGreater(float(np.ptp(colors[:, 2])), 0.1)
    test.assertTrue(bool(np.all(colors >= -1.0e-6)))
    test.assertTrue(bool(np.all(colors <= 1.0 + 1.0e-6)))


def test_elastic_rendering_respects_layer_transform_and_visibility(test, device):
    """Check that elastic meshes follow the viewer layer transform and visibility."""
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 0.0]]],
        mode_mass=[1.0],
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body_position = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(*body_position), wp.quat_identity()),
        mass=1.0,
        inertia=_identity_inertia(),
        modal_basis=basis,
    )
    builder.add_shape_box(body, hx=0.05, hy=0.04, hz=0.03)
    builder.color()
    model = builder.finalize(device=device)

    viewer = _ElasticMeshColorProbe()
    viewer.activate("elastic")
    viewer.set_model(model)
    layer_offset = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    viewer.set_layer_transform("elastic", tuple(layer_offset))
    viewer.log_state(model.state())

    test.assertEqual(len(viewer.mesh_points), 1)
    points = next(iter(viewer.mesh_points.values()))
    expected = model.elastic_shape_vertex_local.numpy() + body_position + layer_offset
    np.testing.assert_allclose(points, expected, atol=1.0e-6)

    viewer.set_layer_visible("elastic", False)
    viewer.log_state(model.state())
    test.assertTrue(all(viewer.mesh_hidden.values()))


def test_torsion_render_shape_sampling(test, device):
    """Check the torsion mode values sampled at a tip vertex of an elastic box mesh."""
    length = 1.0
    hy = 0.06
    hz = 0.025

    def torsion_shape_fn(x):
        s = float(x[0] + 0.5 * length)
        twist = s / length
        return np.array([[0.0, -float(x[2]) * twist, float(x[1]) * twist]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), mode_count=1, mode_shape_fn=torsion_shape_fn)
    builder.add_shape_box(body, hx=0.5 * length, hy=hy, hz=hz)
    builder.color()
    model = builder.finalize(device=device)

    np.testing.assert_allclose(torsion_shape_fn(np.array([0.0, 0.0, 0.0], dtype=np.float32))[0], 0.0, atol=1.0e-7)

    local = model.elastic_shape_vertex_local.numpy()
    phi = model.elastic_shape_vertex_phi.numpy().reshape((-1, 3))
    tip = int(np.argmin(np.abs(local[:, 0] - 0.5 * length) + np.abs(local[:, 1] - hy) + np.abs(local[:, 2] - hz)))
    np.testing.assert_allclose(phi[tip], [0.0, -hz, hy], atol=1.0e-6)


def test_finite_torsion_exemplar_preserves_volume(test, device):
    """Check that the finite torsion exemplar preserves volume while linear torsion does not."""
    length = 1.0
    hy = 0.085
    hz = 0.05
    tip_twist = math.pi / 2.0
    vertices, indices = box_surface_mesh(length, hy, hz)
    rest_volume = mesh_volume(vertices, indices)

    finite_vertices = vertices + finite_torsion_displacement(vertices, length, tip_twist)
    finite_ratio = mesh_volume(finite_vertices, indices) / rest_volume
    np.testing.assert_allclose(finite_ratio, 1.0, atol=2.0e-3)

    s = np.clip(vertices[:, 0] + 0.5 * length, 0.0, length)
    theta = tip_twist * s / length
    linear_vertices = vertices.copy()
    linear_vertices[:, 1] -= vertices[:, 2] * theta
    linear_vertices[:, 2] += vertices[:, 1] * theta
    linear_ratio = mesh_volume(linear_vertices, indices) / rest_volume
    test.assertGreater(linear_ratio, 1.5)


def test_finite_torsion_pod_modes_project_exemplar(test, device):
    """Check that POD modes of finite torsion snapshots reproduce the exemplar and its volume."""
    length = 1.0
    hy = 0.085
    hz = 0.05
    tip_twist = math.pi / 2.0
    mode_count = 8
    vertices, indices = box_surface_mesh(length, hy, hz)
    sample_points = beam_render_sample_points(
        length,
        hy,
        hz,
        extra_points=((-0.5 * length, 0.0, 0.0), (0.5 * length, 0.0, 0.0)),
    )
    snapshot_amplitudes = np.linspace(-1.0, 1.0, 17, dtype=np.float32)
    snapshot_amplitudes = snapshot_amplitudes[np.abs(snapshot_amplitudes) > 1.0e-6]
    snapshots = np.asarray(
        [
            finite_torsion_displacement(sample_points, length, tip_twist * float(amplitude))
            for amplitude in snapshot_amplitudes
        ],
        dtype=np.float32,
    )

    basis = newton.ModalGeneratorPOD(
        sample_points=sample_points,
        displacements=snapshots,
        mode_count=mode_count,
        total_mass=1.0,
        stiffness_scale=1.0,
    ).build()
    phi = basis.sample_phi.reshape((sample_points.shape[0], mode_count, 3))
    projection_matrix = np.transpose(phi, (0, 2, 1)).reshape((-1, mode_count))
    target = finite_torsion_displacement(sample_points, length, tip_twist)
    q, *_ = np.linalg.lstsq(projection_matrix, target.reshape(-1), rcond=None)
    projected = np.einsum("smc,m->sc", phi, q)

    polar_moment = (4.0 / 3.0) * hy * hz**3 + (4.0 / 3.0) * hz * hy**3
    modal_mass, modal_stiffness = beam_torsion_linear_modal_properties(
        length,
        hy,
        hz,
        density=500.0,
        shear_modulus=4.0e4,
    )
    np.testing.assert_allclose(modal_mass, 500.0 * polar_moment * length / 3.0, rtol=1.0e-6)
    np.testing.assert_allclose(modal_stiffness, 4.0e4 * polar_moment / length, rtol=1.0e-6)

    test.assertEqual(basis.mode_count, mode_count)
    test.assertLess(float(np.max(np.linalg.norm(projected - target, axis=1))), 1.0e-5)
    ratio = mesh_volume(vertices + projected[: vertices.shape[0]], indices) / mesh_volume(vertices, indices)
    np.testing.assert_allclose(ratio, 1.0, atol=2.0e-3)


def test_cantilever_tip_load_solution(test, device):
    """Check the static tip deflection of a massless cantilever mode under a modal tip load."""
    length = 0.8
    ei = 0.32
    tip_load = 0.2
    stiffness = 3.0 * ei / (length**3)

    def cantilever_shape_fn(x):
        s = float(x[0] + 0.5 * length)
        s = min(max(s, 0.0), length)
        phi = (s * s * (3.0 * length - s)) / (2.0 * length**3)
        slope = (3.0 * s * (2.0 * length - s)) / (2.0 * length**3)
        return np.array([[-float(x[2]) * slope, 0.0, phi]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_mass=[0.0],
        mode_stiffness=[stiffness],
        mode_damping=[0.0],
        mode_shape_fn=cantilever_shape_fn,
    )
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    qd_start = int(model.joint_qd_start.numpy()[owner_joint])
    jf = control.joint_f.numpy()
    jf[qd_start + 6] = tip_load
    control.joint_f.assign(jf)

    solver = newton.solvers.SolverVBD(model, iterations=0)
    solver.step(state_0, state_1, control, None, 0.01)

    expected_tip_deflection = tip_load * length**3 / (3.0 * ei)
    np.testing.assert_allclose(state_1.joint_q.numpy()[q_start + 7], expected_tip_deflection, atol=1.0e-7)
    tip_phi = cantilever_shape_fn(np.array([0.5 * length, 0.0, 0.0], dtype=np.float32))[0]
    np.testing.assert_allclose(tip_phi, [0.0, 0.0, 1.0], atol=1.0e-7)


def test_cantilever_modal_vibration_solution(test, device):
    """Check a free cantilever mode against the implicit-Euler recurrence over 180 steps."""
    length = 0.9
    basis = newton.ModalGeneratorBeam(
        length=length,
        half_width_y=0.045,
        half_width_z=0.035,
        mode_specs=[
            {
                "type": newton.ModalGeneratorBeam.Mode.BENDING_Z,
                "boundary": newton.ModalGeneratorBeam.Boundary.CANTILEVER_TIP,
            }
        ],
        density=250.0,
        young_modulus=3.2e7,
        damping_ratio=0.001,
    ).build()

    mode_mass = float(basis.mode_mass[0])
    stiffness = float(basis.mode_stiffness[0])
    damping = float(basis.mode_damping[0])
    q_expected = 0.09
    qd_expected = 0.0
    dt = 1.0 / 480.0
    min_q = q_expected

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mode_q=[q_expected],
        mode_qd=[qd_expected],
        modal_basis=basis,
    )
    builder.add_shape_box(body, hx=0.5 * length, hy=0.045, hz=0.035)
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    qd_start = int(model.joint_qd_start.numpy()[owner_joint])

    solver = newton.solvers.SolverVBD(model, iterations=0)
    for _ in range(180):
        solver.step(state_0, state_1, control, None, dt)
        denom = mode_mass + dt * damping + dt * dt * stiffness
        qd_expected = (mode_mass * qd_expected - dt * stiffness * q_expected) / denom
        q_expected = q_expected + dt * qd_expected
        min_q = min(min_q, q_expected)
        state_0, state_1 = state_1, state_0

    np.testing.assert_allclose(state_0.joint_q.numpy()[q_start + 7], q_expected, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(state_0.joint_qd.numpy()[qd_start + 6], qd_expected, rtol=1.0e-6, atol=1.0e-5)
    test.assertLess(min_q, -0.01)


def test_elastic_modal_implicit_solution_vbd(test, device):
    """Check one implicit-Euler modal step with applied force, stiffness, and damping."""
    mode_mass = 2.0
    stiffness = 18.0
    damping = 0.5
    q0 = 0.1
    v0 = -0.2
    force = 1.3
    dt = 0.01

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_mass=[mode_mass],
        mode_stiffness=[stiffness],
        mode_damping=[damping],
        mode_q=[q0],
        mode_qd=[v0],
    )
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    qd_start = int(model.joint_qd_start.numpy()[owner_joint])
    jf = control.joint_f.numpy()
    jf[qd_start + 6] = force
    control.joint_f.assign(jf)

    solver = newton.solvers.SolverVBD(model, iterations=0)
    solver.step(state_0, state_1, control, None, dt)

    v_expected = (mode_mass * v0 + dt * (force - stiffness * q0)) / (mode_mass + dt * damping + dt * dt * stiffness)
    q_expected = q0 + dt * v_expected
    np.testing.assert_allclose(state_0.joint_q.numpy()[q_start + 7], q0, rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(state_0.joint_qd.numpy()[qd_start + 6], v0, rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(state_1.joint_q.numpy()[q_start + 7], q_expected, rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(state_1.joint_qd.numpy()[qd_start + 6], v_expected, rtol=1.0e-6, atol=1.0e-7)


def test_elastic_gravity_modal_force(test, device):
    """Check the modal gravity load on the modes of a kinematic elastic frame."""
    g = 9.81
    stiffness = 18.0
    dt = 0.01
    basis = newton.ModalBasis(
        sample_points=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]],
        sample_mass=[1.0, 1.0],
        mode_stiffness=[stiffness],
        mode_damping=[0.0],
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -g))
    body = builder.add_body_elastic(modal_basis=basis, mode_q=[0.0], mode_qd=[0.0], is_kinematic=True)
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])

    solver = newton.solvers.SolverVBD(model, iterations=16)
    solver.step(state_0, state_1, control, None, dt)

    coupling_z = 2.0
    modal_mass = 2.0
    modal_force = coupling_z * (-g)
    q_expected = modal_force / (stiffness + modal_mass / dt**2)
    np.testing.assert_allclose(state_1.joint_q.numpy()[q_start + 7], q_expected, rtol=1.0e-4, atol=1.0e-9)


def test_elastic_free_fall_no_modal_force(test, device):
    """Check that free fall does not excite the modes of an elastic body.

    The mode bends the outer samples relative to the inner ones, so it carries a nonzero linear
    coupling (its gravity load) without being a rigid-body motion, and the frame uses the basis
    rigid properties so the reduced mass is positive definite and the coupled solve is exercised.
    """
    g = 9.81
    dt = 0.005
    basis = newton.ModalBasis(
        sample_points=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.5, 0.0]],
        sample_phi=[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]],
        sample_mass=[1.0, 1.0, 1.0, 1.0],
        mode_stiffness=[18.0],
        mode_damping=[0.0],
    )
    test.assertGreater(float(np.linalg.norm(basis.mode_coupling_linear[0])), 1.0)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -g))
    body = builder.add_body_elastic(modal_basis=basis, mode_q=[0.0], mode_qd=[0.0])
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])

    solver = newton.solvers.SolverVBD(model, iterations=16)
    test.assertTrue(bool(solver.elastic_implicit_mass_coupling.numpy()[0]))
    max_abs_q = 0.0
    for _ in range(40):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        max_abs_q = max(max_abs_q, abs(float(state_0.joint_q.numpy()[q_start + 7])))
    test.assertLess(max_abs_q, 1.0e-3)


def _check_elastic_euler_modal_force(test, device, compliant: bool):
    length = 0.8
    n = 17
    xs = np.linspace(0.0, length, n, dtype=np.float32)
    sample_points = np.column_stack([xs, np.zeros(n), np.zeros(n)]).astype(np.float32)
    sample_phi = np.zeros((n, 1, 3), dtype=np.float32)
    sample_phi[:, 0, 2] = (xs * xs * (3.0 * length - xs)) / (2.0 * length**3)
    basis = newton.ModalBasis(
        sample_points=sample_points,
        sample_phi=sample_phi,
        sample_mass=np.full(n, 1.0 / n, dtype=np.float32),
        mode_stiffness=[40.0],
        mode_damping=[3.0],
    )

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = builder.add_body(xform=wp.transform_identity(), mass=2.0, inertia=_identity_inertia(), is_kinematic=True)
    # The samples lie on the x axis, so the basis has no inertia about x; the frame keeps the basis
    # mass and COM and carries a nominal inertia at least as large as the samples' to keep the
    # reduced mass positive definite.
    beam = builder.add_body_elastic(
        xform=wp.transform_identity(), inertia=_identity_inertia(), modal_basis=basis, inertia_tolerance=None
    )
    builder.add_joint_fixed(
        parent=base, child=beam, parent_xform=wp.transform_identity(), child_xform=wp.transform_identity()
    )
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    base_q_start = base_qd_start = -1
    for j in range(len(joint_child)):
        if int(joint_child[j]) == base and int(joint_parent[j]) == -1:
            base_q_start = int(model.joint_q_start.numpy()[j])
            base_qd_start = int(model.joint_qd_start.numpy()[j])
    owner = int(model.elastic_joint.numpy()[0])
    q_index = int(model.joint_q_start.numpy()[owner]) + 7

    stiffness = float(model.elastic_mode_stiffness.numpy()[0])
    coupling_angular = np.array(basis.mode_coupling_angular[0], dtype=np.float64)
    coupling_centrifugal = np.array(basis.mode_coupling_centrifugal[0], dtype=np.float64)

    joint_k = 1.0e7
    solver = _vbd_fixed_joint_solver(model, compliant=compliant, iterations=24, linear_k=joint_k, angular_k=joint_k)
    test.assertTrue(bool(solver.elastic_implicit_mass_coupling.numpy()[0]))
    dt = 1.0 / 480.0
    alpha = 2.0

    max_abs_q = 0.0
    t = 0.0
    for _ in range(240):
        omega = alpha * t
        angle = 0.5 * alpha * t * t
        quat = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
        targets = {base: (wp.vec3(0.0, 0.0, 0.0), quat)}
        velocities = {base: (wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, omega, 0.0))}
        apply_kinematic_targets(state_0, {base: base_q_start}, targets, velocities, {base: base_qd_start})
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        max_abs_q = max(max_abs_q, abs(float(state_0.joint_q.numpy()[q_index])))
        t += dt

    q_final = float(state_0.joint_q.numpy()[q_index])
    omega_final = alpha * t
    euler_force = float(np.dot(coupling_angular, np.array([0.0, alpha, 0.0])))
    centrifugal_force = omega_final * omega_final * (coupling_centrifugal[0, 0] + coupling_centrifugal[2, 2])
    q_expected = (euler_force + centrifugal_force) / stiffness

    test.assertGreater(max_abs_q, 5.0e-3)
    test.assertGreater(q_final, 0.0)
    np.testing.assert_allclose(q_final, q_expected, rtol=0.05, atol=1.0e-4)


def test_elastic_euler_modal_force(test, device):
    """Drive a bending mode with the Euler and centrifugal loads of a spin-up."""
    for compliant in _RIGID_ALM_MODES:
        with test.subTest(rigid_compliant_alm=compliant):
            _check_elastic_euler_modal_force(test, device, compliant)


def _elastic_free_joint_starts(model, body):
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    for j in range(len(joint_child)):
        if int(joint_child[j]) == body and int(joint_parent[j]) == -1:
            return int(model.joint_q_start.numpy()[j]), int(model.joint_qd_start.numpy()[j])
    raise RuntimeError("no free joint")


def _check_elastic_centrifugal_modal_force(test, device, compliant: bool):
    length = 1.0
    n = 21
    xs = np.linspace(0.0, length, n, dtype=np.float32)
    sample_points = np.column_stack([xs, np.zeros(n), np.zeros(n)]).astype(np.float32)
    sample_phi = np.zeros((n, 1, 3), dtype=np.float32)
    sample_phi[:, 0, 0] = xs / length
    stiffness = 50.0
    basis = newton.ModalBasis(
        sample_points=sample_points,
        sample_phi=sample_phi,
        sample_mass=np.full(n, 1.0 / n, dtype=np.float32),
        mode_stiffness=[stiffness],
        mode_damping=[3.0],
    )
    m_xx = float(basis.mode_coupling_centrifugal[0][0, 0])
    p_xx = float(basis.mode_coupling_quadratic[0, 0][0, 0])

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = builder.add_body(xform=wp.transform_identity(), mass=2.0, inertia=_identity_inertia(), is_kinematic=True)
    # The frame takes the basis mass and COM. The samples lie on the x axis, so the basis has no inertia
    # about x; the frame carries a nominal inertia at least as large as the samples' to keep the reduced
    # mass positive definite. A frame lighter than the samples would make it indefinite and leave the
    # frame and modes coupled explicitly, which diverges unless the clamp is effectively rigid.
    beam = builder.add_body_elastic(
        xform=wp.transform_identity(),
        inertia=_identity_inertia(),
        modal_basis=basis,
        inertia_tolerance=None,
    )
    builder.add_joint_fixed(
        parent=base, child=beam, parent_xform=wp.transform_identity(), child_xform=wp.transform_identity()
    )
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    base_q_start, base_qd_start = _elastic_free_joint_starts(model, base)
    owner = int(model.elastic_joint.numpy()[0])
    q_index = int(model.joint_q_start.numpy()[owner]) + 7

    solver = _make_vbd_solver(model, compliant=compliant, iterations=24)
    test.assertTrue(bool(solver.elastic_implicit_mass_coupling.numpy()[0]))
    dt = 1.0 / 240.0
    omega = 4.0
    velocities = {base: (wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, omega))}
    t = 0.0
    for _ in range(600):
        quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), omega * t)
        targets = {base: (wp.vec3(0.0, 0.0, 0.0), quat)}
        apply_kinematic_targets(state_0, {base: base_q_start}, targets, velocities, {base: base_qd_start})
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        t += dt

    q_final = float(state_0.joint_q.numpy()[q_index])
    expected = omega * omega * m_xx / (stiffness - omega * omega * p_xx)
    test.assertGreater(q_final, 0.0)
    test.assertAlmostEqual(q_final, expected, delta=0.1 * expected)


def test_elastic_centrifugal_modal_force(test, device):
    """Stretch an axial mode of a beam spun about z to its centrifugal equilibrium.

    The exact kinetic energy gives ``k q = w^2 (B_xx + P_xx q)``: the deformed samples also feel the
    centrifugal load, which softens the mode to ``q = w^2 B_xx / (k - w^2 P_xx)``.
    """
    for compliant in _RIGID_ALM_MODES:
        with test.subTest(rigid_compliant_alm=compliant):
            _check_elastic_centrifugal_modal_force(test, device, compliant)


def test_elastic_coriolis_modal_force(test, device):
    """Check that frame spin couples one mode's velocity into another through the Coriolis term."""
    length = 1.0
    n = 21
    xs = np.linspace(0.0, length, n, dtype=np.float32)
    sample_points = np.column_stack([xs, np.zeros(n), np.zeros(n)]).astype(np.float32)
    sample_phi = np.zeros((n, 2, 3), dtype=np.float32)
    sample_phi[:, 0, 2] = xs / length
    sample_phi[:, 1, 1] = xs / length
    basis = newton.ModalBasis(
        sample_points=sample_points,
        sample_phi=sample_phi,
        sample_mass=np.full(n, 1.0 / n, dtype=np.float32),
        mode_stiffness=[30.0, 30.0],
        mode_damping=[0.0, 0.0],
    )

    def max_abs_mode0(omega):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        base = builder.add_body(xform=wp.transform_identity(), mass=2.0, inertia=_identity_inertia(), is_kinematic=True)
        # Basis mass and COM; a nominal inertia at least as large as the samples' (which have
        # none about x) keeps the reduced mass positive definite.
        beam = builder.add_body_elastic(
            xform=wp.transform_identity(),
            inertia=_identity_inertia(),
            modal_basis=basis,
            mode_qd=[0.0, 1.0],
            inertia_tolerance=None,
        )
        builder.add_joint_fixed(
            parent=base, child=beam, parent_xform=wp.transform_identity(), child_xform=wp.transform_identity()
        )
        builder.color()
        model = builder.finalize(device=device)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        base_q_start, base_qd_start = _elastic_free_joint_starts(model, base)
        owner = int(model.elastic_joint.numpy()[0])
        q0_index = int(model.joint_q_start.numpy()[owner]) + 7
        solver = newton.solvers.SolverVBD(model, iterations=24)
        test.assertTrue(bool(solver.elastic_implicit_mass_coupling.numpy()[0]))
        dt = 1.0 / 240.0
        velocities = {base: (wp.vec3(0.0, 0.0, 0.0), wp.vec3(omega, 0.0, 0.0))}
        peak = 0.0
        t = 0.0
        for _ in range(500):
            quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), omega * t)
            targets = {base: (wp.vec3(0.0, 0.0, 0.0), quat)}
            apply_kinematic_targets(state_0, {base: base_q_start}, targets, velocities, {base: base_qd_start})
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0
            peak = max(peak, abs(float(state_0.joint_q.numpy()[q0_index])))
            t += dt
        return peak

    spinning = max_abs_mode0(6.0)
    still = max_abs_mode0(0.0)
    test.assertLess(still, 1.0e-3)
    test.assertGreater(spinning, 1.0e-2)
    test.assertGreater(spinning, 10.0 * still)


def test_elastic_frame_coupling_conserves_com(test, device):
    """Check that a released modal deflection moves the frame while the deformed COM stays fixed."""
    total_mass = 1.0
    stiffness = 50.0
    deflection = 0.1
    n = 21
    length = 1.0
    xs = np.linspace(-0.5 * length, 0.5 * length, n, dtype=np.float32)
    sample_points = np.column_stack([xs, np.zeros(n), np.zeros(n)]).astype(np.float32)
    sample_phi = np.zeros((n, 1, 3), dtype=np.float32)
    sample_phi[:, 0, 2] = 1.0 - (2.0 * xs / length) ** 2
    basis = newton.ModalBasis(
        sample_points=sample_points,
        sample_phi=sample_phi,
        sample_mass=np.full(n, total_mass / n, dtype=np.float32),
        mode_stiffness=[stiffness],
        mode_damping=[0.0],
    )
    com_factor = float(basis.mode_coupling_linear[0][2]) / total_mass
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    # The samples lie on the x axis and have no inertia about it, so the frame carries a nominal
    # inertia at least as large as the samples' to keep the reduced mass positive definite.
    beam = builder.add_body_elastic(
        xform=wp.transform_identity(),
        com=wp.vec3(0.0, 0.0, 0.0),
        mass=total_mass,
        inertia=_identity_inertia(),
        mode_q=[deflection],
        modal_basis=basis,
        inertia_tolerance=None,
    )
    builder.color()
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    q_start, _ = _elastic_free_joint_starts(model, beam)
    frame_z_index = q_start + 2
    mode_index = q_start + 7

    solver = newton.solvers.SolverVBD(model, iterations=24)
    test.assertTrue(bool(solver.elastic_implicit_mass_coupling.numpy()[0]))
    dt = 1.0 / 240.0

    def frame_and_com():
        jq = state_0.joint_q.numpy()
        frame_z = float(jq[frame_z_index])
        return frame_z, frame_z + com_factor * float(jq[mode_index])

    frame_z0, com0 = frame_and_com()
    max_com_drift = 0.0
    max_frame_move = 0.0
    for _ in range(150):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        frame_z, com = frame_and_com()
        max_com_drift = max(max_com_drift, abs(com - com0))
        max_frame_move = max(max_frame_move, abs(frame_z - frame_z0))

    test.assertGreater(max_frame_move, 0.05)
    test.assertLess(max_com_drift, 1.0e-3 * max_frame_move)


def test_elastic_frame_coupling_conserves_angular_momentum(test, device):
    """Conserve the full angular momentum of a free beam whose bending mode rotates its frame.

    With ``S = 0`` the angular momentum about the rest COM is ``I(q) w - G(q) qdot``; for this planar
    motion its y component is ``(I_yy + Mf q^2) w_y - G_y qdot``. The linear mode shape is a rotation
    of the sample line, so the frame carries an inertia ``I_yy`` larger than the samples' to keep the
    reduced mass positive definite; the explicit inertia is therefore not compared with the basis.
    """
    total_mass = 1.0
    stiffness = 10.0
    deflection = 0.1
    inertia_yy = 0.3
    n = 11
    length = 1.0
    xs = np.linspace(-0.5 * length, 0.5 * length, n, dtype=np.float32)
    sample_points = np.column_stack([xs, np.zeros(n), np.zeros(n)]).astype(np.float32)
    sample_phi = np.zeros((n, 1, 3), dtype=np.float32)
    sample_phi[:, 0, 2] = xs / length
    basis = newton.ModalBasis(
        sample_points=sample_points,
        sample_phi=sample_phi,
        sample_mass=np.full(n, total_mass / n, dtype=np.float32),
        mode_stiffness=[stiffness],
        mode_damping=[0.0],
    )
    angular_y = float(basis.mode_coupling_angular[0][1])
    modal_mass = float(basis.mode_mass[0])
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    beam = builder.add_body_elastic(
        xform=wp.transform_identity(),
        com=wp.vec3(0.0, 0.0, 0.0),
        mass=total_mass,
        inertia=wp.mat33(inertia_yy, 0.0, 0.0, 0.0, inertia_yy, 0.0, 0.0, 0.0, inertia_yy),
        mode_q=[deflection],
        modal_basis=basis,
        inertia_tolerance=None,
    )
    builder.color()
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    q_start, qd_start = _elastic_free_joint_starts(model, beam)
    frame_wy_index = qd_start + 4
    mode_vel_index = qd_start + 6
    mode_index = q_start + 7

    solver = newton.solvers.SolverVBD(model, iterations=24)
    test.assertTrue(bool(solver.elastic_implicit_mass_coupling.numpy()[0]))
    dt = 1.0 / 240.0
    max_total_h = 0.0
    max_frame_spin = 0.0
    modal_h_scale = 0.0
    for _ in range(150):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        jqd = state_0.joint_qd.numpy()
        frame_wy = float(jqd[frame_wy_index])
        mode_vel = float(jqd[mode_vel_index])
        mode_q = float(state_0.joint_q.numpy()[mode_index])
        total_h = (inertia_yy + modal_mass * mode_q * mode_q) * frame_wy - angular_y * mode_vel
        max_total_h = max(max_total_h, abs(total_h))
        max_frame_spin = max(max_frame_spin, abs(frame_wy))
        modal_h_scale = max(modal_h_scale, abs(angular_y * mode_vel))

    test.assertGreater(max_frame_spin, 0.05)
    test.assertLess(max_total_h, 1.0e-3 * max(modal_h_scale, 1.0e-9))


def _rod_model(
    device,
    mode: str = "cos",
    *,
    com=(0.0, 0.0, 0.0),
    position=(0.0, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0),
    spin=(0.0, 0.0, 0.0),
    q0: float = 0.02,
    qd0: float = 0.1,
    stiffness: float = 30.0,
    gravity=(0.0, 0.0, 0.0),
    rigid_control: bool = False,
):
    """Build the one-mode reference rod (see :func:`_reduced_elastic_reference.rod_fixture`).

    The physical body does not depend on ``com``: the samples are shifted by ``com`` and the frame
    origin is placed at ``position - com``, so the rest COM starts at ``position``. With
    ``rigid_control`` a plain rigid body with the rod's mass and inertia is added at the same position.
    """
    points, mass, phi = elastic_reference.rod_fixture(mode)
    offset = np.asarray(com, dtype=np.float32)
    local_points = (points + offset).astype(np.float32)
    inertia = elastic_reference.lumped_inertia(points, mass, np.zeros(3))
    basis = newton.ModalBasis(
        sample_points=local_points,
        sample_phi=phi,
        sample_mass=mass,
        mode_stiffness=[stiffness],
        mode_damping=[0.0],
    )
    builder = newton.ModelBuilder(gravity=gravity)
    origin = np.asarray(position, dtype=np.float32) - offset
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(*origin), wp.quat_identity()),
        com=wp.vec3(*offset),
        mass=float(mass.sum()),
        inertia=wp.mat33(*inertia.ravel()),
        modal_basis=basis,
        mode_q=[q0],
        mode_qd=[qd0],
    )
    rigid = -1
    if rigid_control:
        rigid = builder.add_body(
            xform=wp.transform(wp.vec3(*np.asarray(position, dtype=np.float32)), wp.quat_identity()),
            mass=float(mass.sum()),
            inertia=wp.mat33(*inertia.ravel()),
        )
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    q_start, qd_start = _elastic_free_joint_starts(model, body)
    joint_qd = state.joint_qd.numpy()
    joint_qd[qd_start : qd_start + 3] = velocity
    joint_qd[qd_start + 3 : qd_start + 6] = spin
    if rigid_control:
        _rigid_q_start, rigid_qd_start = _elastic_free_joint_starts(model, rigid)
        joint_qd[rigid_qd_start : rigid_qd_start + 3] = velocity
        joint_qd[rigid_qd_start + 3 : rigid_qd_start + 6] = spin
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    data = {
        "points": local_points,
        "mass": mass,
        "phi": phi,
        "com": offset.astype(np.float64),
        "q_start": q_start,
        "qd_start": qd_start,
        "rigid": rigid,
    }
    return model, state, data


def _rod_joint_state(state, data):
    """Return the elastic body's ``(joint_q, joint_qd)`` as float64 arrays."""
    q_start, qd_start = data["q_start"], data["qd_start"]
    return (
        state.joint_q.numpy()[q_start : q_start + 8].astype(np.float64),
        state.joint_qd.numpy()[qd_start : qd_start + 7].astype(np.float64),
    )


def _rod_physical(state, data):
    """Return the rod's sampled COM, linear momentum, and angular momentum about the COM."""
    joint_q, joint_qd = _rod_joint_state(state, data)
    return elastic_reference.physical_state(data["points"], data["mass"], data["phi"], data["com"], joint_q, joint_qd)


def _simulate_rod(device, steps: int, dt: float, *, iterations: int = 24, record: bool = False, **kwargs):
    """Step the reference rod with :class:`~newton.solvers.SolverVBD` and summarize the trajectory.

    Returns a dict with the final joint state, the maximum ballistic COM error, the maximum drift of the
    angular momentum about the COM, and (with ``record``) the COM and quaternion series.
    """
    model, state_0, data = _rod_model(device, **kwargs)
    state_1 = model.state()
    control = model.control()
    solver = newton.solvers.SolverVBD(model, iterations=iterations)
    com_0, momentum_0, angular_0 = _rod_physical(state_0, data)
    total_mass = float(np.sum(data["mass"]))
    gravity = np.asarray(kwargs.get("gravity", (0.0, 0.0, 0.0)), dtype=np.float64)
    ballistic = 0.0
    angular_drift = 0.0
    coms = [com_0]
    quats = [_rod_joint_state(state_0, data)[0][3:7]]
    for step in range(steps):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        t = (step + 1) * dt
        com, _momentum, angular = _rod_physical(state_0, data)
        # Backward Euler advances a uniform acceleration as g t (t + dt) / 2.
        expected = com_0 + t * momentum_0 / total_mass + 0.5 * t * (t + dt) * gravity
        ballistic = max(ballistic, float(np.linalg.norm(com - expected)))
        angular_drift = max(angular_drift, float(np.linalg.norm(angular - angular_0)))
        if record:
            coms.append(com)
            quats.append(_rod_joint_state(state_0, data)[0][3:7])
    joint_q, joint_qd = _rod_joint_state(state_0, data)
    result = {
        "joint_q": joint_q,
        "joint_qd": joint_qd,
        "ballistic": ballistic,
        "angular_drift": angular_drift,
        "momentum_0": momentum_0,
        "state": state_0,
        "data": data,
    }
    if record:
        result["com"] = np.asarray(coms)
        result["quat"] = np.asarray(quats)
    return result


def _launch_frame_coupling(
    device,
    basis,
    com,
    dt,
    rot_prev,
    omega_prev,
    omega,
    q_prev,
    qd_prev,
    q,
    *,
    velocity=(0.0, 0.0, 0.0),
    accel_rel=(0.0, 0.0, 0.0),
):
    """Launch ``accumulate_elastic_frame_coupling`` for one body and return ``(rot, force, torque)``.

    The mode-pair sums come from ``compute_elastic_mode_sums``, as in the solver. The iterate
    rotation is ``exp(dt omega) rot_prev``, the frame origin moves by ``dt velocity``, and the
    inertial target is displaced so that ``(x_c - x_c*) / dt^2`` equals ``accel_rel``.
    """
    mode_count = basis.mode_count

    def array(values, dtype):
        return wp.array(values, dtype=dtype, device=device)

    zero_index = array([0], wp.int32)
    rot = elastic_reference.quat_multiply(elastic_reference.quat_from_rotvec(dt * np.asarray(omega)), rot_prev)
    position = dt * np.asarray(velocity, dtype=np.float64)
    target = position - dt * dt * np.asarray(accel_rel, dtype=np.float64)
    forces = wp.zeros(1, dtype=wp.vec3, device=device)
    torques = wp.zeros(1, dtype=wp.vec3, device=device)
    coupling_linear = array(basis.mode_coupling_linear, wp.vec3)
    coupling_centrifugal = array(basis.mode_coupling_centrifugal, wp.mat33)
    com_array = array([wp.vec3(*com)], wp.vec3)
    joint_q = array(np.r_[position, rot, q], float)
    joint_q_prev = array(np.r_[np.zeros(3), rot_prev, q_prev], float)
    joint_qd_prev = array(np.r_[velocity, omega_prev, qd_prev], float)
    sums = []
    for q_sum in (joint_q, joint_q_prev):
        outputs = [array(np.zeros((mode_count, 3)), wp.vec3), array(np.zeros((mode_count, 3)), wp.vec3)]
        outputs += [array(np.zeros((mode_count, 3, 3)), wp.mat33), array(np.zeros(mode_count), float)]
        wp.launch(
            compute_elastic_mode_sums,
            dim=(1, mode_count),
            inputs=[
                dt,
                zero_index,
                zero_index,
                zero_index,
                zero_index,
                array([mode_count], wp.int32),
                mode_count,
                com_array,
                coupling_linear,
                array(basis.mode_coupling_angular, wp.vec3),
                array(basis.mode_coupling_coriolis.reshape((-1, 3)), wp.vec3),
                array(basis.mode_coupling_quadratic.reshape((-1, 3, 3)), wp.mat33),
                array(basis.mode_mass_matrix.ravel(), float),
                zero_index,
                zero_index,
                q_sum,
                joint_q_prev,
                joint_qd_prev,
            ],
            outputs=outputs,
            device=device,
        )
        sums.append(outputs)
    (torque_q, _coriolis_q, quadratic_q, _mass_offdiagonal), (torque_q_prev, _, quadratic_q_prev, _) = sums

    wp.launch(
        accumulate_elastic_frame_coupling,
        dim=1,
        inputs=[
            dt,
            zero_index,
            zero_index,
            zero_index,
            array([mode_count], wp.int32),
            coupling_linear,
            coupling_centrifugal,
            torque_q,
            quadratic_q,
            torque_q_prev,
            quadratic_q_prev,
            array([wp.transform(wp.vec3(*position), wp.quat(*rot))], wp.transform),
            array([wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(*rot_prev))], wp.transform),
            array([wp.transform(wp.vec3(*target), wp.quat(*rot))], wp.transform),
            com_array,
            array([1.0 / float(basis.rigid_mass)], float),
            array([wp.spatial_vector()], wp.spatial_vector),
            array([wp.vec3(0.0, 0.0, 0.0)], wp.vec3),
            zero_index,
            0,
            zero_index,
            zero_index,
            joint_q,
            joint_q_prev,
            joint_qd_prev,
        ],
        outputs=[forces, torques],
        device=device,
    )
    return rot, forces.numpy()[0].astype(np.float64), torques.numpy()[0].astype(np.float64)


def test_elastic_frame_coupling_wrench_matches_reference(test, device):
    """Check the frame coupling wrench against brute-force sample momenta.

    The kernel must return ``F = -dP_e/dt`` and ``tau = -dh_e/dt - s x (a_c - g)`` with the elastic
    excess momenta summed directly over the samples, approach the continuous transport
    ``-R [S qdd + 2 W S qd + (W^2 + A) S q]`` for a small step, and ignore a uniform frame velocity
    (the frozen-inertia kernel returned a torque change of (0, 0, 0.6) N m for ``v = (0, 3, 0)``,
    ``S = (1, 0, 0)``, ``qdot = 0.2``).
    """
    rng = np.random.default_rng(42)
    sample_count, mode_count = 24, 2
    points = rng.uniform(-0.5, 0.5, (sample_count, 3)).astype(np.float32)
    mass = rng.uniform(0.02, 0.06, sample_count).astype(np.float32)
    phi = (0.5 * rng.normal(size=(sample_count, mode_count, 3))).astype(np.float32)
    basis = newton.ModalBasis(sample_points=points, sample_phi=phi, sample_mass=mass)
    com = np.asarray(basis.rigid_com, dtype=np.float64)
    inertia_0 = elastic_reference.lumped_inertia(points, mass, com)

    def excess_momenta(rot, omega, q, qd):
        R = elastic_reference.quat_to_matrix(rot)
        y = (points - com + np.einsum("smc,m->sc", phi, q)) @ R.T
        u = np.cross(omega, y) + np.einsum("smc,m->sc", phi, qd) @ R.T
        m = mass.astype(np.float64)[:, None]
        linear = (m * u).sum(0)
        angular = (m * np.cross(y, u)).sum(0) - R @ inertia_0 @ R.T @ omega
        return linear, angular, (m * y).sum(0)

    dt = 0.01
    rot_prev = elastic_reference.quat_from_rotvec([0.3, -0.5, 0.2])
    omega_prev = np.array([1.2, -0.7, 2.1])
    omega = np.array([1.5, -0.4, 1.8])
    q_prev = np.array([0.04, -0.03])
    qd_prev = np.array([0.6, -0.3])
    q = np.array([0.047, -0.031])
    accel_rel = np.array([0.8, -1.3, 0.4])
    rot, force, torque = _launch_frame_coupling(
        device, basis, com, dt, rot_prev, omega_prev, omega, q_prev, qd_prev, q, accel_rel=accel_rel
    )
    linear_1, angular_1, s_1 = excess_momenta(rot, omega, q, (q - q_prev) / dt)
    linear_0, angular_0, _s_0 = excess_momenta(rot_prev, omega_prev, q_prev, qd_prev)
    expected_force = -(linear_1 - linear_0) / dt
    expected_torque = -(angular_1 - angular_0) / dt - np.cross(s_1, accel_rel)
    scale = max(np.linalg.norm(expected_force), np.linalg.norm(expected_torque))
    np.testing.assert_allclose(force, expected_force, rtol=0.0, atol=1.0e-5 * scale)
    np.testing.assert_allclose(torque, expected_torque, rtol=0.0, atol=1.0e-5 * scale)

    # Small step at constant spin and modal acceleration: continuous translational transport. The
    # step and the modal data are powers of two so that float32 represents q exactly; the remaining
    # deviation is O(dt |w|) plus the float32 resolution of the quaternion step.
    dt = 2.0**-10
    omega = np.array([0.0, 0.0, 5.0])
    qdd = np.array([2.0, -1.5])
    q_prev = np.array([0.03125, -0.015625])
    qd_prev = np.array([0.375, 0.25])
    q = q_prev + dt * qd_prev + dt * dt * qdd
    rot, force, _torque = _launch_frame_coupling(device, basis, com, dt, rot_prev, omega, omega, q_prev, qd_prev, q)
    R = elastic_reference.quat_to_matrix(rot)
    omega_skew = elastic_reference.skew(R.T @ omega)
    s_bar = basis.mode_coupling_linear.astype(np.float64).T
    qd = qd_prev + dt * qdd
    continuous = -R @ (s_bar @ qdd + 2.0 * omega_skew @ s_bar @ qd + omega_skew @ omega_skew @ s_bar @ q)
    np.testing.assert_allclose(force, continuous, rtol=0.0, atol=1.0e-2 * np.linalg.norm(continuous))

    # A uniform frame velocity changes neither the force nor the torque.
    single = newton.ModalBasis(
        sample_points=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        sample_phi=[[[0.5, 0.0, 0.0]], [[0.5, 0.0, 0.0]]],
        sample_mass=[1.0, 1.0],
    )
    np.testing.assert_allclose(single.mode_coupling_linear[0], [1.0, 0.0, 0.0], atol=1.0e-7)
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    results = [
        _launch_frame_coupling(
            device,
            single,
            np.zeros(3),
            0.01,
            identity,
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
            np.zeros(1),
            np.array([0.2]),
            np.array([0.0025]),
            velocity=velocity,
        )
        for velocity in ((0.0, 0.0, 0.0), (0.0, 3.0, 0.0))
    ]
    test.assertGreater(np.linalg.norm(results[0][1]), 0.1)
    test.assertGreater(np.linalg.norm(results[0][2]), 1.0e-4)
    np.testing.assert_array_equal(results[1][1], results[0][1])
    np.testing.assert_array_equal(results[1][2], results[0][2])


def test_elastic_reduced_mass_matches_brute_force(test, device):
    """Check the assembled coupled block against ``M(q) = sum_s m_s J_s^T J_s``.

    With ``J_s = [1, -skew(y_s), Phi_s]`` and ``y_s = (r_s - c) + Phi_s q``, this pins the sign and
    transpose of ``I(q)``, the translation-rotation block ``-skew(S q)``, the deformed rotation-mode
    columns ``-(G_i + c x S_i - sum_j C_ij q_j)``, and the full modal mass matrix.
    """
    rng = np.random.default_rng(7)
    sample_count, mode_count = 30, 3
    points = rng.uniform(-0.5, 0.5, (sample_count, 3)).astype(np.float32)
    mass = rng.uniform(0.02, 0.06, sample_count).astype(np.float32)
    phi = (0.5 * rng.normal(size=(sample_count, mode_count, 3))).astype(np.float32)
    basis = newton.ModalBasis(
        sample_points=points,
        sample_phi=phi,
        sample_mass=mass,
        mode_stiffness=[0.0] * mode_count,
        mode_damping=[0.0] * mode_count,
    )
    test.assertGreater(basis.mode_mass_coupling_ratio, 0.05)
    com = np.asarray(basis.rigid_com, dtype=np.float64)
    inertia = elastic_reference.lumped_inertia(points, mass, com)
    q = rng.uniform(-0.1, 0.1, mode_count)
    rot = elastic_reference.quat_from_rotvec(rng.normal(size=3))

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.3, -0.2, 0.1), wp.quat(*rot)),
        com=wp.vec3(*com),
        mass=float(mass.sum()),
        inertia=wp.mat33(*inertia.ravel()),
        modal_basis=basis,
        mode_q=q.tolist(),
    )
    builder.color()
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    solver = newton.solvers.SolverVBD(model, iterations=1)
    test.assertTrue(bool(solver.elastic_implicit_mass_coupling.numpy()[0]))

    # At rest without loads the single assembly happens at the initial pose and modes.
    dt = 0.01
    solver.step(state_0, state_1, model.control(), None, dt)
    assembled = solver.elastic_body_block_matrix.numpy()[0].astype(np.float64) * dt * dt
    expected = elastic_reference.reduced_mass_brute_force(points, mass, phi, com, q)
    scale = float(np.max(np.abs(np.diag(expected))))
    # The solve reads the upper triangle.
    np.testing.assert_allclose(np.triu(assembled), np.triu(expected), rtol=0.0, atol=1.0e-5 * scale)


def _serial_elastic_block_reference(model, solver, state_in, state_out, control, dt):
    """Assemble the coupled block of elastic body 0 serially in float64 from the documented formulas.

    Reads the iterate (``state_in.body_q``, ``state_out.joint_q``), the step-start state
    (``solver.body_q_prev``, ``state_in.joint_q``, ``state_in.joint_qd``), the inertial target or the
    tracked displacement from it, the accumulated body wrench, and the per-attachment joint terms of
    the solver, and returns the
    upper triangle of the block and its gradient before contacts and symmetrization, and the
    rotation angle of every elastic attachment.
    """
    ref = elastic_reference
    max_modes = int(model.elastic_max_mode_count)
    width = max_modes + 6
    body = int(model.elastic_body.numpy()[0])
    mode_count = int(model.elastic_mode_count.numpy()[0])
    owner = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner]) + 7
    body_qd_start = int(model.joint_qd_start.numpy()[owner])
    qd_start = body_qd_start + 6

    def f64(array):
        return array.numpy().astype(np.float64)

    S = f64(model.elastic_mode_coupling_linear)[:mode_count]
    G = f64(model.elastic_mode_coupling_angular)[:mode_count]
    B = f64(model.elastic_mode_coupling_centrifugal)[:mode_count]
    C = f64(model.elastic_mode_coupling_coriolis).reshape(-1, max_modes, max_modes, 3)[0, :mode_count, :mode_count]
    P = f64(model.elastic_mode_coupling_quadratic).reshape(-1, max_modes, max_modes, 3, 3)[0, :mode_count, :mode_count]
    Mf = f64(model.elastic_mode_mass_matrix).reshape(-1, max_modes, max_modes)[0, :mode_count, :mode_count]
    mode_mass = f64(model.elastic_mode_mass)[:mode_count]
    stiffness = f64(model.elastic_mode_stiffness)[:mode_count]
    damping = f64(model.elastic_mode_damping)[:mode_count]
    com = f64(model.body_com)[body]
    mass = float(model.body_mass.numpy()[body])
    inertia = f64(model.body_inertia)[body]

    pose = f64(state_in.body_q)[body]
    pose_prev = f64(solver.body_q_prev)[body]
    target = f64(solver.body_inertia_q)[body]
    R = ref.quat_to_matrix(pose[3:7])
    R_star = ref.quat_to_matrix(target[3:7])
    q = f64(state_out.joint_q)[q_start : q_start + mode_count]
    q_prev = f64(state_in.joint_q)[q_start : q_start + mode_count]
    qd_prev = f64(state_in.joint_qd)[qd_start : qd_start + mode_count]
    omega_prev = f64(state_in.joint_qd)[body_qd_start + 3 : body_qd_start + 6]
    joint_f = f64(control.joint_f)[qd_start : qd_start + mode_count]
    inv_dt = 1.0 / dt
    inv_dt_sq = inv_dt * inv_dt

    matrix = np.zeros((width, width))
    grad = np.zeros(width)
    com_current = pose[:3] + R @ com
    com_star = target[:3] + R_star @ com
    displacement = com_current - com_star
    rot_inv = np.r_[-pose[3:6], pose[6]]
    rotation = -ref.rotvec_from_quat(ref.quat_multiply(rot_inv, target[3:7]))
    if solver._use_body_displacement and solver._body_uses_displacement.numpy()[body]:
        # The frame update tracks the displacement from the inertial target (d, phi in world axes).
        displacement = f64(solver._body_disp)[body]
        rotation = R.T @ f64(solver._body_disp_ang)[body]
    force = -displacement * mass * inv_dt_sq + f64(solver.body_forces)[body]
    torque = -R @ (inertia @ rotation) * inv_dt_sq + f64(solver.body_torques)[body]
    h_ll = f64(solver.body_hessian_ll)[body] + mass * inv_dt_sq * np.eye(3)
    h_al = f64(solver.body_hessian_al)[body]
    h_aa = f64(solver.body_hessian_aa)[body] + R @ inertia @ R.T * inv_dt_sq
    matrix[:3, :3] = R.T @ h_ll @ R
    matrix[3:6, :3] = R.T @ h_al @ R
    matrix[:3, 3:6] = (R.T @ h_al @ R).T
    matrix[3:6, 3:6] = R.T @ h_aa @ R
    grad[:3] = -R.T @ force
    grad[3:6] = -R.T @ torque

    # Deformation-dependent inertia about the rest COM.
    Bc = B - np.einsum("ia,b->iab", S, com)
    D = np.einsum("i,j,ijab->ab", q, q, P)
    inertia_delta = sum(q[i] * (2.0 * np.trace(Bc[i]) * np.eye(3) - Bc[i] - Bc[i].T) for i in range(mode_count))
    inertia_delta = inertia_delta + np.trace(D) * np.eye(3) - 0.5 * (D + D.T)
    matrix[3:6, 3:6] += inertia_delta * inv_dt_sq
    linear_angular = -ref.skew(S.T @ q) * inv_dt_sq
    matrix[:3, 3:6] += linear_angular
    matrix[3:6, :3] += linear_angular.T
    G_q = G + np.cross(com, S) - np.einsum("ija,j->ia", C, q)
    matrix[:3, 6 : 6 + mode_count] += S.T * inv_dt_sq
    matrix[3:6, 6 : 6 + mode_count] -= G_q.T * inv_dt_sq

    # Modal rows.
    omega = ref.rotvec_from_quat(ref.quat_multiply(pose[3:7], np.r_[-pose_prev[3:6], pose_prev[6]])) * inv_dt
    alpha_local = R.T @ (omega - omega_prev) * inv_dt
    omega_local = R.T @ omega
    omega_skew_sq = ref.skew(omega_local) @ ref.skew(omega_local)
    accel_rel = displacement * inv_dt_sq + f64(solver._body_f_integrated)[body][:3] / mass
    accel_local = R.T @ accel_rel
    q_dot = (q - q_prev) * inv_dt
    for i in range(mode_count):
        modal_force = (
            joint_f[i]
            - S[i] @ accel_local
            + G_q[i] @ alpha_local
            - np.sum(omega_skew_sq * (Bc[i] + np.einsum("jab,j->ab", P[i], q)))
            - 2.0 * omega_local @ (C[i].T @ q_dot)
        )
        offdiagonal = sum(Mf[i, j] * (q[j] - q_prev[j] - dt * qd_prev[j]) for j in range(mode_count) if j != i)
        grad[6 + i] = (
            stiffness[i] * q[i]
            - modal_force
            + mode_mass[i] * inv_dt_sq * (q[i] - q_prev[i] - dt * qd_prev[i])
            + damping[i] * inv_dt * (q[i] - q_prev[i])
            + offdiagonal * inv_dt_sq
        )
        matrix[6 + i, 6 + i] = stiffness[i] + mode_mass[i] * inv_dt_sq + damping[i] * inv_dt
        for j in range(i + 1, mode_count):
            matrix[6 + i, 6 + j] = Mf[i, j] * inv_dt_sq
    for row in range(6 + mode_count, width):
        matrix[row, row] = 1.0

    # Joints at deformed elastic attachments, from the solver's per-attachment terms.
    terms = solver._elastic_endpoint_joint_terms
    adjacency = solver.rigid_adjacency
    offsets = adjacency.body_adj_joints_offsets.numpy()
    adjacent = adjacency.body_adj_joints.numpy()
    parent = model.joint_parent.numpy()
    parent_endpoint = model.joint_parent_elastic_endpoint.numpy()
    child_endpoint = model.joint_child_elastic_endpoint.numpy()
    endpoint_phi = f64(model.elastic_endpoint_phi)
    endpoint_psi = f64(model.elastic_endpoint_psi)
    endpoint_angles = []
    for entry in range(offsets[body], offsets[body + 1]):
        joint = adjacent[entry]
        endpoint = parent_endpoint[joint] if parent[joint] == body else child_endpoint[joint]
        if endpoint < 0:
            continue
        F = f64(terms.force)[entry]
        T = f64(terms.torque)[entry]
        T_end = f64(terms.endpoint_torque)[entry]
        H_ll = f64(terms.hessian_ll)[entry]
        H_al = f64(terms.hessian_al)[entry]
        H_aa = f64(terms.hessian_aa)[entry]
        H_end = f64(terms.endpoint_hessian_aa)[entry]
        theta = endpoint_psi[endpoint * max_modes : endpoint * max_modes + mode_count].T @ q
        angle = np.linalg.norm(theta)
        K = ref.skew(theta)
        jacobian = np.eye(3) + (1.0 - np.cos(angle)) / angle**2 * K + (angle - np.sin(angle)) / angle**3 * K @ K
        endpoint_angles.append(angle)
        np.testing.assert_allclose(f64(terms.rotation_jacobian)[entry], jacobian, rtol=0.0, atol=1.0e-6)
        grad[:3] -= F
        grad[3:6] -= T
        matrix[:3, :3] += H_ll
        matrix[3:6, :3] += H_al
        matrix[:3, 3:6] += H_al.T
        matrix[3:6, 3:6] += H_aa
        phi = endpoint_phi[endpoint * max_modes : endpoint * max_modes + mode_count]
        psi = endpoint_psi[endpoint * max_modes : endpoint * max_modes + mode_count] @ jacobian.T
        grad[6 : 6 + mode_count] -= phi @ F + psi @ T_end
        matrix[:3, 6 : 6 + mode_count] += H_ll @ phi.T
        matrix[3:6, 6 : 6 + mode_count] += H_al @ phi.T + H_end @ psi.T
        matrix[6 : 6 + mode_count, 6 : 6 + mode_count] += np.triu(phi @ H_ll @ phi.T + psi @ H_end @ psi.T)

    diagonal = np.diag(matrix).copy()
    matrix[np.diag_indices(width)] = diagonal + 1.0e-9 * (np.abs(diagonal) + 1.0)
    return np.triu(matrix), grad, endpoint_angles


def test_elastic_parallel_assembly_matches_serial_reference(test, device):
    """Match the column-parallel block assembly to a serial float64 assembly of the same state.

    A free five-mode body with non-orthogonal modes, an offset rest COM, spin, modal forcing, a fixed
    joint at one deformed end and a revolute joint to a rigid body at the other is assembled at a
    random iterate. Every frame, coupling and modal entry of the upper triangle and the gradient must
    match ``M(q)``, ``I(q)``, ``G(q)``, the modal force and the projected joint terms computed serially.
    """
    rng = np.random.default_rng(31)
    sample_count, mode_count = 24, 5
    points = rng.uniform((-0.5, -0.1, -0.1), (0.5, 0.1, 0.1), (sample_count, 3)).astype(np.float32)
    points[0], points[1] = (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)
    mass = rng.uniform(0.02, 0.06, sample_count).astype(np.float32)
    phi = (0.3 * rng.normal(size=(sample_count, mode_count, 3))).astype(np.float32)
    psi = (0.8 * rng.normal(size=(sample_count, mode_count, 3))).astype(np.float32)
    basis = newton.ModalBasis(
        sample_points=points,
        sample_phi=phi,
        sample_psi=psi,
        sample_mass=mass,
        mode_stiffness=list(rng.uniform(5.0, 50.0, mode_count)),
        mode_damping=list(rng.uniform(0.0, 0.2, mode_count)),
    )
    test.assertGreater(basis.mode_mass_coupling_ratio, 0.05)
    com = np.asarray(basis.rigid_com, dtype=np.float64)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.2, -0.1, 0.3), wp.quat_from_axis_angle(wp.vec3(0.3, 0.8, -0.5), 0.7)),
        com=wp.vec3(*com),
        mass=float(mass.sum()),
        inertia=wp.mat33(*elastic_reference.lumped_inertia(points, mass, com).ravel()),
        modal_basis=basis,
        mode_q=list(rng.uniform(-0.05, 0.05, mode_count)),
        mode_qd=list(rng.uniform(-0.5, 0.5, mode_count)),
    )
    builder.add_joint_fixed(
        parent=-1,
        child=body,
        parent_xform=wp.transform(wp.vec3(-0.25, -0.1, 0.3), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
    )
    tip = builder.add_body(xform=wp.transform(wp.vec3(0.7, -0.1, 0.3), wp.quat_identity()), mass=0.3)
    builder.add_shape_sphere(tip, radius=0.05)
    builder.add_joint_revolute(
        parent=body,
        child=tip,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)
    control = model.control()
    owner = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner]) + 7
    qd_start = int(model.joint_qd_start.numpy()[owner])
    joint_f = control.joint_f.numpy()
    joint_f[qd_start + 6 : qd_start + 6 + mode_count] = rng.uniform(-1.0, 1.0, mode_count)
    control.joint_f.assign(joint_f)

    solver = newton.solvers.SolverVBD(model, iterations=2)
    state_in, state_out = model.state(), model.state()
    dt = 1.0 / 120.0
    solver.step(state_in, state_out, control, None, dt)
    state_in, state_out = state_out, state_in
    # A random iterate away from the step start: rotated and displaced frame, perturbed modes.
    joint_qd = state_in.joint_qd.numpy()
    joint_qd[qd_start + 3 : qd_start + 6] = (1.5, -2.0, 2.5)
    state_in.joint_qd.assign(joint_qd)
    body_q = state_in.body_q.numpy()
    pose = body_q[body].astype(np.float64)
    turn = elastic_reference.quat_from_rotvec(dt * np.array([1.4, -2.2, 2.7]))
    body_q[body, 3:7] = elastic_reference.quat_multiply(turn, pose[3:7])
    body_q[body, :3] += (0.002, -0.001, 0.003)
    state_in.body_q.assign(body_q)
    solver._init_body_displacement(state_in)
    joint_q = state_in.joint_q.numpy()
    joint_q[q_start : q_start + mode_count] += rng.uniform(-0.01, 0.01, mode_count)
    state_out.joint_q.assign(joint_q)

    solver.body_forces.zero_()
    solver.body_torques.zero_()
    solver.body_hessian_ll.zero_()
    solver.body_hessian_al.zero_()
    solver.body_hessian_aa.zero_()
    solver._accumulate_elastic_frame_coupling(state_in, state_out, dt)
    color = next(group for group in model.body_color_groups if body in group.numpy())
    solver._launch_assemble_elastic_bodies(color, state_in, state_out, control, None, dt)

    width = solver.elastic_body_block_width
    assembled = np.triu(solver.elastic_body_block_matrix.numpy()[0, :width, :width].astype(np.float64))
    assembled_grad = solver.elastic_body_block_grad.numpy()[0, :width].astype(np.float64)
    expected, expected_grad, endpoint_angles = _serial_elastic_block_reference(
        model, solver, state_in, state_out, control, dt
    )
    test.assertEqual(len(endpoint_angles), 2)
    test.assertGreater(min(endpoint_angles), 1.0e-2)
    test.assertGreater(np.abs(expected[:6, 6:]).max(), 1.0)
    # Float32 assembly: the largest deviations are about 1.4e-7 (block) and 9e-8 (gradient) of the scale.
    scale = float(np.abs(np.diag(expected)).max())
    np.testing.assert_allclose(assembled, expected, rtol=0.0, atol=5.0e-7 * scale)
    grad_scale = float(np.abs(expected_grad).max())
    np.testing.assert_allclose(assembled_grad, expected_grad, rtol=0.0, atol=1.0e-6 * grad_scale)


def test_elastic_spinning_com_is_ballistic(test, device):
    """Keep the physical center of mass of a spinning, deforming free body on its ballistic path.

    The initial physical momentum ``P0 = w x (R S q0)`` is nonzero although the stored rest-COM
    velocity is zero, so the sampled COM must travel ``t P0 / m`` (about 0.33 m in 1 s); conserving
    ``m v_c + R S qdot`` instead leaves it near its start. The samples lie on the y axis and have no
    inertia about it, so the frame carries a nominal isotropic inertia, which the COM path does not
    depend on.
    """
    total_mass = 1.0
    stiffness = 60.0
    deflection = 0.1
    spin = 5.0
    n = 21
    length = 1.0
    ys = np.linspace(-0.5 * length, 0.5 * length, n, dtype=np.float32)
    sample_points = np.column_stack([np.zeros(n), ys, np.zeros(n)]).astype(np.float32)
    sample_phi = np.zeros((n, 1, 3), dtype=np.float32)
    sample_phi[:, 0, 0] = 1.0 - (2.0 * ys / length) ** 2
    basis = newton.ModalBasis(
        sample_points=sample_points,
        sample_phi=sample_phi,
        sample_mass=np.full(n, total_mass / n, dtype=np.float32),
        mode_stiffness=[stiffness],
        mode_damping=[0.0],
    )
    s_bar = np.array(basis.mode_coupling_linear[0], dtype=np.float64)

    def run(dt: float, duration: float):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        beam = builder.add_body_elastic(
            xform=wp.transform_identity(),
            com=wp.vec3(0.0, 0.0, 0.0),
            mass=total_mass,
            inertia=wp.mat33(0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2),
            mode_q=[deflection],
            modal_basis=basis,
            inertia_tolerance=None,
        )
        builder.color()
        model = builder.finalize(device=device)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        q_start, qd_start = _elastic_free_joint_starts(model, beam)
        mode_index = q_start + 7
        qd0 = state_0.joint_qd.numpy()
        qd0[qd_start + 5] = spin
        state_0.joint_qd.assign(qd0)
        solver = newton.solvers.SolverVBD(model, iterations=24)
        test.assertTrue(bool(solver.elastic_implicit_mass_coupling.numpy()[0]))

        def true_com():
            bq = state_0.body_q.numpy()[beam].astype(np.float64)
            q_mode = float(state_0.joint_q.numpy()[mode_index])
            return bq[:3] + quat_rotate(bq[3:7], s_bar * q_mode) / total_mass

        com_0 = true_com()
        momentum_0 = np.cross([0.0, 0.0, spin], s_bar * deflection)
        max_error = 0.0
        max_mode_excursion = 0.0
        max_spin_angle = 0.0
        steps = round(duration / dt)
        for step in range(steps):
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0
            t = (step + 1) * dt
            max_error = max(max_error, float(np.linalg.norm(true_com() - com_0 - t * momentum_0 / total_mass)))
            max_mode_excursion = max(max_mode_excursion, abs(float(state_0.joint_q.numpy()[mode_index])))
            bq = state_0.body_q.numpy()[beam]
            max_spin_angle = max(max_spin_angle, 2.0 * math.atan2(abs(float(bq[5])), abs(float(bq[6]))))
        travel = float(np.linalg.norm(true_com() - com_0))
        return max_error, max_mode_excursion, max_spin_angle, travel, float(np.linalg.norm(momentum_0))

    error_coarse, mode_excursion, spin_angle, travel, momentum = run(1.0 / 240.0, 1.0)
    test.assertGreater(momentum, 0.3)
    test.assertGreater(travel, 0.3)
    test.assertGreater(spin_angle, 1.0)
    test.assertGreater(mode_excursion, 0.5 * deflection)
    test.assertLess(error_coarse, 5.0e-3)
    error_fine = run(1.0 / 480.0, 1.0)[0]
    test.assertLessEqual(error_fine, error_coarse / 1.5)


def test_elastic_free_body_boost_invariance(test, device):
    """Make a free elastic body's rotation and deformation independent of a uniform boost and of gravity.

    A uniform velocity or a uniform gravity field must only translate the body: rotation, modal
    coordinate, and the angular momentum about the COM stay as in the unboosted, weightless run. The
    frozen-inertia formulation rotated this ``cos`` rod by about 2 rad in 0.2 s under a 1 m/s z boost.
    """
    dt = 1.0 / 960.0
    steps = 192
    reference = _simulate_rod(device, steps, dt, mode="cos")
    floor = max(reference["angular_drift"], 1.0e-9)
    cases = [{"velocity": boost} for boost in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))]
    cases.append({"gravity": (0.0, 0.0, -9.81)})
    for case in cases:
        result = _simulate_rod(device, steps, dt, mode="cos", **case)
        rotation = elastic_reference.rotation_angle_between(reference["joint_q"][3:7], result["joint_q"][3:7])
        test.assertLessEqual(rotation, 1.0e-6, msg=str(case))
        test.assertLessEqual(abs(result["joint_q"][7] - reference["joint_q"][7]), 1.0e-5, msg=str(case))
        test.assertLessEqual(result["angular_drift"], max(2.0 * floor, 1.0e-6), msg=str(case))
        test.assertLessEqual(result["ballistic"], 1.0e-5, msg=str(case))


def test_elastic_free_body_origin_offset_invariance(test, device):
    """Make a free elastic body's motion independent of where its frame origin is placed.

    The same physical rod is described with its frame origin offset by +-0.1 m along each axis from the
    rest COM; the sampled COM trajectory and the attitude must match the centered description, both
    spinning and at rest. The frozen-inertia torque ``-v_origin x S qdot`` depended on this choice.
    """
    dt = 1.0 / 480.0
    steps = 96
    for spin in ((0.0, 0.0, 0.0), (0.0, 0.0, 5.0)):
        reference = _simulate_rod(device, steps, dt, mode="cos", spin=spin, record=True)
        for axis in range(3):
            for sign in (-1.0, 1.0):
                com = [0.0, 0.0, 0.0]
                com[axis] = 0.1 * sign
                result = _simulate_rod(device, steps, dt, mode="cos", spin=spin, com=com, record=True)
                com_error = float(np.max(np.linalg.norm(result["com"] - reference["com"], axis=1)))
                rotation_error = max(
                    elastic_reference.rotation_angle_between(a, b)
                    for a, b in zip(reference["quat"], result["quat"], strict=True)
                )
                # Poses store the frame origin, so each update rounds the origin-COM offset (float32 floor ~1e-6 m).
                test.assertLessEqual(com_error, 2.0e-6, msg=f"spin={spin} com={com}")
                test.assertLessEqual(rotation_error, 1.0e-5, msg=f"spin={spin} com={com}")


def test_elastic_asymmetric_zero_spin_counterexample(test, device):
    """Keep the COM of a non-spinning body with ``S != 0`` and ``G != 0`` ballistic.

    Adding only the missing ``2 W S qdot`` transport term to the frozen-inertia frame force makes this
    case about four times worse than before (7.9e-4 m vs 1.95e-4 m); the momentum-difference form is
    exact up to the integrator.
    """
    result = _simulate_rod(device, 192, 1.0 / 960.0, mode="cos_offset")
    test.assertLess(result["ballistic"], 5.0e-5)


def test_elastic_random_state_matches_reference(test, device):
    """Follow a float64 Kane-equation reference from random free-body states.

    Four seeded states (``|v| <= 2`` m/s, ``|w| <= 3`` rad/s, ``|q| <= 0.05``, ``|qdot| <= 0.5``) of the
    ``cos_offset`` rod are stepped for 0.05 s. The backward-Euler integrator is first order, so the
    attitude error is bounded at two step sizes and must shrink with the step; the COM, which the
    momentum-difference form keeps on the exact ballistic path, has a tight bound.
    """
    rng = np.random.default_rng(1234)
    duration = 0.05
    for _ in range(4):
        velocity = rng.uniform(-2.0, 2.0, 3) / np.sqrt(3.0)
        spin = rng.uniform(-3.0, 3.0, 3) / np.sqrt(3.0)
        q0 = float(rng.uniform(-0.05, 0.05))
        qd0 = float(rng.uniform(-0.5, 0.5))
        errors = []
        for dt in (1.0 / 960.0, 1.0 / 1920.0):
            result = _simulate_rod(
                device,
                round(duration / dt),
                dt,
                iterations=48,
                mode="cos_offset",
                velocity=velocity,
                spin=spin,
                q0=q0,
                qd0=qd0,
            )
            data = result["data"]
            joint_q_0 = np.r_[np.zeros(3), 0.0, 0.0, 0.0, 1.0, q0]
            joint_qd_0 = np.r_[velocity, spin, qd0]
            start = elastic_reference.newton_to_reference(joint_q_0, joint_qd_0, data["com"], 1)
            end = elastic_reference.integrate_reference(
                data["points"], data["mass"], data["phi"], [30.0], start, duration, 1.0 / 7680.0
            )
            ref_q, ref_qd = elastic_reference.reference_to_newton(end, data["com"], 1)
            ref_com = elastic_reference.physical_state(
                data["points"], data["mass"], data["phi"], data["com"], ref_q, ref_qd
            )[0]
            com = elastic_reference.physical_state(
                data["points"], data["mass"], data["phi"], data["com"], result["joint_q"], result["joint_qd"]
            )[0]
            errors.append(
                (
                    elastic_reference.rotation_angle_between(ref_q[3:7], result["joint_q"][3:7]),
                    float(np.linalg.norm(com - ref_com)),
                    abs(result["joint_q"][7] - ref_q[7]) / max(abs(ref_q[7]), 1.0e-3),
                )
            )
        (rot_coarse, _com_coarse, _q_coarse), (rot_fine, com_fine, q_fine) = errors
        test.assertLess(rot_fine, 3.0e-3)
        test.assertLess(com_fine, 1.0e-5)
        test.assertLess(q_fine, 1.0e-2)
        test.assertLess(rot_fine, rot_coarse / 1.6)


def _large_coordinate_run(device, y: float, steps: int, dt: float):
    """Drift the reference rod and a rigid twin along y from ``(0, y, 0)``; return errors and the final state."""
    velocity = np.array([0.0, 0.0573, 0.0])
    model, state_0, data = _rod_model(device, "cos", position=(0.0, y, 0.0), velocity=velocity, rigid_control=True)
    state_1 = model.state()
    control = model.control()
    solver = newton.solvers.SolverVBD(model, iterations=24)
    rigid = data["rigid"]
    com_0, momentum_0, _ = _rod_physical(state_0, data)
    total_mass = float(np.sum(data["mass"]))
    rigid_0 = state_0.body_q.numpy()[rigid][:3].astype(np.float64)
    elastic_error = 0.0
    rigid_error = 0.0
    for step in range(steps):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        t = (step + 1) * dt
        com = _rod_physical(state_0, data)[0]
        elastic_error = max(elastic_error, float(np.linalg.norm(com - com_0 - t * momentum_0 / total_mass)))
        rigid_pos = state_0.body_q.numpy()[rigid][:3].astype(np.float64)
        rigid_error = max(rigid_error, float(np.linalg.norm(rigid_pos - rigid_0 - t * velocity)))
    return elastic_error, rigid_error, _rod_joint_state(state_0, data)[0]


def test_elastic_large_coordinate_invariance(test, device):
    """Keep a free elastic body's attitude and deformation independent of its absolute position.

    The rod drifts along y from y = 0, 10 and 100 m. Its frame acceleration is taken from the rigid
    row's displacement from the inertial target, not from double-differenced absolute float32
    positions, so the modal coordinate and the attitude match the run at the origin; at y = 0 the
    COM stays ballistic.
    """
    dt = 1.0 / 960.0
    steps = 192
    elastic_error, _rigid_error, reference = _large_coordinate_run(device, 0.0, steps, dt)
    test.assertLessEqual(elastic_error, 1.0e-6)
    for y in (10.0, 100.0):
        _elastic_error, _rigid_error, joint_q = _large_coordinate_run(device, y, steps, dt)
        test.assertLessEqual(abs(joint_q[7] - reference[7]), 1.0e-6, msg=f"y={y}")
        test.assertLessEqual(
            elastic_reference.rotation_angle_between(reference[3:7], joint_q[3:7]), 1.0e-6, msg=f"y={y}"
        )


def test_elastic_large_coordinate_ballistic_com(test, device):
    """Keep a free elastic body's COM at a rigid body's float32 error level 100 m from the origin.

    The frame's per-step response to the elastic coupling force, ``dt^2 F / m`` (about 7e-7 m here),
    is below half the float32 spacing at 100 m (3.8e-6 m). Applied to the absolute pose it was
    rounded away and the COM drifted by about 34 mm in 0.2 s while a rigid twin stayed within 1 mm;
    the frame update now accumulates the displacement from the inertial target separately.
    """
    elastic_error, rigid_error, _joint_q = _large_coordinate_run(device, 100.0, 192, 1.0 / 960.0)
    test.assertLessEqual(elastic_error, max(3.0 * rigid_error, 1.0e-6))
    test.assertLessEqual(elastic_error, 2.0e-3)


def test_elastic_displacement_form_translation_invariance(test, device):
    """Make the modes and attitude of a drifting free elastic body bitwise independent of a 100 m offset.

    The frame update accumulates its displacement from the inertial target and the frame
    velocity is the predicted velocity plus that displacement, so no absolute position enters the
    frame, modal or coupling rows: the modal coordinate and the attitude series at y = 100 m must
    equal those at y = 0 bit for bit, and the COM must stay within three times a rigid twin's error.
    """
    dt = 1.0 / 960.0
    steps = 96
    velocity = np.array([0.0, 0.0573, 0.0])
    series = {}
    for y in (0.0, 100.0):
        model, state_0, data = _rod_model(device, "cos", position=(0.0, y, 0.0), velocity=velocity, rigid_control=True)
        state_1 = model.state()
        control = model.control()
        solver = newton.solvers.SolverVBD(model, iterations=24)
        test.assertEqual(solver._use_body_displacement, 1)
        com_0, momentum_0, _ = _rod_physical(state_0, data)
        total_mass = float(np.sum(data["mass"]))
        rigid = data["rigid"]
        rigid_0 = state_0.body_q.numpy()[rigid][:3].astype(np.float64)
        attitude, elastic_error, rigid_error = [], 0.0, 0.0
        for step in range(steps):
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0
            t = (step + 1) * dt
            joint_q = state_0.joint_q.numpy()
            attitude.append(joint_q[data["q_start"] + 3 : data["q_start"] + 8].copy())
            com = _rod_physical(state_0, data)[0]
            elastic_error = max(elastic_error, float(np.linalg.norm(com - com_0 - t * momentum_0 / total_mass)))
            rigid_pos = state_0.body_q.numpy()[rigid][:3].astype(np.float64)
            rigid_error = max(rigid_error, float(np.linalg.norm(rigid_pos - rigid_0 - t * velocity)))
        series[y] = np.asarray(attitude)
        test.assertLessEqual(elastic_error, max(3.0 * rigid_error, 1.0e-6), msg=f"y={y}")
    np.testing.assert_array_equal(series[100.0].view(np.uint32), series[0.0].view(np.uint32))


def _uncoupled_elastic_model(device):
    """Build a free elastic body whose basis has no sample mass, so every coupling integral is zero.

    Must stay identical to ``uncoupled_model`` in the script that recorded the golden trajectories
    (see :func:`test_elastic_uncoupled_body_matches_base_trajectory`).
    """
    points = np.array([[-0.4, 0.0, 0.0], [0.0, 0.05, 0.0], [0.4, 0.0, -0.05]], dtype=np.float32)
    phi = np.array(
        [[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, -0.5]], [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=np.float32,
    )
    basis = newton.ModalBasis(
        sample_points=points, sample_phi=phi, mode_mass=[0.5, 0.8], mode_stiffness=[20.0, 45.0], mode_damping=[0.1, 0.0]
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.1, -0.2, 1.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.6, 0.8), 0.4)),
        com=wp.vec3(0.05, -0.02, 0.03),
        mass=1.3,
        inertia=wp.mat33(0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4),
        modal_basis=basis,
        mode_q=[0.02, -0.01],
        mode_qd=[0.4, -0.3],
    )
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    joint_qd = state.joint_qd.numpy()
    joint_qd[:6] = [0.3, -0.2, 0.5, 1.0, 2.0, -1.5]
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control = model.control()
    joint_f = control.joint_f.numpy()
    joint_f[6:8] = [0.1, -0.2]
    control.joint_f.assign(joint_f)
    return model, state, control


def test_elastic_uncoupled_body_matches_base_trajectory(test, device):
    """Reproduce the pre-consistent-inertia modes when every coupling integral is zero.

    With ``S = G = B = C = P = 0`` and a diagonal modal mass the frame and the modes are independent,
    so the deformation-dependent inertia terms must add exact zeros. The golden ``joint_q`` series
    (180 steps) was recorded on the CPU with the frozen-inertia kernels (the integration base before this change,
    Warp 1.18.0); on the CPU its modal coordinates must be reproduced bit for bit (CUDA contracts
    floating-point operations differently: within 1e-6).

    The frame is a free rigid body under gravity, so its rest COM must follow the backward-Euler
    ballistic path. The recording's absolute-position frame update rounded its velocity at the float32
    spacing of its position and drifted 3.6e-5 m from that path; the displacement-form update stays
    within 1.5e-6 m, so the frame is checked against the path, and its attitude against the
    recording within 5e-5 (the base kernels' own CUDA-CPU spread is 2.9e-5).
    """
    model, state_0, control = _uncoupled_elastic_model(device)
    state_1 = model.state()
    solver = newton.solvers.SolverVBD(model, iterations=16)
    dt = 1.0 / 240.0
    trajectory = []
    for _ in range(180):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        trajectory.append(state_0.joint_q.numpy().copy())
    trajectory = np.asarray(trajectory, dtype=np.float32)
    golden_dir = os.path.join(os.path.dirname(__file__), "golden_data", "test_reduced_elastic_body")
    golden = np.load(os.path.join(golden_dir, "uncoupled_joint_q_cpu.npy"))
    if wp.get_device(device).is_cuda:
        np.testing.assert_allclose(trajectory[:, 7:], golden[:, 7:], rtol=0.0, atol=1.0e-6)
    else:
        np.testing.assert_array_equal(trajectory[:, 7:].view(np.uint32), golden[:, 7:].view(np.uint32))
    np.testing.assert_allclose(trajectory[:, 3:7], golden[:, 3:7], rtol=0.0, atol=5.0e-5)

    com = np.array([0.05, -0.02, 0.03])
    initial = model.state().joint_q.numpy().astype(np.float64)
    initial_com = initial[:3] + quat_rotate(initial[3:7], com)
    steps = np.arange(1, len(trajectory) + 1)[:, None]
    gravity = np.array([0.0, 0.0, -9.81])
    expected = initial_com + steps * dt * np.array([0.3, -0.2, 0.5]) + gravity * dt * dt * steps * (steps + 1) / 2.0
    rest_com = np.array([row[:3] + quat_rotate(row[3:7], com) for row in trajectory.astype(np.float64)])
    test.assertLess(float(np.max(np.linalg.norm(rest_com - expected, axis=1))), 3.0e-6)


def test_elastic_nonorthogonal_modes_use_full_mass_matrix(test, device):
    """Accelerate non-mass-orthogonal modes with the full modal mass matrix.

    Two sampled modes with ``Mf = [[1, 1], [1, 2]]`` on a kinematic frame receive a modal force
    ``Q = (0, 1)``. The modal acceleration is ``Mf^-1 Q = (-1, 1)``; the diagonal masses alone would
    give ``(0, 0.5)``.
    """
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    phi = [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    basis = newton.ModalBasis(
        sample_points=points, sample_phi=phi, sample_mass=[1.0, 1.0], mode_stiffness=[0.0, 0.0], mode_damping=[0.0, 0.0]
    )
    np.testing.assert_allclose(basis.mode_mass_matrix, [[1.0, 1.0], [1.0, 2.0]], atol=1.0e-7)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    # The kinematic frame's inertia does not enter; two samples on a line have none about it.
    body = builder.add_body_elastic(
        inertia=_identity_inertia(), modal_basis=basis, is_kinematic=True, inertia_tolerance=None
    )
    builder.color()
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    _q_start, qd_start = _elastic_free_joint_starts(model, body)
    joint_f = control.joint_f.numpy()
    joint_f[qd_start + 7] = 1.0
    control.joint_f.assign(joint_f)
    solver = newton.solvers.SolverVBD(model, iterations=8)
    dt = 1.0e-3
    solver.step(state_0, state_1, control, None, dt)
    qdd = state_1.joint_qd.numpy()[qd_start + 6 : qd_start + 8] / dt
    np.testing.assert_allclose(qdd, [-1.0, 1.0], rtol=1.0e-4, atol=1.0e-4)


def test_vbd_revolute_uses_elastic_endpoint(test, device):
    """Check that a revolute joint holds the deformed elastic endpoint rather than the rest anchor."""
    eta = 0.2
    rest_anchor = -0.5

    def shape_fn(x):
        return np.array([[x[0] / abs(rest_anchor), 0.0, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(abs(rest_anchor) + eta, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_q=[eta],
        mode_shape_fn=shape_fn,
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(rest_anchor, 0.0, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    before = _deformed_endpoint_world(model, state_0, joint, "child")
    np.testing.assert_allclose(before, [0.0, 0.0, 0.0], atol=1.0e-6)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=8,
        rigid_compliant_alm=True,
        rigid_joint_linear_ke=1.0e5,
        rigid_joint_angular_ke=1.0e5,
    )
    solver.step(state_0, state_1, control, None, 0.01)

    np.testing.assert_allclose(solver.joint_C0_lin.numpy()[joint], [0.0, 0.0, 0.0], atol=1.0e-6)
    after = _deformed_endpoint_world(model, state_1, joint, "child")
    np.testing.assert_allclose(after, [0.0, 0.0, 0.0], atol=1.0e-4)
    np.testing.assert_allclose(state_1.body_q.numpy()[body, :3], [abs(rest_anchor) + eta, 0.0, 0.0], atol=1.0e-4)


def _check_vbd_revolute_constraint_solves_elastic_mode(test, device, compliant: bool):
    rest_anchor = -0.5
    target_anchor = 0.1

    def shape_fn(x):
        return np.array([[x[0] / abs(rest_anchor), 0.0, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(abs(rest_anchor), 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_mass=[1.0e-6],
        mode_q=[0.0],
        mode_shape_fn=shape_fn,
        is_kinematic=True,
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(target_anchor, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(rest_anchor, 0.0, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])

    legacy = {} if compliant else {"rigid_joint_linear_k_start": 1.0e6}
    solver = _make_vbd_solver(
        model,
        compliant=compliant,
        iterations=4,
        rigid_joint_linear_ke=1.0e6,
        rigid_joint_linear_kd=0.0,
        **legacy,
    )
    solver.step(state_0, state_1, control, None, 0.01)

    np.testing.assert_allclose(state_1.joint_q.numpy()[q_start + 7], -target_anchor, atol=1.0e-4)
    after = _deformed_endpoint_world(model, state_1, joint, "child")
    np.testing.assert_allclose(after, [target_anchor, 0.0, 0.0], atol=1.0e-4)


def test_vbd_revolute_constraint_solves_elastic_mode(test, device):
    """Satisfy a revolute anchor by deforming the elastic endpoint of a kinematic frame."""
    for compliant in _RIGID_ALM_MODES:
        with test.subTest(rigid_compliant_alm=compliant):
            _check_vbd_revolute_constraint_solves_elastic_mode(test, device, compliant)


def test_vbd_fixed_joint_stiffness_pins_penalties(test, device):
    """Check that ``rigid_joint_adaptive_stiffness=False`` pins joint penalties at the authored stiffness."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=_identity_inertia(),
    )
    builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
    )
    builder.color()
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=2,
        rigid_joint_linear_ke=1234.0,
        rigid_joint_angular_ke=567.0,
        rigid_joint_adaptive_stiffness=False,
    )

    np.testing.assert_allclose(solver.joint_penalty_k.numpy(), solver.joint_penalty_k_max.numpy())
    np.testing.assert_allclose(solver.joint_penalty_k_min.numpy(), solver.joint_penalty_k_max.numpy())
    np.testing.assert_allclose(solver.joint_penalty_k.numpy()[:2], [1234.0, 567.0])

    state_0 = model.state()
    state_1 = model.state()
    state_0.body_q.assign([wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity())])
    solver.step(state_0, state_1, model.control(), None, 0.01)

    np.testing.assert_allclose(solver.joint_penalty_k.numpy(), solver.joint_penalty_k_max.numpy())
    test.assertGreater(np.linalg.norm(solver.joint_lambda_lin.numpy()[0]), 0.0)


def test_vbd_elastic_modal_force_matches_joint_projection(test, device):
    """Check that joint forces project onto modal forces for several joint types on either side."""
    for joint_kind in ("ball", "fixed", "revolute", "prismatic"):
        _assert_elastic_modal_projection_matches_joint_force(test, device, joint_kind, elastic_side="child")
    _assert_elastic_modal_projection_matches_joint_force(test, device, "revolute", elastic_side="parent")


def _check_vbd_elastic_joint_assembles_one_coupled_block(test, device, compliant: bool):
    dt = 0.01
    joint_k = 1000.0
    frame_x = 0.1
    mode_q = -0.04
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[1.0, 0.0, 0.0]]],
        sample_mass=[1.0],
        mode_stiffness=[0.0],
        mode_damping=[0.0],
    )

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(frame_x, 0.0, 0.0), wp.quat_identity()),
        mass=3.0,
        inertia=_identity_inertia(),
        mode_q=[mode_q],
        modal_basis=basis,
        inertia_tolerance=None,
    )
    builder.add_joint_fixed(parent=-1, child=body)
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    solver = _vbd_fixed_joint_solver(model, compliant=compliant, iterations=1, linear_k=joint_k, angular_k=joint_k)
    solver.step(state_0, state_1, model.control(), None, dt)
    linear_k = _first_solve_joint_k(solver, 0, 0, joint_k, compliant)

    matrix_upper = solver.elastic_body_block_matrix.numpy()[0]
    matrix = np.triu(matrix_upper) + np.triu(matrix_upper, 1).T
    grad = solver.elastic_body_block_grad.numpy()[0]
    delta = solver.elastic_body_block_delta.numpy()[0]
    expected_frame_diagonal = 3.0 / (dt * dt) + linear_k
    expected_cross = 1.0 / (dt * dt) + linear_k
    expected_modal_diagonal = 1.0 / (dt * dt) + linear_k

    test.assertTrue(bool(solver.elastic_implicit_mass_coupling.numpy()[0]))
    np.testing.assert_allclose(matrix[0, 0], expected_frame_diagonal, rtol=1.0e-6)
    np.testing.assert_allclose(matrix[0, 6], expected_cross, rtol=1.0e-6)
    np.testing.assert_allclose(matrix[6, 6], expected_modal_diagonal, rtol=1.0e-6)
    np.testing.assert_allclose(delta, np.linalg.solve(matrix, -grad), rtol=2.0e-6, atol=1.0e-8)

    body_x = float(state_1.body_q.numpy()[body, 0])
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    solved_mode_q = float(state_1.joint_q.numpy()[q_start + 7])
    np.testing.assert_allclose(body_x, frame_x + delta[0], rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(solved_mode_q, mode_q + delta[6], rtol=1.0e-6, atol=1.0e-7)


def test_vbd_elastic_joint_assembles_one_coupled_block(test, device):
    """Assemble frame, modal, and frame-mode coupling rows of an elastic joint in one block.

    The frame carries 2 kg of rigid mass beyond the single 1 kg sample, so the frame and modal
    diagonals differ while the reduced mass stays positive definite.
    """
    for compliant in _RIGID_ALM_MODES:
        with test.subTest(rigid_compliant_alm=compliant):
            _check_vbd_elastic_joint_assembles_one_coupled_block(test, device, compliant)


def test_vbd_elastic_body_rejects_external_rigid_solver(test, device):
    """Check that elastic bodies are rejected with ``integrate_with_external_rigid_solver``.

    A single sample has no rotational inertia, so the frame carries extra rigid mass and a nominal
    inertia rather than the basis rigid properties.
    """
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[1.0, 0.0, 0.0]]],
        sample_mass=[1.0],
        mode_stiffness=[0.0],
        mode_damping=[0.0],
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=3.0, inertia=_identity_inertia(), mode_q=[0.0], modal_basis=basis, inertia_tolerance=None
    )
    builder.add_joint_fixed(parent=-1, child=body)
    builder.color()
    model = builder.finalize(device=device)

    with test.assertRaises(NotImplementedError):
        newton.solvers.SolverVBD(model, iterations=1, integrate_with_external_rigid_solver=True)


def test_vbd_elastic_body_block_sparse_joints_scope(test, device):
    """Solve articulated elastic bodies in the block-sparse solve and reject the unsupported cases.

    An elastic body's joints must lie in its articulation, as for rigid bodies; its owner joint
    may stay outside. Joint mimics are not combined with elastic articulation bodies. A single
    sample has no rotational inertia, so the frames carry extra rigid mass and a nominal inertia
    rather than the basis rigid properties.
    """
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[1.0, 0.0, 0.0]]],
        sample_mass=[1.0],
        mode_stiffness=[50.0],
        mode_damping=[0.0],
    )

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=3.0, inertia=_identity_inertia(), mode_q=[0.0], modal_basis=basis, inertia_tolerance=None
    )
    builder.add_joint_fixed(parent=-1, child=body)
    builder.color()
    model = builder.finalize(device=device)
    with test.assertRaisesRegex(ValueError, "outside the declared articulation ranges"):
        newton.solvers.SolverVBD(model, iterations=1, rigid_articulation_solve="block_sparse_joints")

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    body = builder.add_link_elastic(
        mass=3.0, inertia=_identity_inertia(), mode_q=[0.01], modal_basis=basis, inertia_tolerance=None
    )
    builder.add_articulation([builder.add_joint_fixed(parent=-1, child=body)])
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=4, rigid_articulation_solve="block_sparse_joints")
    test.assertEqual(solver.rigid_articulation_sparse_layout.elastic_body_count, 1)
    state_0, state_1 = model.state(), model.state()
    for _ in range(4):
        solver.step(state_0, state_1, None, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    test.assertTrue(np.isfinite(state_0.joint_q.numpy()).all())
    test.assertLess(float(np.linalg.norm(state_0.body_q.numpy()[body, :3])), 1.0e-3)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link_elastic(
        mass=3.0, inertia=_identity_inertia(), mode_q=[0.0], modal_basis=basis, inertia_tolerance=None
    )
    links = [builder.add_link(mass=1.0, inertia=_identity_inertia()) for _ in range(2)]
    joints = [builder.add_joint_revolute(parent=-1, child=body, axis=(0.0, 0.0, 1.0))]
    joints += [builder.add_joint_revolute(parent=body, child=link, axis=(0.0, 0.0, 1.0)) for link in links]
    builder.set_joint_mimic(joints[2], joints[1], coeffs=(0.0, 1.0))
    builder.add_articulation(joints)
    builder.color()
    model = builder.finalize(device=device)
    with test.assertRaisesRegex(NotImplementedError, "mimic"):
        newton.solvers.SolverVBD(model, iterations=1, rigid_articulation_solve="block_sparse_joints")


def _no_displacement_inputs(device, body_count: int) -> list:
    """Return the displacement-form inputs of the block-solve kernels with displacement tracking off."""
    return [
        wp.array([wp.transform_identity()] * body_count, dtype=wp.transform, device=device),
        wp.zeros(body_count, dtype=wp.vec3, device=device),
        wp.zeros(body_count, dtype=wp.vec3, device=device),
        wp.zeros(body_count, dtype=wp.int32, device=device),
        0,
    ]


def test_vbd_elastic_block_solve_falls_back_when_indefinite(test, device):
    """An indefinite block degrades to a Jacobi step instead of writing NaN."""
    width = 7
    kernel = create_solve_elastic_body_tiled(width)

    rng = np.random.default_rng(0)
    matrix = np.zeros((2, width, width), dtype=np.float32)
    grad = np.zeros((2, width), dtype=np.float32)
    for block in range(2):
        a = rng.standard_normal((width, width))
        matrix[block] = a @ a.T + width * np.eye(width)
        grad[block] = rng.standard_normal(width)
    matrix[1, 3, 3] = -5.0

    zeros_int = wp.zeros(1, dtype=wp.int32, device=device)
    delta = wp.zeros((2, width), dtype=float, device=device)
    metrics = [wp.zeros(2, dtype=float, device=device) for _ in range(5)]
    wp.launch_tiled(
        kernel=kernel,
        dim=[2],
        inputs=[
            0.01,
            False,
            wp.array([0, 1], dtype=wp.int32, device=device),
            wp.array([0, 1], dtype=wp.int32, device=device),
            wp.array([0, 0], dtype=wp.int32, device=device),
            wp.array([1, 1], dtype=wp.int32, device=device),
            wp.zeros(2, dtype=float, device=device),
            wp.zeros(2, dtype=wp.vec3, device=device),
            wp.array([wp.transform_identity()] * 2, dtype=wp.transform, device=device),
            zeros_int,
            zeros_int,
            wp.zeros(8, dtype=float, device=device),
            wp.array(grad, dtype=float, device=device),
            delta,
            wp.array(matrix, dtype=float, device=device),
            *metrics,
            1.0,
            *_no_displacement_inputs(device, 2),
        ],
        outputs=[
            wp.array([wp.transform_identity()] * 2, dtype=wp.transform, device=device),
            wp.zeros(8, dtype=float, device=device),
            wp.zeros(8, dtype=float, device=device),
        ],
        block_dim=32,
        device=device,
    )

    solved = delta.numpy()
    test.assertTrue(bool(np.isfinite(solved).all()))

    np.testing.assert_allclose(solved[0], np.linalg.solve(matrix[0], -grad[0]), rtol=1.0e-4, atol=1.0e-5)

    diagonal = np.diag(matrix[1])
    expected_fallback = np.where(diagonal > 0.0, -grad[1] / np.where(diagonal > 0.0, diagonal, 1.0), 0.0)
    np.testing.assert_allclose(solved[1], expected_fallback, rtol=1.0e-5, atol=1.0e-6)


def _launch_elastic_block_solve(device, width, matrix, grad, mode_counts, block_dim=32, modal_outputs=None):
    """Launch the tiled block solve on symmetric ``matrix`` blocks and return the deltas.

    Every body shares one zero-initialized modal coordinate range; ``modal_outputs``, when given, is a
    list that receives the updated ``joint_q`` and ``joint_qd`` arrays.
    """
    block_count, stored_width = grad.shape
    kernel = create_solve_elastic_body_tiled(width)
    delta = wp.zeros((block_count, stored_width), dtype=float, device=device)
    metrics = [wp.zeros(block_count, dtype=float, device=device) for _ in range(5)]
    body_ids = wp.array(np.arange(block_count), dtype=wp.int32, device=device)
    mode_dof_count = int(np.max(mode_counts)) + 1
    joint_q = wp.zeros(mode_dof_count + 7, dtype=float, device=device)
    joint_qd = wp.zeros(mode_dof_count + 6, dtype=float, device=device)
    wp.launch_tiled(
        kernel=kernel,
        dim=[block_count],
        inputs=[
            0.01,
            False,
            body_ids,
            body_ids,
            wp.zeros(block_count, dtype=wp.int32, device=device),
            wp.array(mode_counts, dtype=wp.int32, device=device),
            wp.zeros(block_count, dtype=float, device=device),
            wp.zeros(block_count, dtype=wp.vec3, device=device),
            wp.array([wp.transform_identity()] * block_count, dtype=wp.transform, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
            wp.zeros(mode_dof_count + 7, dtype=float, device=device),
            wp.array(grad, dtype=float, device=device),
            delta,
            wp.array(matrix, dtype=float, device=device),
            *metrics,
            1.0,
            *_no_displacement_inputs(device, block_count),
        ],
        outputs=[
            wp.array([wp.transform_identity()] * block_count, dtype=wp.transform, device=device),
            joint_q,
            joint_qd,
        ],
        block_dim=block_dim,
        device=device,
    )
    if modal_outputs is not None:
        modal_outputs.extend((joint_q.numpy(), joint_qd.numpy()))
    return delta.numpy()


def test_vbd_elastic_block_solve_supports_representative_widths(test, device):
    """Tiled Cholesky solves block widths from small bases up to the shared-memory limit."""
    rng = np.random.default_rng(11)

    for width in (17, 22, 38, 70, 111, 157):
        a = rng.standard_normal((width, width))
        matrix = (a @ a.T + width * np.eye(width)).astype(np.float32)[None]
        grad = rng.standard_normal((1, width)).astype(np.float32)

        delta = _launch_elastic_block_solve(device, width, matrix, grad, [width - 6])

        np.testing.assert_allclose(
            delta[0],
            np.linalg.solve(matrix[0].astype(np.float64), -grad[0]),
            rtol=2.0e-4,
            atol=2.0e-5,
            err_msg=f"width={width}",
        )


def test_vbd_elastic_block_solve_reads_leading_sub_block(test, device):
    """A narrow kernel solves the leading sub-block of wider block storage."""
    rng = np.random.default_rng(5)
    width = 14
    stored_width = 70
    matrix = np.zeros((2, stored_width, stored_width), dtype=np.float32)
    grad = np.zeros((2, stored_width), dtype=np.float32)
    for block in range(2):
        a = rng.standard_normal((width, width))
        matrix[block] = np.eye(stored_width) + rng.standard_normal((stored_width, stored_width))
        matrix[block, :width, :width] = a @ a.T + width * np.eye(width)
        grad[block, :width] = rng.standard_normal(width)

    delta = _launch_elastic_block_solve(device, width, matrix, grad, [2, 2])

    for block in range(2):
        expected = np.linalg.solve(matrix[block, :width, :width].astype(np.float64), -grad[block, :width])
        np.testing.assert_allclose(delta[block, :width], expected, rtol=2.0e-4, atol=2.0e-5)
        np.testing.assert_array_equal(delta[block, width:], 0.0)


def _elastic_block_test_basis(mode_count: int, seed: int) -> newton.ModalBasis:
    """Return a random lumped modal basis with ``mode_count`` modes.

    The samples have more displacement DOFs than the frame and the modes together, so the random
    modes are independent of each other and of rigid-body motion, and the reduced mass with the
    basis rigid properties is positive definite.
    """
    rng = np.random.default_rng(seed)
    sample_count = mode_count // 3 + 4
    return newton.ModalBasis(
        sample_points=rng.uniform(-0.5, 0.5, (sample_count, 3)),
        sample_phi=0.1 * rng.standard_normal((sample_count, mode_count, 3)),
        sample_mass=[1.0 / sample_count] * sample_count,
        mode_stiffness=list(np.linspace(10.0, 100.0, mode_count)),
        mode_damping=[0.0] * mode_count,
    )


def _build_elastic_block_test_model(device, mode_counts):
    """Build free reduced elastic bodies with the given modal counts in a single color group."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    for index, mode_count in enumerate(mode_counts):
        builder.add_body_elastic(
            xform=wp.transform((2.0 * index, 0.0, 0.0), wp.quat_identity()),
            mode_q=[0.01] * mode_count,
            modal_basis=_elastic_block_test_basis(mode_count, seed=index),
            label=f"elastic_{index}",
        )
    builder.color()
    return builder.finalize(device=device)


def test_vbd_elastic_block_solve_buckets_mixed_widths(test, device):
    """Bodies with small and large bases in one color launch per-width buckets."""
    model = _build_elastic_block_test_model(device, [2, 64])
    test.assertEqual(len(model.body_color_groups), 1)

    solver = newton.solvers.SolverVBD(model, iterations=1)
    test.assertTrue(bool(np.all(solver.elastic_implicit_mass_coupling.numpy())))
    test.assertEqual(solver.elastic_body_block_width, 70)
    test.assertEqual(sorted(solver._elastic_block_solve_launches), [(0, 14, 1, "tiled"), (0, 70, 1, "tiled")])

    state_0 = model.state()
    state_1 = model.state()
    solver.step(state_0, state_1, model.control(), None, 1.0e-2)

    matrix = solver.elastic_body_block_matrix.numpy()
    grad = solver.elastic_body_block_grad.numpy()
    delta = solver.elastic_body_block_delta.numpy()
    test.assertTrue(bool(np.isfinite(delta).all()))
    for elastic_index, width in enumerate((14, 70)):
        block = matrix[elastic_index, :width, :width].astype(np.float64)
        upper = np.triu(block)
        np.testing.assert_allclose(block, upper + np.triu(block, 1).T, rtol=0.0, atol=0.0)
        expected = np.linalg.solve(block, -grad[elastic_index, :width])
        test.assertGreater(float(np.abs(expected).max()), 0.0)
        np.testing.assert_allclose(delta[elastic_index, :width], expected, rtol=2.0e-4, atol=2.0e-6)
        np.testing.assert_array_equal(delta[elastic_index, width:], 0.0)


def _elastic_block_capacity_model(device, mode_count):
    """Build one elastic body with ``mode_count`` independent nodal modes.

    Mode ``i`` moves sample ``i // 3`` along axis ``i % 3``. Three extra unmoved samples anchor the
    rigid-body motion, so the modal mass is diagonal and the reduced mass with the basis rigid
    properties is positive definite.
    """
    moving_count = (mode_count + 2) // 3
    anchors = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    sample_points = [[0.1 * (s + 1), 0.05 * (s % 7), 0.03 * (s % 5) + 0.2] for s in range(moving_count)] + anchors
    sample_phi = np.zeros((len(sample_points), mode_count, 3))
    for mode in range(mode_count):
        sample_phi[mode // 3, mode, mode % 3] = 1.0
    basis = newton.ModalBasis(
        sample_points=sample_points,
        sample_phi=sample_phi,
        sample_mass=[1.0 / len(sample_points)] * len(sample_points),
        mode_stiffness=[1.0] * mode_count,
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body_elastic(modal_basis=basis, label="capacity")
    builder.color()
    return builder.finalize(device=device)


def test_vbd_elastic_block_width_rejects_oversized_basis(test, device):
    """A basis too wide for the measured shared memory budget fails at construction, not at launch."""
    max_width = _max_elastic_block_solve_tiled_width(wp.get_device(device))
    model = _elastic_block_capacity_model(device, max_width - 6 + 1)

    with test.assertRaises(ValueError) as raised:
        newton.solvers.SolverVBD(model, iterations=1, elastic_block_solver="tiled")
    test.assertIn("shared memory", str(raised.exception))
    test.assertIn(f"at most {max_width - 6} modes", str(raised.exception))


def test_vbd_elastic_block_solver_rejects_unknown_mode(test, device):
    """Check that an unknown ``elastic_block_solver`` value raises."""
    model = _elastic_block_capacity_model(device, 2)
    with test.assertRaisesRegex(ValueError, "elastic_block_solver"):
        newton.solvers.SolverVBD(model, iterations=1, elastic_block_solver="lu")


def _step_elastic_block_capacity_solver(test, model, solver, steps=2):
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, 1.0e-2)
        state_0, state_1 = state_1, state_0
    test.assertTrue(bool(solver.elastic_implicit_mass_coupling.numpy()[0]))
    test.assertTrue(bool(np.isfinite(state_0.joint_q.numpy()).all()))
    test.assertTrue(bool(np.isfinite(solver.elastic_body_block_delta.numpy()).all()))


def test_vbd_elastic_block_auto_falls_back_to_dense(test, device):
    """Past the tiled limit, the default solver warns once and uses the dense block solve."""
    max_width = _max_elastic_block_solve_tiled_width(wp.get_device(device))
    model = _elastic_block_capacity_model(device, max_width - 6 + 1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = newton.solvers.SolverVBD(model, iterations=1)
    dense_warnings = [warning for warning in caught if "dense block solve" in str(warning.message)]
    test.assertEqual(len(dense_warnings), 1, [str(warning.message) for warning in caught])
    message = str(dense_warnings[0].message)
    test.assertIn(f"[{max_width + 1}]", message)
    test.assertIn("shared memory", message)
    test.assertEqual(solver._elastic_block_solve_launches, [(0, max_width + 1, 1, "dense")])

    _step_elastic_block_capacity_solver(test, model, solver)


def test_vbd_elastic_block_capacity_dense_beyond_tiled_limit(test, device):
    """Bases one mode past the tiled limit and with 256 modes construct and step with the dense solve."""
    max_width = _max_elastic_block_solve_tiled_width(wp.get_device(device))
    for mode_count in (max_width - 6 + 1, 256):
        model = _elastic_block_capacity_model(device, mode_count)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            solver = newton.solvers.SolverVBD(model, iterations=2)
        test.assertEqual(solver._elastic_block_solve_launches[0][3], "dense", f"mode_count={mode_count}")
        _step_elastic_block_capacity_solver(test, model, solver)


def test_vbd_elastic_block_dense_matches_tiled_step(test, device):
    """Forcing the dense block solve reproduces the tiled solve's step."""
    results = {}
    for block_solver in ("tiled", "dense"):
        model = _build_elastic_block_test_model(device, [2, 64])
        solver = newton.solvers.SolverVBD(model, iterations=2, elastic_block_solver=block_solver)
        test.assertTrue(bool(np.all(solver.elastic_implicit_mass_coupling.numpy())))
        test.assertEqual({entry[3] for entry in solver._elastic_block_solve_launches}, {block_solver})
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        for _ in range(3):
            solver.step(state_0, state_1, control, None, 1.0e-2)
            state_0, state_1 = state_1, state_0
        results[block_solver] = (state_0.joint_q.numpy(), state_0.body_q.numpy())

    np.testing.assert_allclose(results["dense"][0], results["tiled"][0], rtol=1.0e-4, atol=1.0e-6)
    np.testing.assert_allclose(results["dense"][1], results["tiled"][1], rtol=1.0e-4, atol=1.0e-6)


def _launch_elastic_block_solve_dense(device, width, matrix, grad, mode_counts, modal_outputs=None):
    """Run the dense block solve on symmetric ``matrix`` blocks and return the deltas.

    ``modal_outputs`` is handled as in :func:`_launch_elastic_block_solve`.
    """
    block_count = grad.shape[0]
    dense_width = elastic_block_dense_width(width)
    stored_matrix = np.tile(np.eye(dense_width, dtype=np.float32), (block_count, 1, 1))
    stored_matrix[:, :width, :width] = matrix
    stored_grad = np.zeros((block_count, dense_width), dtype=np.float32)
    stored_grad[:, :width] = grad

    body_ids = wp.array(np.arange(block_count), dtype=wp.int32, device=device)
    grad_array = wp.array(stored_grad, dtype=float, device=device)
    matrix_array = wp.array(stored_matrix, dtype=float, device=device)
    delta = wp.zeros((block_count, dense_width), dtype=float, device=device)
    factor = wp.zeros((block_count, dense_width, dense_width), dtype=float, device=device)
    mode_dof_count = int(np.max(mode_counts)) + 1
    joint_q = wp.zeros(mode_dof_count + 7, dtype=float, device=device)
    joint_qd = wp.zeros(mode_dof_count + 6, dtype=float, device=device)
    launch_solve_elastic_body_dense(
        width,
        body_ids,
        body_ids,
        matrix_array,
        factor,
        [
            0.01,
            False,
            body_ids,
            body_ids,
            wp.zeros(block_count, dtype=wp.int32, device=device),
            wp.array(mode_counts, dtype=wp.int32, device=device),
            wp.zeros(block_count, dtype=float, device=device),
            wp.zeros(block_count, dtype=wp.vec3, device=device),
            wp.array([wp.transform_identity()] * block_count, dtype=wp.transform, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
            wp.zeros(mode_dof_count + 7, dtype=float, device=device),
            grad_array,
            delta,
            matrix_array,
            *[wp.zeros(block_count, dtype=float, device=device) for _ in range(5)],
            1.0,
            *_no_displacement_inputs(device, block_count),
            factor,
            grad_array.reshape((block_count, dense_width, 1)),
        ],
        [
            wp.array([wp.transform_identity()] * block_count, dtype=wp.transform, device=device),
            joint_q,
            joint_qd,
        ],
        device,
    )
    if modal_outputs is not None:
        modal_outputs.extend((joint_q.numpy(), joint_qd.numpy()))
    return delta.numpy()[:, :width]


def test_vbd_elastic_block_dense_solve_matches_numpy(test, device):
    """The panel Cholesky solve matches a dense reference past the tiled shared-memory limit."""
    rng = np.random.default_rng(17)
    widths = (262, 1030) if wp.get_device(device).is_cuda else (70, 262)
    for width in widths:
        a = rng.standard_normal((width, width))
        matrix = (a @ a.T / width + np.eye(width)).astype(np.float32)[None]
        grad = rng.standard_normal((1, width)).astype(np.float32)

        delta = _launch_elastic_block_solve_dense(device, width, matrix, grad, [width - 6])

        np.testing.assert_allclose(
            delta[0],
            np.linalg.solve(matrix[0].astype(np.float64), -grad[0]),
            rtol=2.0e-4,
            atol=2.0e-5,
            err_msg=f"width={width}",
        )


def test_vbd_elastic_block_dense_solve_falls_back_when_indefinite(test, device):
    """An indefinite block degrades to a Jacobi step in the dense solve as in the tiled one."""
    width = 70
    rng = np.random.default_rng(23)
    matrix = np.zeros((2, width, width), dtype=np.float32)
    grad = rng.standard_normal((2, width)).astype(np.float32)
    for block in range(2):
        a = rng.standard_normal((width, width))
        matrix[block] = a @ a.T / width + np.eye(width)
    matrix[1, 40, 40] = -5.0

    delta = _launch_elastic_block_solve_dense(device, width, matrix, grad, [width - 6] * 2)

    test.assertTrue(bool(np.isfinite(delta).all()))
    np.testing.assert_allclose(
        delta[0], np.linalg.solve(matrix[0].astype(np.float64), -grad[0]), rtol=2.0e-4, atol=2.0e-5
    )
    diagonal = np.diag(matrix[1])
    expected_fallback = np.where(diagonal > 0.0, -grad[1] / np.where(diagonal > 0.0, diagonal, 1.0), 0.0)
    np.testing.assert_allclose(delta[1], expected_fallback, rtol=1.0e-5, atol=1.0e-6)


def test_vbd_elastic_block_solve_applies_update_once(test, device):
    """Increment every modal coordinate exactly once in multi-warp tiled and dense block solves.

    The modal update reads and writes ``joint_q`` in place. Executed by every thread of a CUDA block
    with more than one warp, threads that read a coordinate after another warp updated it applied
    the increment twice (about 40 % of the maximum increment at widths 126 and 157 with 128 and
    256 threads).
    """
    rng = np.random.default_rng(29)
    cases = [("tiled", 70, 32), ("tiled", 126, 128), ("tiled", 157, 256), ("dense", 262, 0)]
    for path, width, block_dim in cases:
        a = rng.standard_normal((width, width))
        matrix = (a @ a.T / width + np.eye(width)).astype(np.float32)[None]
        grad = rng.standard_normal((1, width)).astype(np.float32)
        outputs = []
        if path == "tiled":
            delta = _launch_elastic_block_solve(
                device, width, matrix, grad, [width - 6], block_dim=block_dim, modal_outputs=outputs
            )[0]
        else:
            delta = _launch_elastic_block_solve_dense(device, width, matrix, grad, [width - 6], modal_outputs=outputs)[
                0
            ]
        joint_q, joint_qd = outputs
        modes = delta[6:width]
        np.testing.assert_array_equal(joint_q[7 : 7 + width - 6], modes, err_msg=f"{path} width={width}")
        np.testing.assert_allclose(joint_qd[6:width], modes / 0.01, rtol=1.0e-6, err_msg=f"{path} width={width}")


def test_vbd_elastic_block_capacity_steps_at_tiled_limit(test, device):
    """The widest basis that fits the measured shared memory budget constructs and steps."""
    max_width = _max_elastic_block_solve_tiled_width(wp.get_device(device))
    model = _elastic_block_capacity_model(device, max_width - 6)
    solver = newton.solvers.SolverVBD(model, iterations=1)
    test.assertEqual(solver.elastic_body_block_width, max_width)
    test.assertEqual(solver._elastic_block_solve_launches, [(0, max_width, 1, "tiled")])

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    for _ in range(2):
        solver.step(state_0, state_1, control, None, 1.0e-2)
        state_0, state_1 = state_1, state_0
    test.assertTrue(bool(np.isfinite(state_0.joint_q.numpy()).all()))
    test.assertTrue(bool(np.isfinite(solver.elastic_body_block_delta.numpy()).all()))


@wp.kernel
def _elastic_reduced_mobility_probe(
    mobility: ElasticReducedMobility,
    mode_count: int,
    frame_jacobian_a: wp.array[wp.spatial_vector],
    frame_jacobian_b: wp.array[wp.spatial_vector],
    linear_a: wp.array[wp.vec3],
    angular_a: wp.array[wp.vec3],
    linear_b: wp.array[wp.vec3],
    angular_b: wp.array[wp.vec3],
    sample_phi: wp.array[wp.vec3],
    sample_psi: wp.array[wp.vec3],
    max_mode_count: int,
    out: wp.array[float],
):
    i = wp.tid()
    out[i] = _elastic_reduced_mobility_pair(
        mobility,
        0,
        mode_count,
        frame_jacobian_a[i],
        frame_jacobian_b[i],
        1,
        sample_phi,
        sample_psi,
        max_mode_count,
        linear_a[i],
        angular_a[i],
        linear_b[i],
        angular_b[i],
    )


def test_vbd_elastic_reduced_mobility_matches_inverse(test, device):
    """The Schur-form mobility equals ``J_a M^-1 J_b^T`` for diagonal and dense modal mass."""
    rng = np.random.default_rng(29)
    mode_count = 12
    max_mode_count = 16
    width = 6 + mode_count
    pair_count = 32

    # Sample 1 of two samples, so the per-sample offset into phi/psi is exercised.
    phi = rng.standard_normal((mode_count, 3))
    psi = rng.standard_normal((mode_count, 3))
    sample_phi = np.zeros((2 * max_mode_count, 3))
    sample_psi = np.zeros((2 * max_mode_count, 3))
    sample_phi[max_mode_count : max_mode_count + mode_count] = phi
    sample_psi[max_mode_count : max_mode_count + mode_count] = psi
    frame_a = rng.standard_normal((pair_count, 6))
    frame_b = rng.standard_normal((pair_count, 6))
    linear_a, angular_a, linear_b, angular_b = (rng.standard_normal((pair_count, 3)) for _ in range(4))
    angular_a[: pair_count // 2] = 0.0  # point rows, as used by contacts
    angular_b[: pair_count // 2] = 0.0
    jacobian_a = np.concatenate([frame_a, linear_a @ phi.T + angular_a @ psi.T], axis=1)
    jacobian_b = np.concatenate([frame_b, linear_b @ phi.T + angular_b @ psi.T], axis=1)

    for dense in (False, True):
        basis = np.linalg.qr(rng.standard_normal((width, width)))[0]
        reduced_mass = basis @ np.diag(rng.uniform(1.0, 4.0, width)) @ basis.T
        if not dense:
            # Make the modal block diagonal while keeping the matrix positive definite.
            modal_block = reduced_mass[6:, 6:]
            reduced_mass[6:, 6:] = np.diag(np.diag(modal_block)) + np.eye(mode_count) * np.abs(modal_block).sum()
        frame_inv, modal_inv = _elastic_schur_inverse_blocks(
            reduced_mass[:6, :6], reduced_mass[:6, 6:], reduced_mass[6:, 6:]
        )
        modal_inv_dense = np.zeros((1, max_mode_count, max_mode_count))
        modal_inv_dense[0, :mode_count, :mode_count] = modal_inv
        modal_inv_diagonal = np.zeros((1, max_mode_count))
        modal_inv_diagonal[0, :mode_count] = np.diag(modal_inv)
        cross = np.zeros((1, max_mode_count, 6))
        cross[0, :mode_count] = reduced_mass[:6, 6:].T

        mobility = ElasticReducedMobility()
        mobility.frame_inv_schur = wp.array(frame_inv[None], dtype=wp.spatial_matrix, device=device)
        mobility.frame_modal_mass = wp.array(cross, dtype=wp.spatial_vector, device=device)
        mobility.modal_inv_mass = wp.array(modal_inv_diagonal, dtype=float, device=device)
        mobility.modal_inv_mass_dense = wp.array(modal_inv_dense, dtype=float, device=device)
        mobility.modal_mass_is_dense = int(dense)

        out = wp.zeros(pair_count, dtype=float, device=device)
        wp.launch(
            _elastic_reduced_mobility_probe,
            dim=pair_count,
            inputs=[
                mobility,
                mode_count,
                wp.array(frame_a, dtype=wp.spatial_vector, device=device),
                wp.array(frame_b, dtype=wp.spatial_vector, device=device),
                *(wp.array(v, dtype=wp.vec3, device=device) for v in (linear_a, angular_a, linear_b, angular_b)),
                wp.array(sample_phi, dtype=wp.vec3, device=device),
                wp.array(sample_psi, dtype=wp.vec3, device=device),
                max_mode_count,
            ],
            outputs=[out],
            device=device,
        )

        expected = np.einsum("pi,ij,pj->p", jacobian_a, np.linalg.inv(reduced_mass), jacobian_b)
        scale = np.max(np.abs(expected))
        np.testing.assert_allclose(out.numpy(), expected, rtol=1.0e-5, atol=1.0e-5 * scale, err_msg=f"dense={dense}")


def test_vbd_elastic_block_metrics_cover_frame_rows(test, device):
    """Convergence metrics span the whole block, not just the modal rows."""
    width = 7
    kernel = create_solve_elastic_body_tiled(width)

    rng = np.random.default_rng(3)
    a = rng.standard_normal((width, width))
    matrix = (a @ a.T + width * np.eye(width)).astype(np.float32)[None]
    grad = rng.standard_normal((1, width)).astype(np.float32)

    zeros_int = wp.zeros(1, dtype=wp.int32, device=device)
    delta = wp.zeros((1, width), dtype=float, device=device)
    metrics = [wp.zeros(1, dtype=float, device=device) for _ in range(5)]
    wp.launch_tiled(
        kernel=kernel,
        dim=[1],
        inputs=[
            0.01,
            True,
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.ones(1, dtype=float, device=device),
            wp.zeros(1, dtype=wp.vec3, device=device),
            wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
            zeros_int,
            zeros_int,
            wp.zeros(8, dtype=float, device=device),
            wp.array(grad, dtype=float, device=device),
            delta,
            wp.array(matrix, dtype=float, device=device),
            *metrics,
            1.0,
            *_no_displacement_inputs(device, 1),
        ],
        outputs=[
            wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
            wp.zeros(8, dtype=float, device=device),
            wp.zeros(8, dtype=float, device=device),
        ],
        block_dim=32,
        device=device,
    )

    initial_residual = float(metrics[0].numpy()[0])
    modal_only = float(abs(grad[0, 6]))
    test.assertAlmostEqual(initial_residual, float(np.linalg.norm(grad[0])), delta=1.0e-4)
    test.assertGreater(initial_residual, 2.0 * modal_only)

    solve_residual = float(metrics[1].numpy()[0])
    test.assertLess(solve_residual / initial_residual, 1.0e-5)

    update_max = float(metrics[4].numpy()[0])
    test.assertAlmostEqual(update_max, float(np.abs(delta.numpy()[0]).max()), delta=1.0e-5)


def test_vbd_implicit_mass_coupling_requires_spd_block(test, device):
    """Disable implicit frame/mode coupling and warn when the reduced mass matrix is indefinite.

    The frames deliberately differ from the 2 kg sample: a lighter frame makes the reduced mass
    indefinite and a heavier one keeps it positive definite.
    """
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[1.0, 0.0, 0.0]]],
        sample_mass=[2.0],
        mode_stiffness=[1.0],
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    frame = {"inertia": _identity_inertia(), "modal_basis": basis, "inertia_tolerance": None}
    builder.add_body_elastic(mass=1.0, label="indefinite", **frame)
    builder.add_body_elastic(mass=3.0, label="positive_definite", **frame)
    builder.color()
    model = builder.finalize(device=device)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = newton.solvers.SolverVBD(model, iterations=1)
    np.testing.assert_array_equal(solver.elastic_implicit_mass_coupling.numpy(), [False, True])
    messages = [str(w.message) for w in caught if "indefinite reduced mass" in str(w.message)]
    test.assertEqual(len(messages), 1)
    test.assertIn("'indefinite'", messages[0])
    test.assertIn("smallest eigenvalue", messages[0])


def test_vbd_massless_mode_keeps_decoupled_solve_without_warning(test, device):
    """Keep a massless mode in the decoupled solve without the indefinite reduced mass warning.

    A mode with zero modal mass and no inertia coupling leaves the reduced mass singular rather than
    indefinite. A massless mode that still couples to the frame makes it indefinite and warns.
    """

    def build(coupled: bool):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        frame = {"mass": 1.0, "inertia": _identity_inertia(), "mode_mass": [0.0], "mode_stiffness": [10.0]}
        if coupled:
            basis = newton.ModalBasis(
                sample_points=[[0.0, 0.0, 0.0]], sample_phi=[[[1.0, 0.0, 0.0]]], sample_mass=[1.0]
            )
            builder.add_body_elastic(modal_basis=basis, inertia_tolerance=None, label="coupled", **frame)
        else:
            builder.add_body_elastic(
                mode_count=1,
                mode_shape_fn=lambda _x: np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                label="massless",
                **frame,
            )
        builder.color()
        return builder.finalize(device=device)

    model = build(coupled=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = newton.solvers.SolverVBD(model, iterations=1)
    test.assertEqual([str(w.message) for w in caught if "indefinite" in str(w.message)], [])
    test.assertFalse(bool(solver.elastic_implicit_mass_coupling.numpy()[0]))
    state_0, state_1 = model.state(), model.state()
    solver.step(state_0, state_1, model.control(), None, 0.01)
    test.assertTrue(bool(np.isfinite(state_1.joint_q.numpy()).all()))

    model = build(coupled=True)
    with test.assertWarnsRegex(UserWarning, "'coupled'.*indefinite reduced mass"):
        solver = newton.solvers.SolverVBD(model, iterations=1)
    test.assertFalse(bool(solver.elastic_implicit_mass_coupling.numpy()[0]))


def _craig_bampton_clamped_trajectory(device, *, iterations: int, compliant: bool) -> np.ndarray:
    """Return 40 steps of modal coordinates of a Craig-Bampton body clamped at both interfaces."""
    interface_positions, mass, stiffness, damping, sample_points, recovery, _ = _build_craig_bampton_test_data()
    generator = newton.ModalGeneratorCraigBampton(
        interface_positions=interface_positions,
        interface_names=["left", "right"],
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        damping_matrix=damping,
        sample_points=sample_points,
        recovery_matrix=recovery,
    )
    basis = generator.build()
    mode_qd = np.zeros(basis.mode_count, dtype=np.float32)
    mode_qd[0] = 1.0

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=generator.mass,
        com=wp.vec3(*generator.com),
        inertia=wp.mat33(*generator.inertia.flatten()),
        modal_basis=basis,
        mode_qd=mode_qd,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Adding a FIXED joint.*", category=UserWarning)
        for position in interface_positions:
            xform = wp.transform(wp.vec3(*position), wp.quat_identity())
            builder.add_joint_fixed(parent=-1, child=body, parent_xform=xform, child_xform=xform)
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    solver = _vbd_fixed_joint_solver(model, compliant=compliant, iterations=iterations, linear_k=1.0e6, angular_k=1.0e6)

    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint]) + 7
    trajectory = []
    for _ in range(40):
        solver.step(state_0, state_1, control, None, 1.0e-3)
        state_0, state_1 = state_1, state_0
        trajectory.append(state_0.joint_q.numpy()[q_start : q_start + basis.mode_count].copy())
    return np.asarray(trajectory)


def _relative_difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_vbd_craig_bampton_coupled_solve_iteration_invariant(test, device):
    """Reproduce the same Craig-Bampton trajectory for one and eight coupled solves per step.

    Legacy AVBD only: with a fixed penalty and no multiplier updates each step is one linear solve, so
    extra iterations must not change it. Compliant ALM updates its multipliers every iteration by
    design; ``test_vbd_craig_bampton_coupled_solve_converges_compliant_alm`` covers that path.
    """
    one_iteration = _craig_bampton_clamped_trajectory(device, iterations=1, compliant=False)
    eight_iterations = _craig_bampton_clamped_trajectory(device, iterations=8, compliant=False)
    test.assertLess(_relative_difference(one_iteration, eight_iterations), 5.0e-5)


def test_vbd_craig_bampton_coupled_solve_converges_compliant_alm(test, device):
    """Converge the clamped Craig-Bampton trajectory under compliant ALM to the legacy linear solve.

    Compliant ALM carries part of each clamp reaction in multipliers, so one iteration differs from the
    converged trajectory; more iterations must settle on the fixed-penalty answer.
    """
    legacy = _craig_bampton_clamped_trajectory(device, iterations=8, compliant=False)
    converged = _craig_bampton_clamped_trajectory(device, iterations=_ALM_CONVERGED_ITERATIONS, compliant=True)
    refined = _craig_bampton_clamped_trajectory(device, iterations=2 * _ALM_CONVERGED_ITERATIONS, compliant=True)
    test.assertLess(_relative_difference(converged, refined), 5.0e-5)
    test.assertLess(_relative_difference(converged, legacy), 5.0e-5)


def _check_vbd_elastic_modal_joint_damping_projection(test, device, compliant: bool):
    dt = 0.01
    mode_mass = 1.0
    mode_qd = 0.5
    joint_k = 100.0
    joint_kd = 0.2

    def shape_fn(_x):
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_mass=[mode_mass],
        mode_stiffness=[0.0],
        mode_damping=[0.0],
        mode_q=[0.0],
        mode_qd=[mode_qd],
        mode_shape_fn=shape_fn,
        is_kinematic=True,
    )
    builder.add_joint_ball(
        parent=-1,
        child=body,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    solver = _vbd_fixed_joint_solver(
        model,
        compliant=compliant,
        iterations=_ALM_CONVERGED_ITERATIONS if compliant else 1,
        linear_k=joint_k,
        angular_k=0.0,
        linear_kd=joint_kd,
    )
    solver.step(state_0, state_1, model.control(), None, dt)

    q_start = int(model.joint_q_start.numpy()[int(model.elastic_joint.numpy()[0])])
    q_integrated = dt * mode_qd
    modal_inertia = mode_mass / (dt * dt)
    joint_damping_h = joint_kd / dt
    h = modal_inertia + joint_k + joint_damping_h
    grad = joint_k * q_integrated + joint_kd * mode_qd
    q_expected = q_integrated - grad / h

    np.testing.assert_allclose(state_1.joint_q.numpy()[q_start + 7], q_expected, rtol=1.0e-6, atol=1.0e-7)


def test_vbd_elastic_modal_joint_damping_projection(test, device):
    """Project joint damping onto the modal velocity of an elastic endpoint."""
    for compliant in _RIGID_ALM_MODES:
        with test.subTest(rigid_compliant_alm=compliant):
            _check_vbd_elastic_modal_joint_damping_projection(test, device, compliant)


def _check_vbd_elastic_angular_constraint_without_linear_stiffness(test, device, compliant: bool):
    dt = 0.01
    mode_mass = 0.05
    mode_q = 0.3
    angular_k = 100.0
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 0.0]]],
        sample_psi=[[[1.0, 0.0, 0.0]]],
        mode_mass=[mode_mass],
        mode_stiffness=[0.0],
        mode_damping=[0.0],
    )

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_q=[mode_q],
        modal_basis=basis,
        is_kinematic=True,
    )
    builder.add_joint_fixed(parent=-1, child=body)
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    solver = _vbd_fixed_joint_solver(
        model,
        compliant=compliant,
        iterations=_ALM_CONVERGED_ITERATIONS if compliant else 1,
        linear_k=0.0,
        angular_k=angular_k,
    )
    solver.step(state_0, state_1, model.control(), None, dt)

    q_start = int(model.joint_q_start.numpy()[int(model.elastic_joint.numpy()[0])]) + 7
    modal_inertia = mode_mass / (dt * dt)
    expected = mode_q - angular_k * mode_q / (modal_inertia + angular_k)
    np.testing.assert_allclose(state_1.joint_q.numpy()[q_start], expected, rtol=1.0e-6, atol=1.0e-7)


def test_vbd_elastic_angular_constraint_without_linear_stiffness(test, device):
    """Solve a fixed joint's angular row through endpoint rotation when the linear row is disabled."""
    for compliant in _RIGID_ALM_MODES:
        with test.subTest(rigid_compliant_alm=compliant):
            _check_vbd_elastic_angular_constraint_without_linear_stiffness(test, device, compliant)


def _check_vbd_rigid_angular_damping_includes_modal_velocity(test, device, compliant: bool):
    dt = 0.01
    angular_k = 100.0

    def solve_parent_angle(angular_kd: float) -> float:
        basis = newton.ModalBasis(
            sample_points=[[0.0, 0.0, 0.0]],
            sample_phi=[[[0.0, 0.0, 0.0]]],
            sample_psi=[[[1.0, 0.0, 0.0]]],
            mode_mass=[0.05],
            mode_stiffness=[0.0],
            mode_damping=[0.0],
        )
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        parent = builder.add_body(mass=1.0, inertia=_identity_inertia())
        child = builder.add_body_elastic(
            mass=1.0,
            inertia=_identity_inertia(),
            mode_q=[0.0],
            mode_qd=[1.0],
            modal_basis=basis,
            is_kinematic=True,
        )
        builder.add_joint_fixed(parent=parent, child=child)
        builder.color()
        model = builder.finalize(device=device)

        state_0 = model.state()
        state_1 = model.state()
        solver = _vbd_fixed_joint_solver(
            model, compliant=compliant, iterations=1, linear_k=0.0, angular_k=angular_k, angular_kd=angular_kd
        )
        solver.step(state_0, state_1, model.control(), None, dt)
        q = state_1.body_q.numpy()[parent, 3:7]
        return 2.0 * math.atan2(abs(float(q[0])), abs(float(q[3])))

    undamped_angle = solve_parent_angle(0.0)
    damped_angle = solve_parent_angle(0.2)
    test.assertGreater(
        damped_angle,
        1.1 * undamped_angle,
        f"modal angular damping did not react on the rigid parent: {damped_angle:.3e} vs {undamped_angle:.3e}",
    )


def test_vbd_rigid_angular_damping_includes_modal_velocity(test, device):
    """React joint angular damping of an elastic endpoint's modal velocity on the rigid parent."""
    for compliant in _RIGID_ALM_MODES:
        with test.subTest(rigid_compliant_alm=compliant):
            _check_vbd_rigid_angular_damping_includes_modal_velocity(test, device, compliant)


def _check_vbd_elastic_angular_projection_uses_exponential_jacobian(test, device, compliant: bool):
    dt = 0.01
    mode_mass = np.array([1.0, 1.0], dtype=np.float32)
    mode_q = np.array([0.6, 0.4], dtype=np.float32)
    psi = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    angular_k = 100.0
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[np.zeros((2, 3), dtype=np.float32)],
        sample_psi=[psi],
        mode_mass=mode_mass,
        mode_stiffness=[0.0, 0.0],
        mode_damping=[0.0, 0.0],
    )

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_q=mode_q,
        modal_basis=basis,
        is_kinematic=True,
    )
    builder.add_joint_revolute(parent=-1, child=body, axis=(1.0, 0.0, 0.0))
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    solver = _vbd_fixed_joint_solver(model, compliant=compliant, iterations=1, linear_k=1.0, angular_k=angular_k)
    solver.step(state_0, state_1, model.control(), None, dt)
    angular_k = _first_solve_joint_k(solver, 0, 1, angular_k, compliant)

    theta = psi.T @ mode_q
    projector = np.diag([0.0, 1.0, 1.0])
    expected_grad = angular_k * psi @ projector @ theta
    expected_hessian = np.diag(mode_mass / (dt * dt)) + angular_k * psi @ projector @ psi.T
    actual_grad = solver.elastic_body_block_grad.numpy()[0, 6:8]
    actual_hessian = solver.elastic_body_block_matrix.numpy()[0, 6:8, 6:8]
    np.testing.assert_allclose(actual_grad, expected_grad, rtol=5.0e-4, atol=2.0e-2)
    np.testing.assert_allclose(actual_hessian, expected_hessian, rtol=1.0e-5, atol=6.0e-2)


def test_vbd_elastic_angular_projection_uses_exponential_jacobian(test, device):
    """Project an elastic endpoint's angular joint row through the exponential-map Jacobian."""
    for compliant in _RIGID_ALM_MODES:
        with test.subTest(rigid_compliant_alm=compliant):
            _check_vbd_elastic_angular_projection_uses_exponential_jacobian(test, device, compliant)


_REVOLUTE_ELASTIC_DT = 0.01
_REVOLUTE_ELASTIC_MODE_MASS = 1.0
_REVOLUTE_ELASTIC_DRIVE_K = 200.0


def _build_revolute_elastic_mode_model(
    device, *, mode_q: float, target_pos: float, target_ke: float, limit_upper: float
):
    """Build a kinematic elastic body whose revolute joint rotates a single modal endpoint about x."""
    mode_mass, drive_k = _REVOLUTE_ELASTIC_MODE_MASS, _REVOLUTE_ELASTIC_DRIVE_K
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[0.0, 0.0, 0.0]]],
        sample_psi=[[[1.0, 0.0, 0.0]]],
        mode_mass=[mode_mass],
        mode_stiffness=[0.0],
        mode_damping=[0.0],
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_q=[mode_q],
        modal_basis=basis,
        is_kinematic=True,
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(1.0, 0.0, 0.0),
        target_pos=target_pos,
        target_ke=target_ke,
        target_kd=0.0,
        limit_lower=-1.0,
        limit_upper=limit_upper,
        limit_ke=drive_k,
        limit_kd=0.0,
    )
    builder.color()
    return builder.finalize(device=device), joint


def _solve_revolute_elastic_mode(
    device, *, compliant: bool, mode_q: float, target_pos: float, target_ke: float, limit_upper: float
) -> tuple[float, float, float]:
    """Run one VBD step; return the modal coordinate, revolute coordinate [rad], and reported axis torque [N·m].

    Legacy AVBD reproduces the single-solve closed form in one iteration; compliant ALM reaches it once
    its drive/limit multipliers converge.
    """
    model, joint = _build_revolute_elastic_mode_model(
        device, mode_q=mode_q, target_pos=target_pos, target_ke=target_ke, limit_upper=limit_upper
    )
    state_0 = model.state()
    state_1 = model.state()
    solver = _vbd_fixed_joint_solver(
        model,
        compliant=compliant,
        iterations=_ALM_CONVERGED_ITERATIONS if compliant else 1,
        linear_k=1.0,
        angular_k=100.0,
    )
    body_q_prev = wp.clone(state_0.body_q)
    solver.step(state_0, state_1, model.control(), None, _REVOLUTE_ELASTIC_DT)
    joint_wrench = wp.zeros(model.joint_count, dtype=wp.spatial_vector, device=model.device)
    solver.eval_joint_forces(state_1, joint_wrench, body_q_prev=body_q_prev, dt=_REVOLUTE_ELASTIC_DT)
    q_start = int(model.joint_q_start.numpy()[int(model.elastic_joint.numpy()[0])]) + 7
    joint_q = state_1.joint_q.numpy()
    # The revolute axis is x in the child joint frame.
    axis_torque = float(joint_wrench.numpy()[joint][3])
    return float(joint_q[q_start]), float(joint_q[int(model.joint_q_start.numpy()[joint])]), axis_torque


def test_vbd_revolute_drive_projects_into_elastic_modes(test, device):
    """Drive a revolute joint whose only mobility is its elastic endpoint rotation.

    Under compliant ALM the drive row's support must include the endpoint's modal mobility; with rigid
    mobility only, the kinematic frame gives zero support and the drive is dropped.
    """
    dt, mode_mass, drive_k = _REVOLUTE_ELASTIC_DT, _REVOLUTE_ELASTIC_MODE_MASS, _REVOLUTE_ELASTIC_DRIVE_K
    modal_inertia = mode_mass / (dt * dt)
    drive_expected = drive_k * 0.25 / (modal_inertia + drive_k)
    for compliant in _RIGID_ALM_MODES:
        with test.subTest(rigid_compliant_alm=compliant):
            drive_result, _angle, _torque = _solve_revolute_elastic_mode(
                device, compliant=compliant, mode_q=0.0, target_pos=0.25, target_ke=drive_k, limit_upper=1.0
            )
            np.testing.assert_allclose(drive_result, drive_expected, rtol=1.0e-6, atol=1.0e-7)


def test_vbd_revolute_limit_projects_into_elastic_modes(test, device):
    """Engage a revolute limit with rotation that only comes from the elastic endpoint.

    The continuous joint coordinate that drives limits is measured at the deformed attachment frame, so
    endpoint rotation ``psi * q`` beyond the upper bound pushes the mode back and the reported joint
    reaction carries the limit torque.
    """
    dt, mode_mass, drive_k = _REVOLUTE_ELASTIC_DT, _REVOLUTE_ELASTIC_MODE_MASS, _REVOLUTE_ELASTIC_DRIVE_K
    modal_inertia = mode_mass / (dt * dt)
    limit_expected = 0.25 - drive_k * (0.25 - 0.1) / (modal_inertia + drive_k)
    for compliant in _RIGID_ALM_MODES:
        with test.subTest(rigid_compliant_alm=compliant):
            limit_result, angle, axis_torque = _solve_revolute_elastic_mode(
                device, compliant=compliant, mode_q=0.25, target_pos=0.0, target_ke=0.0, limit_upper=0.1
            )
            np.testing.assert_allclose(limit_result, limit_expected, rtol=1.0e-6, atol=1.0e-7)
            np.testing.assert_allclose(angle, limit_result, rtol=1.0e-5)
            np.testing.assert_allclose(axis_torque, -drive_k * (angle - 0.1), rtol=1.0e-4)


def test_vbd_revolute_coordinate_includes_elastic_endpoint_rotation(test, device):
    """Report a revolute coordinate that includes the elastic endpoint's modal rotation."""
    for compliant in _RIGID_ALM_MODES:
        with test.subTest(rigid_compliant_alm=compliant):
            mode_result, angle, _torque = _solve_revolute_elastic_mode(
                device, compliant=compliant, mode_q=0.25, target_pos=0.0, target_ke=0.0, limit_upper=1.0
            )
            # A free mode keeps its rotation; the kinematic frame does not rotate.
            np.testing.assert_allclose(mode_result, 0.25, rtol=1.0e-6)
            np.testing.assert_allclose(angle, mode_result, rtol=1.0e-5, atol=1.0e-6)


def _rod_elastic_endpoint_frame_rows(device, *, compliant: bool, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Assemble one step of a ROD joint at an elastic endpoint and return the frame rows of its block.

    Args:
        device: Warp device.
        compliant: Rigid VBD formulation.
        mode: ``"elastic"`` deforms the attachment through the endpoint's ``phi``/``psi``;
            ``"baked"`` zeroes them and bakes the same deformed frame into ``joint_X_c``;
            ``"rest"`` zeroes them and keeps the undeformed frame.

    Returns:
        The frame gradient rows ``[6]`` and frame Hessian block ``[6, 6]``.
    """
    dt = 0.01
    mode_q = 0.08
    anchor = np.array([0.0, 0.0, 0.2], dtype=np.float32)
    phi = np.array([0.03, -0.02, 0.05], dtype=np.float32)
    psi = np.array([0.4, -0.3, 0.2], dtype=np.float32)
    basis = newton.ModalBasis(
        sample_points=[anchor],
        sample_phi=[[phi]],
        sample_psi=[[psi]],
        mode_mass=[1.0],
        mode_stiffness=[0.0],
        mode_damping=[0.0],
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.2), wp.quat_identity()),
        mass=2.0,
        inertia=_identity_inertia(),
        mode_q=[mode_q],
        modal_basis=basis,
    )
    joint = builder.add_joint_rod(
        parent=-1,
        child=body,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(*anchor), wp.quat_identity()),
        stretch_stiffness=2.0e3,
        shear_stiffness=1.5e3,
        bend_stiffness=40.0,
        twist_stiffness=30.0,
    )
    builder.color()
    model = builder.finalize(device=device)
    if int(model.joint_child_elastic_endpoint.numpy()[joint]) < 0:
        raise AssertionError("the ROD child attachment must be an elastic endpoint")

    state_0 = model.state()
    state_1 = model.state()
    legacy = {} if compliant else {"rigid_joint_adaptive_stiffness": False}
    solver = _make_vbd_solver(model, compliant=compliant, iterations=1, **legacy)
    # Rest bend/twist invariants were cached from the undeformed frame; keep them for every variant.
    if mode != "elastic":
        model.elastic_endpoint_phi.zero_()
        model.elastic_endpoint_psi.zero_()
    if mode == "baked":
        theta = psi * mode_q
        angle = float(np.linalg.norm(theta))
        deformed = wp.transform(
            wp.vec3(*(anchor + phi * mode_q)), wp.quat_from_axis_angle(wp.vec3(*(theta / angle)), angle)
        )
        joint_X_c = model.joint_X_c.numpy()
        joint_X_c[joint] = np.array(deformed, dtype=np.float32)
        model.joint_X_c.assign(joint_X_c)
    solver.step(state_0, state_1, model.control(), None, dt)
    grad = solver.elastic_body_block_grad.numpy()[0, :6].copy()
    matrix_upper = solver.elastic_body_block_matrix.numpy()[0]
    matrix = np.triu(matrix_upper) + np.triu(matrix_upper, 1).T
    return grad, matrix[:6, :6].copy()


def test_vbd_rod_joint_uses_elastic_endpoint(test, device):
    """Evaluate ROD stretch, shear, bend, and twist at a deformed elastic endpoint.

    The frame rows must match the same joint with the deformation baked into ``joint_X_c`` and differ
    from the undeformed attachment.
    """
    for compliant in _RIGID_ALM_MODES:
        with test.subTest(rigid_compliant_alm=compliant):
            grad, hessian = _rod_elastic_endpoint_frame_rows(device, compliant=compliant, mode="elastic")
            baked_grad, baked_hessian = _rod_elastic_endpoint_frame_rows(device, compliant=compliant, mode="baked")
            rest_grad, _rest_hessian = _rod_elastic_endpoint_frame_rows(device, compliant=compliant, mode="rest")
            test.assertGreater(float(np.linalg.norm(baked_grad - rest_grad)), 1.0)
            np.testing.assert_allclose(grad, baked_grad, rtol=1.0e-4, atol=1.0e-3)
            np.testing.assert_allclose(hessian, baked_hessian, rtol=1.0e-4, atol=1.0e-2)


def test_fourbar_elastic_coupler_geometry(test, device):
    """Check that a four-bar with an axially stretched coupler closes at the deformed endpoints."""
    a, b_rest, c, d = 0.2, 0.5, 0.4, 0.5
    eta = 0.06
    b_eff = b_rest + eta
    theta2 = 0.35
    theta3, theta4 = _solve_fourbar(theta2, a, b_eff, c, d)

    A = np.array([0.0, 0.0, 0.0])
    D = np.array([d, 0.0, 0.0])
    e2 = np.array([math.cos(theta2), math.sin(theta2), 0.0])
    e3 = np.array([math.cos(theta3), math.sin(theta3), 0.0])
    e4 = np.array([math.cos(theta4), math.sin(theta4), 0.0])
    B = A + a * e2
    C = B + b_eff * e3
    np.testing.assert_allclose(C, D + c * e4, atol=1.0e-6)

    def axial_shape_fn(x):
        return np.array([[x[0] / b_rest, 0.0, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    crank = builder.add_body(
        xform=wp.transform(wp.vec3(*(A + 0.5 * a * e2)), _quat_from_angle_z(theta2)),
        mass=1.0,
        inertia=_identity_inertia(),
    )
    coupler = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(*(B + 0.5 * b_eff * e3)), _quat_from_angle_z(theta3)),
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_q=[eta],
        mode_shape_fn=axial_shape_fn,
    )
    rocker = builder.add_body(
        xform=wp.transform(wp.vec3(*(D + 0.5 * c * e4)), _quat_from_angle_z(theta4)),
        mass=1.0,
        inertia=_identity_inertia(),
    )

    j_ab = builder.add_joint_revolute(
        parent=crank,
        child=coupler,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(a / 2.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-b_rest / 2.0, 0.0, 0.0), wp.quat_identity()),
    )
    j_bc = builder.add_joint_revolute(
        parent=coupler,
        child=rocker,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(b_rest / 2.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(c / 2.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()

    p_ab_parent = _deformed_endpoint_world(model, state, j_ab, "parent")
    p_ab_child = _deformed_endpoint_world(model, state, j_ab, "child")
    p_bc_parent = _deformed_endpoint_world(model, state, j_bc, "parent")
    p_bc_child = _deformed_endpoint_world(model, state, j_bc, "child")

    np.testing.assert_allclose(p_ab_parent, B, atol=2.0e-6)
    np.testing.assert_allclose(p_ab_child, B, atol=2.0e-6)
    np.testing.assert_allclose(p_bc_parent, C, atol=2.0e-6)
    np.testing.assert_allclose(p_bc_child, C, atol=2.0e-6)
    np.testing.assert_allclose(np.linalg.norm(p_bc_parent - p_ab_child), b_eff, atol=2.0e-6)


def _check_vbd_prismatic_rotates_elastic_child(test, device, compliant: bool):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02)
    parent = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=inertia,
    )
    child = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.4, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=inertia,
        mode_count=1,
    )
    parent_joint = builder.add_joint_revolute(
        parent=-1,
        child=parent,
        axis=(0.0, 0.0, 1.0),
    )
    builder.add_joint_prismatic(
        parent=parent,
        child=child,
        axis=(1.0, 0.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.4, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
        target_pos=0.0,
    )
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    parent_qd_start = int(model.joint_qd_start.numpy()[parent_joint])
    joint_f = control.joint_f.numpy()
    joint_f[parent_qd_start] = 4.0
    control.joint_f.assign(joint_f)

    legacy = {} if compliant else {"rigid_joint_linear_k_start": 1.0e5, "rigid_joint_angular_k_start": 1.0e5}
    solver = _make_vbd_solver(
        model,
        compliant=compliant,
        iterations=20,
        rigid_joint_linear_ke=1.0e6,
        rigid_joint_angular_ke=1.0e6,
        **legacy,
    )
    for _ in range(60):
        solver.step(state_0, state_1, control, None, 1.0 / 180.0)
        state_0, state_1 = state_1, state_0

    body_q = state_0.body_q.numpy()
    parent_angle = 2.0 * math.atan2(float(body_q[parent, 5]), float(body_q[parent, 6]))
    child_angle = 2.0 * math.atan2(float(body_q[child, 5]), float(body_q[child, 6]))
    test.assertGreater(abs(parent_angle), 0.1)
    test.assertLess(_quat_angle_error(body_q[parent, 3:7], body_q[child, 3:7]), 1.0e-4)
    np.testing.assert_allclose(child_angle, parent_angle, atol=1.0e-4)


def test_vbd_prismatic_rotates_elastic_child(test, device):
    """Carry an elastic prismatic child around with its rotating parent."""
    for compliant in _RIGID_ALM_MODES:
        with test.subTest(rigid_compliant_alm=compliant):
            _check_vbd_prismatic_rotates_elastic_child(test, device, compliant)


def test_crank_slider_elastic_analytic_geometry(test, device):
    """Check crank-slider joint endpoints against the analytic geometry with a stretched elastic rod."""
    crank_length = 0.24
    rod_rest_length = 0.64
    axial_q = 0.05
    theta = 0.72
    rod_length = rod_rest_length + axial_q
    slider_x = crank_length * math.cos(theta) + math.sqrt(
        rod_length * rod_length - (crank_length * math.sin(theta)) ** 2
    )

    crank_pin = np.array([crank_length * math.cos(theta), crank_length * math.sin(theta), 0.0])
    slider_pin = np.array([slider_x, 0.0, 0.0])
    rod_theta = math.atan2(float(slider_pin[1] - crank_pin[1]), float(slider_pin[0] - crank_pin[0]))

    basis = newton.ModalGeneratorBeam(
        length=rod_rest_length,
        half_width_y=0.03,
        half_width_z=0.02,
        mode_specs=[{"type": newton.ModalGeneratorBeam.Mode.AXIAL}],
    ).build(sample_points=[[-0.5 * rod_rest_length, 0.0, 0.0], [0.5 * rod_rest_length, 0.0, 0.0]])

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    crank = builder.add_body(
        xform=wp.transform(
            wp.vec3(*(0.5 * crank_pin)),
            _quat_from_angle_z(theta),
        ),
        mass=1.0,
        inertia=_identity_inertia(),
    )
    rod = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(*(0.5 * (crank_pin + slider_pin))), _quat_from_angle_z(rod_theta)),
        mode_q=[axial_q],
        modal_basis=basis,
    )
    slider = builder.add_body(
        xform=wp.transform(wp.vec3(*slider_pin), wp.quat_identity()),
        mass=1.0,
        inertia=_identity_inertia(),
    )
    j_crank_rod = builder.add_joint_revolute(
        parent=crank,
        child=rod,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.5 * crank_length, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5 * rod_rest_length, 0.0, 0.0), wp.quat_identity()),
    )
    j_rod_slider = builder.add_joint_revolute(
        parent=rod,
        child=slider,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.5 * rod_rest_length, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()

    p_crank = joint_endpoint_world(model, state, j_crank_rod, "parent")
    p_rod_left = joint_endpoint_world(model, state, j_crank_rod, "child")
    p_rod_right = joint_endpoint_world(model, state, j_rod_slider, "parent")
    p_slider = joint_endpoint_world(model, state, j_rod_slider, "child")

    np.testing.assert_allclose(p_crank, crank_pin, atol=1.0e-6)
    np.testing.assert_allclose(p_rod_left, crank_pin, atol=1.0e-6)
    np.testing.assert_allclose(p_rod_right, slider_pin, atol=1.0e-6)
    np.testing.assert_allclose(p_slider, slider_pin, atol=1.0e-6)
    np.testing.assert_allclose(np.linalg.norm(p_rod_right - p_rod_left), rod_length, atol=1.0e-6)


def test_elastic_multiple_attachment_samples(test, device):
    """Check that three joints reuse their coincident basis samples and close at the deformed endpoints."""
    basis = newton.ModalBasis(
        sample_points=[[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        sample_phi=[
            [[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.25, 0.0]],
            [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        mode_mass=[1.0, 1.0],
    )

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_q=[0.2, -0.4],
        modal_basis=basis,
    )
    j_left = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(-0.6, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
    )
    j_mid = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.1, 0.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )
    j_right = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.6, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()

    endpoint_samples = model.elastic_endpoint_sample.numpy()
    child_endpoints = model.joint_child_elastic_endpoint.numpy()
    sample_ids = [int(endpoint_samples[int(child_endpoints[j])]) for j in (j_left, j_mid, j_right)]
    test.assertEqual(sample_ids, [0, 1, 2])

    np.testing.assert_allclose(joint_endpoint_world(model, state, j_left, "child"), [-0.6, 0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(joint_endpoint_world(model, state, j_mid, "child"), [0.0, 0.1, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(joint_endpoint_world(model, state, j_right, "child"), [0.6, 0.0, 0.0], atol=1.0e-6)


def test_matrix_rom_example(test, device):
    """Run the matrix ROM example headless and check its final state."""
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = MatrixROMExample(viewer, None)
        for _ in range(100):
            example.step()
            example.render()
        example.test_final()


def test_dipper_arm_example(test, device):
    """Run the dipper arm example headless and check its final state."""
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        # The kinematic actuator barrel keeps its free root joint, since kinematic bodies must be
        # roots; its base revolute duplicates it only to record the pivot.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Adding a REVOLUTE joint", category=UserWarning)
            example = DipperExample(viewer, None)
        for _ in range(120):
            example.step()
            example.render()
        example.test_final()


def test_gravity_coupling_example(test, device):
    """Run the gravity coupling example headless and check its final state."""
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = GravityCouplingExample(viewer, None)
        for _ in range(120):
            example.step()
            example.render()
        example.test_final()


def test_base_excitation_example(test, device):
    """Run the base excitation example headless and check its final state."""
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = BaseExcitationExample(viewer, None)
        for _ in range(120):
            example.step()
            example.render()
        example.test_final()


def test_base_rotation_example(test, device):
    """Run the base rotation example headless and check its final state."""
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = BaseRotationExample(viewer, None)
        for _ in range(120):
            example.step()
            example.render()
        example.test_final()


def test_clamp_moment_example(test, device):
    """Run the clamp moment example headless and check its final state."""
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = ClampMomentExample(viewer, None)
        for _ in range(150):
            example.step()
            example.render()
        example.test_final()


def test_centrifugal_example(test, device):
    """Run the centrifugal example headless and check its final state."""
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = CentrifugalExample(viewer, None)
        for _ in range(150):
            example.step()
            example.render()
        example.test_final()


def test_coriolis_example(test, device):
    """Run the Coriolis example headless and check its final state."""
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = CoriolisExample(viewer, None)
        for _ in range(180):
            example.step()
            example.render()
        example.test_final()


def test_frame_coupling_example(test, device):
    """Run the frame coupling example headless and check its final state."""
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = FrameCouplingExample(viewer, None)
        for _ in range(180):
            example.step()
            example.render()
        example.test_final()


def test_angular_frame_coupling_example(test, device):
    """Run the angular frame coupling example headless and check its final state."""
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = AngularFrameCouplingExample(viewer, None)
        for _ in range(180):
            example.step()
            example.render()
        example.test_final()


def _run_reduced_elastic_contact_example(example_cls, frame_count: int, device):
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        with _legacy_alm_allowed():
            example = example_cls(viewer, None)
        for _ in range(frame_count):
            example.step()
            example.render()
        example.test_final()


def test_elastic_wall_contact_example(test, device):
    """Run the wall contact example headless and check its final state."""
    _run_reduced_elastic_contact_example(WallContactExample, 90, device)


def test_elastic_gripper_contact_example(test, device):
    """Run the gripper contact example headless and check its final state."""
    _run_reduced_elastic_contact_example(GripperContactExample, 120, device)


def test_elastic_scraper_contact_example(test, device):
    """Run the scraper contact example headless and check its final state."""
    _run_reduced_elastic_contact_example(ScraperContactExample, 120, device)


def test_elastic_chair_stick_slip_example(test, device):
    """Run the chair stick-slip example when its asset is cached and check its final state."""
    asset_dir = chair_asset_cache_dir()
    if not all((asset_dir / filename).is_file() for filename in (CHAIR_GLTF_NAME, CHAIR_BIN_NAME)):
        test.skipTest("Poly Haven chair asset is not cached; the offline unit suite does not download it")
    _run_reduced_elastic_contact_example(ChairStickSlipExample, 240, device)


def _lattice_craig_bampton(direct_interface: bool, fixed_mode_count: int = 8):
    """Reduce a 27-node spring lattice with one interface at its top-face centroid node.

    The interface reference point coincides with the physical node at the top-face
    centroid. A distributing (RBE3-style) interface constrains the least-squares
    rigid fit of the top-face nodes, so fixed-interface modes still move that node.
    A direct (RBE2-style) interface constrains the node's own translation and the
    fitted rotation, so the node's recovered motion equals the interface motion.

    Returns:
        ``(points, interface_position, center_node, reduced_mass, reduced_stiffness,
        recovery)``, with every node a recovery sample.
    """
    points, mass, stiffness = _spring_lattice((3, 3, 3), 0.1, 1000.0, 0.01)
    node_count = points.shape[0]

    def skew(value):
        x, y, z = value
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)

    def rigid_rows(nodes, center):
        return np.vstack([np.column_stack((np.eye(3), -skew(node - center))) for node in nodes])

    top = np.nonzero(np.isclose(points[:, 2], points[:, 2].max()))[0]
    interface_position = points[top].mean(axis=0)
    center_node = int(top[np.argmin(np.linalg.norm(points[top] - interface_position, axis=1))])
    fit_rows = rigid_rows(points[top], interface_position)
    fit = np.linalg.solve(fit_rows.T @ fit_rows, fit_rows.T)
    constraint = np.zeros((6, 3 * node_count), dtype=np.float64)
    for j, node in enumerate(top):
        constraint[:, 3 * node : 3 * node + 3] = fit[:, 3 * j : 3 * j + 3]
    if direct_interface:
        constraint[:3] = 0.0
        constraint[:3, 3 * center_node : 3 * center_node + 3] = np.eye(3)
    rigid_all = rigid_rows(points, interface_position)
    np.testing.assert_allclose(constraint @ rigid_all, np.eye(6), atol=1.0e-12)

    # Fixed-interface modes span the null space of the interface constraint.
    _, _, vt = np.linalg.svd(constraint)
    null = vt[6:].T
    null_mass = null.T @ mass @ null
    null_stiffness = null.T @ stiffness @ null
    chol = np.linalg.cholesky(null_mass)
    standard = np.linalg.solve(chol, np.linalg.solve(chol, null_stiffness).T).T
    _, vectors = np.linalg.eigh(0.5 * (standard + standard.T))
    fixed_modes = null @ np.linalg.solve(chol.T, vectors[:, :fixed_mode_count])

    transform = np.hstack((rigid_all, fixed_modes))
    return (
        points,
        interface_position,
        center_node,
        transform.T @ mass @ transform,
        transform.T @ stiffness @ transform,
        transform,
    )


def test_elastic_endpoint_uses_attachment_frame(test, device):
    """Check that a joint at a distributing interface uses the frame values but keeps the material sample."""
    points, interface_position, center_node, reduced_mass, reduced_stiffness, recovery = _lattice_craig_bampton(
        direct_interface=False
    )
    generator = newton.ModalGeneratorCraigBampton(
        interface_positions=[interface_position],
        interface_names=["mount"],
        mass_matrix=reduced_mass,
        stiffness_matrix=reduced_stiffness,
        sample_points=points,
        recovery_matrix=recovery,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Craig-Bampton interfaces", category=UserWarning)
        basis = generator.build()

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(modal_basis=basis)
    xform = wp.transform(wp.vec3(*interface_position), wp.quat_identity())
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Adding a FIXED joint.*", category=UserWarning)
        joint = builder.add_joint_fixed(parent=-1, child=body, parent_xform=xform, child_xform=xform)
    builder.color()
    model = builder.finalize(device=device)

    endpoint = int(model.joint_child_elastic_endpoint.numpy()[joint])
    mode_count = basis.mode_count
    max_modes = int(model.elastic_max_mode_count)
    endpoint_phi = model.elastic_endpoint_phi.numpy()[endpoint * max_modes : endpoint * max_modes + mode_count]
    endpoint_psi = model.elastic_endpoint_psi.numpy()[endpoint * max_modes : endpoint * max_modes + mode_count]
    np.testing.assert_allclose(endpoint_phi, basis.frame_phi[0], atol=1.0e-7)
    np.testing.assert_allclose(endpoint_psi, basis.frame_psi[0], atol=1.0e-7)
    # The endpoint keeps the coincident material sample for render and contact bookkeeping.
    test.assertEqual(int(model.elastic_endpoint_sample.numpy()[endpoint]), center_node)
    test.assertGreater(float(np.max(np.abs(basis.sample_phi[center_node] - basis.frame_phi[0]))), 0.1)


def test_modal_generator_beam_gravity_sag_matches_sampled_mass(test, device):
    """Sag a clamped beam-generator cantilever under gravity like the same shapes with lumped masses."""
    generator = newton.ModalGeneratorBeam(
        length=1.0, half_width_y=0.02, half_width_z=0.02, density=1000.0, young_modulus=2.0e8
    )
    beam = generator.build()
    rigid_mass = generator.density * generator.area * generator.length
    # Midpoint-rule centerline masses plus massless exact root and tip samples, so the clamped
    # endpoint is not inverse-distance interpolated.
    count = 64
    x = np.concatenate((-0.5 + (np.arange(count) + 0.5) / count, [-0.5, 0.5]))
    centerline = np.stack((x, np.zeros_like(x), np.zeros_like(x)), axis=1).astype(np.float32)
    phi, psi = generator._evaluate_modes(centerline)
    sampled = newton.ModalBasis(
        sample_points=centerline,
        sample_phi=phi,
        sample_psi=psi,
        sample_mass=np.concatenate((np.full(count, rigid_mass / count), [0.0, 0.0])),
        mode_mass=beam.mode_mass,
        mode_stiffness=beam.mode_stiffness,
    )
    test.assertIsNotNone(beam.mode_coupling_linear)
    beam_s = float(beam.mode_coupling_linear[0, 2])
    sampled_s = float(sampled.mode_coupling_linear[0, 2])
    test.assertLess(abs(beam_s - sampled_s), 2.0e-2 * abs(sampled_s))

    stiffness = float(beam.mode_stiffness[0])
    damping = [2.0 * 0.5 * math.sqrt(stiffness * float(beam.mode_mass[0]))]
    inertia = wp.mat33(*np.diag([1.0e-4, rigid_mass / 12.0, rigid_mass / 12.0]).ravel())

    def sag(basis):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = builder.add_link_elastic(
            xform=wp.transform(wp.vec3(0.5, 0.0, 1.0), wp.quat_identity()),
            mass=rigid_mass,
            inertia=inertia,
            modal_basis=basis,
            mode_damping=damping,
        )
        joint = builder.add_joint_fixed(
            parent=-1,
            child=body,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([builder.body_elastic_joint[body], joint])
        builder.color()
        model = builder.finalize(device=device)
        state_0, state_1, control = model.state(), model.state(), model.control()
        solver = newton.solvers.SolverVBD(model, iterations=16)
        q_start = int(model.joint_q_start.numpy()[int(model.elastic_joint.numpy()[0])]) + 7
        for _ in range(300):
            solver.step(state_0, state_1, control, None, 1.0 / 600.0)
            state_0, state_1 = state_1, state_0
        return float(state_0.joint_q.numpy()[q_start])

    beam_q = sag(beam)
    sampled_q = sag(sampled)
    static_q = -9.81 * sampled_s / stiffness
    test.assertLess(sampled_q, 0.0)
    test.assertLess(abs(beam_q - sampled_q), 5.0e-2 * abs(sampled_q))
    test.assertLess(abs(beam_q - static_q), 5.0e-2 * abs(static_q))


class TestReducedElasticBody(unittest.TestCase):
    def test_reduced_inertia_pairs_match_brute_force(self):
        """Match the GEMM pair tensor and derived integrals to brute-force sums for lumped and consistent mass."""
        rng = np.random.default_rng(7)
        points = rng.uniform(-1.0, 1.0, size=(7, 3))
        phi = rng.uniform(-1.0, 1.0, size=(7, 3, 3))
        mass = rng.uniform(0.5, 1.5, size=7)
        factor = rng.uniform(-0.3, 0.3, size=(21, 21))
        consistent = factor @ factor.T + np.eye(21)
        cases = {
            "lumped": (None, mass, mass[:, None, None] * phi),
            "consistent": (
                (consistent @ phi.transpose(0, 2, 1).reshape((21, 3))).reshape((7, 3, 3)).transpose(0, 2, 1),
                None,
                None,
            ),
        }
        for name, (mass_phi, lumped_mass, lumped_mass_phi) in cases.items():
            with self.subTest(name):
                weighted = lumped_mass_phi if mass_phi is None else mass_phi
                # Two-sample chunks exercise the chunked accumulation.
                result = _reduced_inertia_from_pairs(points, phi, mass_phi, lumped_mass, chunk_bytes=2 * 8 * 9)
                expected_pairs = np.einsum("sic,sjd->ijcd", phi, weighted)
                expected_coriolis = np.cross(phi[:, None, :, :], weighted[:, :, None, :]).sum(axis=0)
                np.testing.assert_allclose(result.mode_coupling_quadratic, expected_pairs, rtol=1.0e-6, atol=1.0e-6)
                np.testing.assert_allclose(result.mode_coupling_coriolis, expected_coriolis, rtol=1.0e-6, atol=1.0e-6)
                np.testing.assert_allclose(
                    result.mode_mass_matrix, np.einsum("ijcc->ij", expected_pairs), rtol=1.0e-6, atol=1.0e-6
                )
                np.testing.assert_allclose(result.mode_coupling_linear, weighted.sum(axis=0), rtol=1.0e-6, atol=1.0e-6)
                np.testing.assert_allclose(
                    result.mode_coupling_angular,
                    np.cross(weighted, points[:, None, :]).sum(axis=0),
                    rtol=1.0e-6,
                    atol=1.0e-6,
                )
                np.testing.assert_allclose(
                    result.mode_coupling_centrifugal,
                    np.einsum("smc,sd->mcd", weighted, points),
                    rtol=1.0e-6,
                    atol=1.0e-6,
                )
                if name == "lumped":
                    np.testing.assert_allclose(
                        expected_pairs, expected_pairs.transpose(1, 0, 3, 2), rtol=1.0e-12, atol=1.0e-12
                    )
                    com = mass @ points / mass.sum()
                    offsets = points - com
                    inertia = np.einsum("s,sk,sk->", mass, offsets, offsets) * np.eye(3) - np.einsum(
                        "s,sa,sb->ab", mass, offsets, offsets
                    )
                    self.assertAlmostEqual(result.rigid_mass, float(mass.sum()), places=12)
                    np.testing.assert_allclose(result.rigid_com, com, rtol=1.0e-6, atol=1.0e-6)
                    np.testing.assert_allclose(result.rigid_inertia, inertia, rtol=1.0e-6, atol=1.0e-6)
                else:
                    # A consistent mass matrix breaks the lumped P_ji = P_ij^T symmetry.
                    self.assertGreater(
                        float(np.max(np.abs(expected_pairs - expected_pairs.transpose(1, 0, 3, 2)))), 1.0e-3
                    )
                    self.assertIsNone(result.rigid_mass)

    def test_reduced_inertia_pairs_bounded_memory(self):
        """Form the lumped mode-pair integrals for 2000 samples and 256 modes without a quartic temporary."""
        rng = np.random.default_rng(3)
        sample_count, mode_count = 2000, 256
        points = rng.uniform(-1.0, 1.0, size=(sample_count, 3)).astype(np.float32)
        phi = rng.uniform(-1.0, 1.0, size=(sample_count, mode_count, 3)).astype(np.float32)
        mass = np.full(sample_count, 1.0 / sample_count, dtype=np.float32)

        tracemalloc.start()
        try:
            basis = newton.ModalBasis(sample_points=points, sample_phi=phi, sample_mass=mass)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        # The former [sample_count, mode_count, mode_count, 3] float64 temporary alone needs 3.1 GB.
        self.assertLess(peak, 200 * 1024 * 1024)
        self.assertEqual(basis.mode_coupling_quadratic.shape, (mode_count, mode_count, 3, 3))
        self.assertEqual(basis.mode_mass_matrix.shape, (mode_count, mode_count))
        np.testing.assert_allclose(basis.mode_mass, np.diag(basis.mode_mass_matrix), atol=0.0)

    def test_modal_basis_mode_mass_matrix_coupled_modes(self):
        """Expose the full sampled modal mass matrix and warn when an explicit diagonal discards its coupling."""
        points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        phi = [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            basis = newton.ModalBasis(sample_points=points, sample_phi=phi, sample_mass=[1.0, 1.0])
            clone = basis.copy()
        np.testing.assert_allclose(basis.mode_mass_matrix, [[1.0, 1.0], [1.0, 2.0]], atol=1.0e-7)
        np.testing.assert_allclose(basis.mode_mass, [1.0, 2.0], atol=1.0e-7)
        self.assertAlmostEqual(basis.mode_mass_coupling_ratio, 1.0 / math.sqrt(2.0), places=6)
        np.testing.assert_allclose(clone.mode_mass_matrix, basis.mode_mass_matrix, atol=0.0)
        np.testing.assert_allclose(clone.mode_coupling_quadratic, basis.mode_coupling_quadratic, atol=0.0)
        self.assertEqual(clone.rigid_mass, basis.rigid_mass)

        with self.assertWarnsRegex(UserWarning, "not mass-orthogonal"):
            explicit = newton.ModalBasis(
                sample_points=points, sample_phi=phi, sample_mass=[1.0, 1.0], mode_mass=[1.0, 2.0]
            )
        np.testing.assert_allclose(explicit.mode_mass_matrix, np.diag([1.0, 2.0]), atol=0.0)
        self.assertAlmostEqual(explicit.mode_mass_coupling_ratio, 1.0 / math.sqrt(2.0), places=6)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            newton.ModalBasis(
                sample_points=points,
                sample_phi=phi,
                sample_mass=[1.0, 1.0],
                mode_mass=[1.0, 2.0],
                mode_coupling_tolerance=0.8,
            )
        with self.assertRaises(ValueError):
            newton.ModalBasis(
                sample_points=points, sample_phi=phi, mode_mass=[1.0, 3.0], mode_mass_matrix=[[1.0, 1.0], [1.0, 2.0]]
            )

    def test_modal_basis_lumped_coriolis_matches_legacy_cross(self):
        """Match the lumped Coriolis integral to the brute-force cross-product sum for 500 samples and 32 modes."""
        rng = np.random.default_rng(11)
        sample_count, mode_count = 500, 32
        points = rng.uniform(-1.0, 1.0, size=(sample_count, 3)).astype(np.float32)
        phi = rng.uniform(-1.0, 1.0, size=(sample_count, mode_count, 3)).astype(np.float32)
        mass = rng.uniform(0.5, 1.5, size=sample_count).astype(np.float32) / sample_count
        basis = newton.ModalBasis(sample_points=points, sample_phi=phi, sample_mass=mass)

        phi64 = phi.astype(np.float64)
        expected = np.einsum(
            "s,sijc->ijc", mass.astype(np.float64), np.cross(phi64[:, None, :, :], phi64[:, :, None, :])
        )
        np.testing.assert_allclose(basis.mode_coupling_coriolis, expected, rtol=1.0e-5, atol=1.0e-6)
        np.testing.assert_allclose(
            basis.mode_coupling_coriolis, -basis.mode_coupling_coriolis.transpose(1, 0, 2), atol=1.0e-6
        )

    def test_modal_generator_craig_bampton_sample_mass(self):
        """Integrate lumped pair terms over Craig-Bampton recovery samples while keeping exact reduced-matrix values."""
        interface_positions, mass, stiffness, _, sample_points, recovery, _ = _build_craig_bampton_test_data()
        sample_mass = np.linspace(0.5, 1.0, sample_points.shape[0])
        frozen = newton.ModalGeneratorCraigBampton(
            interface_positions=interface_positions,
            mass_matrix=mass,
            stiffness_matrix=stiffness,
            sample_points=sample_points,
            recovery_matrix=recovery,
        ).build()
        generator = newton.ModalGeneratorCraigBampton(
            interface_positions=interface_positions,
            mass_matrix=mass,
            stiffness_matrix=stiffness,
            sample_points=sample_points,
            recovery_matrix=recovery,
            sample_mass=sample_mass,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            basis = generator.build()

        self.assertIsNone(frozen.mode_coupling_centrifugal)
        self.assertIsNone(frozen.mode_coupling_coriolis)
        self.assertIsNone(frozen.mode_coupling_quadratic)
        self.assertIsNotNone(basis.mode_coupling_centrifugal)
        self.assertIsNotNone(basis.mode_coupling_coriolis)
        self.assertIsNotNone(basis.mode_coupling_quadratic)

        np.testing.assert_allclose(basis.mode_mass, frozen.mode_mass, atol=0.0)
        np.testing.assert_allclose(basis.mode_mass, 1.0, atol=1.0e-6)
        np.testing.assert_allclose(basis.mode_mass_matrix, np.eye(basis.mode_count), atol=1.0e-6)
        np.testing.assert_allclose(basis.mode_coupling_linear, frozen.mode_coupling_linear, atol=0.0)
        np.testing.assert_allclose(basis.mode_coupling_angular, frozen.mode_coupling_angular, atol=0.0)
        for current in (basis, frozen):
            self.assertAlmostEqual(current.rigid_mass, generator.mass, places=6)
            np.testing.assert_allclose(current.rigid_com, generator.com, atol=1.0e-6)
            np.testing.assert_allclose(current.rigid_inertia, generator.inertia, atol=1.0e-6)

        # Appended interface samples carry zero mass, so the pair tensor uses the recovery samples only.
        sample_count = sample_points.shape[0]
        np.testing.assert_allclose(basis.sample_mass[sample_count:], 0.0, atol=0.0)
        phi = basis.sample_phi[:sample_count].astype(np.float64)
        expected_pairs = np.einsum("s,sic,sjd->ijcd", sample_mass, phi, phi)
        np.testing.assert_allclose(basis.mode_coupling_quadratic, expected_pairs, rtol=1.0e-5, atol=1.0e-6)

    def test_modal_generator_fem_consistent_mass_pairs(self):
        """Integrate FEM mode-pair inertia and rigid properties from the consistent mass matrix."""
        nodes = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        bar = np.array([[2.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 2.0]]) / 12.0
        mass = np.kron(bar, np.eye(3))
        stiffness = np.kron(np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]]) * 50.0, np.eye(3))
        generator = newton.ModalGeneratorFEM(
            node_positions=nodes,
            mass_matrix=mass,
            stiffness_matrix=stiffness,
            sample_node_indices=[0, 1, 2],
            fixed_node_indices=[0],
            mode_count=4,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            basis = generator.build()

        modes = generator.modal_matrix
        nodal_phi = modes.reshape((3, 3, -1)).transpose(0, 2, 1)
        mass_phi = (mass @ modes).reshape((3, 3, -1)).transpose(0, 2, 1)
        expected_pairs = np.einsum("sic,sjd->ijcd", nodal_phi, mass_phi)
        expected_coriolis = np.cross(nodal_phi[:, None, :, :], mass_phi[:, :, None, :]).sum(axis=0)
        np.testing.assert_allclose(basis.mode_coupling_quadratic, expected_pairs, rtol=1.0e-5, atol=1.0e-6)
        np.testing.assert_allclose(basis.mode_coupling_coriolis, expected_coriolis, rtol=1.0e-5, atol=1.0e-6)
        np.testing.assert_allclose(basis.mode_mass_matrix, np.eye(4), atol=1.0e-6)
        self.assertLess(basis.mode_mass_coupling_ratio, 1.0e-6)

        node_mass = bar.sum(axis=1)
        self.assertAlmostEqual(basis.rigid_mass, float(node_mass.sum()), places=6)
        np.testing.assert_allclose(basis.rigid_com, node_mass @ nodes / node_mass.sum(), atol=1.0e-6)
        consistent_ixx = float(nodes[:, 0] @ bar @ nodes[:, 0]) - basis.rigid_mass * float(basis.rigid_com[0]) ** 2
        np.testing.assert_allclose(basis.rigid_inertia, np.diag([0.0, consistent_ixx, consistent_ixx]), atol=1.0e-6)

    def test_modal_generator_craig_bampton_distributing_interface(self):
        """Keep recovered material samples at a coincident distributing interface and expose the frame."""
        points, interface_position, center_node, reduced_mass, reduced_stiffness, recovery = _lattice_craig_bampton(
            direct_interface=False
        )
        generator = newton.ModalGeneratorCraigBampton(
            interface_positions=[interface_position],
            mass_matrix=reduced_mass,
            stiffness_matrix=reduced_stiffness,
            sample_points=points,
            recovery_matrix=recovery,
        )
        with self.assertWarnsRegex(UserWarning, "distributing interfaces"):
            basis = generator.build()

        node_count = points.shape[0]
        modes = generator.modal_matrix
        expected_phi = (recovery @ modes).reshape((node_count, 3, -1)).transpose(0, 2, 1)
        np.testing.assert_allclose(basis.sample_phi[:node_count], expected_phi, atol=1.0e-6)
        self.assertEqual(basis.sample_count, node_count)
        self.assertEqual(generator.interface_sample_indices["interface_0"], center_node)
        self.assertGreater(
            generator.interface_recovery_deviation["interface_0"], generator.interface_recovery_tolerance
        )

        self.assertEqual(basis.frame_count, 1)
        self.assertEqual(basis.frame_labels, ["interface_0"])
        np.testing.assert_allclose(basis.frame_points[0], interface_position, atol=1.0e-7)
        np.testing.assert_allclose(basis.frame_phi[0], modes[0:3].T, atol=1.0e-6)
        np.testing.assert_allclose(basis.frame_psi[0], modes[3:6].T, atol=1.0e-6)
        self.assertEqual(basis.find_frame(interface_position), 0)
        self.assertEqual(basis.find_frame(interface_position + 1.0e-3), -1)
        self.assertGreater(float(np.max(np.abs(expected_phi[center_node] - modes[0:3].T))), 0.1)
        self.assertAlmostEqual(basis.rigid_mass, node_count * 0.01, places=6)
        self.assertEqual(basis.copy().frame_count, 1)

    def test_modal_generator_craig_bampton_direct_interface(self):
        """Reuse a coincident recovery sample of a direct interface and take the exact interface rotation."""
        points, interface_position, center_node, reduced_mass, reduced_stiffness, recovery = _lattice_craig_bampton(
            direct_interface=True
        )
        generator = newton.ModalGeneratorCraigBampton(
            interface_positions=[interface_position],
            mass_matrix=reduced_mass,
            stiffness_matrix=reduced_stiffness,
            sample_points=points,
            recovery_matrix=recovery,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            basis = generator.build()

        node_count = points.shape[0]
        modes = generator.modal_matrix
        expected_phi = (recovery @ modes).reshape((node_count, 3, -1)).transpose(0, 2, 1)
        np.testing.assert_allclose(basis.sample_phi[:node_count], expected_phi, atol=1.0e-6)
        self.assertEqual(basis.sample_count, node_count)
        self.assertEqual(generator.interface_sample_indices["interface_0"], center_node)
        self.assertLess(generator.interface_recovery_deviation["interface_0"], generator.interface_recovery_tolerance)
        np.testing.assert_allclose(basis.sample_phi[center_node], modes[0:3].T, atol=1.0e-6)
        np.testing.assert_allclose(basis.sample_psi[center_node], modes[3:6].T, atol=1.0e-6)
        np.testing.assert_allclose(basis.frame_phi[0], modes[0:3].T, atol=1.0e-6)
        np.testing.assert_allclose(basis.frame_psi[0], modes[3:6].T, atol=1.0e-6)

    def test_modal_basis_attachment_frames(self):
        """Store massless attachment frames that never enter interpolation or inertia integrals."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        phi = np.array([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]], dtype=np.float32)
        basis = newton.ModalBasis(sample_points=points, sample_phi=phi, sample_mass=[1.0, 1.0])
        self.assertEqual(basis.frame_count, 0)
        self.assertEqual(basis.frame_phi.shape, (0, 1, 3))
        coupling = basis.mode_coupling_linear.copy()

        index = basis.add_frame([0.5, 0.0, 0.0], [[0.0, 3.0, 0.0]], [[1.0, 0.0, 0.0]], label="pad")
        self.assertEqual(index, 0)
        self.assertEqual(basis.find_frame([0.5, 0.0, 0.0]), 0)
        self.assertEqual(basis.find_frame([0.6, 0.0, 0.0]), -1)
        self.assertEqual(basis.frame_labels, ["pad"])
        np.testing.assert_allclose(basis.evaluate([0.5, 0.0, 0.0])[0], [[0.0, 0.0, 0.5]], atol=1.0e-6)
        np.testing.assert_allclose(basis.mode_coupling_linear, coupling, atol=0.0)
        self.assertEqual(basis.sample_count, 2)

        copied = basis.copy()
        copied.add_frame([0.0, 1.0, 0.0], [[0.0, 0.0, 0.0]])
        self.assertEqual(basis.frame_count, 1)
        self.assertEqual(copied.frame_count, 2)
        np.testing.assert_allclose(copied.frame_psi[1], 0.0, atol=0.0)
        with self.assertRaises(ValueError):
            basis.add_frame([0.0, 0.0, 0.0], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            newton.ModalBasis(sample_points=points, sample_phi=phi, frame_points=[[0.0, 0.0, 0.0]])

    def test_modal_basis_mass_orthonormalize(self):
        """Transform coupled modes to unit modal mass without changing the physical displacement field."""
        rng = np.random.default_rng(7)
        points = rng.uniform(-0.5, 0.5, size=(12, 3)).astype(np.float32)
        phi = rng.normal(size=(12, 3, 3)).astype(np.float32)
        sample_mass = rng.uniform(0.5, 1.5, size=12).astype(np.float32)
        basis = newton.ModalBasis(sample_points=points, sample_phi=phi, sample_mass=sample_mass)
        basis.add_frame([0.0, 0.0, 0.0], phi[0], phi[1], label="mount")
        self.assertGreater(basis.mode_mass_coupling_ratio, 0.1)

        with self.assertWarnsRegex(UserWarning, "rescales the modal coordinates"):
            unit = basis.mass_orthonormalize()

        mass64 = sample_mass.astype(np.float64)
        phi_unit = unit.sample_phi[:12].astype(np.float64)
        np.testing.assert_allclose(np.einsum("s,sic,sjc->ij", mass64, phi_unit, phi_unit), np.eye(3), atol=1.0e-6)
        np.testing.assert_allclose(unit.mode_mass, 1.0, atol=0.0)
        np.testing.assert_allclose(unit.mode_mass_matrix, np.eye(3), atol=0.0)
        np.testing.assert_allclose(np.einsum("ijcc->ij", unit.mode_coupling_quadratic), np.eye(3), atol=1.0e-5)
        self.assertLess(unit.mode_mass_coupling_ratio, 1.0e-5)

        factor = np.linalg.cholesky(basis.mode_mass_matrix.astype(np.float64))
        q = rng.normal(size=3)
        q_unit = factor.T @ q
        np.testing.assert_allclose(
            np.einsum("smc,m->sc", unit.sample_phi.astype(np.float64), q_unit),
            np.einsum("smc,m->sc", phi.astype(np.float64), q),
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            np.einsum("mc,m->c", unit.frame_phi[0].astype(np.float64), q_unit),
            np.einsum("mc,m->c", phi[0].astype(np.float64), q),
            atol=1.0e-5,
        )
        np.testing.assert_allclose(unit.mode_coupling_linear.T @ q_unit, basis.mode_coupling_linear.T @ q, atol=1.0e-5)
        self.assertEqual(unit.rigid_mass, basis.rigid_mass)

        stiff = newton.ModalBasis(
            sample_points=points, sample_phi=phi, sample_mass=sample_mass, mode_stiffness=[1.0, 2.0, 3.0]
        )
        with self.assertRaisesRegex(ValueError, "non-diagonal"):
            stiff.mass_orthonormalize()

        diagonal = newton.ModalBasis(
            sample_points=points[:1], sample_phi=phi[:1, :1], mode_mass=[4.0], mode_stiffness=[8.0]
        )
        with self.assertWarnsRegex(UserWarning, "rescales"):
            rescaled = diagonal.mass_orthonormalize()
        np.testing.assert_allclose(rescaled.mode_stiffness, [2.0], rtol=1.0e-6)
        np.testing.assert_allclose(rescaled.sample_phi, phi[:1, :1] / 2.0, rtol=1.0e-6)

    def test_modal_generator_beam_rigid_properties(self):
        """Integrate beam rigid properties and inertia couplings over the volume, not the sample points."""
        generator = newton.ModalGeneratorBeam(
            length=1.2,
            half_width_y=0.03,
            half_width_z=0.05,
            density=800.0,
            mode_specs=[
                {"type": "bending_z"},
                {"type": "bending_z", "order": 2},
                {"type": "bending_y"},
                {"type": "torsion"},
                {"type": "axial", "boundary": "fixed-free"},
            ],
        )
        surface = np.array([[-0.6, 0.03, 0.05], [0.6, -0.03, 0.05], [0.0, 0.03, -0.05]], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            basis = generator.build(sample_points=surface)
        self.assertIsNone(basis.sample_mass)

        density, area, length = generator.density, generator.area, generator.length
        i_y, i_z = generator.area_moment_y, generator.area_moment_z
        self.assertAlmostEqual(basis.rigid_mass, density * area * length, places=5)
        np.testing.assert_allclose(basis.rigid_com, 0.0, atol=0.0)
        expected_inertia = density * np.diag(
            [
                length * (i_y + i_z),
                area * length**3 / 12.0 + length * i_y,
                area * length**3 / 12.0 + length * i_z,
            ]
        )
        np.testing.assert_allclose(basis.rigid_inertia, expected_inertia, rtol=1.0e-6)

        # Like the centerline modal mass, the pair integrals neglect bending rotary inertia.
        trace = np.einsum("iicc->i", basis.mode_coupling_quadratic.astype(np.float64))
        np.testing.assert_allclose(trace, basis.mode_mass, rtol=1.0e-5)
        self.assertLess(basis.mode_mass_coupling_ratio, 1.0e-5)

        # The reduced mass matrix of a frame with the basis's rigid properties is positive definite.
        frame_mass = np.zeros((6, 6))
        frame_mass[:3, :3] = basis.rigid_mass * np.eye(3)
        frame_mass[3:, 3:] = basis.rigid_inertia
        cross = np.hstack((basis.mode_coupling_linear, basis.mode_coupling_angular)).T
        reduced_mass = np.block([[frame_mass, cross], [cross.T, basis.mode_mass_matrix]])
        self.assertGreater(float(np.linalg.eigvalsh(reduced_mass.astype(np.float64))[0]), 0.0)

        # Linear coupling matches a fine centerline line-mass integral.
        count = 4000
        x = -0.5 * length + length * (np.arange(count) + 0.5) / count
        line = np.stack((x, np.zeros(count), np.zeros(count)), axis=1)
        line_phi, _ = generator._evaluate_modes(line)
        expected_linear = density * area * length / count * line_phi.sum(axis=0)
        np.testing.assert_allclose(basis.mode_coupling_linear, expected_linear, rtol=1.0e-4, atol=1.0e-6)
        self.assertGreater(abs(float(basis.mode_coupling_linear[0, 2])), 0.1)

        # A section given by its area and second moments uses the equivalent rectangle; the torsion
        # pair integral uses I_y + I_z even when an explicit polar_moment differs.
        section = newton.ModalGeneratorBeam(
            length=1.0,
            mode_specs=[{"type": "torsion"}],
            density=1000.0,
            area=2.0e-3,
            area_moment_y=3.0e-7,
            area_moment_z=1.0e-7,
            polar_moment=2.0e-7,
        ).build()
        np.testing.assert_allclose(
            np.diag(section.rigid_inertia),
            1000.0 * np.array([4.0e-7, 2.0e-3 / 12.0 + 3.0e-7, 2.0e-3 / 12.0 + 1.0e-7]),
            rtol=1.0e-6,
        )
        torsion_trace = float(np.einsum("cc->", section.mode_coupling_quadratic[0, 0]))
        np.testing.assert_allclose(torsion_trace, 1000.0 * 4.0e-7 / 2.0, rtol=1.0e-5)
        np.testing.assert_allclose(section.mode_mass, [1000.0 * 2.0e-7 / 2.0], rtol=1.0e-6)

    def test_add_link_elastic_defaults_inertia_from_basis(self):
        """Default omitted elastic frame mass properties to the modal basis rigid properties."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
        phi = np.zeros((3, 1, 3), dtype=np.float32)
        phi[1, 0, 2] = 1.0
        basis = newton.ModalBasis(sample_points=points, sample_phi=phi, sample_mass=[1.0, 2.0, 1.0])
        builder = newton.ModelBuilder()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            body = builder.add_body_elastic(modal_basis=basis, label="defaults")
            armature_body = builder.add_link_elastic(modal_basis=basis, armature=0.5)
            explicit = builder.add_link_elastic(mass=4.0, modal_basis=basis)
            kinematic = builder.add_link_elastic(mass=0.0, modal_basis=basis)
            unsampled = builder.add_link_elastic(modal_basis=newton.ModalBasis(sample_points=points, sample_phi=phi))

        self.assertAlmostEqual(builder.body_mass[body], basis.rigid_mass, places=6)
        np.testing.assert_allclose(np.asarray(builder.body_com[body]), basis.rigid_com, atol=1.0e-7)
        np.testing.assert_allclose(
            np.asarray(builder.body_inertia[body]).reshape((3, 3)), basis.rigid_inertia, atol=1.0e-7
        )
        np.testing.assert_allclose(
            np.asarray(builder.body_inertia[armature_body]).reshape((3, 3)),
            basis.rigid_inertia + 0.5 * np.eye(3),
            atol=1.0e-7,
        )
        self.assertEqual(builder.body_mass[explicit], 4.0)
        np.testing.assert_allclose(
            np.asarray(builder.body_inertia[explicit]).reshape((3, 3)), basis.rigid_inertia, atol=1.0e-7
        )
        self.assertEqual(builder.body_mass[kinematic], 0.0)
        np.testing.assert_allclose(np.asarray(builder.body_inertia[kinematic]).reshape((3, 3)), 0.0, atol=0.0)
        self.assertEqual(builder.body_mass[unsampled], 0.0)

    def test_add_link_elastic_warns_on_inconsistent_inertia(self):
        """Warn when explicit elastic frame mass properties disagree with the modal basis."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
        phi = np.zeros((3, 1, 3), dtype=np.float32)
        phi[1, 0, 2] = 1.0
        basis = newton.ModalBasis(sample_points=points, sample_phi=phi, sample_mass=[0.25, 0.5, 0.25])
        inertia = wp.mat33(*basis.rigid_inertia.ravel())
        com = wp.vec3(*basis.rigid_com)
        builder = newton.ModelBuilder()

        # A 0.2 kg frame on a 1 kg lumped basis makes the reduced mass indefinite.
        with self.assertWarnsRegex(UserWarning, r"'light'.*mass 0\.2 kg vs basis 1 kg"):
            builder.add_body_elastic(mass=0.2, com=com, inertia=inertia, modal_basis=basis, label="light")
        with self.assertWarnsRegex(UserWarning, "'offset'.*com"):
            builder.add_body_elastic(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), modal_basis=basis, label="offset")
        with self.assertWarnsRegex(UserWarning, "'spun'.*inertia"):
            builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), modal_basis=basis, label="spun")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            builder.add_body_elastic(mass=1.02, com=com, inertia=inertia, modal_basis=basis)
            builder.add_body_elastic(mass=0.2, modal_basis=basis, inertia_tolerance=None)


for device in devices:
    add_function_test(
        TestReducedElasticBody, "test_modal_basis_add_sample", test_modal_basis_add_sample, devices=[device]
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_basis_add_samples_preserves_mass_invariant",
        test_modal_basis_add_samples_preserves_mass_invariant,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_beam_samples",
        test_modal_generator_beam_samples,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_pod_rank_one",
        test_modal_generator_pod_rank_one,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_pod_explicit_zero_modes",
        test_modal_generator_pod_explicit_zero_modes,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_beam_linear_torsion_properties",
        test_modal_generator_beam_linear_torsion_properties,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_fem_matrix_rom",
        test_modal_generator_fem_matrix_rom,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_craig_bampton_interface_modes",
        test_modal_generator_craig_bampton_interface_modes,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_craig_bampton_single_interface",
        test_modal_generator_craig_bampton_single_interface,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_craig_bampton_nonclassical_damping",
        test_modal_generator_craig_bampton_nonclassical_damping,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_beam_cantilever_eigenfunctions",
        test_modal_generator_beam_cantilever_eigenfunctions,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_beam_cantilever_static_compat",
        test_modal_generator_beam_cantilever_static_compat,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_beam_axial_fixed_free_properties",
        test_modal_generator_beam_axial_fixed_free_properties,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_fem_free_free_rigid_modes_removed",
        test_modal_generator_fem_free_free_rigid_modes_removed,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_fem_discard_skips_elastic_modes",
        test_modal_generator_fem_discard_skips_elastic_modes,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_fem_rejects_mechanism",
        test_modal_generator_fem_rejects_mechanism,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_fem_damping_projection",
        test_modal_generator_fem_damping_projection,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_craig_bampton_damping_projection_error",
        test_modal_generator_craig_bampton_damping_projection_error,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_craig_bampton_negative_damping_warns",
        test_modal_generator_craig_bampton_negative_damping_warns,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_basis_lumped_inertia_coupling",
        test_modal_basis_lumped_inertia_coupling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_basis_coupling_unavailable_without_mass",
        test_modal_basis_coupling_unavailable_without_mass,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_basis_coupling_explicit_override",
        test_modal_basis_coupling_explicit_override,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_fem_inertia_coupling",
        test_modal_generator_fem_inertia_coupling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_pod_inertia_coupling",
        test_modal_generator_pod_inertia_coupling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_pod_coupling_matches_lumped",
        test_modal_generator_pod_coupling_matches_lumped,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_basis_copy_preserves_coupling",
        test_modal_basis_copy_preserves_coupling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_basis_mode_mass_from_sample_mass",
        test_modal_basis_mode_mass_from_sample_mass,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_basis_mode_mass_explicit_wins",
        test_modal_basis_mode_mass_explicit_wins,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_mode_coupling_arrays",
        test_elastic_mode_coupling_arrays,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_coriolis_array_padding",
        test_elastic_coriolis_array_padding,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_mode_coupling_arrays_merge",
        test_elastic_mode_coupling_arrays_merge,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_coriolis_array_merge_different_mode_count",
        test_elastic_coriolis_array_merge_different_mode_count,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_mode_pair_arrays_padding",
        test_elastic_mode_pair_arrays_padding,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_gravity_modal_force",
        test_elastic_gravity_modal_force,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_free_fall_no_modal_force",
        test_elastic_free_fall_no_modal_force,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_euler_modal_force",
        test_elastic_euler_modal_force,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_centrifugal_modal_force",
        test_elastic_centrifugal_modal_force,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_coriolis_modal_force",
        test_elastic_coriolis_modal_force,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_frame_coupling_conserves_com",
        test_elastic_frame_coupling_conserves_com,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_frame_coupling_conserves_angular_momentum",
        test_elastic_frame_coupling_conserves_angular_momentum,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_frame_coupling_wrench_matches_reference",
        test_elastic_frame_coupling_wrench_matches_reference,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_reduced_mass_matches_brute_force",
        test_elastic_reduced_mass_matches_brute_force,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_parallel_assembly_matches_serial_reference",
        test_elastic_parallel_assembly_matches_serial_reference,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_spinning_com_is_ballistic",
        test_elastic_spinning_com_is_ballistic,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_free_body_boost_invariance",
        test_elastic_free_body_boost_invariance,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_free_body_origin_offset_invariance",
        test_elastic_free_body_origin_offset_invariance,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_asymmetric_zero_spin_counterexample",
        test_elastic_asymmetric_zero_spin_counterexample,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_random_state_matches_reference",
        test_elastic_random_state_matches_reference,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_large_coordinate_invariance",
        test_elastic_large_coordinate_invariance,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_large_coordinate_ballistic_com",
        test_elastic_large_coordinate_ballistic_com,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_displacement_form_translation_invariance",
        test_elastic_displacement_form_translation_invariance,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_uncoupled_body_matches_base_trajectory",
        test_elastic_uncoupled_body_matches_base_trajectory,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_nonorthogonal_modes_use_full_mass_matrix",
        test_elastic_nonorthogonal_modes_use_full_mass_matrix,
        devices=[device],
    )
    add_function_test(TestReducedElasticBody, "test_elastic_link_layout", test_elastic_link_layout, devices=[device])
    add_function_test(
        TestReducedElasticBody,
        "test_add_joint_elastic_initializes_relative_frame_coordinates",
        test_add_joint_elastic_initializes_relative_frame_coordinates,
        devices=[device],
    )
    add_function_test(TestReducedElasticBody, "test_elastic_fk_sync", test_elastic_fk_sync, devices=[device])
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_fk_ik_com_velocity_round_trip",
        test_elastic_fk_ik_com_velocity_round_trip,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_endpoint_shape_sampling",
        test_elastic_endpoint_shape_sampling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_endpoint_modal_basis_sampling",
        test_elastic_endpoint_modal_basis_sampling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_endpoint_angular_sampling",
        test_elastic_endpoint_angular_sampling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_endpoint_angular_zero_without_beam",
        test_elastic_endpoint_angular_zero_without_beam,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_clamp_moment_reaction",
        test_elastic_clamp_moment_reaction,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_clamp_moment_reaction_parent_side",
        test_elastic_clamp_moment_reaction_parent_side,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_clamp_moment_conserves_angular_momentum",
        test_elastic_clamp_moment_conserves_angular_momentum,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_estimate_sample_psi_rotation_and_translation",
        test_estimate_sample_psi_rotation_and_translation,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_estimate_sample_psi_degenerate_warns",
        test_estimate_sample_psi_degenerate_warns,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_sampled_psi",
        test_modal_generator_sampled_psi,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_pod_derives_psi",
        test_modal_generator_pod_derives_psi,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_fem_derives_psi",
        test_modal_generator_fem_derives_psi,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_basis_shared_by_elastic_bodies",
        test_modal_basis_shared_by_elastic_bodies,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_render_shape_sampling",
        test_elastic_render_shape_sampling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_shape_mesh_sampling",
        test_elastic_shape_mesh_sampling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_shape_box_winding",
        test_elastic_shape_box_winding,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_shape_box_exact_modal_samples",
        test_elastic_shape_box_exact_modal_samples,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_surface_contact_generation",
        test_elastic_surface_contact_generation,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_surface_contact_work_uses_pair_vertex_ranges",
        test_elastic_surface_contact_work_uses_pair_vertex_ranges,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_surface_contact_deterministic_sort_preserves_samples",
        test_elastic_surface_contact_deterministic_sort_preserves_samples,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_surface_contact_reduction_preserves_modal_samples",
        test_elastic_surface_contact_reduction_preserves_modal_samples,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_surface_contact_uses_deformed_vertex",
        test_elastic_surface_contact_uses_deformed_vertex,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_contact_solves_modal_penetration",
        test_vbd_elastic_contact_solves_modal_penetration,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_body_relaxation_is_general_and_validated",
        test_vbd_elastic_body_relaxation_is_general_and_validated,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_compliant_contact_solves_modal_penetration",
        test_vbd_elastic_compliant_contact_solves_modal_penetration,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_compliant_contact_uses_modal_mobility_for_rho_and_forces",
        test_vbd_elastic_compliant_contact_uses_modal_mobility_for_rho_and_forces,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_contact_overflow_assembles_consistent_block",
        test_vbd_elastic_contact_overflow_assembles_consistent_block,
        devices=[device],
        check_output=False,
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_contact_local_mat33_projection_matches_world",
        test_elastic_contact_local_mat33_projection_matches_world,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_strain_visualization_colors",
        test_elastic_strain_visualization_colors,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_rendering_respects_layer_transform_and_visibility",
        test_elastic_rendering_respects_layer_transform_and_visibility,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_torsion_render_shape_sampling",
        test_torsion_render_shape_sampling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_finite_torsion_exemplar_preserves_volume",
        test_finite_torsion_exemplar_preserves_volume,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_finite_torsion_pod_modes_project_exemplar",
        test_finite_torsion_pod_modes_project_exemplar,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_cantilever_tip_load_solution",
        test_cantilever_tip_load_solution,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_cantilever_modal_vibration_solution",
        test_cantilever_modal_vibration_solution,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_modal_implicit_solution_vbd",
        test_elastic_modal_implicit_solution_vbd,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_revolute_uses_elastic_endpoint",
        test_vbd_revolute_uses_elastic_endpoint,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_revolute_constraint_solves_elastic_mode",
        test_vbd_revolute_constraint_solves_elastic_mode,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_fixed_joint_stiffness_pins_penalties",
        test_vbd_fixed_joint_stiffness_pins_penalties,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_modal_force_matches_joint_projection",
        test_vbd_elastic_modal_force_matches_joint_projection,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_joint_assembles_one_coupled_block",
        test_vbd_elastic_joint_assembles_one_coupled_block,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_body_rejects_external_rigid_solver",
        test_vbd_elastic_body_rejects_external_rigid_solver,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_body_block_sparse_joints_scope",
        test_vbd_elastic_body_block_sparse_joints_scope,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_block_solve_falls_back_when_indefinite",
        test_vbd_elastic_block_solve_falls_back_when_indefinite,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_block_solve_supports_representative_widths",
        test_vbd_elastic_block_solve_supports_representative_widths,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_block_width_rejects_oversized_basis",
        test_vbd_elastic_block_width_rejects_oversized_basis,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_block_solve_reads_leading_sub_block",
        test_vbd_elastic_block_solve_reads_leading_sub_block,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_block_solve_buckets_mixed_widths",
        test_vbd_elastic_block_solve_buckets_mixed_widths,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_block_capacity_steps_at_tiled_limit",
        test_vbd_elastic_block_capacity_steps_at_tiled_limit,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_block_solver_rejects_unknown_mode",
        test_vbd_elastic_block_solver_rejects_unknown_mode,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_block_auto_falls_back_to_dense",
        test_vbd_elastic_block_auto_falls_back_to_dense,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_block_capacity_dense_beyond_tiled_limit",
        test_vbd_elastic_block_capacity_dense_beyond_tiled_limit,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_block_dense_matches_tiled_step",
        test_vbd_elastic_block_dense_matches_tiled_step,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_block_dense_solve_matches_numpy",
        test_vbd_elastic_block_dense_solve_matches_numpy,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_block_dense_solve_falls_back_when_indefinite",
        test_vbd_elastic_block_dense_solve_falls_back_when_indefinite,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_block_solve_applies_update_once",
        test_vbd_elastic_block_solve_applies_update_once,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_reduced_mobility_matches_inverse",
        test_vbd_elastic_reduced_mobility_matches_inverse,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_block_metrics_cover_frame_rows",
        test_vbd_elastic_block_metrics_cover_frame_rows,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_implicit_mass_coupling_requires_spd_block",
        test_vbd_implicit_mass_coupling_requires_spd_block,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_fem_wide_spectrum_keeps_soft_modes",
        test_modal_generator_fem_wide_spectrum_keeps_soft_modes,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_massless_mode_keeps_decoupled_solve_without_warning",
        test_vbd_massless_mode_keeps_decoupled_solve_without_warning,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_craig_bampton_coupled_solve_iteration_invariant",
        test_vbd_craig_bampton_coupled_solve_iteration_invariant,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_craig_bampton_coupled_solve_converges_compliant_alm",
        test_vbd_craig_bampton_coupled_solve_converges_compliant_alm,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_modal_joint_damping_projection",
        test_vbd_elastic_modal_joint_damping_projection,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_angular_constraint_without_linear_stiffness",
        test_vbd_elastic_angular_constraint_without_linear_stiffness,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_rigid_angular_damping_includes_modal_velocity",
        test_vbd_rigid_angular_damping_includes_modal_velocity,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_elastic_angular_projection_uses_exponential_jacobian",
        test_vbd_elastic_angular_projection_uses_exponential_jacobian,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_revolute_drive_projects_into_elastic_modes",
        test_vbd_revolute_drive_projects_into_elastic_modes,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_revolute_limit_projects_into_elastic_modes",
        test_vbd_revolute_limit_projects_into_elastic_modes,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_revolute_coordinate_includes_elastic_endpoint_rotation",
        test_vbd_revolute_coordinate_includes_elastic_endpoint_rotation,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_rod_joint_uses_elastic_endpoint",
        test_vbd_rod_joint_uses_elastic_endpoint,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_fourbar_elastic_coupler_geometry",
        test_fourbar_elastic_coupler_geometry,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_prismatic_rotates_elastic_child",
        test_vbd_prismatic_rotates_elastic_child,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_crank_slider_elastic_analytic_geometry",
        test_crank_slider_elastic_analytic_geometry,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_multiple_attachment_samples",
        test_elastic_multiple_attachment_samples,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_matrix_rom_example",
        test_matrix_rom_example,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_dipper_arm_example",
        test_dipper_arm_example,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_gravity_coupling_example",
        test_gravity_coupling_example,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_base_excitation_example",
        test_base_excitation_example,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_base_rotation_example",
        test_base_rotation_example,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_clamp_moment_example",
        test_clamp_moment_example,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_centrifugal_example",
        test_centrifugal_example,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_coriolis_example",
        test_coriolis_example,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_frame_coupling_example",
        test_frame_coupling_example,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_angular_frame_coupling_example",
        test_angular_frame_coupling_example,
        devices=[device],
    )
    if wp.get_device(device).is_cuda:
        add_function_test(
            TestReducedElasticBody,
            "test_elastic_wall_contact_example",
            test_elastic_wall_contact_example,
            devices=[device],
        )
        add_function_test(
            TestReducedElasticBody,
            "test_elastic_gripper_contact_example",
            test_elastic_gripper_contact_example,
            devices=[device],
        )
        add_function_test(
            TestReducedElasticBody,
            "test_elastic_scraper_contact_example",
            test_elastic_scraper_contact_example,
            devices=[device],
        )
        add_function_test(
            TestReducedElasticBody,
            "test_elastic_chair_stick_slip_example",
            test_elastic_chair_stick_slip_example,
            devices=[device],
        )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_endpoint_uses_attachment_frame",
        test_elastic_endpoint_uses_attachment_frame,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_beam_gravity_sag_matches_sampled_mass",
        test_modal_generator_beam_gravity_sag_matches_sampled_mass,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main()
