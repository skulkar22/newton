# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import builtins
import os
import re
import subprocess
import sys
import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.tests.unittest_utils
from newton._src.solvers.semi_implicit import kernels_particle as semi_implicit_particle_kernels
from newton._src.solvers.solver import _set_module_options_if_changed
from newton._src.solvers.vbd import (
    particle_vbd_kernels,
    reduced_elastic_kernels,
    reduced_elastic_sparse_kernels,
    rigid_sparse_articulation_kernels,
    rigid_vbd_kernels,
    vbd_coupling_kernels,
)
from newton._src.solvers.xpbd import kernels as xpbd_kernels
from newton.tests._reduced_elastic_fixtures import elastic_plank_on_ground, grid_surface_mesh
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, warp_version_at_least

DETERMINISTIC_MODE = wp.DeterministicMode.RUN_TO_RUN
ELASTIC_ROLLOUT_STEPS = 60
ELASTIC_ROLLOUT_SPEED_MAX = 0.05

# Warp 1.19 rejects MuJoCo Warp kernels that read and atomically update one array
# (_accumulate_subtreemass, _qLD_acc) in deterministic mode.
_MUJOCO_WARP_DETERMINISM_UNSUPPORTED = warp_version_at_least("1.19.0.dev0")


def _skip_unsupported_mujoco_determinism(test):
    if _MUJOCO_WARP_DETERMINISM_UNSUPPORTED:
        test.skipTest("MuJoCo Warp does not compile in Warp 1.19 deterministic mode")


def _run_isolated(
    test,
    function_name,
    *args,
    module_name="newton.tests.determinism.test_solver_determinism",
    deterministic_mode=None,
    timeout=600,
):
    # Deterministic kernels and GPU runtime state persist for the life of the process.
    call_args = ", ".join(repr(arg) for arg in args)
    code = f"from {module_name} import {function_name}; {function_name}({call_args})"
    if deterministic_mode is not None:
        # Dependency guards import backends while registering tests.
        # Configure Warp before importing the test module in the fresh child.
        code = (
            "import sys; import warp as wp; assert 'mujoco_warp' not in sys.modules; "
            f"wp.config.deterministic = wp.DeterministicMode.{deterministic_mode.name}; " + code
        )

    env = os.environ.copy()
    env.pop("PYTHONWARNINGS", None)
    warning_args = newton.tests.unittest_utils.get_strict_warning_args()
    if newton.tests.unittest_utils.strict_warnings:
        policy = "import warnings; warnings.filterwarnings('error', module=r'newton(\\.|$)'); "
        for message in newton.tests.unittest_utils.allowed_deprecation_warnings:
            policy += (
                f"warnings.filterwarnings('default', message={re.escape(message)!r}, category=DeprecationWarning); "
            )
        code = policy + code

    result = subprocess.run(
        [sys.executable, *warning_args, "-c", code],
        capture_output=True,
        env=env,
        text=True,
        timeout=timeout,
        check=False,
    )
    test.assertEqual(
        result.returncode,
        0,
        f"{function_name} subprocess failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )
    sys.stderr.write(result.stderr)


def _snapshot(state, fields):
    return {field: np.array(getattr(state, field).numpy(), copy=True) for field in fields}


def _assert_snapshots_equal(first, second):
    if first.keys() != second.keys():
        raise AssertionError(f"snapshot fields differ: {first.keys()} != {second.keys()}")
    for field in first:
        np.testing.assert_array_equal(first[field], second[field], err_msg=f"non-deterministic {field}")


def _build_soft_body(device):
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(-0.2, -0.2, 0.5),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, -0.1),
        dim_x=3,
        dim_y=3,
        dim_z=3,
        cell_x=0.1,
        cell_y=0.1,
        cell_z=0.1,
        density=1.0e3,
        k_mu=5.0e4,
        k_lambda=5.0e4,
        k_damp=1.0e-2,
        fix_left=True,
        particle_radius=0.01,
    )
    builder.color()
    return builder.finalize(device=device)


def _make_particle_solver(model, solver_name):
    if solver_name == "xpbd":
        return newton.solvers.SolverXPBD(model, iterations=5, deterministic=DETERMINISTIC_MODE)
    if solver_name == "semi_implicit":
        return newton.solvers.SolverSemiImplicit(model, deterministic=DETERMINISTIC_MODE)
    if solver_name == "vbd":
        return newton.solvers.SolverVBD(
            model,
            iterations=5,
            particle_enable_self_contact=True,
            particle_enable_tile_solve=False,
            particle_vertex_contact_buffer_size=64,
            particle_edge_contact_buffer_size=64,
            deterministic=DETERMINISTIC_MODE,
        )
    raise ValueError(f"Unsupported particle solver: {solver_name}")


def _run_particle_rollout(device, solver_name):
    model = _build_soft_body(device)
    solver = _make_particle_solver(model, solver_name)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    contacts = newton.CollisionPipeline(model).contacts()

    for _ in range(20):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0

    return _snapshot(state_0, ("particle_q", "particle_qd"))


def _check_particle_determinism(device, solver_name):
    with wp.ScopedDevice(device):
        first = _run_particle_rollout(device, solver_name)
        second = _run_particle_rollout(device, solver_name)
    _assert_snapshots_equal(first, second)


def test_particle_determinism(test, device, solver_name):
    _run_isolated(test, "_check_particle_determinism", str(device), solver_name)


def _build_branching_articulation(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    root = builder.add_link()
    builder.add_shape_box(root, hx=0.15, hy=0.1, hz=0.1)
    joints = [builder.add_joint_revolute(parent=-1, child=root, axis=newton.Axis.Z)]

    for y in (-0.45, -0.15, 0.15, 0.45):
        child = builder.add_link()
        builder.add_shape_box(child, hx=0.2, hy=0.05, hz=0.05)
        joints.append(
            builder.add_joint_revolute(
                parent=root,
                child=child,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.0, y, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
            )
        )

    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    joint_q = np.linspace(-0.2, 0.2, model.joint_coord_count, dtype=np.float32)
    model.joint_q.assign(joint_q)
    return model


def _check_mujoco_sparse_articulation_construction(device):
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        joints = []
        parent = -1
        for _ in range(70):
            child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)))
            joints.append(builder.add_joint_revolute(parent=parent, child=child, axis=newton.Axis.Z))
            parent = child
        builder.add_articulation(joints)

        model = builder.finalize(device=device)
        state_in, state_out = model.state(), model.state()
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        solver = newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_cpu=False,
            disable_contacts=True,
            iterations=1,
            ls_iterations=1,
            deterministic=DETERMINISTIC_MODE,
        )

        solver.step(state_in, state_out, model.control(), None, 1.0 / 240.0)
        if not np.isfinite(state_out.joint_q.numpy()).all():
            raise AssertionError("MuJoCo sparse articulation produced non-finite joint coordinates")


def test_mujoco_sparse_articulation_construction(test, device):
    _skip_unsupported_mujoco_determinism(test)
    _run_isolated(test, "_check_mujoco_sparse_articulation_construction", str(device))


def _make_articulation_solver(model, solver_name):
    if solver_name == "featherstone":
        return newton.solvers.SolverFeatherstone(model, deterministic=DETERMINISTIC_MODE)
    if solver_name == "mujoco":
        return newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_cpu=False,
            disable_contacts=True,
            iterations=10,
            ls_iterations=5,
            deterministic=DETERMINISTIC_MODE,
        )
    raise ValueError(f"Unsupported articulation solver: {solver_name}")


def _run_articulation_rollout(device, solver_name):
    model = _build_branching_articulation(device)
    solver = _make_articulation_solver(model, solver_name)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    control.joint_f.assign(np.linspace(1.0, 5.0, model.joint_dof_count, dtype=np.float32))
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    for _ in range(20):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0

    return _snapshot(state_0, ("body_q", "body_qd", "joint_q", "joint_qd"))


def _check_articulation_determinism(device, solver_name):
    with wp.ScopedDevice(device):
        first = _run_articulation_rollout(device, solver_name)
        second = _run_articulation_rollout(device, solver_name)
    _assert_snapshots_equal(first, second)


def test_articulation_determinism(test, device, solver_name):
    if solver_name == "mujoco":
        _skip_unsupported_mujoco_determinism(test)
    _run_isolated(test, "_check_articulation_determinism", str(device), solver_name)


def _run_elastic_contact_rollout(
    device, deterministic, ground, contact_buffer_size=128, steps=ELASTIC_ROLLOUT_STEPS, **solver_kwargs
):
    # 81 grid vertices on one elastic body: more contacts than modal-assembly threads, so each
    # thread scatters several contacts x modes^2 records. The 1 mm drop impacts after 9 steps.
    model, body = elastic_plank_on_ground(device, ground=ground, count_x=9, count_y=9, height=0.001)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=4,
        rigid_body_contact_buffer_size=contact_buffer_size,
        deterministic=deterministic,
        **solver_kwargs,
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    fields = ("body_q", "body_qd", "joint_q", "joint_qd")
    history = {field: [] for field in fields}
    max_contacts = 0
    for _ in range(steps):
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        max_contacts = max(max_contacts, int(contacts.rigid_contact_count.numpy()[0]))
        solver.step(state_0, state_1, control, contacts, 1.0 / 600.0)
        state_0, state_1 = state_1, state_0
        for field, values in _snapshot(state_0, fields).items():
            history[field].append(values)

    if max_contacts <= 2 * reduced_elastic_kernels._NUM_ELASTIC_CONTACT_THREADS_PER_BODY:
        raise AssertionError(f"elastic plank made only {max_contacts} contacts")
    body_qd = history["body_qd"][-1][body]
    if not np.isfinite(body_qd).all() or np.linalg.norm(body_qd[:3]) > ELASTIC_ROLLOUT_SPEED_MAX:
        raise AssertionError(f"elastic plank did not settle after the impact: body_qd={body_qd}")
    return {field: np.stack(values) for field, values in history.items()}


def _check_elastic_contact_determinism(device, ground):
    with wp.ScopedDevice(device):
        first = _run_elastic_contact_rollout(device, DETERMINISTIC_MODE, ground)
        second = _run_elastic_contact_rollout(device, DETERMINISTIC_MODE, ground)
    _assert_snapshots_equal(first, second)


def test_elastic_contact_determinism(test, device, ground):
    """Repeat a reduced elastic body's drop onto the ground bit for bit in deterministic mode.

    The plank falls onto a static triangle mesh or a ground plane, impacts, and settles. Every
    step's body state and modal coordinates must match between two rollouts. The collision
    pipeline's deterministic option fixes the contact order.
    """
    _run_isolated(test, "_check_elastic_contact_determinism", str(device), ground)


def _check_elastic_contact_default_mode(device, ground):
    with wp.ScopedDevice(device):
        _run_elastic_contact_rollout(device, wp.DeterministicMode.NOT_GUARANTEED, ground)


def test_elastic_contact_default_mode(test, device, ground):
    """Run the elastic drop onto the ground without a deterministic mode."""
    _run_isolated(test, "_check_elastic_contact_default_mode", str(device), ground)


def _check_iteration_metrics_keep_deterministic_solve(device):
    with wp.ScopedDevice(device):
        plain = _run_elastic_contact_rollout(device, DETERMINISTIC_MODE, "mesh")
        recorded = _run_elastic_contact_rollout(
            device, DETERMINISTIC_MODE, "mesh", solver_iteration_metrics=True, elastic_solver_metrics=True
        )
    _assert_snapshots_equal(plain, recorded)


def test_iteration_metrics_keep_deterministic_solve(test, device):
    """Produce bit-identical deterministic rollouts with and without recording iteration metrics."""
    _run_isolated(test, "_check_iteration_metrics_keep_deterministic_solve", str(device))


def _check_elastic_contact_list_overflow_error(device):
    with wp.ScopedDevice(device):
        try:
            _run_elastic_contact_rollout(device, DETERMINISTIC_MODE, "mesh", contact_buffer_size=16, steps=20)
        except ValueError as error:
            if "rigid_body_contact_buffer_size" not in str(error):
                raise AssertionError(f"unexpected message: {error}") from error
            return
    raise AssertionError("an overflowing elastic contact list did not raise")


def test_elastic_contact_list_overflow_error(test, device):
    """Name the needed contact list size when an elastic body overflows it in deterministic mode."""
    _run_isolated(test, "_check_elastic_contact_list_overflow_error", str(device))


def _build_elastic_joined_pair(device, mode_order=12, ground=False):
    # Two beams with 4 * mode_order modes each, joined to each other: the sparse Schur columns of
    # the joint scatter one record per mode of a body from each thread.
    beam = newton.ModalGeneratorBeam
    specs = []
    for order in range(1, mode_order + 1):
        specs += [
            {"type": beam.Mode.BENDING_Y, "boundary": beam.Boundary.PINNED_PINNED, "order": order},
            {"type": beam.Mode.BENDING_Z, "boundary": beam.Boundary.PINNED_PINNED, "order": order},
            {"type": beam.Mode.TORSION, "boundary": beam.Boundary.FIXED_FREE, "order": order},
            {"type": beam.Mode.AXIAL, "boundary": beam.Boundary.FIXED_FREE, "order": order},
        ]
    length = 0.6
    basis = beam(length=length, half_width_y=0.03, half_width_z=0.02, young_modulus=2.0e6, mode_specs=specs).build()
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    if ground:
        # Any shape gives the solver per-body contact lists, which the contact records are sized from.
        builder.add_ground_plane()
    links = [
        builder.add_link_elastic(
            xform=wp.transform(wp.vec3((i + 0.5) * length, 0.0, 0.0), wp.quat_identity()), modal_basis=basis
        )
        for i in range(2)
    ]
    half = wp.transform(wp.vec3(0.5 * length, 0.0, 0.0), wp.quat_identity())
    minus_half = wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity())
    root = builder.add_joint_revolute(parent=-1, child=links[0], axis=(0.0, 0.0, 1.0), child_xform=minus_half)
    builder.add_joint_revolute(
        parent=links[0], child=links[1], axis=(0.0, 0.0, 1.0), parent_xform=half, child_xform=minus_half
    )
    builder.add_articulation([root, root + 1])
    builder.color()
    return builder.finalize(device=device)


def _check_elastic_sparse_schur_records(device):
    with wp.ScopedDevice(device):
        model = _build_elastic_joined_pair(device)
        max_modes = int(model.elastic_max_mode_count)
        if max_modes != 48:
            raise AssertionError(f"expected 48 modes, got {max_modes}")
        solver = newton.solvers.SolverVBD(
            model,
            iterations=1,
            rigid_articulation_solve="block_sparse_joints",
            rigid_articulation_elastic_solve="schur",
            deterministic=DETERMINISTIC_MODE,
        )
        if wp.get_module_options(module=reduced_elastic_sparse_kernels)["deterministic_max_records"] != max_modes:
            raise AssertionError("the sparse elastic module records were not sized from the mode count")
        # Launch only the Schur column kernel, with the solver's own buffers; a full deterministic
        # step compiles the whole sparse articulation module, which takes many minutes on a cold cache.
        layout = solver.rigid_articulation_sparse_layout
        state = model.state()
        wp.launch(
            kernel=reduced_elastic_sparse_kernels.build_elastic_sparse_schur_columns,
            dim=layout.elastic_group_count * max_modes,
            inputs=[
                layout.elastic_bodies,
                layout.elastic_group_owner,
                layout.elastic_group_joint,
                layout.elastic_group_body,
                layout.elastic_group_neighbor_elastic,
                model.body_elastic_index,
                state.body_q,
                model.body_com,
                model.elastic_joint,
                model.elastic_mode_count,
                max_modes,
                model.joint_q_start,
                model.joint_parent,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_parent_elastic_endpoint,
                model.joint_child_elastic_endpoint,
                model.elastic_endpoint_phi,
                model.elastic_endpoint_psi,
                layout.joint_cross_slot,
                state.joint_q,
                solver._rigid_articulation_sparse_values_scalar,
            ],
            outputs=[solver.elastic_body_block_matrix, solver._sparse_elastic_b],
        )
        if not np.isfinite(solver.elastic_body_block_matrix.numpy()).all():
            raise AssertionError("non-finite elastic block entries")


def test_elastic_sparse_schur_records(test, device):
    """Scatter the modal coupling of two 48-mode elastic links joined to each other in deterministic mode."""
    _run_isolated(test, "_check_elastic_sparse_schur_records", str(device))


# Contact list size that puts one 48-mode body's contact assembly launch under the deterministic
# scatter limit and a launch of both joined links above it: 600 contacts per thread x 1464 records.
_JOINED_PAIR_GUARD_BUFFER_SIZE = 19200


def _check_elastic_sparse_coupled_records(device):
    with wp.ScopedDevice(device):
        model = _build_elastic_joined_pair(device, ground=True)
        max_modes = int(model.elastic_max_mode_count)
        solver = newton.solvers.SolverVBD(
            model, iterations=1, rigid_articulation_solve="block_sparse_joints", deterministic=DETERMINISTIC_MODE
        )
        layout = solver.rigid_articulation_sparse_layout
        if layout.elastic_solve != "coupled" or layout.elastic_body_count != 0 or layout.mixed_elastic_body_count != 2:
            raise AssertionError("the default elastic solve must put both joined links in one coupled articulation")
        expected = reduced_elastic_kernels.elastic_contact_deterministic_max_records(
            max_modes, int(solver.body_body_contact_buffer_pre_alloc)
        )
        if (
            expected == 0
            or wp.get_module_options(module=reduced_elastic_kernels)["deterministic_max_records"] != expected
        ):
            raise AssertionError("the elastic contact assembly records were not sized for the coupled solve")
        # The two links sit in different colors, so only the coupled launch holds both: the
        # scatter guard must count that launch. The local solve launches one link per color.
        for mode, backend, raises in (
            ("block_sparse_joints", "coupled", True),
            ("block_sparse_levels", "coupled", True),
            ("block_sparse_joints", "schur", True),
            ("local", "coupled", False),
        ):
            try:
                newton.solvers.SolverVBD(
                    model,
                    iterations=1,
                    rigid_articulation_solve=mode,
                    rigid_articulation_elastic_solve=backend,
                    rigid_body_contact_buffer_size=_JOINED_PAIR_GUARD_BUFFER_SIZE,
                    deterministic=DETERMINISTIC_MODE,
                )
            except ValueError as error:
                if not raises:
                    raise AssertionError(f"{mode}/{backend}: unexpected scatter limit error: {error}") from error
                if "2 elastic bodies per launch" not in str(error):
                    raise AssertionError(f"{mode}/{backend}: unexpected message: {error}") from error
                continue
            if raises:
                raise AssertionError(f"{mode}/{backend}: the scatter limit did not count the joint launch")


def test_elastic_sparse_coupled_records(test, device):
    """Size the deterministic elastic contact records and bound their memory for the coupled solve."""
    _run_isolated(test, "_check_elastic_sparse_coupled_records", str(device))


def _elastic_plank_with_rigid_tip(device):
    """Return the elastic plank of :func:`elastic_plank_on_ground` hinged to a rigid tip in one articulation.

    The plank (9 x 9 grid contact samples, 1 mm above a ground plane) and the tip form a mixed
    rigid/elastic articulation, which the default ``rigid_articulation_elastic_solve="coupled"``
    solves in one variable-block system.
    """
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.ke = 1.0e5
    cfg.kd = 0.0
    cfg.mu = 0.5
    cfg.gap = 0.01
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=cfg)
    beam = newton.ModalGeneratorBeam
    basis = beam(
        length=0.4,
        half_width_y=0.1,
        half_width_z=0.005,
        mode_specs=[
            {"type": beam.Mode.BENDING_Z, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
            {"type": beam.Mode.BENDING_Z, "boundary": beam.Boundary.PINNED_PINNED, "order": 2},
            {"type": beam.Mode.TORSION, "boundary": beam.Boundary.LINEAR},
        ],
        young_modulus=1.0e7,
        damping_ratio=0.05,
    ).build()
    plank = builder.add_link_elastic(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.001), wp.quat_identity()), modal_basis=basis
    )
    builder.add_shape_mesh(plank, mesh=grid_surface_mesh(0.2, 0.1, 9, 9), cfg=cfg)
    tip = builder.add_link(
        xform=wp.transform(wp.vec3(0.25, 0.0, 0.03), wp.quat_identity()),
        mass=0.1,
        inertia=wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-4),
    )
    root = builder.add_joint_free(plank)
    builder.add_joint_revolute(
        parent=plank,
        child=tip,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.05, 0.0, -0.03), wp.quat_identity()),
    )
    builder.add_articulation([root, root + 1])
    builder.color()
    return builder.finalize(device=device)


def _run_coupled_elastic_contact_rollout(device, steps=30):
    model = _elastic_plank_with_rigid_tip(device)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=4,
        rigid_body_contact_buffer_size=128,
        rigid_articulation_solve="block_sparse_joints",
        deterministic=DETERMINISTIC_MODE,
    )
    layout = solver.rigid_articulation_sparse_layout
    if layout.elastic_solve != "coupled" or layout.mixed_elastic_body_count != 1:
        raise AssertionError("the plank must be solved in a coupled mixed articulation")
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    fields = ("body_q", "body_qd", "joint_q", "joint_qd")
    history = {field: [] for field in fields}
    max_contacts = 0
    for _ in range(steps):
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        max_contacts = max(max_contacts, int(contacts.rigid_contact_count.numpy()[0]))
        solver.step(state_0, state_1, control, contacts, 1.0 / 600.0)
        state_0, state_1 = state_1, state_0
        for field, values in _snapshot(state_0, fields).items():
            history[field].append(values)
    if max_contacts <= 2 * reduced_elastic_kernels._NUM_ELASTIC_CONTACT_THREADS_PER_BODY:
        raise AssertionError(f"elastic plank made only {max_contacts} contacts")
    if not all(np.isfinite(values[-1]).all() for values in history.values()):
        raise AssertionError("non-finite coupled rollout")
    return {field: np.stack(values) for field, values in history.items()}


def _check_coupled_elastic_contact_determinism(device):
    with wp.ScopedDevice(device):
        first = _run_coupled_elastic_contact_rollout(device)
        second = _run_coupled_elastic_contact_rollout(device)
    _assert_snapshots_equal(first, second)


def test_coupled_elastic_contact_determinism(test, device):
    """Repeat the contacts of an elastic link in a coupled articulation bit for bit in deterministic mode.

    The coupled solve assembles the contacts of all mixed elastic bodies in one launch, with the
    deterministic records sized from the mode count and the contact list size. A cold kernel cache
    compiles the deterministic sparse articulation modules, which takes several minutes.
    """
    # A cold kernel cache compiles the deterministic sparse articulation modules first (450 s
    # measured on a shared RTX PRO 6000 Blackwell GPU).
    _run_isolated(test, "_check_coupled_elastic_contact_determinism", str(device), timeout=1800)


class TestSolverDeterminism(unittest.TestCase):
    pass


def _emit_warning_for_policy_test(category_name, message):
    warnings.warn_explicit(message, getattr(builtins, category_name), __file__, 1, module=__name__)


class TestSolverDeterminismWarnings(unittest.TestCase):
    def test_allowlisted_deprecations_override_newton_error_filter(self):
        """Allow only acknowledged deprecations ahead of the Newton error filter."""
        allowed_prefix = "dependency.old_api is deprecated"
        with (
            mock.patch.object(newton.tests.unittest_utils, "strict_warnings", True),
            mock.patch.object(newton.tests.unittest_utils, "allowed_deprecation_warnings", (allowed_prefix,)),
        ):
            _run_isolated(self, "_emit_warning_for_policy_test", "DeprecationWarning", f"{allowed_prefix}; use new_api")
            for category, message in (
                ("DeprecationWarning", "unexpected deprecation"),
                ("UserWarning", allowed_prefix),
            ):
                with self.subTest(category=category), self.assertRaisesRegex(AssertionError, category):
                    _run_isolated(self, "_emit_warning_for_policy_test", category, message)


class TestSolverDeterminismOptions(unittest.TestCase):
    def setUp(self):
        self._modules = (
            xpbd_kernels,
            particle_vbd_kernels,
            semi_implicit_particle_kernels,
            vbd_coupling_kernels,
            rigid_vbd_kernels,
            rigid_sparse_articulation_kernels,
            reduced_elastic_kernels,
            reduced_elastic_sparse_kernels,
            rigid_vbd_kernels.accumulate_body_body_contacts_per_body.module,
            rigid_vbd_kernels.compute_rigid_contact_forces.module,
            rigid_vbd_kernels.update_duals_body_body_contacts.module,
        )
        self._saved_config = wp.config.deterministic
        self._saved_options = {
            module: {
                "deterministic": wp.get_module_options(module=module)["deterministic"],
                "deterministic_max_records": wp.get_module_options(module=module)["deterministic_max_records"],
            }
            for module in self._modules
        }

    def tearDown(self):
        wp.config.deterministic = self._saved_config
        for module, options in self._saved_options.items():
            wp.set_module_options(options, module=module)

    def test_xpbd_resets_inherited_module_options(self):
        with wp.ScopedDevice("cpu"):
            model = _build_soft_body("cpu")
            newton.solvers.SolverXPBD(model, deterministic=DETERMINISTIC_MODE)
            options = wp.get_module_options(module=xpbd_kernels)
            self.assertEqual(options["deterministic"], DETERMINISTIC_MODE)
            self.assertEqual(options["deterministic_max_records"], 0)

            wp.config.deterministic = wp.DeterministicMode.NOT_GUARANTEED
            newton.solvers.SolverXPBD(model)
            options = wp.get_module_options(module=xpbd_kernels)
            self.assertEqual(options["deterministic"], wp.DeterministicMode.NOT_GUARANTEED)
            self.assertEqual(options["deterministic_max_records"], 0)

    def test_matching_module_options_are_not_reapplied(self):
        options = {
            "deterministic": wp.get_module_options(module=xpbd_kernels)["deterministic"],
            "deterministic_max_records": wp.get_module_options(module=xpbd_kernels)["deterministic_max_records"],
        }
        with mock.patch.object(wp, "set_module_options") as set_module_options:
            _set_module_options_if_changed(options, module=xpbd_kernels)
        set_module_options.assert_not_called()

    def test_current_solver_skips_module_option_checks(self):
        model = newton.ModelBuilder().finalize(device="cpu")
        solver = newton.solvers.SolverXPBD(model)
        with mock.patch.object(wp, "get_module_options") as get_module_options:
            solver._apply_module_options()
        get_module_options.assert_not_called()

    def test_live_solvers_reapply_module_options_before_step(self):
        with wp.ScopedDevice("cpu"):
            model = newton.ModelBuilder().finalize(device="cpu")
            deterministic_solver = newton.solvers.SolverXPBD(model, deterministic=DETERMINISTIC_MODE)
            default_solver = newton.solvers.SolverXPBD(
                model,
                deterministic=wp.DeterministicMode.NOT_GUARANTEED,
            )
            state_0, state_1 = model.state(), model.state()

            deterministic_solver.step(state_0, state_1, None, None, 1.0 / 60.0)
            self.assertEqual(wp.get_module_options(module=xpbd_kernels)["deterministic"], DETERMINISTIC_MODE)

            default_solver.step(state_1, state_0, None, None, 1.0 / 60.0)
            self.assertEqual(
                wp.get_module_options(module=xpbd_kernels)["deterministic"],
                wp.DeterministicMode.NOT_GUARANTEED,
            )

    def test_shared_semi_implicit_modules_reset_record_budget(self):
        with wp.ScopedDevice("cpu"):
            model = _build_soft_body("cpu")
            for solver_type in (newton.solvers.SolverSemiImplicit, newton.solvers.SolverFeatherstone):
                with self.subTest(solver_type=solver_type.__name__):
                    wp.set_module_options(
                        {"deterministic_max_records": 7},
                        module=semi_implicit_particle_kernels,
                    )
                    solver_type(model, deterministic=DETERMINISTIC_MODE)
                    options = wp.get_module_options(module=semi_implicit_particle_kernels)
                    self.assertEqual(options["deterministic"], DETERMINISTIC_MODE)
                    self.assertEqual(options["deterministic_max_records"], 0)

    def test_vbd_resets_inherited_module_options(self):
        with wp.ScopedDevice("cpu"):
            model = _build_soft_body("cpu")
            newton.solvers.SolverVBD(
                model,
                particle_enable_self_contact=True,
                particle_enable_tile_solve=False,
                particle_vertex_contact_buffer_size=64,
                particle_edge_contact_buffer_size=64,
                deterministic=DETERMINISTIC_MODE,
            )
            options = wp.get_module_options(module=particle_vbd_kernels)
            self.assertEqual(options["deterministic"], DETERMINISTIC_MODE)
            records_per_buffer = (64 + particle_vbd_kernels.NUM_THREADS_PER_COLLISION_PRIMITIVE - 1) // (
                particle_vbd_kernels.NUM_THREADS_PER_COLLISION_PRIMITIVE
            )
            self.assertEqual(options["deterministic_max_records"], 8 * records_per_buffer)

            wp.config.deterministic = wp.DeterministicMode.NOT_GUARANTEED
            newton.solvers.SolverVBD(
                model,
                particle_enable_self_contact=False,
                particle_enable_tile_solve=False,
            )
            options = wp.get_module_options(module=particle_vbd_kernels)
            self.assertEqual(options["deterministic"], wp.DeterministicMode.NOT_GUARANTEED)
            self.assertEqual(options["deterministic_max_records"], 0)

    def test_vbd_sizes_elastic_contact_scatter_records(self):
        """Size the elastic contact scatter records only in deterministic mode with an elastic body."""
        with wp.ScopedDevice("cpu"):
            model, _body = elastic_plank_on_ground("cpu")
            newton.solvers.SolverVBD(model, rigid_body_contact_buffer_size=64, deterministic=DETERMINISTIC_MODE)
            options = wp.get_module_options(module=reduced_elastic_kernels)
            self.assertEqual(options["deterministic"], DETERMINISTIC_MODE)
            # 64 list entries over 32 threads give 2 contacts per thread, each with 6 * 3 + 3 * 4 / 2 = 24
            # block-matrix records for the plank's 3 modes.
            self.assertEqual(options["deterministic_max_records"], 48)
            for module in (rigid_vbd_kernels, reduced_elastic_sparse_kernels):
                self.assertEqual(wp.get_module_options(module=module)["deterministic_max_records"], 0)

            newton.solvers.SolverVBD(
                model,
                rigid_body_contact_buffer_size=64,
                deterministic=wp.DeterministicMode.NOT_GUARANTEED,
            )
            options = wp.get_module_options(module=reduced_elastic_kernels)
            self.assertEqual(options["deterministic"], wp.DeterministicMode.NOT_GUARANTEED)
            self.assertEqual(options["deterministic_max_records"], 0)

            builder = newton.ModelBuilder()
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
            builder.add_ground_plane()
            builder.color()
            rigid_model = builder.finalize(device="cpu")
            newton.solvers.SolverVBD(rigid_model, deterministic=DETERMINISTIC_MODE)
            options = wp.get_module_options(module=reduced_elastic_kernels)
            self.assertEqual(options["deterministic"], DETERMINISTIC_MODE)
            self.assertEqual(options["deterministic_max_records"], 0)

    def test_vbd_rejects_oversized_elastic_contact_scatter_buffers(self):
        """Raise at construction when the elastic contact scatter buffers would exceed the memory limit."""
        with wp.ScopedDevice("cpu"):
            model, _body = elastic_plank_on_ground("cpu")
            with self.assertRaisesRegex(ValueError, "rigid_body_contact_buffer_size"):
                newton.solvers.SolverVBD(
                    model, rigid_body_contact_buffer_size=1 << 21, deterministic=DETERMINISTIC_MODE
                )
            newton.solvers.SolverVBD(model, rigid_body_contact_buffer_size=1 << 21)

    def test_vbd_coupling_hook_reapplies_deterministic_options(self):
        with wp.ScopedDevice("cpu"):
            model = _build_soft_body("cpu")
            deterministic_solver = newton.solvers.SolverVBD(
                model,
                particle_enable_self_contact=True,
                particle_enable_tile_solve=False,
                particle_vertex_contact_buffer_size=64,
                particle_edge_contact_buffer_size=64,
                deterministic=DETERMINISTIC_MODE,
            )
            newton.solvers.SolverVBD(
                model,
                particle_enable_self_contact=False,
                particle_enable_tile_solve=False,
                deterministic=wp.DeterministicMode.NOT_GUARANTEED,
            )

            deterministic_solver.coupling_notify_input_state_update(
                model.state(),
                newton.StateFlags.BODY_Q,
            )

            options = wp.get_module_options(module=vbd_coupling_kernels)
            records_per_buffer = (64 + particle_vbd_kernels.NUM_THREADS_PER_COLLISION_PRIMITIVE - 1) // (
                particle_vbd_kernels.NUM_THREADS_PER_COLLISION_PRIMITIVE
            )
            self.assertEqual(options["deterministic"], DETERMINISTIC_MODE)
            self.assertEqual(options["deterministic_max_records"], 5 * records_per_buffer)


devices = get_cuda_test_devices(mode="basic")
for solver_name in ("xpbd", "semi_implicit", "vbd"):
    add_function_test(
        TestSolverDeterminism,
        f"test_particle_determinism_{solver_name}",
        test_particle_determinism,
        devices=devices,
        solver_name=solver_name,
        check_output=False,
    )

for solver_name in ("featherstone", "mujoco"):
    add_function_test(
        TestSolverDeterminism,
        f"test_articulation_determinism_{solver_name}",
        test_articulation_determinism,
        devices=devices,
        solver_name=solver_name,
        check_output=False,
    )

for ground in ("mesh", "plane"):
    add_function_test(
        TestSolverDeterminism,
        f"test_elastic_contact_determinism_{ground}",
        test_elastic_contact_determinism,
        devices=devices,
        ground=ground,
        check_output=False,
    )

add_function_test(
    TestSolverDeterminism,
    "test_elastic_contact_list_overflow_error",
    test_elastic_contact_list_overflow_error,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverDeterminism,
    "test_elastic_sparse_schur_records",
    test_elastic_sparse_schur_records,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverDeterminism,
    "test_elastic_sparse_coupled_records",
    test_elastic_sparse_coupled_records,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverDeterminism,
    "test_coupled_elastic_contact_determinism",
    test_coupled_elastic_contact_determinism,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverDeterminism,
    "test_iteration_metrics_keep_deterministic_solve",
    test_iteration_metrics_keep_deterministic_solve,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverDeterminism,
    "test_elastic_contact_default_mode_mesh",
    test_elastic_contact_default_mode,
    devices=devices,
    ground="mesh",
    check_output=False,
)

add_function_test(
    TestSolverDeterminism,
    "test_mujoco_sparse_articulation_construction",
    test_mujoco_sparse_articulation_construction,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
