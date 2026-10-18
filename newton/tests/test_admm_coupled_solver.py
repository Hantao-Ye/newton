# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ADMM-coupled solvers.

These tests validate generic :class:`SolverCoupledADMM` ADMM plumbing against a
cloth-plus-rigid-body scene.
"""

from __future__ import annotations

import inspect
import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.collide import _TRIANGLE_PAIRS_MIN_CAPACITY
from newton._src.solvers.coupled.interface import CouplingInterface
from newton.solvers import (
    SolverBase,
    SolverMuJoCo,
    SolverSemiImplicit,
    SolverVBD,
    SolverXPBD,
)
from newton.solvers.experimental.coupled import (
    SolverCoupled,
    SolverCoupledADMM,
)


@wp.kernel(enable_backward=False)
def _set_admm_plane_angle_kernel(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], angle: float):
    body_q[0] = wp.transform(
        wp.vec3(0.0, 0.0, 0.0),
        wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle),
    )
    body_qd[0] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class _CustomAdmmParticleCopySolver(SolverBase, CouplingInterface):
    """Base test solver that copies particle state."""

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        if state_in.particle_q is not None and state_out.particle_q is not None:
            wp.copy(state_out.particle_q, state_in.particle_q)
            wp.copy(state_out.particle_qd, state_in.particle_qd)
        if state_in.body_q is not None and state_out.body_q is not None:
            wp.copy(state_out.body_q, state_in.body_q)
            wp.copy(state_out.body_qd, state_in.body_qd)


class _KinematicAdmmPlaneSolver(_CustomAdmmParticleCopySolver):
    """Test solver that prescribes a fixed kinematic plane angle."""

    def __init__(self, model, angle):
        super().__init__(model)
        self.angle = float(angle)

    def step(self, state_in, state_out, control, contacts, dt):
        super().step(state_in, state_out, control, contacts, dt)
        wp.launch(
            _set_admm_plane_angle_kernel,
            dim=1,
            inputs=[state_out.body_q, state_out.body_qd, self.angle],
            device=self.model.device,
        )


def _build_cloth_rigid_scene(
    rigid_pos: tuple[float, float, float] = (0.0, 0.0, 1.5),
    rigid_mass: float = 0.05,
    cloth_pos: tuple[float, float, float] = (-0.25, -0.25, 1.5),
    dim_xy: int = 5,
    fix_cloth_edges: bool = True,
) -> tuple[newton.Model, int, int, int]:
    """Build a pinned cloth + free rigid body scene for attachment tests."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    rigid_start = builder.body_count
    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(*rigid_pos), q=wp.quat_identity()),
        mass=rigid_mass,
        inertia=wp.mat33(np.eye(3) * 0.001),
    )
    builder.add_shape_box(body, hx=0.03, hy=0.03, hz=0.03)
    rigid_end = builder.body_count

    particle_start = builder.particle_count
    builder.add_cloth_grid(
        pos=wp.vec3(*cloth_pos),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        fix_left=fix_cloth_edges,
        fix_right=fix_cloth_edges,
        dim_x=dim_xy,
        dim_y=dim_xy,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.05,
        tri_ke=1.0e4,
        tri_ka=1.0e4,
        tri_kd=1e-2,
        edge_ke=0.01,
        edge_kd=1e-2,
        particle_radius=0.01,
    )
    center = dim_xy // 2
    particle_idx = particle_start + center * (dim_xy + 1) + center
    builder.color()
    model = builder.finalize()
    return model, rigid_start, rigid_end, particle_idx


def _make_solver(
    model: newton.Model,
    rigid_start: int,
    rigid_end: int,
    admm_iters: int = 5,
    rho: float = 50.0,
    gamma: float = 0.0,
    baumgarte: float = 0.1,
):
    """Standard MuJoCo/VBD ADMM configuration used across tests."""
    mjc_ids = wp.array(list(range(rigid_start, rigid_end)), dtype=int)
    vbd_ids = wp.array(
        [i for i in range(model.body_count) if i < rigid_start or i >= rigid_end],
        dtype=int,
    )
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="mjc",
                solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, njmax=20),
                bodies=[int(i) for i in mjc_ids.numpy()],
                joints=list(range(model.joint_count)),
            ),
            SolverCoupled.Entry(
                name="vbd",
                solver=lambda v: SolverVBD(model=v, iterations=5, rigid_compliant_alm=True),
                bodies=[int(i) for i in vbd_ids.numpy()],
                particles=list(range(model.particle_count)),
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=admm_iters,
            rho=rho,
            gamma=gamma,
            baumgarte=baumgarte,
        ),
    )


def _run(solver, model: newton.Model, n_steps: int = 30, dt: float = 1.0 / 60.0):
    """Run ``n_steps`` of simulation and return (body_q, particle_q)."""
    state_0 = model.state()
    state_1 = model.state()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    for _ in range(n_steps):
        state_0.clear_forces()
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

    return state_0.body_q.numpy().copy(), state_0.particle_q.numpy().copy()


def _build_two_particle_scene() -> newton.Model:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_particle(pos=(-0.5, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
    builder.add_particle(pos=(0.5, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
    builder.color()
    return builder.finalize(device="cpu")


def _build_two_particle_contact_scene(
    gap: float = -0.1,
    vel_a: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vel_b: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 0.05,
) -> newton.Model:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_particle(pos=(gap, 0.0, 0.0), vel=vel_a, mass=1.0, radius=radius)
    builder.add_particle(pos=(0.0, 0.0, 0.0), vel=vel_b, mass=1.0, radius=radius)
    builder.color()
    return builder.finalize(device="cpu")


def _run_particles(solver, model: newton.Model, n_steps: int = 5, dt: float = 1.0 / 60.0):
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    for _ in range(n_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    return state_0.particle_q.numpy().copy()


def _make_vbd_xpbd_particle_solver(model: newton.Model):
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="vbd",
                solver=lambda v: SolverVBD(model=v, iterations=2),
                particles=[0],
            ),
            SolverCoupled.Entry(
                name="xpbd",
                solver=lambda v: SolverXPBD(model=v, iterations=2),
                particles=[1],
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=8,
            rho=20.0,
            baumgarte=0.2,
        ),
    )


def _make_semi_particle_solver(model: newton.Model):
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="a",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                particles=[0],
            ),
            SolverCoupled.Entry(
                name="b",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                particles=[1],
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=10,
            rho=30.0,
            baumgarte=0.5,
        ),
    )


def _build_body_particle_contact_scene() -> newton.Model:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body(
        xform=wp.transform(p=wp.vec3(-0.1, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
    )
    builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
    builder.color()
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    return model


def _build_body_particle_attachment_scene(enabled: bool = True) -> newton.Model:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
    )
    particle = builder.add_particle(pos=(0.3, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
    builder.add_attachment_particle_body(
        particle,
        body,
        stiffness=500.0,
        enabled=enabled,
    )
    builder.color()
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    return model


def _build_two_world_body_particle_attachment_scene() -> newton.Model:
    """Build two worlds with one rigid-particle attachment each."""
    world = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = world.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
    particle = world.add_particle(pos=(0.3, 0.0, 0.0), vel=wp.vec3(), mass=1.0, radius=0.0)
    world.add_attachment_particle_body(particle, body, stiffness=500.0)
    world.color()

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_world(world)
    builder.add_world(world)
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    return model


def _build_two_body_contact_scene(gap: float = -0.1) -> newton.Model:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body(
        xform=wp.transform(p=wp.vec3(gap, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
    )
    builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
    )
    builder.color()
    return builder.finalize(device="cpu")


def _build_collision_contact_scene() -> tuple[newton.Model, int, int, int]:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    tray_body = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        mass=0.1,
        inertia=wp.mat33(np.eye(3) * 0.01),
    )

    tray_cfg = newton.ModelBuilder.ShapeConfig()
    tray_cfg.has_shape_collision = False
    tray_cfg.has_particle_collision = True
    tray_shape = builder.add_shape_box(
        tray_body,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.025), q=wp.quat_identity()),
        hx=0.1,
        hy=0.1,
        hz=0.025,
        cfg=tray_cfg,
    )
    particle = builder.add_particle(
        pos=(0.0, 0.0, 0.12),
        vel=(0.0, 0.0, -0.5),
        mass=0.025,
        radius=0.025,
    )
    builder.color()
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    model.soft_contact_ke = 0.0
    model.soft_contact_kd = 0.0
    model.soft_contact_kf = 0.0
    model.soft_contact_mu = 0.0
    return model, particle, tray_body, tray_shape


def _build_multi_world_contact_candidate_solver(world_count: int) -> tuple[newton.Model, SolverCoupledADMM]:
    """Replicate a falling box and two particle sets, coupled by rigid-particle and particle-particle contacts.

    Worlds are spaced 1 m apart along x, and the box falls between the particles without touching them.
    All three entries use :class:`SolverSemiImplicit`, which does not support inertial property refresh,
    so ADMM applies the static proximal lumps accumulated from every contact candidate.
    """
    world = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    body = world.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.3), q=wp.quat_identity()),
        mass=0.2,
        inertia=wp.mat33(np.eye(3) * 1.0e-3),
    )
    world.add_shape_box(body, hx=0.03, hy=0.03, hz=0.03)
    for x in (-0.15, -0.05, 0.05, 0.15):
        world.add_particle(pos=(x, 0.0, 0.0), vel=wp.vec3(), mass=0.01, radius=0.01)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    for world_index in range(world_count):
        builder.add_world(world, xform=wp.transform(p=wp.vec3(float(world_index), 0.0, 0.0), q=wp.quat_identity()))
    model = builder.finalize(device="cpu")
    solver = SolverCoupledADMM(
        model,
        [
            SolverCoupled.Entry(
                "body",
                lambda view: SolverSemiImplicit(view, enable_tri_contact=False),
                bodies=list(range(model.body_count)),
            ),
            SolverCoupled.Entry(
                "a",
                lambda view: SolverSemiImplicit(view, enable_tri_contact=False),
                particles=[particle for particle in range(model.particle_count) if particle % 4 < 2],
            ),
            SolverCoupled.Entry(
                "b",
                lambda view: SolverSemiImplicit(view, enable_tri_contact=False),
                particles=[particle for particle in range(model.particle_count) if particle % 4 >= 2],
            ),
        ],
        SolverCoupledADMM.Config(
            iterations=2,
            rho=50.0,
            gamma=0.1,
            contact_pairs=[
                SolverCoupledADMM.ContactPair(source="body", destination="a"),
                SolverCoupledADMM.ContactPair(source="a", destination="b"),
            ],
        ),
    )
    return model, solver


def _run_body_particle(solver, model: newton.Model, n_steps: int = 4, dt: float = 1.0 / 60.0):
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    for _ in range(n_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    return state_0.body_q.numpy().copy(), state_0.particle_q.numpy().copy()


def _run_bodies(
    solver,
    model: newton.Model,
    n_steps: int = 4,
    dt: float = 1.0 / 60.0,
    body_qd: np.ndarray | None = None,
):
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    if body_qd is not None:
        state_0.body_qd = wp.array(body_qd, dtype=wp.spatial_vector, device=model.device)

    for _ in range(n_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    return state_0.body_q.numpy().copy(), state_0.body_qd.numpy().copy()


def _make_semi_body_particle_solver(model: newton.Model):
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="body",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                bodies=[0],
            ),
            SolverCoupled.Entry(
                name="particle",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                particles=[0],
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=10,
            rho=30.0,
            baumgarte=0.5,
        ),
    )


def _make_semi_body_body_solver(model: newton.Model):
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="a",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                bodies=[0],
            ),
            SolverCoupled.Entry(
                name="b",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                bodies=[1],
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=10,
            rho=30.0,
            baumgarte=0.5,
        ),
    )


def _build_inclined_plane_particle_box_scene(
    angle: float,
    *,
    particle_radius: float = 0.025,
    box_half_extent: float = 0.06,
    penetration: float = 0.002,
) -> tuple[newton.Model, int, int, list[int]]:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    plane_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    plane_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), plane_q),
        mass=0.0,
        inertia=wp.mat33(),
        is_kinematic=True,
    )

    plane_cfg = newton.ModelBuilder.ShapeConfig()
    plane_cfg.has_shape_collision = False
    plane_cfg.has_particle_collision = True
    plane_shape = builder.add_shape_plane(
        body=plane_body,
        xform=wp.transform_identity(),
        width=2.0,
        length=2.0,
        cfg=plane_cfg,
    )

    n = np.array([math.sin(angle), 0.0, math.cos(angle)], dtype=np.float32)
    tangent = np.array([math.cos(angle), 0.0, -math.sin(angle)], dtype=np.float32)
    binormal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    center = (particle_radius - penetration) * n

    particle_ids = []
    for tangent_sign in (-1.0, 1.0):
        for binormal_sign in (-1.0, 1.0):
            pos = center + tangent_sign * box_half_extent * tangent + binormal_sign * box_half_extent * binormal
            particle_ids.append(
                builder.add_particle(
                    pos=tuple(float(x) for x in pos),
                    vel=(0.0, 0.0, 0.0),
                    mass=0.25,
                    radius=particle_radius,
                )
            )

    builder.color()
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    model.soft_contact_ke = 0.0
    model.soft_contact_kd = 0.0
    model.soft_contact_kf = 0.0
    model.soft_contact_mu = 0.0
    return model, plane_body, plane_shape, particle_ids


def _make_admm_inclined_plane_particle_box_solver(
    model: newton.Model,
    plane_body: int,
    particle_ids: list[int],
    angle: float,
    friction: float,
) -> SolverCoupledADMM:
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="plane",
                solver=lambda v: _KinematicAdmmPlaneSolver(model=v, angle=angle),
                bodies=[plane_body],
            ),
            SolverCoupled.Entry(
                name="box",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                particles=particle_ids,
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=18,
            rho=50.0,
            baumgarte=0.1,
            contact_pairs=[
                SolverCoupledADMM.ContactPair(
                    source="plane",
                    destination="box",
                )
            ],
        ),
    )


def _run_inclined_plane_particle_box(
    angle: float,
    friction: float,
    *,
    steps: int = 120,
    dt: float = 1.0 / 360.0,
) -> tuple[float, float, int]:
    model, plane_body, _, particle_ids = _build_inclined_plane_particle_box_scene(angle)
    # ADMM derives friction from material properties; set both sides so the
    # geometric-mean combine reduces to the requested coefficient.
    model.particle_mu = float(friction)
    model.shape_material_mu = wp.full(model.shape_count, float(friction), dtype=wp.float32, device=model.device)
    solver = _make_admm_inclined_plane_particle_box_solver(
        model,
        plane_body,
        particle_ids,
        angle,
        friction,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    initial_com = np.mean(state_0.particle_q.numpy()[particle_ids], axis=0)
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    final_q = state_0.particle_q.numpy()[particle_ids]
    final_qd = state_0.particle_qd.numpy()[particle_ids]
    final_com = np.mean(final_q, axis=0)
    final_vel = np.mean(final_qd, axis=0)
    tangent = np.array([math.cos(angle), 0.0, -math.sin(angle)], dtype=np.float32)
    displacement = float(np.dot(final_com - initial_com, tangent))
    velocity = float(np.dot(final_vel, tangent))
    return displacement, velocity, solver.collision_contact_count_max


def _rotate_y_np(v: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]], dtype=np.float32)


def _build_inclined_plane_rigid_box_scene(
    angle: float,
    *,
    box_half_height: float = 0.08,
    penetration: float = 0.002,
) -> tuple[newton.Model, int, int]:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    plane_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    plane_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), plane_q),
        mass=0.0,
        inertia=wp.mat33(),
        is_kinematic=True,
    )

    local_center = np.array([0.0, 0.0, box_half_height - penetration], dtype=np.float32)
    box_center = _rotate_y_np(local_center, angle)
    box_body = builder.add_body(
        xform=wp.transform(
            wp.vec3(float(box_center[0]), float(box_center[1]), float(box_center[2])),
            plane_q,
        ),
        mass=1.0,
        inertia=wp.mat33(np.eye(3) * 0.01),
    )
    builder.color()
    return builder.finalize(device="cpu"), plane_body, box_body


def _build_collision_inclined_plane_rigid_box_scene(
    angle: float,
    *,
    box_half_height: float = 0.08,
    penetration: float = 0.004,
) -> tuple[newton.Model, int, int]:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    plane_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    plane_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), plane_q),
        mass=0.0,
        inertia=wp.mat33(),
        is_kinematic=True,
    )

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.has_shape_collision = True
    cfg.has_particle_collision = False
    cfg.density = 0.0
    builder.add_shape_box(
        plane_body,
        xform=wp.transform(wp.vec3(1.0, 0.0, -0.025), wp.quat_identity()),
        hx=3.0,
        hy=0.4,
        hz=0.025,
        cfg=cfg,
    )

    local_center = np.array([0.0, 0.0, box_half_height - penetration], dtype=np.float32)
    box_center = _rotate_y_np(local_center, angle)
    box_body = builder.add_body(
        xform=wp.transform(
            wp.vec3(float(box_center[0]), float(box_center[1]), float(box_center[2])),
            plane_q,
        ),
        mass=1.0,
        inertia=wp.mat33(np.eye(3) * 0.01),
    )
    builder.add_shape_box(
        box_body,
        hx=0.08,
        hy=0.08,
        hz=box_half_height,
        cfg=cfg,
    )
    builder.color()
    return builder.finalize(device="cpu"), plane_body, box_body


def _make_collision_admm_inclined_plane_rigid_box_solver(
    model: newton.Model,
    plane_body: int,
    box_body: int,
    angle: float,
    friction: float,
    *,
    rigid_contact_matching: str = "disabled",
    contact_matching_pos_threshold: float | None = None,
    contact_matching_normal_dot_threshold: float | None = None,
    contact_matching_force_scale: float = 1.0,
) -> SolverCoupledADMM:
    del friction
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="plane",
                solver=lambda v: _KinematicAdmmPlaneSolver(model=v, angle=angle),
                bodies=[plane_body],
            ),
            SolverCoupled.Entry(
                name="box",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                bodies=[box_body],
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=30,
            rho=5.0,
            gamma=0.2,
            baumgarte=0.03,
            rigid_contact_matching=rigid_contact_matching,
            contact_matching_pos_threshold=contact_matching_pos_threshold,
            contact_matching_normal_dot_threshold=contact_matching_normal_dot_threshold,
            contact_matching_force_scale=contact_matching_force_scale,
            contact_pairs=[
                SolverCoupledADMM.ContactPair(
                    source="plane",
                    destination="box",
                )
            ],
        ),
    )


def _run_collision_inclined_plane_rigid_box(
    angle: float,
    friction: float,
    *,
    steps: int = 120,
    dt: float = 1.0 / 360.0,
) -> tuple[float, float, float, int]:
    model, plane_body, box_body = _build_collision_inclined_plane_rigid_box_scene(angle)
    model.shape_material_mu = wp.full(model.shape_count, float(friction), dtype=wp.float32, device=model.device)
    solver = _make_collision_admm_inclined_plane_rigid_box_solver(model, plane_body, box_body, angle, friction)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    initial_pos = state_0.body_q.numpy()[box_body, :3].copy()
    min_gap = math.inf
    normal = _rotate_y_np(np.array([0.0, 0.0, 1.0], dtype=np.float32), angle)
    tangent = _rotate_y_np(np.array([1.0, 0.0, 0.0], dtype=np.float32), angle)
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0
        center_gap = float(np.dot(normal, state_0.body_q.numpy()[box_body, :3]))
        min_gap = min(min_gap, center_gap - 0.08)

    final_pos = state_0.body_q.numpy()[box_body, :3]
    final_qd = state_0.body_qd.numpy()[box_body, :3]
    displacement = float(np.dot(final_pos - initial_pos, tangent))
    velocity = float(np.dot(final_qd, tangent))
    return displacement, velocity, min_gap, solver.collision_contact_count_max


class TestAdmmSmoke(unittest.TestCase):
    """End-to-end: construct, run, verify state advances without NaNs."""

    def test_rejects_invalid_numerical_config(self):
        """Reject invalid ADMM coefficients and collision capacities before allocation."""
        model = _build_two_particle_scene()
        entries = [
            SolverCoupled.Entry(name="a", solver=SolverSemiImplicit, particles=[0]),
            SolverCoupled.Entry(name="b", solver=SolverSemiImplicit, particles=[1]),
        ]
        invalid_configs = (
            ({"iterations": 0}, "iterations"),
            ({"iterations": 1.5}, "iterations"),
            ({"rho": 0.0}, "rho"),
            ({"rho": float("nan")}, "rho"),
            ({"gamma": -1.0}, "gamma"),
            ({"baumgarte": float("inf")}, "baumgarte"),
            ({"joint_stiffness": -1.0}, "joint_stiffness"),
            ({"joint_proximal_mass_scale": 0.0}, "joint_proximal_mass_scale"),
            ({"contact_matching_normal_dot_threshold": 1.1}, "normal_dot_threshold"),
            ({"contact_max_triangle_pairs": 0}, "contact_max_triangle_pairs"),
            ({"contact_max_triangle_pairs": 1.5}, "contact_max_triangle_pairs"),
            ({"contact_max_triangle_pairs": True}, "contact_max_triangle_pairs"),
            ({"contact_reduction_hashtable_size_factor": 0.0}, "contact_reduction_hashtable_size_factor"),
            ({"contact_reduction_hashtable_size_factor": float("nan")}, "contact_reduction_hashtable_size_factor"),
            ({"contact_reduction_hashtable_size_factor": float("inf")}, "contact_reduction_hashtable_size_factor"),
        )
        for kwargs, message in invalid_configs:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                SolverCoupledADMM(model=model, entries=entries, coupling=SolverCoupledADMM.Config(**kwargs))

    def test_contact_capacity_matching_limit(self):
        """Allow larger triangle-pair capacities only when contact matching is disabled."""
        model = _build_two_particle_scene()
        entries = [
            SolverCoupled.Entry(name="a", solver=SolverSemiImplicit, particles=[0]),
            SolverCoupled.Entry(name="b", solver=SolverSemiImplicit, particles=[1]),
        ]
        for matching in ("disabled", "latest", "sticky"):
            for capacity in (None, 2**20 - 1, 2**20, 2**20 + 1):
                with self.subTest(matching=matching, capacity=capacity):
                    config = SolverCoupledADMM.Config(
                        rigid_contact_matching=matching, contact_max_triangle_pairs=capacity
                    )
                    if matching != "disabled" and capacity is not None and capacity >= 2**20:
                        with self.assertRaisesRegex(ValueError, r"contact_max_triangle_pairs.*2\*\*20"):
                            SolverCoupledADMM(model, entries, config)
                    else:
                        SolverCoupledADMM(model, entries, config)

    def test_construct_and_step_no_attachments(self):
        model, rs, re, _ = _build_cloth_rigid_scene()
        solver = _make_solver(model, rs, re, admm_iters=1)

        body_q, particle_q = _run(solver, model, n_steps=10)
        self.assertTrue(np.all(np.isfinite(body_q)))
        self.assertTrue(np.all(np.isfinite(particle_q)))

    def test_admm_iters_idempotent_with_no_coupling(self):
        """With gamma=0 and no attachments the iteration count should not
        change the result (no coupling = idempotent outer loop)."""
        model_a, rs, re, _ = _build_cloth_rigid_scene()
        solver_a = _make_solver(model_a, rs, re, admm_iters=1, gamma=0.0)
        body_a, part_a = _run(solver_a, model_a, n_steps=5)

        model_b, rs, re, _ = _build_cloth_rigid_scene()
        solver_b = _make_solver(model_b, rs, re, admm_iters=4, gamma=0.0)
        body_b, part_b = _run(solver_b, model_b, n_steps=5)

        np.testing.assert_allclose(body_a, body_b, atol=1e-6)
        np.testing.assert_allclose(part_a, part_b, atol=1e-6)


class TestAdmmProximal(unittest.TestCase):
    """Proximal terms affect constrained DOFs only."""

    def test_gamma_does_not_change_unconstrained_freefall(self):
        # Place the rigid body high so it stays in free-fall across the
        # window; with no ADMM constraints, gamma should not alter the result.
        model_ref, rs, re, _ = _build_cloth_rigid_scene(rigid_pos=(0.0, 0.0, 5.0))
        solver_ref = _make_solver(model_ref, rs, re, admm_iters=3, gamma=0.0)
        body_ref, part_ref = _run(solver_ref, model_ref, n_steps=5)

        model_g, rs, re, _ = _build_cloth_rigid_scene(rigid_pos=(0.0, 0.0, 5.0))
        solver_g = _make_solver(model_g, rs, re, admm_iters=3, gamma=5.0)
        body_g, part_g = _run(solver_g, model_g, n_steps=5)

        np.testing.assert_allclose(body_ref, body_g, atol=1.0e-6)
        np.testing.assert_allclose(part_ref, part_g, atol=1.0e-6)
        self.assertTrue(np.all(np.isfinite(body_g)))
        self.assertTrue(np.all(np.isfinite(part_g)))


class TestAdmmGraphCapture(unittest.TestCase):
    """CUDA graph-capture smoke tests for dynamic proximal refresh."""

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires CUDA")
    def test_xpbd_vbd_contact_proximal_refresh_is_graph_capturable(self):
        device = "cuda:0"
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.default_shape_cfg.density = 1000.0
        body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(body=body_a, hx=0.05, hy=0.05, hz=0.05)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.08, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(body=body_b, hx=0.05, hy=0.05, hz=0.05)
        builder.color()
        model = builder.finalize(device=device)
        solver = SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="xpbd",
                    solver=lambda v: SolverXPBD(model=v, iterations=1),
                    bodies=[body_a],
                ),
                SolverCoupled.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(model=v, iterations=1, rigid_compliant_alm=True),
                    bodies=[body_b],
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=1,
                gamma=1.0,
                contact_pairs=[SolverCoupledADMM.ContactPair(source="xpbd", destination="vbd")],
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        solver.step(state_0, state_1, control, contacts=None, dt=1.0 / 120.0)
        state_0, state_1 = state_1, state_0
        wp.synchronize_device(device)

        with wp.ScopedCapture(device=device) as capture:
            solver.step(state_0, state_1, control, contacts=None, dt=1.0 / 120.0)

        self.assertIsNotNone(capture.graph)
        wp.capture_launch(capture.graph)
        q = state_1.body_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)))


class TestAdmmModelJointInterface(unittest.TestCase):
    """Cross-solver model joints are converted to ADMM attachments."""

    def test_revolute_axis_frames_handle_antiparallel_x_axis(self):
        identity_row = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        axis = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        frame_child, frame_parent = SolverCoupledADMM._revolute_axis_frames_from_rows(
            identity_row,
            identity_row,
            axis,
        )

        for frame in (frame_child, frame_parent):
            rotation = wp.transform_get_rotation(frame)
            rotated_axis = wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0))
            np.testing.assert_allclose(np.asarray(rotated_axis), axis, atol=1.0e-6)

    def _build_two_body_joint_scene(
        self,
        joint_type: str = "ball",
        *,
        friction: float = 0.0,
    ) -> tuple[newton.Model, int, int, int]:
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        parent = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3) * 0.01),
        )
        child = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.3, 0.0, 0.0), q=wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3) * 0.01),
        )
        if joint_type == "ball":
            joint = builder.add_joint_ball(
                parent=parent,
                child=child,
                friction=friction,
                collision_filter_parent=False,
            )
        elif joint_type == "fixed":
            joint = builder.add_joint_fixed(parent=parent, child=child, collision_filter_parent=False)
        elif joint_type == "revolute":
            joint = builder.add_joint_revolute(
                parent=parent, child=child, friction=friction, collision_filter_parent=False
            )
        else:
            raise ValueError(joint_type)
        builder.color()
        return builder.finalize(device="cpu"), parent, child, joint

    def _make_two_body_joint_solver(self, model: newton.Model, parent: int, child: int, **coupling_kwargs):
        return SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="parent",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    bodies=[parent],
                ),
                SolverCoupled.Entry(
                    name="child",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    bodies=[child],
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=12,
                rho=40.0,
                baumgarte=0.5,
                joint_stiffness=500.0,
                joint_angular_stiffness=50.0,
                **coupling_kwargs,
            ),
        )

    def test_ball_joint_attachment_closes_anchor_gap(self):
        model, parent, child, _ = self._build_two_body_joint_scene("ball")
        solver = self._make_two_body_joint_solver(model, parent, child)
        initial_gap = abs(model.state().body_q.numpy()[child, 0] - model.state().body_q.numpy()[parent, 0])

        body_q, _ = _run_bodies(solver, model, n_steps=8, dt=1.0 / 120.0)
        final_gap = abs(body_q[child, 0] - body_q[parent, 0])

        self.assertLess(final_gap, 0.5 * initial_gap)

    def test_joint_proxy_shape_collisions_disabled_after_shape_compaction(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        unrelated = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_sphere(body=unrelated, radius=0.05)
        parent = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_sphere(body=parent, radius=0.05)
        child = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_sphere(body=child, radius=0.05)
        builder.add_joint_ball(parent=parent, child=child, collision_filter_parent=False)
        builder.color()
        model = builder.finalize(device="cpu")

        solver = SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="parent",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    bodies=[parent],
                ),
                SolverCoupled.Entry(
                    name="child",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    bodies=[child],
                ),
            ],
            coupling=SolverCoupledADMM.Config(),
        )

        collision_mask = int(
            newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES | newton.ShapeFlags.HYDROELASTIC
        )
        proxy_flag = int(newton.BodyFlags.PROXY)
        for name in ("parent", "child"):
            with self.subTest(name=name):
                view = solver.view(name)
                self.assertEqual(view.shape_count, model.shape_count)
                shape_body = view.shape_body.numpy()
                body_flags = view.body_flags.numpy()
                shape_flags = view.shape_flags.numpy()
                proxy_shapes = [
                    shape_id
                    for shape_id, body_id in enumerate(shape_body)
                    if int(body_id) >= 0 and int(body_flags[int(body_id)]) & proxy_flag
                ]
                owned_shapes = [
                    shape_id
                    for shape_id, body_id in enumerate(shape_body)
                    if int(body_id) >= 0 and shape_id not in proxy_shapes
                ]
                hidden_shapes = [shape_id for shape_id, body_id in enumerate(shape_body) if int(body_id) < 0]

                self.assertEqual(len(proxy_shapes), 1)
                self.assertEqual(len(owned_shapes), 1)
                self.assertEqual(len(hidden_shapes), 1)
                self.assertEqual(int(shape_flags[proxy_shapes[0]]) & collision_mask, 0)
                self.assertNotEqual(int(shape_flags[owned_shapes[0]]) & collision_mask, 0)
                self.assertEqual(int(shape_flags[hidden_shapes[0]]) & collision_mask, 0)

    def test_rejects_cross_solver_joint_owned_by_subsolver(self):
        model, parent, child, joint = self._build_two_body_joint_scene("ball")
        with self.assertRaisesRegex(ValueError, "must not be owned"):
            SolverCoupledADMM(
                model=model,
                entries=[
                    SolverCoupled.Entry(
                        name="parent",
                        solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                        bodies=[parent],
                        joints=[joint],
                    ),
                    SolverCoupled.Entry(
                        name="child",
                        solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                        bodies=[child],
                    ),
                ],
                coupling=SolverCoupledADMM.Config(),
            )


class TestAdmmBodyParticleAttachment(unittest.TestCase):
    """Model attachment rows are converted to rigid-particle ADMM attachments."""

    def test_model_attachment_closes_gap(self):
        """Couple a cross-entry model attachment and pull the endpoints together."""
        model = _build_body_particle_attachment_scene()
        solver = _make_semi_body_particle_solver(model)
        initial_gap = np.linalg.norm(model.state().body_q.numpy()[0, :3] - model.state().particle_q.numpy()[0])

        body_q, particle_q = _run_body_particle(solver, model, n_steps=8, dt=1.0 / 120.0)
        final_gap = np.linalg.norm(body_q[0, :3] - particle_q[0])

        self.assertLess(final_gap, 0.5 * initial_gap)

    def test_deprecated_helper_preserves_legacy_attachment_row(self):
        """Keep the deprecated ADMM helper compatible with its custom-row layout."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        particle = builder.add_particle(pos=(0.3, 0.0, 0.0), vel=wp.vec3(), mass=1.0, radius=0.0)

        with self.assertWarnsRegex(DeprecationWarning, "add_attachment_particle_body"):
            attachment = SolverCoupledADMM.add_body_particle_attachment(builder, body, particle, stiffness=500.0)

        self.assertEqual(attachment, 0)
        self.assertEqual(builder.attachment_particle_body_count, 0)

        builder.color()
        model = builder.finalize(device="cpu")
        self.assertEqual(model.attachment_particle_body_count, 0)
        self.assertEqual(
            model.custom_frequency_counts[SolverCoupledADMM.BODY_PARTICLE_ATTACHMENT_FREQUENCY],
            1,
        )
        self.assertIn(
            SolverCoupledADMM.BODY_PARTICLE_ATTACHMENT_FREQUENCY,
            model.custom_frequency_counts,
        )

    def test_single_unsupported_entry_attachment_is_reported(self):
        """Warn when a same-entry solver cannot apply a model attachment."""
        model = _build_body_particle_attachment_scene()
        with self.assertLogs("newton._src.solvers.coupled.solver_coupled_admm", level="WARNING") as logs:
            solver = SolverCoupledADMM(
                model,
                [
                    SolverCoupled.Entry(
                        "both",
                        lambda view: SolverSemiImplicit(view, enable_tri_contact=False),
                        bodies=range(model.body_count),
                        particles=range(model.particle_count),
                    ),
                ],
                SolverCoupledADMM.Config(iterations=1),
            )

        self.assertEqual(solver._admm_rp_groups, [])
        self.assertTrue(any("only SolverVBD" in message for message in logs.output))

    def test_single_vbd_entry_attachment_is_left_to_vbd(self):
        """Leave a same-entry model attachment to SolverVBD without an attachment warning."""
        model = _build_body_particle_attachment_scene()
        with (
            self.assertNoLogs("newton._src.solvers.coupled.solver_coupled_admm", level="WARNING"),
            self.assertWarns(DeprecationWarning),
        ):
            solver = SolverCoupledADMM(
                model,
                [
                    SolverCoupled.Entry(
                        "both",
                        lambda view: SolverVBD(view, iterations=0, rigid_compliant_alm=False),
                        bodies=range(model.body_count),
                        particles=range(model.particle_count),
                    ),
                ],
                SolverCoupledADMM.Config(iterations=1),
            )

        self.assertEqual(solver._admm_rp_groups, [])
        self.assertEqual(solver.view("both").attachment_particle_body_count, 1)

    def test_unowned_attachment_endpoint_is_reported(self):
        """Warn about attachment rows that no entry can apply."""
        model = _build_body_particle_attachment_scene()
        with self.assertLogs("newton._src.solvers.coupled.solver_coupled_admm", level="WARNING") as logs:
            solver = SolverCoupledADMM(
                model,
                [
                    SolverCoupled.Entry(
                        "body",
                        lambda view: SolverSemiImplicit(view, enable_tri_contact=False),
                        bodies=range(model.body_count),
                    ),
                ],
                SolverCoupledADMM.Config(iterations=1),
            )

        self.assertEqual(solver._admm_rp_groups, [])
        self.assertTrue(any("not owned by any entry" in message for message in logs.output))

    def test_masked_reset_preserves_unselected_attachment_history(self):
        """Clear selected attachment duals and preserve every other row."""
        model = _build_two_world_body_particle_attachment_scene()
        solver = SolverCoupledADMM(
            model,
            [
                SolverCoupled.Entry(
                    "body",
                    lambda view: SolverSemiImplicit(view, enable_tri_contact=False),
                    bodies=range(model.body_count),
                ),
                SolverCoupled.Entry(
                    "particle",
                    lambda view: SolverSemiImplicit(view, enable_tri_contact=False),
                    particles=range(model.particle_count),
                ),
            ],
            SolverCoupledADMM.Config(iterations=1),
        )
        group = solver._admm_rp_groups[0]
        u_before = np.arange(1, group.u.numpy().size + 1, dtype=np.float32).reshape(group.u.numpy().shape)
        lambda_before = u_before + 10.0
        group.u.assign(u_before)
        group.lambda_.assign(lambda_before)

        solver.reset(
            model.state(),
            world_mask=wp.array((False, False, False), dtype=wp.bool, device=model.device),
            flags=0,
        )
        np.testing.assert_array_equal(group.u.numpy(), u_before)
        np.testing.assert_array_equal(group.lambda_.numpy(), lambda_before)

        solver.reset(
            model.state(),
            world_mask=wp.array((True, False, False), dtype=wp.bool, device=model.device),
            flags=0,
        )
        np.testing.assert_array_equal(group.u.numpy()[0], 0.0)
        np.testing.assert_array_equal(group.lambda_.numpy()[0], 0.0)
        np.testing.assert_array_equal(group.u.numpy()[1], u_before[1])
        np.testing.assert_array_equal(group.lambda_.numpy()[1], lambda_before[1])


class TestAdmmExternalForces(unittest.TestCase):
    """External forces set on ``state_in.body_f`` / ``particle_f`` by the
    caller (e.g. a viewer gizmo) must reach the sub-solvers."""

    def test_body_f_reaches_mujoco(self):
        """An upward ``body_f`` on the rigid sphere should slow its fall
        compared to the zero-force baseline."""
        # Baseline: no external force, body falls under gravity.
        model_a, rs, re, _ = _build_cloth_rigid_scene(rigid_pos=(0.0, 0.0, 5.0))
        solver_a = _make_solver(model_a, rs, re, admm_iters=1)
        state_0 = model_a.state()
        state_1 = model_a.state()
        collision_pipeline = newton.CollisionPipeline(model_a)
        contacts = collision_pipeline.contacts()
        control = model_a.control()
        newton.eval_fk(model_a, model_a.joint_q, model_a.joint_qd, state_0)
        for _ in range(5):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver_a.step(state_0, state_1, control, contacts, 1.0 / 60.0)
            state_0, state_1 = state_1, state_0
        z_baseline = state_0.body_q.numpy()[0, 2]

        # With a strong upward body_f applied each step, the body should fall
        # less (or even rise).
        model_b, rs, re, _ = _build_cloth_rigid_scene(rigid_pos=(0.0, 0.0, 5.0))
        solver_b = _make_solver(model_b, rs, re, admm_iters=1)
        state_0 = model_b.state()
        state_1 = model_b.state()
        collision_pipeline = newton.CollisionPipeline(model_b)
        contacts = collision_pipeline.contacts()
        control = model_b.control()
        newton.eval_fk(model_b, model_b.joint_q, model_b.joint_qd, state_0)
        body_idx = rs  # only MuJoCo body
        body_mass = float(model_b.body_mass.numpy()[body_idx])
        upward_force = 5.0 * body_mass * 9.81  # 5 g upward wrench
        for _ in range(5):
            state_0.clear_forces()
            wrench = np.zeros((model_b.body_count, 6), dtype=np.float32)
            wrench[body_idx, 2] = upward_force  # linear z
            state_0.body_f = wp.array(wrench, dtype=wp.spatial_vector, device=model_b.device)
            collision_pipeline.collide(state_0, contacts)
            solver_b.step(state_0, state_1, control, contacts, 1.0 / 60.0)
            state_0, state_1 = state_1, state_0
        z_with_force = state_0.body_q.numpy()[0, 2]

        self.assertGreater(
            z_with_force,
            z_baseline + 0.02,
            f"external body_f didn't reach MuJoCo: baseline z={z_baseline:.4f}, "
            f"with 5g upward force z={z_with_force:.4f}",
        )


class TestAdmmCollisionDetection(unittest.TestCase):
    """Collision-detected ADMM contact constraints."""

    def test_rigid_contact_collision_capacity(self):
        """Apply independent capacity overrides while retaining mesh contacts and matching."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        mesh_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)))
        mesh = newton.Mesh(
            vertices=np.array([[0.0, -0.1, -0.1], [0.0, 0.1, -0.1], [0.0, 0.0, 0.1]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.int32),
            compute_inertia=False,
        )
        builder.add_shape_mesh(body=mesh_body, mesh=mesh)
        sphere_body = builder.add_body(xform=wp.transform(wp.vec3(0.08, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_sphere(body=sphere_body, radius=0.1)
        model = builder.finalize(device="cpu")
        entries = [
            SolverCoupled.Entry(name="mesh", solver=SolverXPBD, bodies=[mesh_body]),
            SolverCoupled.Entry(name="sphere", solver=SolverXPBD, bodies=[sphere_body]),
        ]
        defaults = inspect.signature(newton.CollisionPipeline).parameters
        for capacity, factor, matching in (
            (None, None, "disabled"),
            (8192, None, "latest"),
            (None, 0.5, "sticky"),
            (8192, 256.0, "latest"),
        ):
            with self.subTest(capacity=capacity, factor=factor, matching=matching):
                solver = SolverCoupledADMM(
                    model,
                    entries,
                    SolverCoupledADMM.Config(
                        iterations=1,
                        contact_pairs=[SolverCoupledADMM.ContactPair(source="mesh", destination="sphere")],
                        rigid_contact_matching=matching,
                        contact_max_triangle_pairs=capacity,
                        contact_reduction_hashtable_size_factor=factor,
                    ),
                )
                pipeline = solver._admm_collision_pipeline
                expected_capacity = capacity if capacity is not None else _TRIANGLE_PAIRS_MIN_CAPACITY
                expected_factor = (
                    factor if factor is not None else defaults["contact_reduction_hashtable_size_factor"].default
                )
                self.assertEqual(pipeline.narrow_phase.max_triangle_pairs, expected_capacity)
                requested_slots = max(1024, int(expected_capacity * expected_factor))
                self.assertEqual(
                    pipeline.narrow_phase.global_contact_reducer.hashtable.capacity,
                    1 << (requested_slots - 1).bit_length(),
                )
                self.assertEqual(pipeline.contact_matching, matching)
                state_in, state_out = model.state(), model.state()
                solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
                self.assertGreater(solver.collision_contact_count_max, 0)
                self.assertTrue(np.all(np.isfinite(state_out.body_q.numpy())))

    def test_masked_reset_preserves_unselected_contact_history(self):
        """Clear only dynamic contact duals touching the reset world."""
        world = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        world.add_particle(pos=(-0.04, 0.0, 0.0), vel=wp.vec3(), mass=1.0, radius=0.05)
        world.add_particle(pos=(0.04, 0.0, 0.0), vel=wp.vec3(), mass=1.0, radius=0.05)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_world(world)
        builder.add_world(world, xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()))
        model = builder.finalize(device="cpu")
        solver = SolverCoupledADMM(
            model,
            [
                SolverCoupled.Entry(
                    "a",
                    lambda view: SolverSemiImplicit(view, enable_tri_contact=False),
                    particles=(0, 2),
                ),
                SolverCoupled.Entry(
                    "b",
                    lambda view: SolverSemiImplicit(view, enable_tri_contact=False),
                    particles=(1, 3),
                ),
            ],
            SolverCoupledADMM.Config(
                iterations=1,
                contact_pairs=[SolverCoupledADMM.ContactPair(source="a", destination="b")],
            ),
        )
        solver._refresh_collision_contact_groups(model.state())
        group = solver._admm_dynamic_pp_contact_groups[0]
        active = np.flatnonzero(group.active.numpy())
        self.assertEqual(active.size, 2)
        ids_a = group.particle_ids_a.numpy()[active]
        ids_b = group.particle_ids_b.numpy()[active]
        worlds_a = solver._entries[group.particle_entry_name_a].view.particle_world.numpy()[ids_a]
        worlds_b = solver._entries[group.particle_entry_name_b].view.particle_world.numpy()[ids_b]
        selected = (worlds_a == 0) | (worlds_b == 0)

        u_before = np.arange(1, group.u.numpy().size + 1, dtype=np.float32).reshape(group.u.numpy().shape)
        lambda_before = u_before + 20.0
        group.u.assign(u_before)
        group.lambda_.assign(lambda_before)
        solver.reset(
            model.state(),
            world_mask=wp.array((True, False, False), dtype=wp.bool, device=model.device),
            flags=0,
        )

        after_u = group.u.numpy()[active]
        after_lambda = group.lambda_.numpy()[active]
        np.testing.assert_array_equal(after_u[selected], 0.0)
        np.testing.assert_array_equal(after_lambda[selected], 0.0)
        np.testing.assert_array_equal(after_u[~selected], u_before[active][~selected])
        np.testing.assert_array_equal(after_lambda[~selected], lambda_before[active][~selected])

    def test_rigid_contact_detection_rejects_cross_world_pairs(self):
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.density = 1000.0

        builder.begin_world()
        body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        shape_a = builder.add_shape_box(body=body_a, hx=0.05, hy=0.05, hz=0.05)
        builder.end_world()

        builder.begin_world()
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        shape_b = builder.add_shape_box(body=body_b, hx=0.05, hy=0.05, hz=0.05)
        builder.end_world()

        model = builder.finalize(device="cpu")
        model.shape_contact_pairs = wp.array(
            np.asarray([(shape_a, shape_b)], dtype=np.int32), dtype=wp.vec2i, device=model.device
        )
        model.shape_contact_pair_count = 1

        with self.assertRaisesRegex(ValueError, "same world"):
            SolverCoupledADMM(
                model=model,
                entries=[
                    SolverCoupled.Entry(
                        name="a",
                        solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                        bodies=[body_a],
                    ),
                    SolverCoupled.Entry(
                        name="b",
                        solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                        bodies=[body_b],
                    ),
                ],
                coupling=SolverCoupledADMM.Config(
                    contact_pairs=[SolverCoupledADMM.ContactPair(source="a", destination="b")],
                ),
            )

    def test_collision_contact_candidates_are_world_local(self):
        """Size contact candidates and static proximal lumps per world, independent of the world count.

        Rigid-particle and particle-particle contacts are detected only between compatible worlds, so
        candidates in other worlds must neither enlarge a contact group nor add proximal mass to an entity.
        """
        _, solver_1 = _build_multi_world_contact_candidate_solver(1)
        _, solver_3 = _build_multi_world_contact_candidate_solver(3)

        for kind, groups_1, groups_3 in (
            ("rigid-particle", solver_1._admm_dynamic_rp_contact_groups, solver_3._admm_dynamic_rp_contact_groups),
            ("particle-particle", solver_1._admm_dynamic_pp_contact_groups, solver_3._admm_dynamic_pp_contact_groups),
        ):
            with self.subTest(kind=kind):
                self.assertEqual(len(groups_1), 1)
                self.assertEqual(len(groups_3), 1)
                self.assertEqual(groups_3[0].capacity, 3 * groups_1[0].capacity)

        body_lump_1 = solver_1._admm_buffers["body"].body_proximal_mass.numpy()
        self.assertTrue(np.all(body_lump_1 > 0.0))
        np.testing.assert_allclose(
            solver_3._admm_buffers["body"].body_proximal_mass.numpy(), np.tile(body_lump_1, 3), rtol=1.0e-6
        )
        for name in ("a", "b"):
            with self.subTest(entry=name):
                particle_lump_1 = solver_1._admm_buffers[name].particle_proximal_mass.numpy()
                self.assertGreater(float(np.max(particle_lump_1)), 0.0)
                np.testing.assert_allclose(
                    solver_3._admm_buffers[name].particle_proximal_mass.numpy(),
                    np.tile(particle_lump_1, 3),
                    rtol=1.0e-6,
                )

    def test_static_proximal_lump_does_not_depend_on_world_count(self):
        """Simulate each replicated world exactly like a single-world model."""
        model_1, solver_1 = _build_multi_world_contact_candidate_solver(1)
        model_3, solver_3 = _build_multi_world_contact_candidate_solver(3)

        body_q_1, particle_q_1 = _run_body_particle(solver_1, model_1, n_steps=30)
        body_q_3, particle_q_3 = _run_body_particle(solver_3, model_3, n_steps=30)

        # World i is the single world translated by i along x.
        expected_body_q = np.tile(body_q_1, (3, 1))
        expected_body_q[:, 0] += np.arange(3)
        expected_particle_q = np.tile(particle_q_1, (3, 1))
        expected_particle_q[:, 0] += np.repeat(np.arange(3), model_1.particle_count)
        np.testing.assert_allclose(body_q_3, expected_body_q, atol=1.0e-6)
        np.testing.assert_allclose(particle_q_3, expected_particle_q, atol=1.0e-6)

    def test_collision_particle_particle_contacts_are_refreshed_in_solver(self):
        model = _build_two_particle_contact_scene(gap=-0.08)
        solver = SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="a",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[0],
                ),
                SolverCoupled.Entry(
                    name="b",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[1],
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=10,
                rho=30.0,
                baumgarte=0.5,
                contact_pairs=[
                    SolverCoupledADMM.ContactPair(source="a", destination="b"),
                ],
            ),
        )

        q_contact = _run_particles(solver, model, n_steps=4)

        self.assertGreater(solver.collision_contact_count_max, 0)
        self.assertGreater(q_contact[1, 0] - q_contact[0, 0], 0.08 + 1.0e-3)

    def test_collision_particle_particle_contacts_respect_disabled_model_particle_grid(self):
        model = _build_two_particle_contact_scene(gap=-0.08)
        model.particle_grid = None
        solver = SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="a",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[0],
                ),
                SolverCoupled.Entry(
                    name="b",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[1],
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                contact_pairs=[SolverCoupledADMM.ContactPair(source="a", destination="b")],
            ),
        )

        _run_particles(solver, model, n_steps=1)

        self.assertEqual(solver.collision_contact_count_max, 0)

    def test_collision_frictional_contact_matches_inclined_plane_box_motion(self):
        friction = 0.4
        angle = math.radians(35.0)
        dt = 1.0 / 360.0
        steps = 120
        displacement, velocity, contact_count = _run_inclined_plane_particle_box(
            angle,
            friction,
            steps=steps,
            dt=dt,
        )

        t = steps * dt
        acceleration = 10.0 * (math.sin(angle) - friction * math.cos(angle))
        expected_displacement = 0.5 * acceleration * t * t
        expected_velocity = acceleration * t

        self.assertGreater(contact_count, 0)
        self.assertGreater(acceleration, 0.0)
        self.assertAlmostEqual(displacement, expected_displacement, delta=0.45 * expected_displacement)
        self.assertAlmostEqual(velocity, expected_velocity, delta=0.45 * expected_velocity)

    def test_collision_frictional_contact_holds_subcritical_inclined_box(self):
        friction = 0.4
        angle = math.radians(15.0)
        displacement, velocity, contact_count = _run_inclined_plane_particle_box(
            angle,
            friction,
            steps=120,
            dt=1.0 / 360.0,
        )

        self.assertGreater(contact_count, 0)
        self.assertLess(math.tan(angle), friction)
        self.assertLess(abs(displacement), 0.01)
        self.assertLess(abs(velocity), 0.05)

    def test_collision_rigid_rigid_frictional_contact_matches_inclined_plane_box_motion(self):
        friction = 0.35
        angle = math.radians(24.0)
        steps = 120
        dt = 1.0 / 360.0
        displacement, velocity, min_gap, contact_count = _run_collision_inclined_plane_rigid_box(
            angle,
            friction,
            steps=steps,
            dt=dt,
        )

        t = steps * dt
        acceleration = 10.0 * (math.sin(angle) - friction * math.cos(angle))
        expected_displacement = 0.5 * acceleration * t * t
        expected_velocity = acceleration * t

        self.assertGreater(contact_count, 0)
        self.assertGreater(acceleration, 0.0)
        self.assertAlmostEqual(displacement, expected_displacement, delta=0.65 * expected_displacement)
        self.assertAlmostEqual(velocity, expected_velocity, delta=0.65 * expected_velocity)
        self.assertGreater(min_gap, -0.03)

    def test_collision_particle_shape_contacts_are_refreshed_in_solver(self):
        model, particle, tray_body, _ = _build_collision_contact_scene()
        solver = SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="drop",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[particle],
                ),
                SolverCoupled.Entry(
                    name="tray",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    bodies=[tray_body],
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=12,
                rho=45.0,
                gamma=0.05,
                baumgarte=0.1,
                contact_pairs=[
                    SolverCoupledADMM.ContactPair(
                        source="drop",
                        destination="tray",
                    ),
                ],
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        initial_tray_z = float(state_0.body_q.numpy()[tray_body, 2])
        min_gap = float(state_0.particle_q.numpy()[particle, 2] - initial_tray_z)
        for _ in range(90):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts=None, dt=1.0 / 120.0)
            state_0, state_1 = state_1, state_0
            particle_z = float(state_0.particle_q.numpy()[particle, 2])
            tray_z = float(state_0.body_q.numpy()[tray_body, 2])
            min_gap = min(min_gap, particle_z - tray_z)

        final_particle_z = float(state_0.particle_q.numpy()[particle, 2])
        final_tray_z = float(state_0.body_q.numpy()[tray_body, 2])
        final_gap = final_particle_z - final_tray_z
        self.assertGreater(solver.collision_contact_count_max, 0)
        self.assertLessEqual(min_gap, 0.08)
        self.assertGreater(min_gap, -0.02)
        self.assertGreater(final_gap, 0.02)
        self.assertLess(final_tray_z, initial_tray_z - 1.0e-3)


# ----------------------------------------------------------------------
# Full-surface (edge/face) rigid-soft contact rows
# ----------------------------------------------------------------------

_PAD_HALF = (0.011, 0.004, 0.010)  # a gripper-pad-sized box [m]


def _pad_cloth_world(
    *,
    gravity: float = -10.0,
    particle_mass: float = 0.0,
    box_xy: tuple[float, float] = (0.07, 0.03),
    box_bottom: float = 0.002,
    box_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    box_mass: float = 0.05,
    cells: tuple[int, int] = (1, 1),
    mu: float = 1.0,
    box_half: tuple[float, float, float] = _PAD_HALF,
    box_inertia: float | None = None,
) -> tuple[newton.ModelBuilder, int]:
    """A 0.1 m-cell cloth (radius 3 mm) in the z = 0 plane and a box (pad-sized by default) above it.

    With the default box position the box's footprint covers no cloth vertex: it lies across the
    diagonal edge of a 0.1 m cell, so only edge/face contact can hold it. By default the cloth's
    particles are kinematic but active (mass 0: infinite mass, still colliding), so under gravity the
    box comes to rest on the cloth when something holds it. ``box_inertia`` overrides the solid
    box's inertia with an isotropic one [kg m^2].
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    SolverVBD.register_custom_attributes(builder)
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=cells[0],
        dim_y=cells[1],
        cell_x=0.1,
        cell_y=0.1,
        mass=particle_mass,
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        edge_ke=0.0,
        particle_radius=0.003,
    )
    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(box_xy[0], box_xy[1], box_bottom + box_half[2]), q=wp.quat_identity()),
        mass=box_mass,
        # Solid-box inertia about the centre: m / 3 * (sum of the other two half extents squared).
        inertia=wp.mat33(
            np.eye(3) * box_inertia
            if box_inertia is not None
            else np.diag(
                [
                    box_mass / 3.0 * (box_half[1] ** 2 + box_half[2] ** 2),
                    box_mass / 3.0 * (box_half[0] ** 2 + box_half[2] ** 2),
                    box_mass / 3.0 * (box_half[0] ** 2 + box_half[1] ** 2),
                ]
            )
        ),
    )
    builder.body_qd[body] = wp.spatial_vector(*box_velocity, 0.0, 0.0, 0.0)
    builder.add_shape_box(
        body, hx=box_half[0], hy=box_half[1], hz=box_half[2], cfg=newton.ModelBuilder.ShapeConfig(mu=mu)
    )
    return builder, body


def _finalize_pad_cloth(builder: newton.ModelBuilder, world_count: int = 1, device="cpu") -> newton.Model:
    if world_count > 1:
        multi = newton.ModelBuilder(gravity=builder.gravity)
        SolverVBD.register_custom_attributes(multi)
        for world_index in range(world_count):
            multi.add_world(builder, xform=wp.transform(p=wp.vec3(float(world_index), 0.0, 0.0), q=wp.quat_identity()))
        builder = multi
    builder.color()
    model = builder.finalize(device=device)
    model.particle_mu = 1.0
    model.soft_contact_ke = 0.0
    model.soft_contact_kd = 0.0
    return model


def _pad_cloth_solver(
    model: newton.Model,
    *,
    full_surface: bool,
    cloth_solver: str = "semi",
    redundancy_tolerance: float | None = 5.0e-4,
    iterations: int = 8,
) -> SolverCoupledADMM:
    bodies = list(range(model.body_count))
    particles = list(range(model.particle_count))

    def make_cloth(view):
        if cloth_solver == "vbd":
            return SolverVBD(view, iterations=10)
        return SolverSemiImplicit(view, enable_tri_contact=False)

    return SolverCoupledADMM(
        model,
        [
            SolverCoupled.Entry("box", lambda view: SolverSemiImplicit(view, enable_tri_contact=False), bodies=bodies),
            SolverCoupled.Entry("cloth", make_cloth, particles=particles),
        ],
        SolverCoupledADMM.Config(
            iterations=iterations,
            rho=50.0,
            gamma=0.0,
            baumgarte=0.01,
            full_surface_redundancy_tolerance=redundancy_tolerance,
            contact_pairs=[SolverCoupledADMM.ContactPair("box", "cloth", full_surface=full_surface)],
        ),
    )


def _step_pad_cloth(solver, model, n_steps: int, dt: float = 1.0 / 600.0, observe=None):
    state_0, state_1, control = model.state(), model.state(), model.control()
    for _ in range(n_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0
        if observe is not None:
            observe(state_0)
    return state_0


def _system_momentum(model: newton.Model, state) -> tuple[np.ndarray, float]:
    """Linear momentum [kg m/s] and kinetic energy [J] of every body and particle."""
    pm = model.particle_mass.numpy()
    pv = state.particle_qd.numpy()
    bm = model.body_mass.numpy()
    qd = state.body_qd.numpy()
    bv, bw = qd[:, :3], qd[:, 3:]
    inertia = model.body_inertia.numpy()
    rot = [np.array(wp.quat_to_matrix(wp.quat(*q[3:7]))).reshape(3, 3) for q in state.body_q.numpy()]
    p = (pm[:, None] * pv).sum(axis=0) + (bm[:, None] * bv).sum(axis=0)
    ke = 0.5 * float((pm * (pv * pv).sum(axis=1)).sum()) + 0.5 * float((bm * (bv * bv).sum(axis=1)).sum())
    for b in range(len(bm)):
        world_inertia = rot[b] @ inertia[b] @ rot[b].T
        ke += 0.5 * float(bw[b] @ world_inertia @ bw[b])
    return p, ke


class TestAdmmFullSurfaceContact(unittest.TestCase):
    """Edge/face rigid-soft ADMM rows (``ContactPair(full_surface=True)``)."""

    def test_pad_on_triangle_interior_needs_feature_rows(self):
        """A pad whose footprint covers no cloth vertex: per-particle rows let it fall through, feature rows hold it."""
        bottoms = {}
        for full_surface in (False, True):
            builder, body = _pad_cloth_world()
            model = _finalize_pad_cloth(builder)
            solver = _pad_cloth_solver(model, full_surface=full_surface)
            state = _step_pad_cloth(solver, model, 300)
            cloth_z = float(state.particle_q.numpy()[:, 2].mean())
            bottoms[full_surface] = float(state.body_q.numpy()[body, 2]) - _PAD_HALF[2] - cloth_z
            if full_surface:
                group = solver._admm_dynamic_ef_contact_groups[0]
                self.assertGreater(int(group.active_count_max.numpy()[0]), 0)
                self.assertEqual(solver._admm_dynamic_rp_contact_groups[0].active_count_max.numpy()[0], 0)
            else:
                self.assertEqual(solver._admm_dynamic_ef_contact_groups, [])
        # Without feature rows the pad falls through the cloth plane (free fall for 0.5 s is 1.25 m).
        self.assertLess(bottoms[False], -0.5)
        # With them it rests on the cloth's contact shell (radius 3 mm), within a millimetre.
        self.assertGreater(bottoms[True], 0.002)
        self.assertLess(bottoms[True], 0.0045)

    def test_feature_records_cover_edges_and_faces(self):
        """A pad across an interior cloth edge yields edge rows (two corners) as well as face rows."""
        builder, body = _pad_cloth_world(
            box_xy=(0.1, 0.05),
            cells=(2, 1),
            box_bottom=0.001,
            gravity=0.0,
            particle_mass=1.0,
            box_velocity=(0, 0, -0.05),
        )
        del body
        model = _finalize_pad_cloth(builder)
        solver = _pad_cloth_solver(model, full_surface=True, redundancy_tolerance=None, cloth_solver="vbd")
        _step_pad_cloth(solver, model, 30)
        group = solver._admm_dynamic_ef_contact_groups[0]
        n = int(group.active_count.numpy()[0])
        corners = group.corners.numpy()[:n]
        bary = group.bary.numpy()[:n]
        self.assertGreater(int((corners[:, 2] < 0).sum()), 0)  # edge rows
        self.assertGreater(int((corners[:, 2] >= 0).sum()), 0)  # face rows
        np.testing.assert_allclose(bary.sum(axis=1), 1.0, atol=1.0e-5)

    def test_force_is_distributed_by_barycentric_weight(self):
        """``J^T f``: corner ``a`` receives ``w_a f``; the body receives ``-f`` and its torque about the COM."""
        from newton._src.solvers.coupled.admm_feature_contact import (  # noqa: PLC0415
            contact_ef_accumulate_forces_kernel,
        )

        device = "cpu"
        bary = np.array([0.2, 0.5, 0.3], dtype=np.float32)
        lam = np.array([0.1, -0.2, 0.7], dtype=np.float32)
        u = np.array([0.0, 0.0, 0.05], dtype=np.float32)
        jv = np.array([0.01, 0.0, -0.02], dtype=np.float32)
        W, rho = 0.3, 50.0
        point = np.array([0.01, -0.02, 0.005], dtype=np.float32)
        com = np.array([0.0, 0.0, 0.001], dtype=np.float32)
        particle_f = wp.zeros(4, dtype=wp.vec3, device=device)
        body_f = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        wp.launch(
            contact_ef_accumulate_forces_kernel,
            dim=1,
            inputs=[
                wp.array([1], dtype=int, device=device),
                wp.array([0], dtype=int, device=device),
                wp.array([point], dtype=wp.vec3, device=device),
                wp.array([wp.vec3i(3, 1, 2)], dtype=wp.vec3i, device=device),
                wp.array([bary], dtype=wp.vec3, device=device),
                wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
                wp.array([com], dtype=wp.vec3, device=device),
                rho,
                wp.array([W], dtype=float, device=device),
                wp.array([lam], dtype=wp.vec3, device=device),
                wp.array([u], dtype=wp.vec3, device=device),
                wp.array([jv], dtype=wp.vec3, device=device),
            ],
            outputs=[body_f, particle_f],
            device=device,
        )
        f = W * (lam + rho * W * (u - jv))
        pf = particle_f.numpy()
        np.testing.assert_allclose(pf[3], bary[0] * f, rtol=1e-6)
        np.testing.assert_allclose(pf[1], bary[1] * f, rtol=1e-6)
        np.testing.assert_allclose(pf[2], bary[2] * f, rtol=1e-6)
        np.testing.assert_allclose(pf[0], 0.0)
        bf = body_f.numpy()[0]
        np.testing.assert_allclose(bf[:3], -f, rtol=1e-6)
        np.testing.assert_allclose(bf[3:], np.cross(point - com, -f), rtol=1e-5, atol=1e-9)
        np.testing.assert_allclose(pf.sum(axis=0) + bf[:3], 0.0, atol=1e-7)

    def test_momentum_and_energy(self):
        """A free pad hitting a free triangle interior: linear momentum is kept, kinetic energy does not grow."""
        builder, body = _pad_cloth_world(gravity=0.0, particle_mass=0.01, box_velocity=(0.0, -0.05, -0.2), mu=0.5)
        model = _finalize_pad_cloth(builder)
        solver = _pad_cloth_solver(model, full_surface=True, cloth_solver="semi")
        state_0 = model.state()
        p0, ke0 = _system_momentum(model, state_0)
        rows = []
        state = _step_pad_cloth(
            solver,
            model,
            120,
            observe=lambda s: rows.append(int(solver._admm_dynamic_ef_contact_groups[0].active_count.numpy()[0])),
        )
        p1, ke1 = _system_momentum(model, state)
        self.assertGreater(max(rows), 0)
        np.testing.assert_allclose(p1, p0, atol=1.0e-6 * np.linalg.norm(p0))
        self.assertLessEqual(ke1, ke0 * (1.0 + 1.0e-6))
        # The pad's momentum went into the cloth: the contact acted.
        self.assertGreater(float(state.body_qd.numpy()[body, 2]), -0.2 + 0.01)

    def test_flat_contact_is_left_to_particle_rows(self):
        """A flat plate over every vertex of a cloth: the redundancy filter drops the implied edge/face records.

        Unfiltered, the same contact also gets an edge and a face row per feature, which adds rows (and
        ADMM gain) on the same constraint. The plate moves down into a light, free VBD cloth.
        """
        counts = {}
        for tolerance in (5.0e-4, None):
            builder, body = _pad_cloth_world(
                box_xy=(0.1, 0.1),
                cells=(2, 2),
                box_bottom=0.004,
                box_half=(0.15, 0.15, 0.01),
                gravity=0.0,
                particle_mass=0.001,
                box_velocity=(0.0, 0.0, -0.05),
            )
            model = _finalize_pad_cloth(builder)
            solver = _pad_cloth_solver(model, full_surface=True, redundancy_tolerance=tolerance, cloth_solver="vbd")
            lowest_clearance = []

            def observe(state, body=body, lowest_clearance=lowest_clearance):
                bottom = float(state.body_q.numpy()[body, 2]) - 0.01
                lowest_clearance.append(bottom - float(state.particle_q.numpy()[:, 2].max()))

            _step_pad_cloth(solver, model, 120, observe=observe)
            group = solver._admm_dynamic_ef_contact_groups[0]
            counts[tolerance] = int(group.active_count_max.numpy()[0])
            self.assertGreater(int(solver._admm_dynamic_rp_contact_groups[0].active_count_max.numpy()[0]), 0)
            # The plate pushes the cloth ahead of it: no particle centre gets closer than ~2 mm (radius 3 mm).
            self.assertGreater(min(lowest_clearance), 0.002)
        self.assertEqual(counts[5.0e-4], 0)
        self.assertGreater(counts[None], 0)

    def test_feature_candidates_are_world_local(self):
        """N worlds: N times the single-world candidates, and every world reproduces the single world."""
        finals = {}
        for world_count in (1, 3):
            # A rotationally stiff pad (as on a finger): a solid pad rocks on its contact points and
            # magnifies the float32 rounding of the offset worlds, which is not what this test measures.
            builder, _ = _pad_cloth_world(box_inertia=1.0e-4)
            model = _finalize_pad_cloth(builder, world_count=world_count)
            solver = _pad_cloth_solver(model, full_surface=True)
            group = solver._admm_dynamic_ef_contact_groups[0]
            finals[world_count] = (group.pair_count, _step_pad_cloth(solver, model, 120).body_q.numpy())
        pairs_1, q_1 = finals[1]
        pairs_3, q_3 = finals[3]
        self.assertEqual(pairs_3, 3 * pairs_1)
        # World 0 shares the single world's coordinates and candidate order: bit-identical.
        np.testing.assert_array_equal(q_3[0], q_1[0])

        # Worlds 1 and 2 sit 1 and 2 m away, where float32 rounds their coordinates differently; the resting
        # contact amplifies that to micrometres. Their box corners stay within 0.1 mm of the single world's.
        def corners(q, offset):
            rot = np.array(wp.quat_to_matrix(wp.quat(*q[3:7]))).reshape(3, 3)
            signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=np.float64)
            return (signs * np.array(_PAD_HALF)) @ rot.T + q[:3] - np.array([offset, 0.0, 0.0])

        for world_index in (1, 2):
            deviation = np.abs(corners(q_3[world_index], world_index) - corners(q_1[0], 0.0)).max()
            self.assertLess(deviation, 1.0e-4)

    def test_masked_reset_clears_selected_world_feature_history(self):
        builder, _ = _pad_cloth_world()
        model = _finalize_pad_cloth(builder, world_count=2)
        solver = _pad_cloth_solver(model, full_surface=True)
        state = _step_pad_cloth(solver, model, 60)
        group = solver._admm_dynamic_ef_contact_groups[0]
        n = int(group.active_count.numpy()[0])
        self.assertGreater(n, 0)
        world_of_row = model.particle_world.numpy()[group.particle_ids.numpy()[:n]]
        self.assertEqual(set(world_of_row.tolist()), {0, 1})
        solver.reset(state, world_mask=wp.array([True, False, False], dtype=wp.bool, device=model.device))
        lam = np.linalg.norm(group.lambda_.numpy()[:n], axis=1)
        self.assertTrue(np.all(lam[world_of_row == 0] == 0.0))
        self.assertTrue(np.any(lam[world_of_row == 1] > 0.0))

    def test_flag_off_builds_no_feature_rows(self):
        builder, _ = _pad_cloth_world()
        model = _finalize_pad_cloth(builder)
        solver = _pad_cloth_solver(model, full_surface=False)
        self.assertEqual(solver._admm_dynamic_ef_contact_groups, [])
        self.assertFalse(SolverCoupledADMM.ContactPair("a", "b").full_surface)

    def test_rejects_gamma_with_full_surface(self):
        builder, _ = _pad_cloth_world()
        model = _finalize_pad_cloth(builder)
        with self.assertRaisesRegex(ValueError, "gamma"):
            SolverCoupledADMM(
                model,
                [
                    SolverCoupled.Entry("box", SolverSemiImplicit, bodies=[0]),
                    SolverCoupled.Entry("cloth", SolverVBD, particles=list(range(model.particle_count))),
                ],
                SolverCoupledADMM.Config(
                    gamma=0.1, contact_pairs=[SolverCoupledADMM.ContactPair("box", "cloth", full_surface=True)]
                ),
            )
        with self.assertRaisesRegex(ValueError, "full_surface_redundancy_tolerance"):
            SolverCoupledADMM._validate_config(SolverCoupledADMM.Config(full_surface_redundancy_tolerance=-1.0))

    def test_rejects_full_surface_without_rigid_particle_interface(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for x in (0.0, 0.1):
            builder.add_particle(pos=(x, 0.0, 0.0), vel=wp.vec3(), mass=0.01, radius=0.01)
        model = builder.finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "full_surface=True"):
            SolverCoupledADMM(
                model,
                [
                    SolverCoupled.Entry("a", SolverSemiImplicit, particles=[0]),
                    SolverCoupled.Entry("b", SolverSemiImplicit, particles=[1]),
                ],
                SolverCoupledADMM.Config(contact_pairs=[SolverCoupledADMM.ContactPair("a", "b", full_surface=True)]),
            )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires CUDA")
    def test_feature_rows_are_graph_capturable(self):
        """Captured steps match eager steps (same device, same scene)."""
        device = "cuda:0"
        results = []
        for capture in (False, True):
            builder, body = _pad_cloth_world(box_bottom=0.004)
            model = _finalize_pad_cloth(builder, device=device)
            solver = _pad_cloth_solver(model, full_surface=True)
            state_0, state_1, control = model.state(), model.state(), model.control()
            dt = 1.0 / 600.0

            def step(s0, s1, solver=solver, control=control, dt=dt):
                s0.clear_forces()
                solver.step(s0, s1, control, contacts=None, dt=dt)
                s1.clear_forces()
                solver.step(s1, s0, control, contacts=None, dt=dt)

            step(state_0, state_1)  # warm up (module loads, allocations)
            graph = None
            if capture:
                with wp.ScopedCapture(device=device) as cap:
                    step(state_0, state_1)
                graph = cap.graph
            for _ in range(60):
                if graph is not None:
                    wp.capture_launch(graph)
                else:
                    step(state_0, state_1)
            wp.synchronize_device(device)
            group = solver._admm_dynamic_ef_contact_groups[0]
            self.assertGreater(int(group.active_count_max.numpy()[0]), 0)
            results.append(state_0.body_q.numpy()[body])
        # Eager runs 61 double steps after warm-up, the captured run 1 eager + 60 replays.
        np.testing.assert_allclose(results[1], results[0], atol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
