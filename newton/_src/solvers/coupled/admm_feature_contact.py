# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for ADMM rigid-soft *feature* contact rows (full-surface contact).

A feature row couples a point on a rigid body to a point on a soft edge or triangle,
``x_s = sum_a w_a x_a`` over the record's corner particles (``-1`` padded, as in
:attr:`newton.Contacts.soft_contact_indices`). Its Jacobian is
``J = [w_0 I, w_1 I, w_2 I, -J_body(point)]``: the relative velocity is
``Jv = sum_a w_a v_a - v_body(point)``, corner ``a`` receives ``w_a f`` and the body ``-f``
(and the matching torque). Normals point from the shape toward the soft point, so the soft
side is endpoint A, as in the rigid-particle rows with ``body_sign = -1``. Because
``sum_a w_a = 1``, the row exchanges equal and opposite linear impulses.

The rows share the generic local Coulomb solve and dual update of the other ADMM contact
groups (:func:`contact_u_update_kernel`, :func:`contact_lambda_update_kernel`); only the
geometry-dependent kernels live here. Warm starting is keyed by the detection's
(feature, shape) candidate pair, recovered from the per-thread replay ids of the detection
pass, so it costs O(1) per row.
"""

from __future__ import annotations

import warp as wp

from ...geometry.sdf_texture import TextureSDFData
from ...geometry.soft_contacts_sdf import eval_shape_sdf
from ...math.spatial import velocity_at_point
from .admm_utils import _PARTICLE_FLAG_ACTIVE, _contact_u_min_from_gap, _interface_weight, _rescale_lambda


@wp.func
def _feature_point(corners: wp.vec3i, bary: wp.vec3, x: wp.array[wp.vec3]) -> wp.vec3:
    """Barycentric point ``sum_a bary[a] * x[corners[a]]`` over the non-negative corners."""
    p = bary[0] * x[corners[0]]
    if corners[1] >= 0:
        p = p + bary[1] * x[corners[1]]
    if corners[2] >= 0:
        p = p + bary[2] * x[corners[2]]
    return p


@wp.func
def _feature_radius(corners: wp.vec3i, particle_radius: wp.array[float]) -> float:
    """Largest corner radius, the soft feature's contact shell (as the detection and VBD use it)."""
    r = particle_radius[corners[0]]
    if corners[1] >= 0:
        r = wp.max(r, particle_radius[corners[1]])
    if corners[2] >= 0:
        r = wp.max(r, particle_radius[corners[2]])
    return r


@wp.func
def _feature_effective_mass(corners: wp.vec3i, bary: wp.vec3, particle_mass: wp.array[float]) -> float:
    """Effective mass ``1 / sum_a (w_a^2 / m_a)`` of the barycentric point.

    A corner with nonpositive mass is kinematic (infinite mass) and adds no inverse mass. A
    point whose corners are all kinematic returns 0, the interface-weight convention for an
    infinite mass.
    """
    inv_mass = float(0.0)
    for k in range(3):
        c = corners[k]
        if c >= 0:
            m = particle_mass[c]
            if m > 0.0:
                inv_mass += bary[k] * bary[k] / m
    if inv_mass > 0.0:
        return 1.0 / inv_mass
    return 0.0


@wp.func
def _shape_local_phi(
    geo: wp.int32,
    scale: wp.vec3,
    X_sw: wp.transform,
    x_world: wp.vec3,
    sdf_index: wp.int32,
    texture_sdf_table: wp.array[TextureSDFData],
) -> float:
    _phi_lower, phi, _grad = eval_shape_sdf(geo, scale, wp.transform_point(X_sw, x_world), sdf_index, texture_sdf_table)
    return phi


@wp.kernel(enable_backward=False)
def contact_ef_clear_buffer_kernel(
    soft_contact_count: wp.array[int],
    soft_contact_tids: wp.array[int],
):
    """Clear the feature-contact detection buffer: its counter and every per-thread replay id."""
    t = wp.tid()
    if t == 0:
        soft_contact_count[0] = 0
    soft_contact_tids[t] = -1


@wp.kernel(enable_backward=False)
def contact_ef_map_pairs_kernel(
    soft_contact_tids: wp.array[int],
    soft_contact_max: int,
    contact_pair: wp.array[int],
):
    """Invert the detection's per-thread record ids: ``contact_pair[record] = candidate pair``."""
    t = wp.tid()
    idx = soft_contact_tids[t]
    if idx >= 0 and idx < soft_contact_max:
        contact_pair[idx] = t


@wp.kernel(enable_backward=False)
def contact_ef_store_clear_kernel(
    prev_count: wp.array[int],
    prev_pair: wp.array[int],
    store_active: wp.array[int],
):
    """Forget the warm-start entries written at the previous refresh."""
    i = wp.tid()
    if i >= prev_count[0]:
        return
    t = prev_pair[i]
    if t >= 0:
        store_active[t] = 0


@wp.kernel(enable_backward=False)
def contact_ef_store_write_kernel(
    active_count: wp.array[int],
    pair_id: wp.array[int],
    W: wp.array[float],
    lambda_: wp.array[wp.vec3],
    store_active: wp.array[int],
    store_W: wp.array[float],
    store_lambda: wp.array[wp.vec3],
    prev_pair: wp.array[int],
    prev_count: wp.array[int],
):
    """Store the rows of the step that just ended by candidate pair, for the next fill's warm start."""
    i = wp.tid()
    n = active_count[0]
    if i == 0:
        prev_count[0] = n
    if i >= n:
        return
    t = pair_id[i]
    prev_pair[i] = t
    if t < 0:
        return
    store_active[t] = 1
    store_W[t] = W[i]
    store_lambda[t] = lambda_[i]


@wp.kernel(enable_backward=False)
def contact_ef_fill_kernel(
    soft_contact_count: wp.array[int],
    soft_contact_max: int,
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[int],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    contact_pair: wp.array[int],
    redundancy_filter: int,
    redundancy_tolerance: float,
    particle_flags: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_type: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_table: wp.array[TextureSDFData],
    body_global_to_local: wp.array[int],
    body_mass: wp.array[float],
    particle_mass: wp.array[float],
    shape_material_mu: wp.array[float],
    particle_mu: float,
    store_active: wp.array[int],
    store_W: wp.array[float],
    store_lambda: wp.array[wp.vec3],
    capacity: int,
    active_count: wp.array[int],
    active_count_max: wp.array[int],
    record_count_max: wp.array[int],
    filtered_count: wp.array[int],
    body_id: wp.array[int],
    point_body_local: wp.array[wp.vec3],
    corners_out: wp.array[wp.vec3i],
    bary_out: wp.array[wp.vec3],
    particle_id: wp.array[int],
    shape_id: wp.array[int],
    pair_id: wp.array[int],
    normal: wp.array[wp.vec3],
    W: wp.array[float],
    friction: wp.array[float],
    lambda_: wp.array[wp.vec3],
):
    """Populate feature rows from edge/face soft-contact records.

    A record is kept only when every corner particle is active, as the per-particle pass requires of
    its particle and :class:`SolverCoupled` of every corner it forwards to an entry. With ``redundancy_filter`` set, a record is dropped when its corners' own (vertex) rows
    already imply it: ``phi(x_s) >= sum_a w_a phi(x_a) - redundancy_tolerance``. The signed
    distance of a convex shape is convex, so the inequality holds with equality exactly where
    the SDF is affine over the feature (a flat face under it), where the corners' gaps enforce
    the feature's gap by linearity; it is strict where a shape edge or corner reaches into the
    feature between its corners, the case the per-particle rows miss.
    """
    i = wp.tid()
    if i == 0:
        # Records detected this step, including any beyond the buffer (overflow diagnostics).
        wp.atomic_max(record_count_max, 0, soft_contact_count[0])
    if i >= wp.min(soft_contact_count[0], soft_contact_max):
        return

    corners = soft_contact_indices[i]
    s = soft_contact_shape[i]
    if corners[0] < 0 or corners[1] < 0 or s < 0:
        return
    for k in range(3):
        c = corners[k]
        if c >= 0:
            if (particle_flags[c] & _PARTICLE_FLAG_ACTIVE) == 0:
                return
    b = shape_body[s]
    if b < 0:
        return
    b_local = body_global_to_local[b]
    if b_local < 0:
        return

    n = soft_contact_normal[i]
    n_len = wp.length(n)
    if n_len <= 0.0:
        return
    n = n / n_len
    bary = soft_contact_barycentric[i]

    if redundancy_filter != 0:
        X_sw = wp.transform_inverse(wp.transform_multiply(body_q[b], shape_transform[s]))
        geo = shape_type[s]
        scale = shape_scale[s]
        sdf_index = shape_sdf_index[s]
        phi_point = _shape_local_phi(
            geo, scale, X_sw, _feature_point(corners, bary, particle_q), sdf_index, texture_sdf_table
        )
        phi_corners = bary[0] * _shape_local_phi(geo, scale, X_sw, particle_q[corners[0]], sdf_index, texture_sdf_table)
        phi_corners += bary[1] * _shape_local_phi(
            geo, scale, X_sw, particle_q[corners[1]], sdf_index, texture_sdf_table
        )
        if corners[2] >= 0:
            phi_corners += bary[2] * _shape_local_phi(
                geo, scale, X_sw, particle_q[corners[2]], sdf_index, texture_sdf_table
            )
        if phi_point >= phi_corners - redundancy_tolerance:
            wp.atomic_add(filtered_count, 0, 1)
            return

    dst = wp.atomic_add(active_count, 0, 1)
    if dst >= capacity:
        wp.atomic_min(active_count, 0, capacity)
        wp.atomic_max(active_count_max, 0, capacity)
        return
    wp.atomic_max(active_count_max, 0, dst + 1)

    weight = _interface_weight(body_mass[b], _feature_effective_mass(corners, bary, particle_mass))
    lambda0 = wp.vec3(0.0, 0.0, 0.0)
    t = contact_pair[i]
    if t >= 0:
        if store_active[t] != 0:
            lambda0 = _rescale_lambda(store_lambda[t], store_W[t], weight)

    body_id[dst] = b_local
    point_body_local[dst] = soft_contact_body_pos[i]
    corners_out[dst] = corners
    bary_out[dst] = bary
    particle_id[dst] = corners[0]
    shape_id[dst] = s
    pair_id[dst] = t
    normal[dst] = n
    W[dst] = weight
    # Geometric-mean combining of shape and particle friction, as for the rigid-particle rows.
    friction[dst] = wp.sqrt(shape_material_mu[s] * particle_mu)
    lambda_[dst] = lambda0


@wp.kernel(enable_backward=False)
def contact_ef_compute_Jv_kernel(
    active_count: wp.array[int],
    body_id: wp.array[int],
    point_body_local: wp.array[wp.vec3],
    corners: wp.array[wp.vec3i],
    bary: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_qd: wp.array[wp.spatial_vector],
    particle_qd: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
):
    """Relative velocity ``sum_a w_a v_a - v_body(point)`` of the active feature rows."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    b = body_id[i]
    xform = body_q[b]
    world_pt = wp.transform_point(xform, point_body_local[i])
    arm = world_pt - wp.transform_point(xform, body_com[b])
    body_v = velocity_at_point(body_qd[b], arm)
    Jv[i] = _feature_point(corners[i], bary[i], particle_qd) - body_v


@wp.kernel(enable_backward=False)
def contact_ef_compute_u_min_kernel(
    active_count: wp.array[int],
    body_id: wp.array[int],
    point_body_local: wp.array[wp.vec3],
    corners: wp.array[wp.vec3i],
    bary: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    baumgarte: float,
    dt: float,
    u_min: wp.array[float],
):
    """Minimum normal velocity of the active feature rows from their surface gap."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    c = corners[i]
    body_pt = wp.transform_point(body_q[body_id[i]], point_body_local[i])
    gap = wp.dot(normal[i], _feature_point(c, bary[i], particle_q) - body_pt) - _feature_radius(c, particle_radius)
    u_min[i] = _contact_u_min_from_gap(gap, baumgarte, dt)


@wp.kernel(enable_backward=False)
def contact_ef_accumulate_forces_kernel(
    active_count: wp.array[int],
    body_id: wp.array[int],
    point_body_local: wp.array[wp.vec3],
    corners: wp.array[wp.vec3i],
    bary: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    rho: float,
    W: wp.array[float],
    lambda_k: wp.array[wp.vec3],
    u_k: wp.array[wp.vec3],
    Jv_k: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    particle_f: wp.array[wp.vec3],
):
    """Splat ``J^T f`` of the active feature rows: ``w_a f`` on each corner, ``-f`` on the body."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    b = body_id[i]
    W_i = W[i]
    force = W_i * (lambda_k[i] + rho * W_i * (u_k[i] - Jv_k[i]))

    xform = body_q[b]
    world_pt = wp.transform_point(xform, point_body_local[i])
    arm = world_pt - wp.transform_point(xform, body_com[b])
    force_body = -force
    wp.atomic_add(body_f, b, wp.spatial_vector(force_body, wp.cross(arm, force_body)))

    c = corners[i]
    w = bary[i]
    wp.atomic_add(particle_f, c[0], w[0] * force)
    if c[1] >= 0:
        wp.atomic_add(particle_f, c[1], w[1] * force)
    if c[2] >= 0:
        wp.atomic_add(particle_f, c[2], w[2] * force)
