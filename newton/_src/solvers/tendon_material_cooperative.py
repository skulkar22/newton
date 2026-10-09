# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp-cooperative sigmoid material projection for CUDA.

Every lane of a complete warp participates. The leader controls capstan faces
and minimum-transfer slack allocation; lanes solve coupled span force equations.
FP32 predicts and corrects; FP64 evaluates residuals and certifies publication.
"""

import warp as wp

from .tendon_material_nonlinear import (
    TendonMaterialNonlinearState,
    _exact_block_targets,
    _finish_exact_faces,
    _initialize_component,
    _inverse_extension,
)


@wp.func_native("""
#ifdef __CUDA_ARCH__
__syncwarp(0xffffffffu);
return __shfl_sync(0xffffffffu,value,0);
#else
return value;
#endif
""")
def warp_broadcast(value: int) -> int: ...


@wp.func_native("""
#ifdef __CUDA_ARCH__
for (int d=16;d>0;d/=2) value+=__shfl_down_sync(0xffffffffu,value,d);
return __shfl_sync(0xffffffffu,value,0);
#else
return value;
#endif
""")
def warp_sum(value: wp.float64) -> wp.float64: ...


@wp.func_native("""
#ifdef __CUDA_ARCH__
for (int d=16;d>0;d/=2) value=wp::max(value,__shfl_down_sync(0xffffffffu,value,d));
return __shfl_sync(0xffffffffu,value,0);
#else
return value;
#endif
""")
def warp_max(value: wp.float64) -> wp.float64: ...


@wp.func_native("""
#ifdef __CUDA_ARCH__
auto sum = [&](double v) {
    for(int d=16;d>0;d/=2) v+=__shfl_down_sync(0xffffffffu,v,d);
    return __shfl_sync(0xffffffffu,v,0);
};
auto maximum = [&](double v) {
    for(int d=16;d>0;d/=2) v=wp::max(v,__shfl_down_sync(0xffffffffu,v,d));
    return __shfl_sync(0xffffffffu,v,0);
};
auto minimum = [&](double v) {
    for(int d=16;d>0;d/=2) v=wp::min(v,__shfl_down_sync(0xffffffffu,v,d));
    return __shfl_sync(0xffffffffu,v,0);
};
double capacity=sum(active ? l-floor : 0.0);
if (capacity<=total-floor_sum) return wp::vec3d(-1.0,0.0,lower);
double e=active ? floor+(total-floor_sum)*(l-floor)/capacity : 0.0;
auto evaluate = [&](double x, bool accurate, double &f, double &k) {
    f=0.0; k=1.0;
    if (!active) return;
    if (accurate) {
        double r=wp::max(l-x,1.0e-8), s=wp::max(x,0.0)/r;
        double t=wp::tanh((s-knee)/width), d=0.5*ea_low*(ea_ratio-1.0);
        double ea=ea_low+d*(1.0+t), ds=d*(1.0-t*t)/width;
        f=ea*s;
        float rf=static_cast<float>(r), sf=static_cast<float>(s);
        float factor=(l-x<=1.0e-8 ? 1.0e8f : static_cast<float>(l)/(rf*rf));
        k=static_cast<double>(factor*(static_cast<float>(ea)+sf*static_cast<float>(ds)));
    } else {
        float ll=static_cast<float>(l), xx=static_cast<float>(x);
        float r=wp::max(ll-xx,1.0e-8f), s=wp::max(xx,0.0f)/r;
        float t=wp::tanh((s-static_cast<float>(knee))/static_cast<float>(width));
        float d=0.5f*static_cast<float>(ea_low)*(static_cast<float>(ea_ratio)-1.0f);
        float ea=static_cast<float>(ea_low)+d*(1.0f+t), ds=d*(1.0f-t*t)/static_cast<float>(width);
        f=static_cast<double>(ea*s);
        k=static_cast<double>((ll-xx<=1.0e-8f ? 1.0f/1.0e-8f : ll/(r*r))*(ea+s*ds));
    }
};
double base=lower;

// FP32 does the bulk solve; the FP64 stage below corrects the prediction.
float ll=static_cast<float>(l), bb=static_cast<float>(b), ww=static_cast<float>(w);
float ff=static_cast<float>(floor), ee=static_cast<float>(e), lower32=static_cast<float>(lower);
float low32=static_cast<float>(ea_low), ratio32=static_cast<float>(ea_ratio);
float knee32=static_cast<float>(knee), width32=static_cast<float>(width);
float total32=static_cast<float>(total), base32=lower32;
auto sum32 = [&](float v) {
    for(int d=16;d>0;d/=2) v+=__shfl_down_sync(0xffffffffu,v,d);
    return __shfl_sync(0xffffffffu,v,0);
};
auto max32 = [&](float v) {
    for(int d=16;d>0;d/=2) v=wp::max(v,__shfl_down_sync(0xffffffffu,v,d));
    return __shfl_sync(0xffffffffu,v,0);
};
auto min32 = [&](float v) {
    for(int d=16;d>0;d/=2) v=wp::min(v,__shfl_down_sync(0xffffffffu,v,d));
    return __shfl_sync(0xffffffffu,v,0);
};
auto evaluate32 = [&](float x, float &f, float &k) {
    f=0.0f;k=1.0f;
    if(!active) return;
    float r=wp::max(ll-x,1.0e-8f), s=wp::max(x,0.0f)/r;
    float t=wp::tanh((s-knee32)/width32), d=0.5f*low32*(ratio32-1.0f);
    float ea=low32+d*(1.0f+t), ds=d*(1.0f-t*t)/width32;
    f=ea*s;k=(ll-x<=1.0e-8f ? 1.0f/1.0e-8f : ll/(r*r))*(ea+s*ds);
};
for(int it=0;it<12;++it) {
    float elastic=0.0f,k=1.0f;
    evaluate32(ee,elastic,k);
    float force=elastic+bb, mass=sum32(ee), num=sum32(force/k), den=sum32(ww/k);
    if(it==0) base32=wp::max(lower32,(num+total32-mass)/den);
    float scale32=max32(active ? (wp::abs(elastic)+wp::abs(bb))/ww : 0.0f);
    float error=max32(active ? wp::abs(force/ww-base32) : 0.0f);
    if(error<=2.0e-6f*wp::max(scale32,1.0e-30f)) break;
    float proposed=(num+total32-mass)/den, step=1.0f;
    if(proposed<lower32) step=wp::min(step,0.99f*(base32-lower32)/(base32-proposed));
    float direction=active ? (ww*proposed-force)/k : 0.0f;
    if(direction<0.0f) step=wp::min(step,0.99f*(ee-ff)/(-direction));
    else if(direction>0.0f) step=wp::min(step,0.99f*(ll-ee)/direction);
    step=min32(step);
    bool accepted=false;
    for(int backtrack=0;backtrack<12;++backtrack) {
        float tf=0.0f,tk=1.0f;
        evaluate32(ee+step*direction,tf,tk);
        float trial=max32(active ? wp::abs((tf+bb)/ww-(base32+step*(proposed-base32))) : 0.0f);
        if(trial<=(1.0f-0.0001f*step)*error || trial<=2.0e-6f*wp::max(scale32,1.0e-30f)) {accepted=true;break;}
        step*=0.5f;
    }
    if(!accepted) break;
    ee+=step*direction;
    base32+=step*(proposed-base32);
}
e=wp::max(floor,wp::min(l,static_cast<double>(ee)));
base=wp::max(lower,static_cast<double>(base32));

bool accurate=true, converged=false;

for(int iteration=0;iteration<64;++iteration) {
    // Residual evaluation and extension accumulation retain the small difference
    // between nearly cancelling elastic and damping forces. The linearized
    // correction equations themselves use floats (iterative refinement).
    double elastic=0.0,k64=1.0;
    evaluate(e,true,elastic,k64);
    double force=elastic+b, mass=sum(e);
    float k=static_cast<float>(k64);
    float residual=active ? static_cast<float>(force-w*base) : 0.0f;
    float force_scale=max32(active ? static_cast<float>((wp::abs(elastic)+wp::abs(b))/w) : 0.0f);
    force_scale=wp::max(force_scale,1.0e-30f);
    float error=max32(active ? wp::abs(residual/ww) : 0.0f);
    if(error<=1.0e-10f*force_scale && wp::abs(mass-total)<=32.0*2.220446049250313e-16*(scale+wp::abs(mass))) {
        converged=true;break;
    }
    float num=sum32(residual/k), den=sum32(ww/k);
    float delta_base=(num+static_cast<float>(total-mass))/den;
    float step=1.0f;
    if(base+static_cast<double>(delta_base)<lower)
        step=wp::min(step,0.99f*static_cast<float>(base-lower)/(-delta_base));
    float direction=active ? (ww*delta_base-residual)/k : 0.0f;
    if(direction<0.0f) step=wp::min(step,0.99f*static_cast<float>(e-floor)/(-direction));
    else if(direction>0.0f) step=wp::min(step,0.99f*static_cast<float>(l-e)/direction);
    step=min32(step);
    bool accepted=false;
    for(int line=0;line<32;++line) {
        double trial_base=base+static_cast<double>(step)*static_cast<double>(delta_base), tf=0.0,tk=1.0;
        evaluate(e+static_cast<double>(step)*static_cast<double>(direction),true,tf,tk);
        float trial_error=max32(active ? static_cast<float>(wp::abs((tf+b)/w-trial_base)) : 0.0f);
        if(trial_error<=(1.0f-0.0001f*step)*error || trial_error<=1.0e-10f*force_scale) {accepted=true;break;}
        step*=0.5f;
    }
    if(!accepted) return wp::vec3d(-1.0,e,base);
    e+=static_cast<double>(step)*static_cast<double>(direction);
    base+=static_cast<double>(step)*static_cast<double>(delta_base);
}
return wp::vec3d(converged ? 1.0 : -1.0,e,base);
#else
return wp::vec3d(-1.0,0.0,0.0);
#endif

""")
def warp_newton_block(
    l: wp.float64,
    b: wp.float64,
    w: wp.float64,
    floor: wp.float64,
    total: wp.float64,
    floor_sum: wp.float64,
    scale: wp.float64,
    lower: wp.float64,
    ea_low: wp.float64,
    ea_ratio: wp.float64,
    knee: wp.float64,
    width: wp.float64,
    active: bool,
) -> wp.vec3d: ...


@wp.func
def cooperative_targets(
    state: TendonMaterialNonlinearState,
    n: int,
    offset: int,
    edge_offset: int,
    ea_low: float,
    ea_ratio: float,
    knee: float,
    width: float,
    min_rest: float,
    lane: int,
) -> int:
    first = int(0)
    while first < n:
        last = first
        while last < n - 1 and state.faces[edge_offset + last] != 0:
            last += 1
        code = int(1)
        if first == last:
            if lane == 0:
                code = _exact_block_targets(
                    state, 1, offset + first, edge_offset + first, ea_low, ea_ratio, knee, width, min_rest
                )
            code = warp_broadcast(code)
        else:
            active = lane <= last - first
            index = offset + first + lane
            l = wp.float64(1.0)
            b = wp.float64(0.0)
            w = wp.float64(0.0)
            reference = wp.float64(0.0)
            lower = wp.float64(0.0)
            if active:
                l = wp.float64(state.length[index])
                b = wp.float64(state.damping[index])
                reference = wp.float64(state.reference[index])
                w = wp.float64(1.0)
                for i in range(first, first + lane):
                    cap = wp.float64(state.cap[edge_offset + i])
                    if state.faces[edge_offset + i] > 0:
                        w /= cap
                    else:
                        w *= cap
                lower = wp.max(b, wp.float64(0.0)) / w
            lower = warp_max(lower)
            total = warp_sum(reference)
            scale = warp_sum(wp.abs(reference)) + wp.float64(1.0e-300)
            target = wp.max(lower * w - b, wp.float64(0.0))
            bound = wp.float64(0.0)
            if active:
                bound = wp.max(
                    l * (target / (wp.float64(ea_low) + target)), wp.float64(1.0e-8) * target / wp.float64(ea_low)
                )
            upper_floor_sum = warp_sum(bound)
            taut = total > upper_floor_sum + wp.float64(128.0 * 2.220446049250313e-16) * (scale + upper_floor_sum)
            endpoint = wp.float64(0.0)
            if active and not taut:
                endpoint = _inverse_extension(
                    target, l, wp.float64(ea_low), wp.float64(ea_ratio), wp.float64(knee), wp.float64(width)
                )
            floor_sum = warp_sum(endpoint)
            base = lower
            if total > floor_sum:
                solved = warp_newton_block(
                    l,
                    b,
                    w,
                    endpoint,
                    total,
                    floor_sum,
                    scale,
                    lower,
                    wp.float64(ea_low),
                    wp.float64(ea_ratio),
                    wp.float64(knee),
                    wp.float64(width),
                    active,
                )
                code = int(solved[0])
                endpoint = solved[1]
                base = solved[2]
            bad = wp.float64(0.0)
            if active and code > 0:
                force = base * w
                floor_force = wp.max(b, wp.float64(0.0))
                roundoff = wp.float64(64.0 * 2.220446049250313e-16) * wp.max(force, floor_force)
                plateau = int(wp.abs(force - floor_force) <= roundoff)
                upper_extension = l - wp.float64(min_rest)
                if plateau != 0:
                    endpoint = wp.min(endpoint, upper_extension)
                elif endpoint > upper_extension:
                    bad = wp.float64(1.0)
                if not wp.isfinite(endpoint) or not wp.isfinite(force):
                    bad = wp.float64(1.0)
                state.tension[index] = w
                state.plateau[index] = plateau
                state.endpoint[index] = endpoint
                state.target_log_force[index] = wp.log(wp.max(force, wp.float64(1.0e-300)))
            if warp_max(bad) > wp.float64(0.0):
                code = -1
            code = warp_broadcast(code)
        if code < 0:
            return -1
        first = last + 1
    return 1


@wp.func
def prepare_cooperative(
    state: TendonMaterialNonlinearState,
    n: int,
    row: int,
    span_stride: int,
    edge_stride: int,
    ea_low: float,
    ea_ratio: float,
    knee: float,
    width: float,
    min_rest: float,
    max_iterations: int,
    tolerance: float,
) -> bool:
    state.status[row] = -1
    state.inner_status[row] = 0
    state.outer_iterations[row] = 0
    state.residual[row] = 1.0e30
    state.publication_residual[row] = 1.0e30
    state.storage_error[row] = wp.vec4d(wp.float64(0.0))
    if (
        n < 1
        or n > 32
        or max_iterations < 1
        or max_iterations > 64
        or not wp.isfinite(ea_low)
        or ea_low <= 0.0
        or not wp.isfinite(ea_ratio)
        or ea_ratio < 1.0
        or not wp.isfinite(knee)
        or not wp.isfinite(width)
        or width <= 0.0
        or not wp.isfinite(min_rest)
        or min_rest < 0.0
        or not wp.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        state.valid[row] = 0
        return True
    # Full rolling can put the no-slip trial beyond an individual rest bound;
    # slip must be allowed to repair it. Certify the projected rest lengths,
    # not the trial. A short geometric span can also contain valid slack.
    for i in range(n):
        index = row * span_stride + i
        reference = state.reference[index]
        length = state.length[index]
        damping = state.damping[index]
        if not wp.isfinite(reference) or not wp.isfinite(length) or length < 0.0 or not wp.isfinite(damping):
            state.valid[row] = 0
            return True
        if i < n - 1:
            cap = state.cap[row * edge_stride + i]
            if not wp.isfinite(cap) or cap < 1.0 or cap > 4.0:
                state.valid[row] = 0
                return True
    return _initialize_component(
        state, n, row, span_stride, edge_stride, ea_low, ea_ratio, knee, width, min_rest, tolerance
    )


@wp.func
def publish_cooperative(state: TendonMaterialNonlinearState, n: int, row: int, offset: int, tolerance: float):
    for i in range(n):
        state.output[offset + i] = state.candidate[offset + i]
    state.status[row] = 1
    if state.publication_residual[row] > tolerance:
        state.status[row] = 2
    state.valid[row] = 1


@wp.func
def solve_tendon_material_nonlinear_component(
    state: TendonMaterialNonlinearState,
    n: int,
    allow_warm: int,
    row: int,
    span_stride: int,
    edge_stride: int,
    ea_low: float,
    ea_ratio: float,
    knee: float,
    width: float,
    min_rest: float,
    max_iterations: int,
    tolerance: float,
    lane: int,
):
    # Every lane of a complete warp must call this function together.
    done = int(0)
    if lane == 0:
        done = int(
            prepare_cooperative(
                state,
                n,
                row,
                span_stride,
                edge_stride,
                ea_low,
                ea_ratio,
                knee,
                width,
                min_rest,
                max_iterations,
                tolerance,
            )
        )
    done = warp_broadcast(done)
    if done != 0:
        return
    offset = row * span_stride
    edge_offset = row * edge_stride
    peak = wp.float64(1.0e-300)
    cached = int(0)
    if lane == 0:
        for i in range(n):
            peak = wp.max(peak, state.tension[offset + i])
        cached = int(allow_warm != 0 and state.valid[row] != 0)
        for i in range(n - 1):
            if wp.abs(state.faces[edge_offset + i]) > 1:
                cached = 0
    cached = warp_broadcast(cached)
    attempts = int(0)
    if cached != 0:
        attempts = 1
        if lane == 0:
            state.outer_iterations[row] = attempts
        trial = cooperative_targets(state, n, offset, edge_offset, ea_low, ea_ratio, knee, width, min_rest, lane)
        if trial > 0:
            if lane == 0:
                trial = _finish_exact_faces(
                    state, n, row, offset, edge_offset, ea_low, ea_ratio, knee, width, min_rest, tolerance
                )
                if trial == 0:
                    publish_cooperative(state, n, row, offset, tolerance)
            trial = warp_broadcast(trial)
            if trial == 0:
                return
    if lane == 0:
        for i in range(n):
            state.log_force[offset + i] = wp.log(peak)
        for i in range(n - 1):
            state.faces[edge_offset + i] = 0
        state.valid[row] = 0
        state.status[row] = -4
    done = warp_broadcast(0)
    for iteration in range(max_iterations - attempts):
        if lane == 0:
            state.outer_iterations[row] = attempts + iteration + 1
        trial = cooperative_targets(state, n, offset, edge_offset, ea_low, ea_ratio, knee, width, min_rest, lane)
        if trial < 0:
            if lane == 0:
                state.status[row] = -6
            return
        command = int(1)
        if lane == 0:
            step = wp.float64(1.0)
            blocking_edge = int(-1)
            blocking_face = int(0)
            for i in range(n - 1):
                if state.faces[edge_offset + i] == 0:
                    current = state.log_force[offset + i] - state.log_force[offset + i + 1]
                    target = state.target_log_force[offset + i] - state.target_log_force[offset + i + 1]
                    bound = wp.log(wp.float64(state.cap[edge_offset + i]))
                    trial_step = wp.float64(1.0)
                    face = int(0)
                    if target > bound:
                        trial_step = wp.max(wp.float64(0.0), (bound - current) / (target - current))
                        face = 1
                    elif target < -bound:
                        trial_step = wp.max(wp.float64(0.0), (-bound - current) / (target - current))
                        face = -1
                    if trial_step < step:
                        step = trial_step
                        blocking_edge = i
                        blocking_face = face
            for i in range(n):
                state.log_force[offset + i] += step * (state.target_log_force[offset + i] - state.log_force[offset + i])
            if blocking_edge >= 0:
                state.faces[edge_offset + blocking_edge] = blocking_face
            else:
                release = _finish_exact_faces(
                    state, n, row, offset, edge_offset, ea_low, ea_ratio, knee, width, min_rest, tolerance
                )
                if release < 0:
                    state.status[row] = -5
                    command = -1
                elif release > 0:
                    state.faces[edge_offset + release - 1] = 0
                else:
                    publish_cooperative(state, n, row, offset, tolerance)
                    command = 0
        command = warp_broadcast(command)
        if command != 1:
            return
