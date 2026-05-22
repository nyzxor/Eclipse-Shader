/*
 * Voxel Ray Tracing – DDA-based GI and specular-reflection fallback.
 *
 * Uses the 128^3 block-occupancy grid (imgVoxelMask) populated each frame
 * during the shadow pass, and the LPV to colour indirect hits.
 *
 * Hard requirements: IS_LPV_ENABLED, MC_GL_ARB_shader_image_load_store.
 * Called from composite1.fsh (GI) and specular.glsl (reflection fallback).
 */

#ifndef VOXEL_RT_INCLUDED
#define VOXEL_RT_INCLUDED

#if defined IS_LPV_ENABLED && defined MC_GL_ARB_shader_image_load_store

// Read-only view of the voxel occupancy grid.
// (The writable declaration lives in voxel_common.glsl / RENDER_SHADOW path.)
#ifndef VOXEL_MASK_DECLARED
#define VOXEL_MASK_DECLARED
layout(r16ui) uniform readonly uimage3D imgVoxelMask;
#endif

// Hemisphere sample count for VOXEL_GI (user-adjustable in settings)
#ifndef VOXEL_GI_SAMPLES
    #define VOXEL_GI_SAMPLES 4
#endif
// DDA iteration budget for diffuse GI rays (~1 block per step)
#ifndef VOXEL_GI_STEPS
    #define VOXEL_GI_STEPS 48
#endif
// DDA iteration budget for specular reflection rays
#ifndef VOXEL_REFL_STEPS
    #define VOXEL_REFL_STEPS 96
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  DDA core traversal
// ─────────────────────────────────────────────────────────────────────────────

// Trace a ray through the 128³ block-occupancy voxel grid.
//
//   playerOrigin  feet-relative world-space origin (same coords as feetPlayerPos)
//   rayDir        normalised world-space direction
//   maxSteps      DDA iteration budget (each step crosses exactly one voxel)
//   hitPos        feet-relative position of the hit surface face (out)
//   hitNormal     outward face normal at hit, axis-aligned ±X/Y/Z (out)
//
// Returns true when a solid block is found inside the grid bounds.
bool VoxelDDA(
    vec3  playerOrigin,
    vec3  rayDir,
    int   maxSteps,
    out vec3 hitPos,
    out vec3 hitNormal
) {
    // Small bias along the ray to skip the voxel we are standing inside
    const float BIAS = 0.02;

    // Transform origin to voxel-grid space  [0, VoxelSize)
    vec3  cameraOff = fract(cameraPosition);
    vec3  gO = playerOrigin + cameraOff
             + vec3(VoxelSize3) * 0.5
             + rayDir * BIAS;

    ivec3 voxel = ivec3(floor(gO));
    ivec3 step  = ivec3(
        rayDir.x >= 0.0 ? 1 : -1,
        rayDir.y >= 0.0 ? 1 : -1,
        rayDir.z >= 0.0 ? 1 : -1);

    // Distance between consecutive boundary crossings along each axis
    vec3 deltaDist = vec3(
        abs(rayDir.x) > 1e-7 ? 1.0 / abs(rayDir.x) : 1e10,
        abs(rayDir.y) > 1e-7 ? 1.0 / abs(rayDir.y) : 1e10,
        abs(rayDir.z) > 1e-7 ? 1.0 / abs(rayDir.z) : 1e10);

    // Parametric t to the first boundary crossing on each axis
    vec3 sideDist = max(vec3(
        abs(rayDir.x) > 1e-7
            ? (float(voxel.x) + (step.x > 0 ? 1.0 : 0.0) - gO.x) / rayDir.x : 1e10,
        abs(rayDir.y) > 1e-7
            ? (float(voxel.y) + (step.y > 0 ? 1.0 : 0.0) - gO.y) / rayDir.y : 1e10,
        abs(rayDir.z) > 1e-7
            ? (float(voxel.z) + (step.z > 0 ? 1.0 : 0.0) - gO.z) / rayDir.z : 1e10),
        vec3(0.0));  // never negative at start

    hitNormal = vec3(0.0, 1.0, 0.0);  // default (overwritten on first step)

    for (int i = 0; i < maxSteps; i++) {

        // Exit if ray left the grid
        if (any(lessThan(voxel, ivec3(0))) ||
            any(greaterThanEqual(voxel, ivec3(VoxelSize3)))) break;

        if (imageLoad(imgVoxelMask, voxel).r != uint(BLOCK_EMPTY)) {
            // Face-centre just in front of the solid voxel, in player space
            hitPos = vec3(voxel) + 0.5 + hitNormal * 0.5
                   - cameraOff - vec3(VoxelSize3) * 0.5;
            return true;
        }

        // Step to the closest next boundary
        if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
            sideDist.x += deltaDist.x;
            voxel.x    += step.x;
            hitNormal   = vec3(-float(step.x), 0.0, 0.0);
        } else if (sideDist.y < sideDist.z) {
            sideDist.y += deltaDist.y;
            voxel.y    += step.y;
            hitNormal   = vec3(0.0, -float(step.y), 0.0);
        } else {
            sideDist.z += deltaDist.z;
            voxel.z    += step.z;
            hitNormal   = vec3(0.0, 0.0, -float(step.z));
        }
    }
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Lighting at a voxel hit
// ─────────────────────────────────────────────────────────────────────────────

// Samples the LPV half a block in front of the hit face to get the indirect
// light colour that would illuminate a surface at that position.
vec3 SampleVoxelLighting(vec3 hitPos, vec3 hitNormal) {
    vec4 lpv = SampleLpvLinear(GetLpvPosition(hitPos + hitNormal * 0.5));
    return GetLpvBlockLight(lpv);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Hemisphere global illumination  (drop-in for ApplySSRT)
// ─────────────────────────────────────────────────────────────────────────────

// Traces VOXEL_GI_SAMPLES cosine-weighted hemisphere rays from viewPos and
// returns the estimated indirect irradiance, combining sky occlusion with LPV
// colour bleeding from nearby surfaces.
//
// Signature is identical to ApplySSRT so the call-site in composite1 works
// for both SSRT_AO_GI and VOXEL_GI modes without code duplication.
vec3 ApplyVoxelRT(
    in vec3  unchangedIndirect,
    in vec3  blockLightColor,
    in vec3  minimumLightColor,
    in vec3  viewPos,
    in vec3  normal,
    in vec3  noise,
    in float lightmap,
    in bool  isGrass,
    in bool  isLOD
) {
    vec3 playerPos = mat3(gbufferModelViewInverse) * viewPos
                   + gbufferModelViewInverse[3].xyz;

    int  nrays    = VOXEL_GI_SAMPLES;
    vec3 radiance  = vec3(0.0);
    vec3 occlusion = vec3(0.0);

    for (int i = 0; i < nrays; i++) {
        int  seed   = (frameCounter % 40000) * nrays + i;
        vec2 Xi     = fract(R2_samples(seed) + noise.xy);
        vec3 rayDir = TangentToWorld(normal, cosineHemisphereSample(Xi));

        // Sky contribution for this ray direction (same weighting as ApplySSRT)
        vec3 sky;
        #ifdef OVERWORLD_SHADER
            sky = unchangedIndirect
                * (max(rayDir.y, pow(1.0 - lightmap, 2.0)) * 0.95 + 0.05) * 1.25;
        #else
            sky = unchangedIndirect;
        #endif

        radiance += sky;

        vec3 hitPos, hitNormal;
        if (VoxelDDA(playerPos, rayDir, VOXEL_GI_STEPS, hitPos, hitNormal)) {
            // Occlude the sky contribution and inject the LPV colour bounce
            vec3 bounce = SampleVoxelLighting(hitPos, hitNormal);
            occlusion += sky - bounce * 0.5;
        }
    }

    float threshold = isGrass ? 0.8 : (pow(1.0 - lightmap, 2.0) * 0.9 + 0.1);
    return max((radiance - occlusion) / float(nrays),
               unchangedIndirect * threshold);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Specular reflection fallback  (fills gaps left by screen-space RT)
// ─────────────────────────────────────────────────────────────────────────────

// Called after getEnvironmentReflections() when VOXEL_RT_REFLECTIONS is set.
// enviornmentReflectA: the .a from getEnvironmentReflections()
//   1.0 = SSR fully covered this pixel → nothing to do
//   0.0 = SSR fully missed  → trace a voxel ray
// Returns the extra colour contribution to add to enviornmentReflection.rgb.
vec3 VoxelReflectionFallback(
    vec3  playerPos,
    vec3  reflectedDir,
    float enviornmentReflectA
) {
    float miss = 1.0 - enviornmentReflectA;
    if (miss < 0.01) return vec3(0.0);  // SSR already covered this pixel

    vec3 hitPos, hitNormal;
    if (VoxelDDA(playerPos, reflectedDir, VOXEL_REFL_STEPS, hitPos, hitNormal)) {
        // Scale to the typical sky-reflection brightness range used elsewhere
        return SampleVoxelLighting(hitPos, hitNormal) * 15.0 * miss;
    }
    return vec3(0.0);
}

#endif  // IS_LPV_ENABLED && MC_GL_ARB_shader_image_load_store
#endif  // VOXEL_RT_INCLUDED
