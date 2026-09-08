/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * 
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace
{
    __device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs,
        const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
    {
        glm::vec3 pos = means[idx];
        glm::vec3 dir = glm::normalize(pos - campos);
        glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
        glm::vec3 result = SH_C0 * sh[0];

        if (deg > 0)
        {
            float x = dir.x, y = dir.y, z = dir.z;
            result = result
                - SH_C1 * y * sh[1]
                + SH_C1 * z * sh[2]
                - SH_C1 * x * sh[3];

            if (deg > 1)
            {
                float xx = x*x, yy = y*y, zz = z*z;
                float xy = x*y, yz = y*z, xz = x*z;
                result = result
                    + SH_C2[0] * xy          * sh[4]
                    + SH_C2[1] * yz          * sh[5]
                    + SH_C2[2] * (2.f*zz - xx - yy) * sh[6]
                    + SH_C2[3] * xz          * sh[7]
                    + SH_C2[4] * (xx - yy)   * sh[8];
            }
        }
        result += 0.5f;
        clamped[3*idx+0] = (result.x < 0);
        clamped[3*idx+1] = (result.y < 0);
        clamped[3*idx+2] = (result.z < 0);
        return glm::max(result, 0.0f);
    }

    __device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y,
        float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
    {
        float3 t = transformPoint4x3(mean, viewmatrix);
        const float limx = 1.3f * tan_fovx, limy = 1.3f * tan_fovy;
        const float txtz = t.x / t.z, tytz = t.y / t.z;
        t.x = min(limx, max(-limx, txtz)) * t.z;
        t.y = min(limy, max(-limy, tytz)) * t.z;

        glm::mat3 J = glm::mat3(
            focal_x / t.z, 0.f, -(focal_x * t.x) / (t.z * t.z),
            0.f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
            0, 0, 0);
        glm::mat3 W = glm::mat3(
            viewmatrix[0], viewmatrix[4], viewmatrix[8],
            viewmatrix[1], viewmatrix[5], viewmatrix[9],
            viewmatrix[2], viewmatrix[6], viewmatrix[10]);
        glm::mat3 T   = W * J;
        glm::mat3 Vrk = glm::mat3(
            cov3D[0], cov3D[1], cov3D[2],
            cov3D[1], cov3D[3], cov3D[4],
            cov3D[2], cov3D[4], cov3D[5]);
        glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;
        return {float(cov[0][0]), float(cov[0][1]), float(cov[1][1])};
    }

    __device__ void computeCov3D(const glm::vec3 scale, float mod,
        const glm::vec4 rot, float* cov3D)
    {
        glm::mat3 S = glm::mat3(0.f);
        S[0][0] = mod * scale.x;
        S[1][1] = mod * scale.y;
        S[2][2] = mod * scale.z;

        float r = rot.x, x = rot.y, y = rot.z, z = rot.w;
        glm::mat3 R = glm::mat3(
            1.f - 2.f*(y*y + z*z), 2.f*(x*y - r*z),       2.f*(x*z + r*y),
            2.f*(x*y + r*z),       1.f - 2.f*(x*x + z*z), 2.f*(y*z - r*x),
            2.f*(x*z - r*y),       2.f*(y*z + r*x),       1.f - 2.f*(x*x + y*y));
        glm::mat3 M     = S * R;
        glm::mat3 Sigma = glm::transpose(M) * M;
        cov3D[0] = Sigma[0][0]; cov3D[1] = Sigma[0][1]; cov3D[2] = Sigma[0][2];
        cov3D[3] = Sigma[1][1]; cov3D[4] = Sigma[1][2]; cov3D[5] = Sigma[2][2];
    }

    __device__ inline float2 projectToUV(
        float3 hit_dist, float3 dU, float3 dV,
        float dUU, float dVV, float dUV, float D)
    {
        float bU = hit_dist.x*dU.x + hit_dist.y*dU.y + hit_dist.z*dU.z;
        float bV = hit_dist.x*dV.x + hit_dist.y*dV.y + hit_dist.z*dV.z;
        return {(bU*dVV - bV*dUV) / D,
                (bV*dUU - bU*dUV) / D};
    }
}

namespace FORWARD
{
    template <int C>
    __global__ void preprocessCUDA(
        int P, int D, int M,
        const float* orig_points, const glm::vec3* scales,
        const float scale_modifier, const glm::vec4* rotations,
        const float* opacities, const float* shs, bool* clamped,
        const float* cov3D_precomp, const float* colors_precomp,
        const float* viewmatrix, const float* projmatrix,
        const glm::vec3* cam_pos,
        const int W, int H,
        const float tan_fovx, float tan_fovy,
        const float focal_x, float focal_y,
        int* radii, float2* points_xy_image,
        float* depths, float* cov3Ds, float* rgb,
        float4* conic_opacity,
        const dim3 grid, uint32_t* tiles_touched,
        bool prefiltered, bool antialiasing,
        const float* boxmin, const float* boxmax,
        const float* uvs, const cudaTextureObject_t* tex_objs,
		const int* tex_idx,
        float uvFixedDepth)
    {
        auto idx = cg::this_grid().thread_rank();
        if (idx >= P) return;

        radii[idx]         = 0;
        tiles_touched[idx] = 0;

        float3 p_orig = {orig_points[3*idx], orig_points[3*idx+1], orig_points[3*idx+2]};

        float3 p_view;
        if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
            return;

        // Gaussians suppressed to near-zero opacity contribute nothing but still
        // occupy tiles and consume transmittance T. Early-exit here so tiles_touched
        // stays 0, removing them from the sort/render pipeline entirely.
        if (opacities[idx] < 1.f / 255.f)
            return;

        const float* cov3D_ptr = cov3D_precomp
            ? (cov3D_precomp + idx * 6)
            : (cov3Ds        + idx * 6);
        if (!cov3D_precomp)
            computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);

        float3 cov = computeCov2D(p_orig, focal_x, focal_y,
                                   tan_fovx, tan_fovy, cov3D_ptr, viewmatrix);

        const float filter_size = 0.3f;
        const float det_orig    = fmaxf(1e-8f, cov.x*cov.z - cov.y*cov.y);
        cov.x += filter_size;
        cov.z += filter_size;
        const float det_filter  = cov.x*cov.z - cov.y*cov.y;
        float conv_scaling      = antialiasing ? sqrtf(det_orig / det_filter) : 1.0f;

        float det_inv = 1.f / det_filter;
        float3 conic  = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};
        float  mid    = 0.5f * (cov.x + cov.z);
        float  lambda1   = mid + sqrtf(fmaxf(0.1f, mid*mid - det_filter));
        float  my_radius = ceilf(3.f * sqrtf(lambda1));

        float4 p_hom      = transformPoint4x4(p_orig, projmatrix);
        float2 point_image = {ndc2Pix(p_hom.x / p_hom.w, W),
                               ndc2Pix(p_hom.y / p_hom.w, H)};

        uint2 rect_min, rect_max;
        getRect(point_image, my_radius, rect_min, rect_max, grid);
        if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
            return;

        if (colors_precomp)
        {
            for (int i = 0; i < C; i++)
                rgb[idx*C + i] = colors_precomp[idx*C + i];
        }
        else
        {
            glm::vec3 c = computeColorFromSH(idx, D, M,
                (glm::vec3*)orig_points, *cam_pos, shs, clamped);
            rgb[idx*C + 0] = c.x;
            rgb[idx*C + 1] = c.y;
            rgb[idx*C + 2] = c.z;
        }
        // UV Gaussians get a small depth bias so they sort before (are composited before)
        // any non-UV Gaussian at the same surface position. The natural front-to-back
        // transmittance compositing then causes the UV Gaussian to occlude the non-UV
        // Gaussian, without needing opacity suppression.
        if (uvs && uvs[idx * 2] >= 0.f && uvFixedDepth > 0.f)
            depths[idx] = uvFixedDepth - 0.05f;
        else if (uvs && uvs[idx * 2] >= 0.f)
            depths[idx] = p_view.z - 0.05f;
        else
            depths[idx] = p_view.z;
        radii[idx]           = (int)my_radius;
        points_xy_image[idx] = point_image;
        conic_opacity[idx]   = {conic.x, conic.y, conic.z,
                                 opacities[idx] * conv_scaling};
        tiles_touched[idx]   = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
    }

    template <uint32_t CHANNELS>
    __global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
    renderCUDA(
        const uint2*   __restrict__ ranges,
        const uint32_t* __restrict__ point_list,
        int W, int H,
        const float2* __restrict__ points_xy_image,
        const float*  __restrict__ features,
        const float4* __restrict__ conic_opacity,
        float*        __restrict__ final_T,
        uint32_t*     __restrict__ n_contrib,
        const float*  __restrict__ bg_color,
        float*        __restrict__ out_color,
        const float*  __restrict__ depths,
        float*        __restrict__ invdepth,
        const float*  __restrict__ means3D,
        const float*  __restrict__ dUs,
        const float*  __restrict__ dVs,
        const float*  __restrict__ uvs,
        const float*  __restrict__ surfaceDists,
        const float*  __restrict__ viewmatrix,
        const float*  __restrict__ projmatrix,
        float focal_x, float focal_y,
        const cudaTextureObject_t* tex_objs,
		const int* tex_idx,
        const float*  __restrict__ cov3Ds,
        const cudaTextureObject_t* normal_texs)
    {
        auto block = cg::this_thread_block();
        uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
        uint2 pix = {
            block.group_index().x * BLOCK_X + block.thread_index().x,
            block.group_index().y * BLOCK_Y + block.thread_index().y};

        // No thread may leave this kernel early. Every thread has to keep
        // reaching the block.sync() calls below AND keep refilling its own
        // slot of the collected_* shared arrays each batch. Returning here
        // left out-of-bounds threads out of both, so the still-active threads
        // read stale shared-memory entries -- 16x16 blocks of wrong color
        // that flicker as the camera moves. Flag them done instead (upstream
        // 3DGS does the same) and gate the final write on `inside`.
        const bool inside = (pix.x < W) && (pix.y < H);
        bool done = !inside;

        uint32_t pix_id = W * pix.y + pix.x;
        float2   pixf   = {(float)pix.x, (float)pix.y};
        uint2    range  = ranges[block.group_index().y * horizontal_blocks
                                + block.group_index().x];
        int toDo = range.y - range.x;

        __shared__ int    collected_id[BLOCK_SIZE];
        __shared__ float2 collected_xy[BLOCK_SIZE];
        __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
        __shared__ float2 collected_uv[BLOCK_SIZE];
        __shared__ int    collected_texidx[BLOCK_SIZE];

        const float3 cam_pos = {
            -(viewmatrix[0]*viewmatrix[12] + viewmatrix[1]*viewmatrix[13] + viewmatrix[2]*viewmatrix[14]),
            -(viewmatrix[4]*viewmatrix[12] + viewmatrix[5]*viewmatrix[13] + viewmatrix[6]*viewmatrix[14]),
            -(viewmatrix[8]*viewmatrix[12] + viewmatrix[9]*viewmatrix[13] + viewmatrix[10]*viewmatrix[14])};

        const float rdx = (pixf.x + 0.5f - W * 0.5f) / focal_x;
        const float rdy = (pixf.y + 0.5f - H * 0.5f) / focal_y;
        const float3 ray_d = {
            viewmatrix[0]*rdx + viewmatrix[1]*rdy + viewmatrix[2],
            viewmatrix[4]*rdx + viewmatrix[5]*rdy + viewmatrix[6],
            viewmatrix[8]*rdx + viewmatrix[9]*rdy + viewmatrix[10]};

        const float3 dray_dpx = {
            viewmatrix[0] / focal_x,
            viewmatrix[4] / focal_x,
            viewmatrix[8] / focal_x};
        const float3 dray_dpy = {
            viewmatrix[1] / focal_y,
            viewmatrix[5] / focal_y,
            viewmatrix[9] / focal_y};

        float    T = 1.f;
        float    C[CHANNELS] = {0};
        uint32_t last_contributor = 0;

        for (int i = 0; i < toDo; i += BLOCK_SIZE)
        {
            // Whole-block vote: leave the batch loop only when EVERY thread is
            // done. __syncthreads_count is itself a barrier all threads reach
            // (toDo and the step are uniform across the block), and it returns
            // the same value to all of them, so the block always breaks
            // together -- never one thread at a time.
            if (__syncthreads_count(done) == BLOCK_SIZE)
                break;

            int progress = i + block.thread_rank();
            if (range.x + progress < range.y)
            {
                int coll_id = point_list[range.x + progress];
                collected_id[block.thread_rank()]            = coll_id;
                collected_xy[block.thread_rank()]            = points_xy_image[coll_id];
                collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
                collected_uv[block.thread_rank()] = uvs
                    ? make_float2(uvs[coll_id*2], uvs[coll_id*2+1])
                    : make_float2(-1.f, -1.f);
                collected_texidx[block.thread_rank()] = tex_idx ? tex_idx[coll_id] : -1;
            }
            block.sync();

            for (int j = 0; !done && j < min(BLOCK_SIZE, toDo - i); j++)
            {
                float2 d    = {collected_xy[j].x - pixf.x,
                               collected_xy[j].y - pixf.y};
                float4 con_o = collected_conic_opacity[j];
                float power  = -0.5f * (con_o.x*d.x*d.x + con_o.z*d.y*d.y)
                               - con_o.y * d.x * d.y;
                if (power > 0.f) continue;

                float alpha = fminf(0.99f, con_o.w * expf(power));
                if (alpha < 1.f / 255.f) continue;

                int curr_id = collected_id[j];

                float C_frag[CHANNELS];
                for (int ch = 0; ch < CHANNELS; ch++)
                    C_frag[ch] = features[curr_id * CHANNELS + ch];

                // Which texture this Gaussian samples. Each one belongs to at most a
                // single patch, so the index alone decides -- no search, and patches
                // painted at different times coexist without interfering.
                const int  my_tex     = collected_texidx[j];
                const bool is_uv_gauss = (tex_objs != nullptr && my_tex >= 0
                                          && collected_uv[j].x >= 0.0f);
                bool tex_applied = false;
                bool is_backface = false;

                if (is_uv_gauss)
                {
                    float2 uv  = collected_uv[j];
                    float3 p   = {means3D[curr_id*3], means3D[curr_id*3+1], means3D[curr_id*3+2]};
                    float3 dU  = {dUs[curr_id*3],        dUs[curr_id*3+1],     dUs[curr_id*3+2]};
                    float3 dV  = {dVs[curr_id*3],        dVs[curr_id*3+1],     dVs[curr_id*3+2]};

                    // surface normal n = dU x dV
                    float3 n   = {dU.y*dV.z - dU.z*dV.y,
                                   dU.z*dV.x - dU.x*dV.z,
                                   dU.x*dV.y - dU.y*dV.x};
                    float denom = ray_d.x*n.x + ray_d.y*n.y + ray_d.z*n.z;
                    is_backface = (denom >= -1e-6f);  // ray not pointing into surface

                    if (denom < -1e-6f)   // front-face only: ray must point into surface
                    {
                        // Ray-tangent-plane intersection
                        float t_hit = ((p.x - cam_pos.x)*n.x
                                     + (p.y - cam_pos.y)*n.y
                                     + (p.z - cam_pos.z)*n.z) / denom;

                        if (t_hit > 0.f)
                        {
                            float3 hit_dist = {
                                cam_pos.x + t_hit*ray_d.x - p.x,
                                cam_pos.y + t_hit*ray_d.y - p.y,
                                cam_pos.z + t_hit*ray_d.z - p.z};
                            float dUU = dU.x*dU.x + dU.y*dU.y + dU.z*dU.z;
                            float dVV = dV.x*dV.x + dV.y*dV.y + dV.z*dV.z;
                            float dUV = dU.x*dV.x + dU.y*dV.y + dU.z*dV.z;
                            float D   = dUU*dVV - dUV*dUV;

                            if (fabsf(D) > 1e-10f)
                            {
                                // UV offset at primary ray hit
                                float2 uv_offset = projectToUV(
                                    hit_dist, dU, dV, dUU, dVV, dUV, D);
                                uv.x += uv_offset.x;
                                uv.y += uv_offset.y;

                                float dt_dpx = -t_hit *
                                    (dray_dpx.x*n.x + dray_dpx.y*n.y + dray_dpx.z*n.z) / denom;
                                float dt_dpy = -t_hit *
                                    (dray_dpy.x*n.x + dray_dpy.y*n.y + dray_dpy.z*n.z) / denom;

                                float3 dhit_dpx = {
                                    dt_dpx*ray_d.x + t_hit*dray_dpx.x,
                                    dt_dpx*ray_d.y + t_hit*dray_dpx.y,
                                    dt_dpx*ray_d.z + t_hit*dray_dpx.z};
                                float3 dhit_dpy = {
                                    dt_dpy*ray_d.x + t_hit*dray_dpy.x,
                                    dt_dpy*ray_d.y + t_hit*dray_dpy.y,
                                    dt_dpy*ray_d.z + t_hit*dray_dpy.z};

                                float2 dPdx = projectToUV(
                                    dhit_dpx, dU, dV, dUU, dVV, dUV, D);
                                float2 dPdy = projectToUV(
                                    dhit_dpy, dU, dV, dUU, dVV, dUV, D);

                                // Nothing to sample outside the texture image.
                                if (uv.x < 0.f || uv.x > 1.f ||
                                    uv.y < 0.f || uv.y > 1.f) continue;

                                float4 tc = tex2DGrad<float4>(
                                    tex_objs[my_tex], uv.x, uv.y, dPdx, dPdy);

                                // The texture is masked to the patch triangles: tc.w is 1
                                // inside the patch and ramps to 0 across a ~1-texel edge
                                // (plus the dilation ring). Gate on it so an edge Gaussian's
                                // kernel tail stops splatting textured pixels past the patch
                                // boundary -- that tail is the fuzzy halo around the decal.
                                // For a hard cut instead of an anti-aliased edge, raise this
                                // threshold to 0.5f and drop the `alpha *= tc.w` below.
                                if (tc.w < 1.0f / 255.0f) continue;

                                // Replace SH color with texture color; carry the mask's
                                // edge ramp into alpha for a clean anti-aliased border.
                                C_frag[0] = tc.x;
                                C_frag[1] = tc.y;
                                C_frag[2] = tc.z;
                                alpha *= tc.w;

                                // Lambertian shading from normal map
                                if ((normal_texs != nullptr && my_tex >= 0 && normal_texs[my_tex] != 0)) {
                                    float4 nc = tex2D<float4>(normal_texs[my_tex], uv.x, uv.y);
                                    if (nc.w > 0.5f) {
                                        float3 nw = {nc.x*2.f-1.f, nc.y*2.f-1.f, nc.z*2.f-1.f};
                                        // Headlight: light from camera position toward Gaussian center
                                        float3 toL = {cam_pos.x - p.x, cam_pos.y - p.y, cam_pos.z - p.z};
                                        float tlen = sqrtf(toL.x*toL.x + toL.y*toL.y + toL.z*toL.z);
                                        float3 L = (tlen > 1e-6f)
                                            ? float3{toL.x/tlen, toL.y/tlen, toL.z/tlen}
                                            : float3{0.f, 1.f, 0.f};
                                        float diffuse = fmaxf(0.f, nw.x*L.x + nw.y*L.y + nw.z*L.z);
                                        const float ambient = 0.3f;
                                        float shade = ambient + (1.f - ambient) * diffuse;
                                        C_frag[0] *= shade;
                                        C_frag[1] *= shade;
                                        C_frag[2] *= shade;
                                    }
                                }

                                tex_applied = true;
                            }
                        }
                    }
                }

                // A front-facing UV Gaussian that failed the texture lookup is skipped,
                // to avoid SH-colour haloing at the edges of the patch.
                //
                // A back-facing one instead falls through to its original SH colour: the
                // decal only lives on the front of the surface, so from behind the region
                // must still look like the surface it always was. Skipping it there (the
                // previous behaviour) punched a hole straight through the model wherever a
                // single Gaussian sheet represents both sides of the surface -- the patch
                // area vanished when viewed from the back. The thin-object bleed-through
                // this can reintroduce is the lesser artefact: those same Gaussians are
                // what belongs on the back face anyway.
                if (is_uv_gauss && !tex_applied && !is_backface) continue;

                // Front-to-back compositing, in the tile's global-sort order.
                float test_T = T * (1.f - alpha);
                if (test_T < 0.0001f)
                {
                    // Leave the candidate loop only. This thread stays in the
                    // batch loop (fetching into shared memory, hitting every
                    // barrier) until the whole block votes done above.
                    done = true;
                    break;
                }
                for (int ch = 0; ch < CHANNELS; ch++)
                    C[ch] += C_frag[ch] * alpha * T;
                T = test_T;
                last_contributor++;
            }
            block.sync();
        }

        if (inside)
        {
            final_T[pix_id]    = T;
            n_contrib[pix_id]  = last_contributor;

            for (int ch = 0; ch < CHANNELS; ch++)
                out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
        }
    }

    void render(
        const dim3 grid, dim3 block,
        const uint2* ranges, const uint32_t* point_list,
        int W, int H,
        const float2* means2D, const float* colors,
        const float4* conic_opacity,
        float* final_T, uint32_t* n_contrib,
        const float* bg_color, float* out_color,
        float* depths, float* depth,
        const float* means3D,
        const float* dUs, const float* dVs,
        const float* uvs, const float* surfaceDists,
        const float* viewmatrix, const float* projmatrix,
        float focal_x, float focal_y,
        const cudaTextureObject_t* tex_objs,
		const int* tex_idx,
        const float* cov3Ds,
        const cudaTextureObject_t* normal_texs)
    {
        renderCUDA<NUM_CHANNELS><<<grid, block>>>(
            ranges, point_list, W, H,
            means2D, colors, conic_opacity,
            final_T, n_contrib, bg_color, out_color,
            depths, depth,
            means3D, dUs, dVs, uvs, surfaceDists,
            viewmatrix, projmatrix,
            focal_x, focal_y,
            tex_objs, tex_idx, cov3Ds, normal_texs);
    }

    void preprocess(
        int P, int D, int M,
        const float* means3D, const glm::vec3* scales,
        const float scale_modifier, const glm::vec4* rotations,
        const float* opacities, const float* shs, bool* clamped,
        const float* cov3D_precomp, const float* colors_precomp,
        const float* viewmatrix, const float* projmatrix,
        const glm::vec3* cam_pos,
        const int W, int H,
        const float focal_x, float focal_y,
        const float tan_fovx, float tan_fovy,
        int* radii, float2* means2D,
        float* depths, float* cov3Ds, float* rgb,
        float4* conic_opacity,
        const dim3 grid, uint32_t* tiles_touched,
        bool prefiltered, bool antialiasing,
        const float* boxmin, const float* boxmax,
        const float* uvs, const cudaTextureObject_t* tex_objs,
		const int* tex_idx,
        float uvFixedDepth)
    {
        preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
            P, D, M, means3D, scales, scale_modifier, rotations,
            opacities, shs, clamped,
            cov3D_precomp, colors_precomp,
            viewmatrix, projmatrix, cam_pos,
            W, H, tan_fovx, tan_fovy, focal_x, focal_y,
            radii, means2D, depths, cov3Ds, rgb, conic_opacity,
            grid, tiles_touched, prefiltered, antialiasing,
            boxmin, boxmax, uvs, tex_objs, tex_idx, uvFixedDepth);
    }
}
