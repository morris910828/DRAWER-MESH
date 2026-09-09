#ifndef EXPMAP_SOLVER_SIBR_H
#define EXPMAP_SOLVER_SIBR_H

#include <core/graphics/Mesh.hpp>
#include <core/system/Utils.hpp>
#include <core/system/String.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iostream>
#include <cstring>
#include <cmath>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <imgui/imgui.h>
#include "texture.h"

#ifndef IM_PI
#define IM_PI 3.14159265358979323846f
#endif

// Helper to convert SIBR vector to GLM
inline glm::vec3 toGlm(const sibr::Vector3f& v) { 
    return glm::vec3(v.x(), v.y(), v.z()); 
}

struct ProjectionStats {
    int total = 0;
    int projected = 0;
};

// ============================================================================
// UV-window ellipse geometry, precomputed per projected Gaussian.
//
// Drawing one Gaussian's 1-sigma ellipse used to run the whole
// quaternion -> tangent-plane covariance -> eigen -> SVD chain inline, once
// per Gaussian per frame. At the default ExpMap radius a selection holds
// ~8k Gaussians, so that was ~8k of those chains plus ~8k*32 sin/cos every
// frame -- the reason the viewer dropped from 60 to ~20 fps after a
// selection.
//
// None of that math depends on the camera or on anything else that changes
// per frame. It depends on the Gaussian, plus the canvas scale sc:
// the semi-axes are exactly linear in sc and the orientation is invariant
// under it (sc multiplies both principal vectors equally). So it is computed
// once at sc == 1 when the projection is handed over, and a frame only
// multiplies ra0/rb0 by the current sc.
// ============================================================================
struct EllipseGeom {
    float ra0  = 0.f;   // major semi-axis in canvas pixels at sc == 1
    float rb0  = 0.f;   // minor semi-axis in canvas pixels at sc == 1
    float cosR = 1.f;   // orientation, invariant under sc
    float sinR = 0.f;
    bool  degenerate = true;  // tangent frame unusable -> draw a dot
};

// Per-projected-Gaussian scalars the UV-window filter needs, evaluated once when
// the projection is handed over. Re-running exp()/sigmoid for every Gaussian
// every frame -- together with the std::map lookups the triangle pass used to do
// -- is what dropped the viewer to ~20 fps while the UV window was open.
struct GaussFilterCache {
    float opacitySig  = 0.f;  // sigmoid(opacity)
    float maxScaleExp = 0.f;  // max(exp(scale.x), exp(scale.y), exp(scale.z))
    float surfDist    = 0.f;  // |originalPos - position|
};

// One selected triangle's display UVs plus its precomputed edge ratio, flattened
// out of _displayUVs (a std::map) so RenderUI's per-frame triangle pass is a
// flat-array walk instead of ~9 map lookups + 3 mesh-distance calls per triangle.
struct TriDisplay {
    glm::vec2 uv[3]      = { {0,0}, {0,0}, {0,0} };
    float     maxEdgeRatio = 0.f;
    bool      valid       = false;
};

// Unit-circle samples for ellipse tessellation. The 32 entries are the finest
// level; 16 and 8 segments sub-sample this same table with a stride, so no
// trigonometry runs at draw time at all.
struct UnitCircleLUT {
    float cs[32], sn[32];
    UnitCircleLUT() {
        for (int i = 0; i < 32; ++i) {
            float a = 2.f * IM_PI * (float)i / 32.f;
            cs[i] = std::cos(a);
            sn[i] = std::sin(a);
        }
    }
};
static const UnitCircleLUT kUnitCircle;

// Semi-axes (at sc == 1) and orientation of a Gaussian's 1-sigma ellipse,
// projected onto its triangle's UV tangent frame. Lifted verbatim out of the
// old per-frame drawEllipse body with sc factored out.
inline EllipseGeom ComputeEllipseGeom(const ProjectedGaussian& pg) {
    EllipseGeom g;
    if (glm::length(pg.dU) < 1e-8f || glm::length(pg.dV) < 1e-8f) return g;

    glm::vec3 t1  = glm::normalize(pg.dU);
    glm::vec3 t2r = pg.dV - glm::dot(pg.dV, t1) * t1;
    if (glm::length(t2r) < 1e-8f) return g;
    glm::vec3 t2 = glm::normalize(t2r);

    float qw = pg.rotation.x, qx = pg.rotation.y, qy = pg.rotation.z, qz = pg.rotation.w;
    glm::mat3 R(
        1.f-2.f*(qy*qy+qz*qz), 2.f*(qx*qy+qw*qz),     2.f*(qx*qz-qw*qy),
        2.f*(qx*qy-qw*qz),     1.f-2.f*(qx*qx+qz*qz), 2.f*(qy*qz+qw*qx),
        2.f*(qx*qz+qw*qy),     2.f*(qy*qz-qw*qx),     1.f-2.f*(qx*qx+qy*qy)
    );
    float s0=std::exp(pg.scale.x), s1=std::exp(pg.scale.y), s2=std::exp(pg.scale.z);
    float a0=s0*glm::dot(R[0],t1), a1=s1*glm::dot(R[1],t1), a2=s2*glm::dot(R[2],t1);
    float b0=s0*glm::dot(R[0],t2), b1=s1*glm::dot(R[1],t2), b2=s2*glm::dot(R[2],t2);
    float s11=a0*a0+a1*a1+a2*a2, s12=a0*b0+a1*b1+a2*b2, s22=b0*b0+b1*b1+b2*b2;
    float tr=s11+s22, dif=s11-s22;
    float disc=std::sqrt(std::max(0.f, dif*dif+4.f*s12*s12));
    float sA=std::sqrt(std::max(0.f,(tr+disc)*0.5f));
    float sB=std::sqrt(std::max(0.f,(tr-disc)*0.5f));
    if (sA < 1e-9f) return g;
    float theta=0.5f*std::atan2(2.f*s12, dif);
    float cT=std::cos(theta), sT=std::sin(theta);
    glm::vec3 axA=sA*(cT*t1+sT*t2), axB=sB*(-sT*t1+cT*t2);

    float jA=glm::dot(pg.dU,pg.dU), jB=glm::dot(pg.dU,pg.dV), jC=glm::dot(pg.dV,pg.dV);
    float detJ=jA*jC-jB*jB;
    if (std::abs(detJ) < 1e-14f) return g;
    auto toUV2=[&](const glm::vec3& v)->glm::vec2{
        float du=glm::dot(pg.dU,v), dv=glm::dot(pg.dV,v);
        return glm::vec2((jC*du-jB*dv)/detJ, (-jB*du+jA*dv)/detJ);
    };
    glm::vec2 uvA=toUV2(axA), uvB=toUV2(axB);

    // sc == 1 here; the caller scales ra0/rb0 by the live sc.
    float pxAx=-uvA.y, pxAy=-uvA.x;
    float pxBx=-uvB.y, pxBy=-uvB.x;
    float mp=pxAx*pxAx+pxAy*pxAy;
    float mq=pxAx*pxBx+pxAy*pxBy;
    float mr=pxBx*pxBx+pxBy*pxBy;
    float halfTr=(mp+mr)*0.5f;
    float halfDiff=(mp-mr)*0.5f;
    float discSVD=std::sqrt(halfDiff*halfDiff+mq*mq);
    float ra=std::sqrt(std::max(0.f, halfTr+discSVD));
    float rb=std::sqrt(std::max(0.f, halfTr-discSVD));
    if (ra < 1e-6f) return g;

    float v1x=mq, v1y=halfTr+discSVD-mp;
    float v1len=std::sqrt(v1x*v1x+v1y*v1y);
    float ellRot;
    if (v1len < 1e-9f) {
        ellRot=0.f;
    } else {
        v1x/=v1len; v1y/=v1len;
        float u1x=pxAx*v1x+pxBx*v1y;
        float u1y=pxAy*v1x+pxBy*v1y;
        ellRot=std::atan2(u1y,u1x);
    }

    g.ra0=ra; g.rb0=rb;
    g.cosR=std::cos(ellRot); g.sinR=std::sin(ellRot);
    g.degenerate=false;
    return g;
}

struct TangentFrame {
    glm::vec3 origin = {0, 0, 0};
    glm::mat3 axes = glm::mat3(1);

    TangentFrame() = default;
    
    TangentFrame(const glm::vec3& pos, const glm::vec3& normal) {
        origin = pos;
        glm::vec3 n = glm::normalize(normal);
        glm::vec3 x;
        if (glm::abs(n.x) >= glm::abs(n.y) && glm::abs(n.x) >= glm::abs(n.z)) {
            x = glm::normalize(glm::vec3(-n.y, n.x, 0.0f));
        } else {
            x = glm::normalize(glm::vec3(0.0f, n.z, -n.y));
        }
        glm::vec3 y = glm::cross(n, x);
        axes = glm::mat3(x, y, n);
    }

    glm::vec3 toLocal(const glm::vec3& worldVec) const {
        return glm::transpose(axes) * worldVec;
    }
    
    void alignZAxis(const TangentFrame& target) {
        glm::vec3 fromZ = axes[2];
        glm::vec3 toZ = target.axes[2];
        glm::vec3 axis = glm::cross(fromZ, toZ);
        float sinAngle = glm::length(axis);
        float cosAngle = glm::dot(fromZ, toZ);

        if (sinAngle < 1e-6f) {
            if (cosAngle < 0) axes = glm::mat3(axes[0], -axes[1], -axes[2]);
            return;
        }
        axis = glm::normalize(axis);
        float angle = std::acos(glm::clamp(cosAngle, -1.0f, 1.0f));
        glm::mat3 rot = glm::mat3(glm::rotate(glm::mat4(1.0f), angle, axis));
        axes = rot * axes;
    }
};

struct ExpVertex {
    int id;
    int parentId = -1;
    float cost = 1e9f;
    glm::vec2 uv = {0,0};
    glm::vec3 tangentX = {0, 0, 0}; 
    bool frozen = false;
};

class ExpMapSolverSIBR {
public:
    ~ExpMapSolverSIBR() {
        if (_liveSSBO) glDeleteBuffers(1, &_liveSSBO);
    }

    const std::vector<sibr::Vector3u>& GetActiveTris() const { return _validTriangles; }
    const std::set<int>& GetActiveTriIndices() const { return _validTriangleIndicesSet; }

    // Bumped whenever the active triangle set is replaced or cleared, so
    // renderers can cache GPU buffers built from it instead of rebuilding
    // them every frame.
    uint64_t GetActiveGeneration() const { return _activeGeneration; }
    
    const std::vector<ProjectedGaussian>& GetProjectedGaussians() const { return _projectedGaussians; }

    const std::map<int, glm::vec2>& GetDisplayUVs() const { return _displayUVs; }
    const TextureLoader& GetTextureLoader() const { return _textureLoader; }
    TextureLoader& GetTextureLoader() { return _textureLoader; }

    sibr::Texture2DRGBA::Ptr GetCudaTexture() const { return _cudaTexPtr; }

    // Rebuild the coverage-masked CUDA texture with the patch's texture rotation
    // baked in, so renderCUDA's rotated UV lookup still lands inside the mask.
    // Call whenever the "Texture Rotation" slider moves.
    void RegenerateCudaTexture(float rotDeg) {
        _cudaTexRotDeg = rotDeg;
        if (_validTriangles.empty() || _displayUVs.empty()) return;
        _cudaTexPtr = _textureLoader.generateCudaTexture(_validTriangles, _displayUVs, rotDeg);
    }

    bool ConsumeTextureDirty() {
        if (_textureDirty) { _textureDirty = false; return true; }
        return false;
    }

    float GetSurfaceBlend() const { return _surfaceBlend; }
    bool ConsumeSurfaceBlendDirty() {
        bool d = _surfaceBlendDirty;
        _surfaceBlendDirty = false;
        return d;
    }
    void ResetSurfaceBlend() { _surfaceBlend = 0.0f; _surfaceBlendDirty = true; }

    float GetSurfDistMin() const { return _surfDistMin; }
    float GetSurfDistMax() const { return _surfDistMax; }
    bool ConsumeSurfDistRangeDirty() {
        bool d = _surfDistRangeDirty;
        _surfDistRangeDirty = false;
        return d;
    }

    const ProjectionStats& GetProjectionStats() const { return _projectionStats; }

    void SetMainGaussiansForNextSave(const std::vector<ProjectedGaussian>& g) {
        _pendingMainGaussians = g;
    }

    // Unified splat format: the caller projects every Gaussian itself (each one
    // carries its own mesh_face_idx) and hands the finished set over here.
    void SetProjectedGaussians(const std::vector<ProjectedGaussian>& pts) {
        _projectedGaussians = pts;
        // Precompute each ellipse once here rather than per frame -- see EllipseGeom.
        _ellipseGeom.resize(pts.size());
        for (size_t i = 0; i < pts.size(); ++i)
            _ellipseGeom[i] = ComputeEllipseGeom(pts[i]);

        // Precompute the UV-window filter scalars + a fixed back-to-front draw
        // order, both frame-invariant -- see GaussFilterCache.
        _gaussFilter.resize(pts.size());
        for (size_t i = 0; i < pts.size(); ++i) {
            const ProjectedGaussian& pg = pts[i];
            glm::vec3 se = glm::exp(pg.scale);
            _gaussFilter[i].opacitySig  = 1.f / (1.f + std::exp(-pg.opacity));
            _gaussFilter[i].maxScaleExp = std::max({ se.x, se.y, se.z });
            _gaussFilter[i].surfDist    = glm::length(pg.originalPos - pg.position);
        }
        _gaussDrawOrder.resize(pts.size());
        std::iota(_gaussDrawOrder.begin(), _gaussDrawOrder.end(), 0);
        std::sort(_gaussDrawOrder.begin(), _gaussDrawOrder.end(),
                  [&](int a, int b){ return _gaussFilter[a].surfDist > _gaussFilter[b].surfDist; });

        _projectionStats.total     = (int)pts.size();
        _projectionStats.projected = (int)pts.size();
        float maxD = 0.f;
        for (const auto& pg : pts)
            maxD = std::max(maxD, glm::length(pg.originalPos - pg.position));
        if (maxD > 1e-6f) {
            _computedMaxSurfDist = maxD * 1.1f;
            _surfDistMax = _computedMaxSurfDist;
        }
        // Reset the lower cutoff on every fresh projection, which is what the
        // removed ProjectAndInsertClouds() used to do at the end of Compute().
        _surfDistMin = 0.0f;
        _liveDirty = true;
        _cachedGaussianCoverage = ComputeGaussianCoverage();
    }

    GLuint& GetLiveSSBO() { return _liveSSBO; }
    bool&   GetLiveDirty() { return _liveDirty; }
    
    void ClearRaycastState() {
        _validTriangles.clear();
        _validTriIDs.clear();
        _validTriangleIndicesSet.clear();
        _strokePoints.clear();
        ++_activeGeneration;  // invalidates renderer-side caches of this set
        _displayUVs.clear();
        _vertexData.clear();

        _projectedGaussians.clear();
        _ellipseGeom.clear();
        _gaussFilter.clear();
        _gaussDrawOrder.clear();
        _triDisplayCache.clear();

        _extraNodePositions.clear();
        _extraNodeNormals.clear();

        _projectionStats = ProjectionStats();

        _cudaTexPtr = nullptr;
        if (_liveSSBO) { glDeleteBuffers(1, &_liveSSBO); _liveSSBO = 0; }
        _liveDirty = true;

        _surfDistMin         = 0.0f;
        _surfDistMax         = 1e9f;
        _computedMaxSurfDist = 1.0f;
        _surfDistRangeDirty  = false;
        _cachedGaussianCoverage = 0.f;
    }
    
    // Average of a triangle's three vertex normals, normalized. Falls back to +Y
    // for an out-of-range face id.
    sibr::Vector3f faceAvgNormal(int triID) const {
        sibr::Vector3f n(0, 1, 0);
        if (_mesh && triID >= 0 && triID < (int)_mesh->triangles().size()) {
            const auto& tri = _mesh->triangles()[triID];
            n = (_mesh->normals()[tri[0]] + _mesh->normals()[tri[1]] + _mesh->normals()[tri[2]]) / 3.0f;
            n.normalize();
        }
        return n;
    }

    void OnRaycastHit(const sibr::Vector3f& hitPos, float radius, int hitTriID) {
        if (!_mesh) return;
        Compute(hitPos, faceAvgNormal(hitTriID), radius, hitTriID);
    }

    // Brush selection: a polyline of surface sample points. The frozen region is
    // every vertex within |radius| of the polyline (a swept "capsule"), unwrapped
    // from a single seed frame anchored at the first sample so the UV chart stays
    // continuous. A one-point stroke is identical to OnRaycastHit.
    void OnBrushStroke(const std::vector<sibr::Vector3f>& pts,
                       const std::vector<int>& triIDs, float radius) {
        if (!_mesh || pts.empty()) return;
        std::vector<glm::vec3>      gp(pts.size());
        std::vector<sibr::Vector3f> nrm(pts.size());
        for (size_t i = 0; i < pts.size(); ++i) {
            gp[i]  = toGlm(pts[i]);
            nrm[i] = faceAvgNormal(i < triIDs.size() ? triIDs[i] : -1);
        }
        Compute(gp, nrm, triIDs.empty() ? -1 : triIDs[0], radius);
    }

    // Shortest distance from |p| to the brush polyline (segments between
    // consecutive stroke samples). Degrades to point distance for a 1-point
    // stroke, and is what Compute() uses as its freeze cutoff.
    float distToStroke(const glm::vec3& p) const {
        if (_strokePoints.empty()) return 1e9f;
        if (_strokePoints.size() == 1) return glm::distance(p, _strokePoints[0]);
        float best = 1e9f;
        for (size_t i = 0; i + 1 < _strokePoints.size(); ++i) {
            const glm::vec3& a = _strokePoints[i];
            const glm::vec3& b = _strokePoints[i + 1];
            glm::vec3 ab = b - a;
            float len2 = glm::dot(ab, ab);
            float t = (len2 > 1e-12f) ? glm::clamp(glm::dot(p - a, ab) / len2, 0.f, 1.f) : 0.f;
            best = std::min(best, glm::distance(p, a + t * ab));
        }
        return best;
    }

    const std::vector<glm::vec3>& GetStrokePoints() const { return _strokePoints; }

    void Init(const sibr::Mesh* mesh) {
        _mesh = mesh;
        buildBaseAdjacency();

        bool normalsAreZero = true;
        if (!_mesh->normals().empty()) {
            for (const auto& n : _mesh->normals()) {
                if (n.norm() > 1e-6f) {
                    normalsAreZero = false;
                    break;
                }
            }
        } else {
            normalsAreZero = true; 
        }

        if (normalsAreZero) {
            std::cout << "[INFO] ExpMapSolverSIBR: Detected zero or missing normals. Computing vertex normals..." << std::endl;
            computeVertexNormals();
        }
    }

    const std::vector<sibr::Vector3u>& GetActiveTriangles() const { return _validTriangles; }

    glm::vec3 getPos(int id) const {
        if (id < _mesh->vertices().size()) return toGlm(_mesh->vertices()[id]);
        else return _extraNodePositions[id - _mesh->vertices().size()];
    }

    glm::vec3 getNormal(int id) const {
        if (id < _mesh->vertices().size()) return toGlm(_mesh->normals()[id]);
        else return _extraNodeNormals[id - _mesh->vertices().size()];
    }

    void Compute(const sibr::Vector3f& hitPos, const sibr::Vector3f& hitNormal, float radius, int hitTriID = -1) {
        Compute(std::vector<glm::vec3>{ toGlm(hitPos) },
                std::vector<sibr::Vector3f>{ hitNormal }, hitTriID, radius);
    }

    void Compute(const std::vector<glm::vec3>& strokePts,
                 const std::vector<sibr::Vector3f>& strokeNormals,
                 int seedTriID, float radius) {
        if(!_mesh || strokePts.empty() || strokeNormals.empty()) return;

        _strokePoints = strokePts;
        const sibr::Vector3f hitNormal = strokeNormals[0];
        const int hitTriID = seedTriID;

        _vertexData.clear();
        _displayUVs.clear();
        _validTriangles.clear();
        _validTriIDs.clear();
        _validTriangleIndicesSet.clear();
        ++_activeGeneration;  // invalidates renderer-side caches of this set

        _projectedGaussians.clear();
        _ellipseGeom.clear();
        _gaussFilter.clear();
        _gaussDrawOrder.clear();
        _triDisplayCache.clear();

        _extraNodePositions.clear();
        _extraNodeNormals.clear();
        _projectionStats = ProjectionStats();
        
        _adj = _baseAdj;

        glm::vec3 target = strokePts[0];

        auto comp = [&](int a, int b){ return _vertexData[a].cost > _vertexData[b].cost; };
        std::priority_queue<int, std::vector<int>, decltype(comp)> pq(comp);

        _seedFrame = TangentFrame(target, toGlm(hitNormal));

        if (hitTriID >= 0 && hitTriID < _mesh->triangles().size()) {
            const auto& tri = _mesh->triangles()[hitTriID];
            for (int k = 0; k < 3; ++k) {
                int vIdx = (int)tri[k];
                glm::vec3 vPos = toGlm(_mesh->vertices()[vIdx]);

                ExpVertex vSeed;
                vSeed.id = vIdx;
                vSeed.cost = glm::distance(target, vPos);

                glm::vec3 localPos = _seedFrame.toLocal(vPos - target);
                vSeed.uv = glm::vec2(localPos.x, localPos.y);
                vSeed.tangentX = _seedFrame.axes[0];

                _vertexData[vIdx] = vSeed;
                pq.push(vIdx);
            }
        } else {
            return;
        }

        float maxCostFound = 0.0f;

        while(!pq.empty()) {
            int currIdx = pq.top(); pq.pop();
            if(_vertexData[currIdx].frozen) continue;
            _vertexData[currIdx].frozen = true;

            // Freeze cutoff: within the brush radius of the stroke polyline (a
            // swept capsule), not the geodesic distance from a single seed. For a
            // 1-point stroke distToStroke() is exactly the old radius test.
            if(distToStroke(getPos(currIdx)) > radius) continue;
            glm::vec3 currN = glm::normalize(getNormal(currIdx));
            if (glm::dot(currN, toGlm(hitNormal)) < 0.0f) continue;

            maxCostFound = std::max(maxCostFound, _vertexData[currIdx].cost);

            for(int neighbor : _adj[currIdx]) {
                if(_vertexData.find(neighbor) != _vertexData.end() && _vertexData[neighbor].frozen) continue;

                if(_vertexData.find(neighbor) == _vertexData.end()) {
                    ExpVertex vNew; vNew.id = neighbor; vNew.cost = 1e9f;
                    _vertexData[neighbor] = vNew;
                }
                propagate(currIdx, neighbor);
                pq.push(neighbor);
            }
        }

        if(maxCostFound > 1e-6f) {
            _viewScale = 1.0f;
            _viewOffset = ImVec2(0, 0);

            refineUVsTriangleUnfolding(3);
            refineUVsARAP(8);

            glm::vec2 exactHitUV(0.0f, 0.0f);
            if (hitTriID >= 0 && hitTriID < _mesh->triangles().size()) {
                const auto& t = _mesh->triangles()[hitTriID];
                if (_vertexData.count(t[0]) && _vertexData.count(t[1]) && _vertexData.count(t[2])) {
                    glm::vec3 v0 = toGlm(_mesh->vertices()[t[0]]);
                    glm::vec3 v1 = toGlm(_mesh->vertices()[t[1]]);
                    glm::vec3 v2 = toGlm(_mesh->vertices()[t[2]]);
                    
                    glm::vec3 v0v1 = v1 - v0, v0v2 = v2 - v0, pVec = target - v0;
                    float d00 = glm::dot(v0v1, v0v1), d01 = glm::dot(v0v1, v0v2), d11 = glm::dot(v0v2, v0v2);
                    float d20 = glm::dot(pVec, v0v1), d21 = glm::dot(pVec, v0v2);
                    float denom = d00 * d11 - d01 * d01;
                    
                    if (std::abs(denom) > 1e-9f) {
                        float v = (d11 * d20 - d01 * d21) / denom;
                        float w = (d00 * d21 - d01 * d20) / denom;
                        float u = 1.0f - v - w;
                        exactHitUV = u * _vertexData[t[0]].uv + v * _vertexData[t[1]].uv + w * _vertexData[t[2]].uv;
                    }
                }
            }

            glm::vec2 uvMin(1e9f), uvMax(-1e9f);
            for (auto& [id, vd] : _vertexData) {
                if (vd.frozen) {
                    vd.uv -= exactHitUV; 
                    uvMin = glm::min(uvMin, vd.uv);
                    uvMax = glm::max(uvMax, vd.uv);
                }
            }

            float uvWidth  = uvMax.x - uvMin.x;
            float uvHeight = uvMax.y - uvMin.y;
            float maxDim   = std::max(uvWidth, uvHeight);
            
            _uvScale = (maxDim > 1e-7f) ? (0.95f / maxDim) : 1.0f;

            glm::vec2 uvCenter = (uvMin + uvMax) * 0.5f;

            for (auto& [id, vd] : _vertexData) {
                if (vd.frozen) {
                    _displayUVs[id] = (vd.uv - uvCenter) * _uvScale + glm::vec2(0.5f, 0.5f);
                }
            }

            const auto& tris = _mesh->triangles();

            struct TriCandidate {
                sibr::Vector3u t;
                int   idx;
                float signedAreaUV;
                float maxEdgeRatio;
            };
            std::vector<TriCandidate> cands;
            cands.reserve(2048);

            for (size_t i = 0; i < tris.size(); ++i) {
                const auto& t = tris[i];
                if (!_displayUVs.count(t[0]) || !_displayUVs.count(t[1]) || !_displayUVs.count(t[2]))
                    continue;

                glm::vec2 uv0 = _displayUVs[t[0]], uv1 = _displayUVs[t[1]], uv2 = _displayUVs[t[2]];
                glm::vec3 p0  = toGlm(_mesh->vertices()[t[0]]);
                glm::vec3 p1  = toGlm(_mesh->vertices()[t[1]]);
                glm::vec3 p2  = toGlm(_mesh->vertices()[t[2]]);

                float d01 = glm::distance(p0,p1), d12 = glm::distance(p1,p2), d20 = glm::distance(p2,p0);
                if (d01 < 1e-6f && d12 < 1e-6f && d20 < 1e-6f) continue;

                float signedAreaUV = (uv1.x-uv0.x)*(uv2.y-uv0.y) - (uv1.y-uv0.y)*(uv2.x-uv0.x);
                if (std::abs(signedAreaUV) < 1e-6f) continue;

                float r01 = (d01 > 1e-6f) ? (glm::distance(uv0,uv1)/_uvScale) / d01 : 0.f;
                float r12 = (d12 > 1e-6f) ? (glm::distance(uv1,uv2)/_uvScale) / d12 : 0.f;
                float r20 = (d20 > 1e-6f) ? (glm::distance(uv2,uv0)/_uvScale) / d20 : 0.f;
                float maxR = std::max({r01, r12, r20});

                cands.push_back({t, (int)i, signedAreaUV, maxR});
            }

            int posW = 0, negW = 0;
            for (auto& c : cands) {
                if (c.signedAreaUV > 0.f) ++posW; else ++negW;
            }
            bool expectPos = (posW >= negW);

            std::vector<float> goodRatios;
            goodRatios.reserve(cands.size());
            for (auto& c : cands) {
                bool ok = expectPos ? (c.signedAreaUV > 0.f) : (c.signedAreaUV < 0.f);
                if (ok) goodRatios.push_back(c.maxEdgeRatio);
            }
            float medRatio = 1.0f;
            if (!goodRatios.empty()) {
                std::sort(goodRatios.begin(), goodRatios.end());
                medRatio = goodRatios[goodRatios.size() / 2];
            }
            _autoThreshold = std::min(std::max(1.5f, medRatio * 2.0f), 12.0f);

            // Pass 1: filter triangles with wrong winding or too-large edge ratio
            struct PassedCand {
                sibr::Vector3u t;
                int   idx;
                float distFromCenter; // distance from UV centroid to center (0.5,0.5), for sorting
            };
            std::vector<PassedCand> passed;
            passed.reserve(cands.size());

            for (auto& c : cands) {
                bool correctWind = expectPos ? (c.signedAreaUV > 0.f) : (c.signedAreaUV < 0.f);
                if (!correctWind)              continue;
                if (c.maxEdgeRatio > _autoThreshold) continue;

                // Drop faces that point away from the surface that was picked. The
                // Dijkstra pass above refuses to *propagate* through a vertex whose
                // normal opposes hitNormal, but that vertex has already been given a
                // UV, so the faces behind it can still be accepted here. On a thin
                // plate that is exactly what happens: the sign in this scene is 29 mm
                // thick, so the geodesic reaches its back face after ~15 mm -- well
                // inside any usable ExpMap radius. The back then shares the front's UV
                // patch, which is why editing the front showed up on the back.
                {
                    const glm::vec3 fv0 = toGlm(_mesh->vertices()[c.t[0]]);
                    const glm::vec3 fv1 = toGlm(_mesh->vertices()[c.t[1]]);
                    const glm::vec3 fv2 = toGlm(_mesh->vertices()[c.t[2]]);
                    const glm::vec3 fn  = glm::cross(fv1 - fv0, fv2 - fv0);
                    const float fnLen = glm::length(fn);
                    // cos > 0 keeps the whole visible hemisphere, so a curved surface
                    // is still unwrapped in one piece; only genuine back faces go.
                    if (fnLen > 1e-12f && glm::dot(fn / fnLen, toGlm(hitNormal)) < 0.0f)
                        continue;
                }

                // UV centroid distance to (0.5, 0.5)
                glm::vec2 uv0 = _displayUVs[c.t[0]], uv1 = _displayUVs[c.t[1]], uv2 = _displayUVs[c.t[2]];
                glm::vec2 centroid = (uv0 + uv1 + uv2) / 3.f;
                float dist = glm::distance(centroid, glm::vec2(0.5f, 0.5f));
                passed.push_back({c.t, c.idx, dist});
            }

            // Sort near-to-far to prioritize triangles close to the hit point
            std::sort(passed.begin(), passed.end(),
                      [](const PassedCand& a, const PassedCand& b){ return a.distFromCenter < b.distFromCenter; });

            // Accept all triangles that passed winding + edge-ratio check.
            // UV overlap detection is O(n^2) and tends to misclassify adjacent
            // triangles on curved surfaces as overlapping, causing holes. Skip it.
            for (auto& pc : passed) {
                _validTriangles.push_back(pc.t);
                _validTriIDs.push_back(pc.idx);
                _validTriangleIndicesSet.insert(pc.idx);
            }

            rebuildTriDisplayCache();

            _cudaTexRotDeg = 0.f;  // a fresh pick starts unrotated
            _cudaTexPtr = _textureLoader.generateCudaTexture(_validTriangles, _displayUVs, _cudaTexRotDeg);
            _cachedGaussianCoverage = ComputeGaussianCoverage();
        }
    }

    // Fraction of the selected patch's UV area that has at least one Gaussian
    // projected onto it. Approximates each Gaussian's UV-space
    // footprint as a circle: world-space radius (mean of exp(scale)) converted
    // to UV units via the local dU/dV jacobian at that Gaussian's triangle.
    float ComputeGaussianCoverage() const {
        if (_validTriangles.empty()) return 0.f;
        constexpr int GRID = 128;
        std::vector<uint8_t> selectedMask(GRID * GRID, 0);
        std::vector<uint8_t> coveredMask(GRID * GRID, 0);

        auto rasterCircle = [&](std::vector<uint8_t>& mask, const glm::vec2& uv, float rUV) {
            glm::vec2 c(uv.x * GRID, uv.y * GRID);
            float rPix = rUV * GRID;
            if (rPix <= 0.f) return;
            int xmin = std::max(0, (int)std::floor(c.x - rPix));
            int xmax = std::min(GRID - 1, (int)std::ceil(c.x + rPix));
            int ymin = std::max(0, (int)std::floor(c.y - rPix));
            int ymax = std::min(GRID - 1, (int)std::ceil(c.y + rPix));
            for (int py = ymin; py <= ymax; ++py) {
                for (int px = xmin; px <= xmax; ++px) {
                    float dx = (px + 0.5f) - c.x, dy = (py + 0.5f) - c.y;
                    if (dx * dx + dy * dy <= rPix * rPix) mask[py * GRID + px] = 1;
                }
            }
        };

        for (const auto& t : _validTriangles) {
            if (!_displayUVs.count(t[0]) || !_displayUVs.count(t[1]) || !_displayUVs.count(t[2])) continue;
            glm::vec2 uv0 = _displayUVs.at(t[0]) * float(GRID);
            glm::vec2 uv1 = _displayUVs.at(t[1]) * float(GRID);
            glm::vec2 uv2 = _displayUVs.at(t[2]) * float(GRID);
            int xmin = std::max(0,        (int)std::floor(std::min({uv0.x, uv1.x, uv2.x})));
            int xmax = std::min(GRID - 1, (int)std::ceil (std::max({uv0.x, uv1.x, uv2.x})));
            int ymin = std::max(0,        (int)std::floor(std::min({uv0.y, uv1.y, uv2.y})));
            int ymax = std::min(GRID - 1, (int)std::ceil (std::max({uv0.y, uv1.y, uv2.y})));
            for (int py = ymin; py <= ymax; ++py) {
                for (int px = xmin; px <= xmax; ++px) {
                    float qx = px + 0.5f, qy = py + 0.5f;
                    float d0 = (uv1.x-uv0.x)*(qy-uv0.y) - (uv1.y-uv0.y)*(qx-uv0.x);
                    float d1 = (uv2.x-uv1.x)*(qy-uv1.y) - (uv2.y-uv1.y)*(qx-uv1.x);
                    float d2 = (uv0.x-uv2.x)*(qy-uv2.y) - (uv0.y-uv2.y)*(qx-uv2.x);
                    if ((d0>=0&&d1>=0&&d2>=0)||(d0<=0&&d1<=0&&d2<=0))
                        selectedMask[py * GRID + px] = 1;
                }
            }
        }

        // Project the Gaussian's 3D covariance onto its local tangent plane (dU/dV)
        // to get true in-plane semi-axes, same math used to draw its ellipse on screen.
        auto rasterGaussian = [&](const ProjectedGaussian& pg) {
            float dUlen = glm::length(pg.dU), dVlen = glm::length(pg.dV);
            if (dUlen < 1e-8f || dVlen < 1e-8f) return;
            glm::vec3 t1 = glm::normalize(pg.dU);
            glm::vec3 t2r = pg.dV - glm::dot(pg.dV, t1) * t1;
            if (glm::length(t2r) < 1e-8f) return;
            glm::vec3 t2 = glm::normalize(t2r);

            float qw=pg.rotation.x,qx=pg.rotation.y,qy=pg.rotation.z,qz=pg.rotation.w;
            glm::mat3 R(
                1.f-2.f*(qy*qy+qz*qz),2.f*(qx*qy+qw*qz),    2.f*(qx*qz-qw*qy),
                2.f*(qx*qy-qw*qz),    1.f-2.f*(qx*qx+qz*qz),2.f*(qy*qz+qw*qx),
                2.f*(qx*qz+qw*qy),    2.f*(qy*qz-qw*qx),    1.f-2.f*(qx*qx+qy*qy)
            );
            float s0=std::exp(pg.scale.x),s1=std::exp(pg.scale.y),s2=std::exp(pg.scale.z);
            float a0=s0*glm::dot(R[0],t1),a1=s1*glm::dot(R[1],t1),a2=s2*glm::dot(R[2],t1);
            float b0=s0*glm::dot(R[0],t2),b1=s1*glm::dot(R[1],t2),b2=s2*glm::dot(R[2],t2);
            float s11=a0*a0+a1*a1+a2*a2,s12=a0*b0+a1*b1+a2*b2,s22=b0*b0+b1*b1+b2*b2;
            float tr=s11+s22,dif=s11-s22;
            float disc=std::sqrt(std::max(0.f,dif*dif+4.f*s12*s12));
            float sA=std::sqrt(std::max(0.f,(tr+disc)*0.5f));
            float sB=std::sqrt(std::max(0.f,(tr-disc)*0.5f));

            // Equal-area circle radius (world units), then to UV units via the
            // local dU/dV jacobian (world length per unit UV).
            float worldRadius = std::sqrt(sA * sB);
            float uvRadius = worldRadius * 0.5f * (1.f / dUlen + 1.f / dVlen);
            rasterCircle(coveredMask, pg.uv, uvRadius);
        };
        for (const auto& pg : _projectedGaussians) rasterGaussian(pg);

        int totalSelected = 0, coveredSelected = 0;
        for (int i = 0; i < GRID * GRID; ++i) {
            if (selectedMask[i]) {
                ++totalSelected;
                if (coveredMask[i]) ++coveredSelected;
            }
        }
        return totalSelected > 0 ? float(coveredSelected) / float(totalSelected) : 0.f;
    }

    void RenderUI() {
        ImGui::SetNextWindowSize(ImVec2(400, 450), ImGuiCond_FirstUseEver);
        if(ImGui::Begin("ExpMap UV Result", nullptr, ImGuiWindowFlags_MenuBar)) {
            
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("File")) {
                    if (ImGui::BeginMenu("Load Background")) {
                        // Base dir = one level above SIBR_viewers (mirrors the old
                        // layout where assets/ was a sibling of SIBR_viewers).
                        std::string targetDir = sibr::parentDirectory(sibr::parentDirectory(sibr::getInstallDirectory())) + "/assets/texture/";
                        std::vector<std::string> imageFiles = _textureLoader.scanForImages(targetDir);
                        
                        if (imageFiles.empty()) {
                            ImGui::MenuItem("(No images in assets/texture)", NULL, false, false);
                        } else {
                            for (const auto& imagePath : imageFiles) {
                                std::string displayName = imagePath;
                                size_t lastSlash = displayName.find_last_of("/\\");
                                if (lastSlash != std::string::npos) {
                                    displayName = displayName.substr(lastSlash + 1);
                                }

                                if (ImGui::MenuItem(displayName.c_str())) {
                                    _textureLoader.LoadImage(imagePath);
                                    // Keep the current rotation: reloading the image mid-edit
                                    // must not un-rotate the coverage mask.
                                    _cudaTexPtr = _textureLoader.generateCudaTexture(_validTriangles, _displayUVs, _cudaTexRotDeg);
                                    _textureDirty = true;
                                }
                            }
                        }
                        ImGui::EndMenu();
                    }
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }

            if (_displayUVs.empty()) {
                ImGui::TextColored(ImVec4(1,1,0,1), "Right-click on the mesh to compute UV.");
            } else {
                ImGui::Text("Tris: %lu | Gaussians: %d/%d",
                            _validTriangles.size(),
                            _projectionStats.projected,
                            _projectionStats.total);
                ImGui::Text("Gaussian Coverage: %.1f%%", _cachedGaussianCoverage * 100.f);

                
                ImGui::SameLine();
                if (ImGui::Button("Reset View")) { _viewScale = 1.0f; _viewOffset = ImVec2(0,0); }
            }
            
            ImGui::Checkbox("Fill", &_drawFilled);
            ImGui::SameLine();
            ImGui::Checkbox("Cull", &_cullBackFace);
            ImGui::SameLine();
            ImGui::Checkbox("BG", &_showBackgroundTexture);
            ImGui::SameLine();
            ImGui::Checkbox("Grid", &_showMeshGrid);
            ImGui::Checkbox("Show Gaussians", &_showGaussianPoints);

            {
                float rangeMax = (_computedMaxSurfDist > 1e-6f) ? _computedMaxSurfDist : 1.0f;
                _surfDistMin = std::clamp(_surfDistMin, 0.0f, rangeMax);
                _surfDistMax = std::clamp(_surfDistMax, _surfDistMin, rangeMax);
                float prevMin = _surfDistMin, prevMax = _surfDistMax;

                ImDrawList* dlUi  = ImGui::GetWindowDrawList();
                ImVec2 basePos    = ImGui::GetCursorScreenPos();
                float  availW     = ImGui::GetContentRegionAvail().x;
                float  w          = availW * (2.f / 3.f);
                if (w < 20.f) w = 20.f;
                float  offsetX    = (availW - w) * 0.5f;
                ImVec2 sp         = { basePos.x + offsetX, basePos.y };
                const float H = 20.f, R = 7.f, trackH = 4.f;
                float midY = sp.y + H * 0.5f;
                float minX = sp.x + (_surfDistMin / rangeMax) * w;
                float maxX = sp.x + (_surfDistMax / rangeMax) * w;
                ImVec2 mouse = ImGui::GetIO().MousePos;

                // Invisible button centred over the track
                ImGui::SetCursorScreenPos(sp);
                ImGui::InvisibleButton("##distrange_track", {w, H});
                if (ImGui::IsItemClicked()) {
                    float dMin = std::abs(mouse.x - minX);
                    float dMax = std::abs(mouse.x - maxX);
                    _draggingMin = (dMin <= dMax);
                    _draggingMax = !_draggingMin;
                }
                if (!ImGui::IsMouseDown(0)) { _draggingMin = false; _draggingMax = false; }
                if (ImGui::IsItemActive()) {
                    float frac = std::clamp((mouse.x - sp.x) / w, 0.f, 1.f);
                    float val  = frac * rangeMax;
                    if      (_draggingMin) _surfDistMin = std::clamp(val, 0.f, _surfDistMax);
                    else if (_draggingMax) _surfDistMax = std::clamp(val, _surfDistMin, rangeMax);
                    minX = sp.x + (_surfDistMin / rangeMax) * w;
                    maxX = sp.x + (_surfDistMax / rangeMax) * w;
                }

                // Draw track background and selected range
                dlUi->AddRectFilled({sp.x, midY-trackH*.5f}, {sp.x+w, midY+trackH*.5f}, IM_COL32(60,60,60,255), trackH*.5f);
                dlUi->AddRectFilled({minX, midY-trackH*.5f}, {maxX,   midY+trackH*.5f}, IM_COL32(100,160,255,200), trackH*.5f);
                // Min handle (red)
                bool minHov = _draggingMin || ImGui::IsMouseHoveringRect({minX-R, midY-R}, {minX+R, midY+R});
                dlUi->AddCircleFilled({minX, midY}, R, IM_COL32(255, minHov?140:80, 80, 255));
                dlUi->AddCircle({minX, midY}, R, IM_COL32(255,255,255,160), 16, 1.5f);
                // Max handle (blue)
                bool maxHov = _draggingMax || ImGui::IsMouseHoveringRect({maxX-R, midY-R}, {maxX+R, midY+R});
                dlUi->AddCircleFilled({maxX, midY}, R, IM_COL32(80, maxHov?140:80, 255, 255));
                dlUi->AddCircle({maxX, midY}, R, IM_COL32(255,255,255,160), 16, 1.5f);

                // Value labels below the bar
                float lblH = ImGui::GetTextLineHeight();
                char lblMin[32], lblMax[32];
                snprintf(lblMin, sizeof(lblMin), "%.4f", _surfDistMin);
                snprintf(lblMax, sizeof(lblMax), "%.4f", _surfDistMax);
                dlUi->AddText({sp.x, sp.y+H+1.f}, IM_COL32(255,100,100,255), lblMin);
                dlUi->AddText({sp.x+w-ImGui::CalcTextSize(lblMax).x, sp.y+H+1.f}, IM_COL32(100,100,255,255), lblMax);

                // Advance cursor past bar + label row
                ImGui::SetCursorScreenPos({basePos.x, sp.y + H + lblH + 4.f});
                ImGui::Dummy({availW, 2.f});

                if (_surfDistMin != prevMin || _surfDistMax != prevMax) _surfDistRangeDirty = true;
            }

            ImVec2 p = ImGui::GetCursorScreenPos();
            ImVec2 sz = ImGui::GetContentRegionAvail();
            if(sz.x < 50) sz.x = 50; 
            if(sz.y < 50) sz.y = 50;
            float dim = std::min(sz.x, sz.y);

            ImGui::InvisibleButton("##uvcanvas", sz);
            bool isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
            ImVec2 mousePos = ImGui::GetMousePos();

            if (isHovered) {
                float wheel = ImGui::GetIO().MouseWheel;
                if (wheel != 0.0f) {
                    float lx = mousePos.x - (p.x + sz.x * 0.5f + _viewOffset.x);
                    float ly = mousePos.y - (p.y + sz.y * 0.5f + _viewOffset.y);
                    
                    float zoomFactor = 1.1f;
                    float oldScale = _viewScale;
                    if (wheel < 0.0f) _viewScale /= zoomFactor;
                    else              _viewScale *= zoomFactor;
                    
                    float ratio = _viewScale / oldScale;
                    _viewOffset.x -= lx * (ratio - 1.0f);
                    _viewOffset.y -= ly * (ratio - 1.0f);
                }
            }

            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->PushClipRect(p, ImVec2(p.x + sz.x, p.y + sz.y), true);

            // The canvas draws thousands of tiny ellipses + up to 5000 triangles
            // every frame. Anti-aliased fills/strokes roughly double the vertex and
            // index count of each one and are the bulk of what still costs frame
            // time here; on a dense debug overlay the aliasing is barely visible.
            // Restored before PopClipRect so the rest of the UI keeps its AA.
            const ImDrawListFlags _savedDrawFlags = dl->Flags;
            dl->Flags &= ~(ImDrawListFlags_AntiAliasedLines | ImDrawListFlags_AntiAliasedFill);

            if (_showBackgroundTexture) {
                const auto& bgTex = _textureLoader.getTexture();
                if (bgTex && bgTex->handle() != 0) {
                    dl->AddImage((void*)(intptr_t)bgTex->handle(), p, ImVec2(p.x + sz.x, p.y + sz.y), ImVec2(0, 1), ImVec2(1, 0));
                } else {
                    dl->AddRectFilled(p, ImVec2(p.x + sz.x, p.y + sz.y), IM_COL32(40, 40, 40, 255));
                }
            } else {
                dl->AddRectFilled(p, ImVec2(p.x + sz.x, p.y + sz.y), IM_COL32(40, 40, 40, 255));
            }

            auto TransformUV = [&](const glm::vec2& uv) -> ImVec2 {
                float lx = (0.5f - uv.y) * dim * _viewScale;
                float ly = (0.5f - uv.x) * dim * _viewScale;
                return ImVec2(p.x + sz.x*0.5f + lx + _viewOffset.x, p.y + sz.y*0.5f + ly + _viewOffset.y);
            };

            const float CANVAS_GUARD = sz.x + sz.y;
            auto ClampPx = [&](ImVec2 v) -> ImVec2 {
                float xMin = p.x - CANVAS_GUARD, xMax = p.x + sz.x + CANVAS_GUARD;
                float yMin = p.y - CANVAS_GUARD, yMax = p.y + sz.y + CANVAS_GUARD;
                v.x = std::max(xMin, std::min(xMax, v.x));
                v.y = std::max(yMin, std::min(yMax, v.y));
                return v;
            };

            const int TRI_BUDGET  = 5000;
            int triStride = std::max(1, (int)_validTriangles.size() / TRI_BUDGET);

            // Past this many Gaussian dots the canvas is just a denser cloud, not
            // more information (the coverage % already quantifies density). The
            // filtered list below is depth-sorted, so an index stride thins it
            // uniformly across depth.
            const int GAUSS_BUDGET = 12000;
            int gaussStride = 1;  // recomputed once the filtered count is known

            float canvasX0 = p.x, canvasX1 = p.x + sz.x;
            float canvasY0 = p.y, canvasY1 = p.y + sz.y;

            static float maxScale3DExp = 1.0f;
            static float minOpacity = 0.0f;

            ImGui::PushItemWidth(120.0f);
            ImGui::SliderFloat("MaxScale3D##ell", &maxScale3DExp, 0.001f, 5.0f, "%.3f");
            ImGui::PopItemWidth();
            ImGui::SameLine();
            ImGui::PushItemWidth(100.0f);
            ImGui::SliderFloat("MinOpac##ell", &minOpacity, 0.0f, 1.0f, "%.2f");
            ImGui::PopItemWidth();
            if (_showMeshGrid && triStride > 1)
                ImGui::TextColored(ImVec4(1,0.6f,0,1), "Tri display thinned x%d", triStride);
            if (_gaussStrideShown > 1)  // value from last frame; one frame stale is invisible
                ImGui::TextColored(ImVec4(1,0.6f,0,1), "Gaussian display thinned x%d", _gaussStrideShown);

            {
                struct GEntry {
                    const ProjectedGaussian* pg;
                    const EllipseGeom*       geom;
                    float absDist;
                };
                std::vector<GEntry> nonSelEntries;

                glm::vec3 hitN = glm::normalize(_seedFrame.axes[2]);
                glm::vec3 hitO = _seedFrame.origin;

                // opacity / scale / surfDist are precomputed in _gaussFilter; this
                // is now pure comparisons instead of ~5 transcendentals per Gaussian.
                auto passFilter = [&](size_t gi) {
                    const ProjectedGaussian& pt = _projectedGaussians[gi];
                    if (pt.uv.x < -0.15f || pt.uv.x > 1.15f || pt.uv.y < -0.15f || pt.uv.y > 1.15f) return false;
                    const GaussFilterCache& fc = _gaussFilter[gi];
                    if (fc.opacitySig < minOpacity) return false;
                    if (fc.maxScaleExp > maxScale3DExp) return false;
                    if (fc.surfDist < _surfDistMin || fc.surfDist > _surfDistMax) return false;
                    return true;
                };

                if (_showGaussianPoints &&
                    _gaussFilter.size() == _projectedGaussians.size() &&
                    _gaussDrawOrder.size() == _projectedGaussians.size()) {
                    nonSelEntries.reserve(_projectedGaussians.size());
                    // _gaussDrawOrder is presorted far->near (by surfDist), so the
                    // filtered list comes out already back-to-front: far (blue)
                    // drawn first, near (red) on top -- no per-frame sort.
                    for (int gi : _gaussDrawOrder) {
                        if (!passFilter((size_t)gi)) continue;
                        nonSelEntries.push_back(
                            GEntry{&_projectedGaussians[gi], &_ellipseGeom[gi], _gaussFilter[gi].surfDist});
                    }
                    gaussStride = std::max(1, (int)nonSelEntries.size() / GAUSS_BUDGET);
                }
                _gaussStrideShown = gaussStride;

                // Per-frame work per Gaussian is now: scale two cached semi-axes,
                // reject against the canvas AABB, then emit N table-driven points.
                // The heavy covariance/eigen/SVD chain moved to ComputeEllipseGeom().
                const float sc = dim * _viewScale;
                auto drawEllipse = [&](const GEntry& e, ImU32 fillCol, ImU32 outlineCol) {
                    const EllipseGeom& g = *e.geom;
                    ImVec2 ctr = TransformUV(e.pg->uv);

                    if (g.degenerate) {
                        if (ctr.x>=canvasX0&&ctr.x<=canvasX1&&ctr.y>=canvasY0&&ctr.y<=canvasY1)
                            dl->AddCircleFilled(ctr, 2.5f, outlineCol);
                        return;
                    }

                    float ra = g.ra0 * sc, rb = g.rb0 * sc;
                    float rmax = (ra > rb) ? ra : rb;

                    // Cheap AABB reject, before any geometry is generated. The old
                    // code built all 32 points first and only then asked whether any
                    // of them landed on the canvas.
                    if (ctr.x + rmax < canvasX0 || ctr.x - rmax > canvasX1 ||
                        ctr.y + rmax < canvasY0 || ctr.y - rmax > canvasY1) return;

                    // Adaptive tessellation. Most selected Gaussians cover only a
                    // few pixels, where a 32-gon fill plus a closed polyline is
                    // indistinguishable from a dot but costs ~100 vertices.
                    if (rmax < 1.5f) {
                        dl->AddRectFilled(ImVec2(ctr.x-1.f, ctr.y-1.f),
                                          ImVec2(ctr.x+1.f, ctr.y+1.f), outlineCol);
                        return;
                    }
                    const int nseg   = (rmax < 4.f) ? 8 : ((rmax < 12.f) ? 16 : 32);
                    const int stride = 32 / nseg;

                    ImVec2 pts[32];
                    for (int k = 0; k < nseg; ++k) {
                        int t = k * stride;
                        float cx = ra*kUnitCircle.cs[t]*g.cosR - rb*kUnitCircle.sn[t]*g.sinR;
                        float cy = ra*kUnitCircle.cs[t]*g.sinR + rb*kUnitCircle.sn[t]*g.cosR;
                        pts[k] = ClampPx(ImVec2(ctr.x+cx, ctr.y+cy));
                    }
                    dl->AddConvexPolyFilled(pts, nseg, fillCol);
                    dl->AddPolyline(pts, nseg, outlineCol, true, 1.0f);
                };

                {
                    const float globalMax = (_computedMaxSurfDist > 1e-6f) ? _computedMaxSurfDist : 1.f;
                    for (size_t ei = 0; ei < nonSelEntries.size(); ei += gaussStride) {
                        const GEntry& e = nonSelEntries[ei];
                        // absDist is already |originalPos - position|, computed when
                        // the entry was built -- no need to take the square root again.
                        float t = glm::clamp(e.absDist / globalMax, 0.f, 1.f);
                        int r = (int)(80.f + 150.f * (1.f - t));
                        int g = 70;
                        int b = (int)(80.f + 150.f * t);
                        int a = (int)(100.f + 130.f * t);
                        drawEllipse(e, IM_COL32(r, g, b, a), IM_COL32(r, g, b, 230));
                    }
                }
            }

            // Display UVs + edge ratios come from _triDisplayCache (built once per
            // selection), so this pass no longer touches _displayUVs (a std::map)
            // or the mesh vertex array per frame.
            if (_showMeshGrid && _triDisplayCache.size() == _validTriangles.size()) {
                for (int ti = 0; ti < (int)_triDisplayCache.size(); ti += triStride) {
                    const TriDisplay& td = _triDisplayCache[ti];
                    if (!td.valid) continue;

                    ImVec2 ip1 = TransformUV(td.uv[0]);
                    ImVec2 ip2 = TransformUV(td.uv[1]);
                    ImVec2 ip3 = TransformUV(td.uv[2]);

                    if (ip1.x < canvasX0 && ip2.x < canvasX0 && ip3.x < canvasX0) continue;
                    if (ip1.x > canvasX1 && ip2.x > canvasX1 && ip3.x > canvasX1) continue;
                    if (ip1.y < canvasY0 && ip2.y < canvasY0 && ip3.y < canvasY0) continue;
                    if (ip1.y > canvasY1 && ip2.y > canvasY1 && ip3.y > canvasY1) continue;

                    if (_cullBackFace) {
                        float area = (ip2.x - ip1.x) * (ip3.y - ip1.y) - (ip3.x - ip1.x) * (ip2.y - ip1.y);
                        if (area > 0.0f) continue;
                    }
                    if (td.maxEdgeRatio > _autoThreshold) continue;

                    ip1 = ClampPx(ip1); ip2 = ClampPx(ip2); ip3 = ClampPx(ip3);

                    if (_drawFilled) dl->AddTriangleFilled(ip1, ip2, ip3, IM_COL32(255, 215, 0, 80));
                    else dl->AddTriangle(ip1, ip2, ip3, IM_COL32(255, 215, 0, 200), 1.0f);
                }
            }

            dl->Flags = _savedDrawFlags;
            dl->PopClipRect();
        }
        ImGui::End();
    }

private:
    // Flatten _validTriangles + _displayUVs (a std::map) into _triDisplayCache,
    // kept index-parallel to _validTriangles. Built once per selection; RenderUI
    // then walks it flat instead of doing map lookups + mesh-distance calls for
    // every triangle every frame.
    void rebuildTriDisplayCache() {
        _triDisplayCache.clear();
        _triDisplayCache.reserve(_validTriangles.size());
        for (const auto& t : _validTriangles) {
            auto i0 = _displayUVs.find((int)t[0]);
            auto i1 = _displayUVs.find((int)t[1]);
            auto i2 = _displayUVs.find((int)t[2]);
            TriDisplay td;
            if (i0 == _displayUVs.end() || i1 == _displayUVs.end() || i2 == _displayUVs.end()) {
                _triDisplayCache.push_back(td);   // invalid, keeps indices aligned
                continue;
            }
            td.uv[0] = i0->second;
            td.uv[1] = i1->second;
            td.uv[2] = i2->second;
            auto edgeRatio = [&](unsigned int a, unsigned int b,
                                 const glm::vec2& ua, const glm::vec2& ub) -> float {
                float d3D = glm::distance(toGlm(_mesh->vertices()[a]), toGlm(_mesh->vertices()[b]));
                if (d3D < 1e-6f) return 0.f;
                return (glm::distance(ua, ub) / _uvScale) / d3D;
            };
            td.maxEdgeRatio = std::max({ edgeRatio(t[0], t[1], td.uv[0], td.uv[1]),
                                         edgeRatio(t[1], t[2], td.uv[1], td.uv[2]),
                                         edgeRatio(t[2], t[0], td.uv[2], td.uv[0]) });
            td.valid = true;
            _triDisplayCache.push_back(td);
        }
    }

    void computeVertexNormals() {
        if (!_mesh || _mesh->vertices().empty() || _mesh->triangles().empty()) {
            std::cerr << "[WARNING] Cannot compute normals: Mesh is invalid or empty." << std::endl;
            return;
        }

        std::vector<sibr::Vector3f>& normals = const_cast<std::vector<sibr::Vector3f>&>(_mesh->normals()); 
        if (normals.size() != _mesh->vertices().size()) {
             normals.assign(_mesh->vertices().size(), sibr::Vector3f(0.0f, 0.0f, 0.0f));
        } else {
            std::fill(normals.begin(), normals.end(), sibr::Vector3f(0.0f, 0.0f, 0.0f)); 
        }

        const auto& vertices = _mesh->vertices();
        const auto& triangles = _mesh->triangles();

        for (const auto& tri : triangles) {
            if (tri.x() >= vertices.size() || tri.y() >= vertices.size() || tri.z() >= vertices.size()) {
                continue;
            }

            const sibr::Vector3f& v0 = vertices[tri.x()];
            const sibr::Vector3f& v1 = vertices[tri.y()];
            const sibr::Vector3f& v2 = vertices[tri.z()];

            sibr::Vector3f edge1 = v1 - v0;
            sibr::Vector3f edge2 = v2 - v0;
            sibr::Vector3f faceNormal = edge1.cross(edge2);

            if (faceNormal.norm() > 1e-6f) {
                faceNormal.normalize();
                normals[tri.x()] += faceNormal;
                normals[tri.y()] += faceNormal;
                normals[tri.z()] += faceNormal;
            }
        }

        for (sibr::Vector3f& n : normals) {
            if (n.norm() > 1e-6f) n.normalize();
            else n = sibr::Vector3f(0.0f, 1.0f, 0.0f); 
        }
    }
    
    static bool segmentsProperlyIntersect(const glm::vec2& p1, const glm::vec2& p2,
                                          const glm::vec2& p3, const glm::vec2& p4) {
        auto cross2D = [](const glm::vec2& a, const glm::vec2& b) {
            return a.x * b.y - a.y * b.x;
        };
        glm::vec2 d = p2 - p1, e = p4 - p3;
        float denom = cross2D(d, e);
        if (std::abs(denom) < 1e-12f) return false;
        glm::vec2 r = p3 - p1;
        float t = cross2D(r, e) / denom;
        float u = cross2D(r, d) / denom;
        const float eps = 1e-5f;
        return t > eps && t < 1.f - eps && u > eps && u < 1.f - eps;
    }

    static bool pointStrictlyInTriangle(const glm::vec2& p,
                                        const glm::vec2& a, const glm::vec2& b, const glm::vec2& c) {
        float d1 = (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y);
        float d2 = (p.x - c.x) * (b.y - c.y) - (b.x - c.x) * (p.y - c.y);
        float d3 = (p.x - a.x) * (c.y - a.y) - (c.x - a.x) * (p.y - a.y);
        const float eps = 1e-6f;
        bool hasNeg = (d1 < -eps) || (d2 < -eps) || (d3 < -eps);
        bool hasPos = (d1 >  eps) || (d2 >  eps) || (d3 >  eps);
        if (hasNeg && hasPos) return false;
        return (std::abs(d1) > eps) && (std::abs(d2) > eps) && (std::abs(d3) > eps);
    }

    static bool uvTrianglesOverlap(const glm::vec2 a[3], const glm::vec2 b[3]) {
        // Check if any vertex of one triangle lies inside the other
        for (int i = 0; i < 3; ++i) {
            if (pointStrictlyInTriangle(a[i], b[0], b[1], b[2])) return true;
            if (pointStrictlyInTriangle(b[i], a[0], a[1], a[2])) return true;
        }
        // Check if any edge pair properly intersects
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (segmentsProperlyIntersect(a[i], a[(i+1)%3], b[j], b[(j+1)%3]))
                    return true;
            }
        }
        return false;
    }

    void refineUVsTriangleUnfolding(int iterations = 8) {
        const int meshSize = (int)_mesh->vertices().size();
        std::vector<std::pair<float, int>> byDist;
        for (auto& [id, vd] : _vertexData) {
            if (id < meshSize && vd.frozen) byDist.push_back({vd.cost, id});
        }
        std::sort(byDist.begin(), byDist.end());

        for (int iter = 0; iter < iterations; ++iter) {
            float blend = 0.5f + 0.3f * (float)iter / (float)iterations;

            for (auto& [cost, vIdx] : byDist) {
                if (cost < 1e-7f) continue;
                if (vIdx >= (int)_vertexToTriangles.size()) continue;

                glm::vec2 uvAccum(0.f, 0.f);
                float     wAccum = 0.f;

                for (int triIdx : _vertexToTriangles[vIdx]) {
                    const auto& tri = _mesh->triangles()[triIdx];
                    int vA = -1, vB = -1;
                    for (int k = 0; k < 3; ++k) {
                        if ((int)tri[k] != vIdx) {
                            if (vA < 0) vA = (int)tri[k]; else vB = (int)tri[k];
                        }
                    }
                    
                    auto itA = _vertexData.find(vA);
                    auto itB = _vertexData.find(vB);
                    if (itA == _vertexData.end() || itB == _vertexData.end() || 
                        !itA->second.frozen || !itB->second.frozen) continue;

                    
                    glm::vec3 pV = getPos(vIdx), pA = getPos(vA), pB = getPos(vB);
                    float dAB = glm::distance(pA, pB);
                    float dVA = glm::distance(pV, pA);
                    float dVB = glm::distance(pV, pB);
                    if (dAB < 1e-7f) continue;

                    float cosA = glm::clamp((dVA*dVA + dAB*dAB - dVB*dVB) / (2.f * dVA * dAB), -1.f, 1.f);
                    float sinA = std::sqrt(1.f - cosA * cosA);

                    glm::vec2 uvA = itA->second.uv, uvB = itB->second.uv;
                    glm::vec2 edgeUV = uvB - uvA;
                    float L_uv = glm::length(edgeUV);
                    if (L_uv < 1e-8f) continue;

                    float localScale = L_uv / dAB;
                    glm::vec2 dir = edgeUV / L_uv;
                    glm::vec2 perp(-dir.y, dir.x);

                    glm::vec3 nA = glm::normalize(getNormal(vA));
                    float side = glm::dot(glm::cross(pB - pA, pV - pA), nA);
                    glm::vec2 uvEst = uvA + dir * (dVA * cosA * localScale) + 
                                    perp * (dVA * sinA * localScale * (side >= 0.f ? 1.f : -1.f));

                    float w = std::max(0.1f, glm::dot(nA, glm::normalize(getNormal(vIdx))));
                    uvAccum += uvEst * w;
                    wAccum += w;
                }

                if (wAccum > 1e-6f) {
                    _vertexData[vIdx].uv = glm::mix(_vertexData[vIdx].uv, uvAccum / wAccum, blend);
                }
            }
        }
    }

    // As-rigid-as-possible UV relaxation -- Liu et al., "A Local/Global Approach
    // to Mesh Parameterization" (2008) -- run on top of the exponential-map UVs.
    //
    // The discrete exp map is a *local* parametrization: propagate() unfolds each
    // vertex across a single parent's tangent plane, so along a geodesic the
    // mesh's curvature piles up as UV shear and fold-over. At a small ExpMap
    // radius that error is invisible; at a large one it is exactly what makes
    // whole rings of far triangles fail the winding / edge-ratio filters in
    // Compute() and drop out -- the "shattered" patch, and the reason the
    // projected Gaussians on those faces scatter.
    //
    // ARAP repairs the global shape: a fixed cotangent-Laplacian solve
    // (factorized once) alternates with a per-triangle best-fit rotation, pulling
    // every triangle's UV edges back toward a rigid copy of its 3D edges. The
    // exp-map UVs are the initial guess, so a few iterations converge. Only
    // translation is left free here; Compute()'s existing recenter + _uvScale
    // normalization pins that down afterwards, unchanged.
    void refineUVsARAP(int iterations) {
        const int meshSize = (int)_mesh->vertices().size();

        // Compact index space over the frozen patch vertices.
        std::vector<int> local2vid;
        local2vid.reserve(_vertexData.size());
        std::unordered_map<int, int> vid2local;
        for (auto& [id, vd] : _vertexData) {
            if (id < meshSize && vd.frozen) {
                vid2local[id] = (int)local2vid.size();
                local2vid.push_back(id);
            }
        }
        const int n = (int)local2vid.size();
        if (n < 8) return;   // exp map is already accurate on a patch this small

        static const int OPP[3][2] = { {1, 2}, {2, 0}, {0, 1} };  // edge opposite corner k

        // Patch triangles: mesh faces whose three corners are all in the patch.
        struct ArapTri {
            int       v[3];     // local vertex indices
            glm::vec2 x[3];     // the 3D triangle laid out isometrically in 2D
            float     cot[3];   // cotangent of the angle at corner k
        };
        std::vector<ArapTri> tris;
        tris.reserve(4096);
        for (const auto& mt : _mesh->triangles()) {
            auto i0 = vid2local.find((int)mt[0]);
            auto i1 = vid2local.find((int)mt[1]);
            auto i2 = vid2local.find((int)mt[2]);
            if (i0 == vid2local.end() || i1 == vid2local.end() || i2 == vid2local.end())
                continue;

            glm::vec3 p0 = toGlm(_mesh->vertices()[mt[0]]);
            glm::vec3 p1 = toGlm(_mesh->vertices()[mt[1]]);
            glm::vec3 p2 = toGlm(_mesh->vertices()[mt[2]]);

            glm::vec3 e01 = p1 - p0;
            float l01 = glm::length(e01);
            if (l01 < 1e-9f) continue;
            glm::vec3 ex = e01 / l01;
            glm::vec3 e02 = p2 - p0;
            float qx = glm::dot(e02, ex);
            float qy = glm::length(e02 - qx * ex);
            if (qy < 1e-9f) continue;   // degenerate face

            ArapTri at;
            at.v[0] = i0->second; at.v[1] = i1->second; at.v[2] = i2->second;
            at.x[0] = glm::vec2(0.f, 0.f);
            at.x[1] = glm::vec2(l01, 0.f);
            at.x[2] = glm::vec2(qx, qy);

            auto cotAt = [](const glm::vec2& a, const glm::vec2& b) {
                float cr = std::abs(a.x * b.y - a.y * b.x);   // 2 * triangle area
                return (cr < 1e-12f) ? 0.f : glm::dot(a, b) / cr;
            };
            at.cot[0] = cotAt(at.x[1] - at.x[0], at.x[2] - at.x[0]);
            at.cot[1] = cotAt(at.x[2] - at.x[1], at.x[0] - at.x[1]);
            at.cot[2] = cotAt(at.x[0] - at.x[2], at.x[1] - at.x[2]);
            // Obtuse corners give negative cotangents that let the global solve
            // fold the patch back on itself; clamp them out. We trade the last
            // few percent of conformality for not shattering.
            for (int k = 0; k < 3; ++k) at.cot[k] = std::max(at.cot[k], 1e-4f);
            tris.push_back(at);
        }
        if (tris.empty()) return;

        // Fixed cotan-Laplacian system. Pin local vertex 0 to (0,0) to remove the
        // translational nullspace -- because the pinned value is the origin no
        // right-hand-side coupling term is needed for it.
        const int pin = 0;
        std::vector<Eigen::Triplet<double>> trip;
        trip.reserve(tris.size() * 12);
        auto addW = [&](int a, int b, double w) {
            if (a == pin || b == pin) {
                int f = (a == pin) ? b : a;           // only the free end gets a term
                trip.emplace_back(f, f, w);
                return;
            }
            trip.emplace_back(a, a,  w);
            trip.emplace_back(b, b,  w);
            trip.emplace_back(a, b, -w);
            trip.emplace_back(b, a, -w);
        };
        for (const auto& at : tris)
            for (int k = 0; k < 3; ++k)
                addW(at.v[OPP[k][0]], at.v[OPP[k][1]], (double)at.cot[k]);
        trip.emplace_back(pin, pin, 1.0);

        Eigen::SparseMatrix<double> A(n, n);
        A.setFromTriplets(trip.begin(), trip.end());
        A.makeCompressed();

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);
        if (solver.info() != Eigen::Success) return;   // keep the exp-map UVs untouched

        Eigen::MatrixX2d U(n, 2);
        for (int i = 0; i < n; ++i) {
            const glm::vec2& uv = _vertexData[local2vid[i]].uv;
            U(i, 0) = uv.x; U(i, 1) = uv.y;
        }

        Eigen::MatrixX2d B(n, 2);
        for (int iter = 0; iter < iterations; ++iter) {
            B.setZero();
            for (const auto& at : tris) {
                // Local step: rotation that best maps the 3D edges onto the UV edges.
                Eigen::Matrix2d S = Eigen::Matrix2d::Zero();
                for (int k = 0; k < 3; ++k) {
                    int a = at.v[OPP[k][0]], b = at.v[OPP[k][1]];
                    Eigen::Vector2d du(U(a, 0) - U(b, 0), U(a, 1) - U(b, 1));
                    Eigen::Vector2d dx(at.x[OPP[k][0]].x - at.x[OPP[k][1]].x,
                                       at.x[OPP[k][0]].y - at.x[OPP[k][1]].y);
                    S += (double)at.cot[k] * du * dx.transpose();
                }
                Eigen::JacobiSVD<Eigen::Matrix2d> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::Matrix2d Vt = svd.matrixV();
                Eigen::Matrix2d R  = svd.matrixU() * Vt.transpose();
                if (R.determinant() < 0.0) {            // reject reflections
                    Vt.col(1) *= -1.0;
                    R = svd.matrixU() * Vt.transpose();
                }
                // Accumulate this triangle's contribution to the global RHS.
                for (int k = 0; k < 3; ++k) {
                    int a = at.v[OPP[k][0]], b = at.v[OPP[k][1]];
                    Eigen::Vector2d dx(at.x[OPP[k][0]].x - at.x[OPP[k][1]].x,
                                       at.x[OPP[k][0]].y - at.x[OPP[k][1]].y);
                    Eigen::Vector2d rhs = (double)at.cot[k] * (R * dx);
                    B(a, 0) += rhs.x(); B(a, 1) += rhs.y();
                    B(b, 0) -= rhs.x(); B(b, 1) -= rhs.y();
                }
            }
            B.row(pin).setZero();

            // Global step: both coordinates reuse the one factorization.
            U.col(0) = solver.solve(B.col(0));
            U.col(1) = solver.solve(B.col(1));
        }

        for (int i = 0; i < n; ++i)
            _vertexData[local2vid[i]].uv = glm::vec2((float)U(i, 0), (float)U(i, 1));
    }

    void buildBaseAdjacency() {
        const int N = (int)_mesh->vertices().size();
        _baseAdj.assign(N, {});
        _vertexToTriangles.assign(N, {});

        const auto& tris = _mesh->triangles();
        for (size_t ti = 0; ti < tris.size(); ++ti) {
            int a = (int)tris[ti][0], b = (int)tris[ti][1], c = (int)tris[ti][2];

            auto addEdge = [&](int u, int v) {
                if (std::find(_baseAdj[u].begin(), _baseAdj[u].end(), v) == _baseAdj[u].end())
                    _baseAdj[u].push_back(v);
                if (std::find(_baseAdj[v].begin(), _baseAdj[v].end(), u) == _baseAdj[v].end())
                    _baseAdj[v].push_back(u);
            };
            addEdge(a, b); addEdge(b, c); addEdge(c, a);

            _vertexToTriangles[a].push_back((int)ti);
            _vertexToTriangles[b].push_back((int)ti);
            _vertexToTriangles[c].push_back((int)ti);
        }
    }

    void propagate(int parentIdx, int currIdx) {
        ExpVertex& parent = _vertexData[parentIdx];
        ExpVertex& curr   = _vertexData[currIdx];

        glm::vec3 posP = getPos(parentIdx);
        glm::vec3 posC = getPos(currIdx);

        float edgeLen = glm::distance(posP, posC);
        float newCost = parent.cost + edgeLen;
        if (newCost >= curr.cost) return;
        curr.cost     = newCost;
        curr.parentId = parentIdx;

        if (parentIdx >= (int)_mesh->vertices().size() || currIdx >= (int)_mesh->vertices().size()) return;

        glm::vec3 nP = glm::normalize(getNormal(parentIdx));

        TangentFrame centerFrame(posP, nP);

        TangentFrame seedAligned = _seedFrame;
        seedAligned.alignZAxis(centerFrame);

        glm::vec3 centerAxisX = centerFrame.axes[0];
        glm::vec3 seedAxisX   = seedAligned.axes[0];
        float cosTheta = glm::clamp(glm::dot(centerAxisX, seedAxisX), -1.f, 1.f);
        float fTmp     = std::max(0.f, 1.f - cosTheta * cosTheta);
        float sinTheta = std::sqrt(fTmp);
        glm::vec3 crossVec = glm::cross(centerAxisX, seedAxisX);
        if (glm::dot(crossVec, nP) < 0.f) sinTheta = -sinTheta;

        glm::mat2 matR(
            glm::vec2( cosTheta, -sinTheta),
            glm::vec2( sinTheta,  cosTheta)
        );

        glm::vec3 posC_proj = posC - nP * glm::dot(posC - posP, nP);
        glm::vec3 localVec  = centerFrame.toLocal(posC_proj - posP);

        curr.uv = parent.uv + matR * glm::vec2(localVec.x, localVec.y);
    }

    void ComputeExtraNodeCosts() {
        size_t meshSize = _mesh->vertices().size();
        for (auto& [nodeID, vData] : _vertexData) {
            if (nodeID < (int)meshSize) continue; 
            
            float minCost = 1e9f;
            int bestParent = -1;
            
            if (nodeID < (int)_adj.size()) {
                for (int neighborID : _adj[nodeID]) {
                    if (neighborID >= (int)meshSize) continue; 
                    if (_vertexData.find(neighborID) == _vertexData.end()) continue;
                    if (!_vertexData[neighborID].frozen) continue;
                    
                    float dist = glm::distance(getPos(nodeID), getPos(neighborID));
                    glm::vec3 nodeN = getNormal(nodeID);
                    glm::vec3 neighborN = getNormal(neighborID);
                    float dotNormal = glm::dot(nodeN, neighborN);
                    
                    if (dotNormal < 0.3f) continue; 
                    
                    float penalty = 1.0f + 5.0f * (1.0f - dotNormal);
                    float edgeCost = dist * penalty;
                    float totalCost = _vertexData[neighborID].cost + edgeCost;
                    
                    if (totalCost < minCost) {
                        minCost = totalCost;
                        bestParent = neighborID;
                    }
                }
            }
            
            if (bestParent != -1) {
                vData.cost = minCost;
                vData.parentId = bestParent;
                vData.frozen = true;
            }
        }
    }

    const sibr::Mesh* _mesh = nullptr;
    std::vector<std::vector<int>> _baseAdj;
    std::vector<std::vector<int>> _adj;
    std::vector<std::vector<int>> _vertexToTriangles;
    std::map<int, ExpVertex>      _vertexData;
    TangentFrame                  _seedFrame;
    std::vector<glm::vec3>        _strokePoints;   // brush polyline; 1 entry == single click

    std::vector<sibr::Vector3u>   _validTriangles;
    std::vector<int>              _validTriIDs;
    std::set<int>                 _validTriangleIndicesSet;

    std::vector<glm::vec3>        _extraNodePositions;
    std::vector<glm::vec3>        _extraNodeNormals;

    float  _uvScale       = 1.0f;
    ImVec2 _viewOffset    = ImVec2(0, 0);
    float  _viewScale     = 1.0f;
    bool   _drawFilled    = true;
    bool   _cullBackFace  = true;
    bool   _showBackgroundTexture = true;
    bool   _showMeshGrid          = true;   // "Grid" -- the yellow triangle overlay on the UV canvas

    bool   _showGaussianPoints = true;
    int    _gaussStrideShown   = 1;   // last frame's Gaussian-dot thinning factor, for the UI label

    std::vector<ProjectedGaussian> _projectedGaussians;
    std::vector<EllipseGeom>       _ellipseGeom;      // parallel to _projectedGaussians
    std::vector<GaussFilterCache>  _gaussFilter;      // parallel to _projectedGaussians
    std::vector<int>               _gaussDrawOrder;   // indices, back-to-front by surfDist
    std::vector<TriDisplay>        _triDisplayCache;  // parallel to _validTriangles

    std::map<int, glm::vec2> _displayUVs;

    TextureLoader _textureLoader;
    sibr::Texture2DRGBA::Ptr _cudaTexPtr = nullptr;
    float _cudaTexRotDeg = 0.f;  // rotation baked into _cudaTexPtr's coverage mask
    bool _textureDirty = false;
    float _cachedGaussianCoverage = 0.f;
    std::vector<ProjectedGaussian> _pendingMainGaussians;

    uint64_t _activeGeneration = 0;

    float _autoThreshold = 3.0f;
    
    ProjectionStats _projectionStats;

    GLuint _liveSSBO = 0;
    bool   _liveDirty = true;

    float _surfaceBlend      = 1.0f;
    bool  _surfaceBlendDirty = false;

    float _surfDistMin         = 0.0f;
    float _surfDistMax         = 1e9f;
    float _computedMaxSurfDist = 1.0f;
    bool  _surfDistRangeDirty  = false;
    bool  _draggingMin         = false;
    bool  _draggingMax         = false;
};

#endif