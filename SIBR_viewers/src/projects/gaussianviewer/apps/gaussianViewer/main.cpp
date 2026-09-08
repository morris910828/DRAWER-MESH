#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include <regex>
#include <array>
#include <cmath>
#include <unordered_map>
#include <opencv2/videoio.hpp>

#include <core/graphics/Window.hpp>
#include <core/view/MultiViewManager.hpp>
#include <core/system/String.hpp>
#include <core/system/Utils.hpp>
#include "projects/gaussianviewer/renderer/GaussianView.hpp"
#include <core/renderer/DepthRenderer.hpp>
#include <core/raycaster/Raycaster.hpp>
#include <core/view/SceneDebugView.hpp>
#include <imgui/imgui_internal.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "ExpMapSolverSIBR.h"

namespace std_fs = std::filesystem;
using namespace sibr;

// Load mesh_face_idx from a new-format splat PLY (vertex element with mesh_face_idx property).
// Returns true and fills |fids| if the property is found, false otherwise.
static bool loadFidsFromSgPly(const std::string& plyPath, std::vector<int>& fids)
{
    std::ifstream f(plyPath, std::ios::binary);
    if (!f.good()) return false;

    auto propBytes = [](const std::string& t) -> size_t {
        if (t == "float" || t == "int" || t == "uint") return 4;
        if (t == "double" || t == "int64" || t == "uint64") return 8;
        if (t == "short"  || t == "ushort") return 2;
        return 1;
    };

    int vertCount  = 0;
    size_t stride  = 0;
    size_t fidOff  = 0;
    bool hasFid    = false;
    bool inVertex  = false;

    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line == "end_header") break;
        std::stringstream ss(line);
        std::string tok; ss >> tok;
        if (tok == "element") {
            std::string name; int cnt; ss >> name >> cnt;
            inVertex = (name == "vertex");
            if (inVertex) { vertCount = cnt; stride = 0; fidOff = 0; hasFid = false; }
        } else if (tok == "property" && inVertex) {
            std::string type, name; ss >> type >> name;
            if (type != "list") {
                if (name == "mesh_face_idx") { fidOff = stride; hasFid = true; }
                stride += propBytes(type);
            }
        }
    }

    if (!hasFid || vertCount == 0) return false;

    fids.resize(vertCount);
    std::vector<char> buf(stride);
    for (int i = 0; i < vertCount; i++) {
        f.read(buf.data(), (std::streamsize)stride);
        int32_t fid = -1;
        std::memcpy(&fid, buf.data() + fidOff, sizeof(int32_t));
        fids[i] = (int)fid;
    }
    std::cout << "Loaded " << vertCount << " mesh_face_idx from: " << plyPath << "\n";
    return true;
}

class MeshGaussianView : public sibr::ViewBase {
public:
    using Ptr = std::shared_ptr<MeshGaussianView>;

    MeshGaussianView(
        GaussianView::Ptr                   gaussianView,
        const sibr::Mesh* mesh,
        sibr::InteractiveCameraHandler::Ptr camHandler,
        const std::string&                  splatPath = ""
    )   : ViewBase(gaussianView->getResolution().x(), gaussianView->getResolution().y()),
          _gaussianView(gaussianView),
          _mesh(mesh),
          _camHandler(camHandler),
          _viewport(0, 0, (float)gaussianView->getResolution().x(), (float)gaussianView->getResolution().y())
    {
        // New unified format: load per-Gaussian face IDs from splat PLY
        if (!splatPath.empty() && _allFids.empty())
            loadFidsFromSgPly(splatPath, _allFids);

        // Re-sort _allFids from PLY order to GPU Morton-sorted order for backface culling.
        const auto& plyToSorted = gaussianView->getPlyToSorted();
        if (!_allFids.empty() && (int)plyToSorted.size() == (int)_allFids.size()) {
            _sortedFids.resize(_allFids.size(), -1);
            for (int i = 0; i < (int)_allFids.size(); ++i)
                _sortedFids[plyToSorted[i]] = _allFids[i];
        }

        // Precompute world-space face normals (used by the ExpMap patch normal-map bake).
        if (mesh && !mesh->triangles().empty()) {
            const auto& verts = mesh->vertices();
            const auto& tris  = mesh->triangles();
            _faceNormals.resize(tris.size());
            for (size_t t = 0; t < tris.size(); ++t) {
                const auto& tri = tris[t];
                sibr::Vector3f e1 = verts[tri[1]] - verts[tri[0]];
                sibr::Vector3f e2 = verts[tri[2]] - verts[tri[0]];
                sibr::Vector3f n  = e1.cross(e2);
                float len = n.norm();
                _faceNormals[t] = (len > 1e-9f) ? (n / len) : sibr::Vector3f(0.f, 0.f, 1.f);
            }
        }

        if (_mesh) {
            _expMapSolver.Init(_mesh);
            _wireframeRenderer.uploadMesh(_mesh);
        }
        _meshColor = sibr::Vector3f(0.0f, 0.6f, 0.0f);
        initGaussianOutlineRenderer();
    }

    ~MeshGaussianView() {
        if (_liveSSBO) glDeleteBuffers(1, &_liveSSBO);
        if (_gaussOutlineVAO)    glDeleteVertexArrays(1, &_gaussOutlineVAO);
        if (_gaussOutlineVBO)    glDeleteBuffers(1, &_gaussOutlineVBO);
        if (_gaussOutlineShader) glDeleteProgram(_gaussOutlineShader);
        if (_normalMapGLTex)     glDeleteTextures(1, &_normalMapGLTex);
    }

    void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override {
        if (!_gaussianView) return;

        if (_recording360) {
            constexpr float kPi = 3.14159265358979323846f;
            float angle    = (2.0f * kPi * _recordFrame) / (float)_recordTotalFrames;
            float elevRad  = _orbitElevation * kPi / 180.0f;
            float horizR   = _orbitRadius * std::cos(elevRad);
            glm::vec3 camPos(
                _orbitCenter.x + horizR * std::sin(angle),
                _orbitCenter.y + _orbitRadius * std::sin(elevRad),
                _orbitCenter.z + horizR * std::cos(angle)
            );
            sibr::Camera orbitCam = eye;
            orbitCam.setLookAt(
                sibr::Vector3f(camPos.x, camPos.y, camPos.z),
                sibr::Vector3f(_orbitCenter.x, _orbitCenter.y, _orbitCenter.z),
                sibr::Vector3f(0.f, 1.f, 0.f)
            );

            _gaussianView->onRenderIBR(dst, orbitCam);
            _viewport = Viewport(0, 0, (float)dst.w(), (float)dst.h());
            dst.bind();
            glViewport(0, 0, dst.w(), dst.h());
            const glm::mat4 mvp = glm::make_mat4(orbitCam.viewproj().data());
            if (_showGaussianOutlines) renderGaussianOutlines(orbitCam);
            _wireframeRenderer.render(_mesh, _expMapSolver.GetActiveTriIndices(), mvp, _showMesh,
                                      _expMapSolver.GetActiveGeneration());

            // Capture frame and write to video
            int W = dst.w(), H = dst.h();
            if (!_videoWriter.isOpened()) {
                std::string videoPath = _videoDir + "/orbit.mp4";
                _videoWriter.open(videoPath,
                    cv::VideoWriter::fourcc('m','p','4','v'), 30.0, cv::Size(W, H));
                if (!_videoWriter.isOpened())
                    std::cerr << "[Record360] Failed to open VideoWriter: " << videoPath << "\n";
            }
            cv::Mat frame(H, W, CV_8UC3);
            glReadPixels(0, 0, W, H, GL_BGR, GL_UNSIGNED_BYTE, frame.data);
            dst.unbind();
            cv::flip(frame, frame, 0);
            if (_videoWriter.isOpened())
                _videoWriter.write(frame);

            _recordFrame++;
            if (_recordFrame >= _recordTotalFrames) {
                _recording360 = false;
                _videoWriter.release();
                std::cout << "[Record360] Done. Video saved to: "
                          << _videoDir << "/orbit.mp4\n";
            }
        } else {
            _gaussianView->onRenderIBR(dst, eye);
            _viewport = Viewport(0, 0, (float)dst.w(), (float)dst.h());
            dst.bind();
            glViewport(0, 0, dst.w(), dst.h());
            const glm::mat4 mvp = glm::make_mat4(eye.viewproj().data());
            if (_showGaussianOutlines) renderGaussianOutlines(eye);
            _wireframeRenderer.render(_mesh, _expMapSolver.GetActiveTriIndices(), mvp, _showMesh,
                                      _expMapSolver.GetActiveGeneration());
            dst.unbind();
        }
    }

    void onUpdate(sibr::Input& input) override {
        if (!_gaussianView) return;
        _gaussianView->onUpdate(input);
        if (input.mouseButton().isReleased(sibr::Mouse::Right) &&
            input.key().isActivated(sibr::Key::LeftShift))
            performRaycast(input);
    }

    void onGUI() override {
        if (!_gaussianView) return;
        _gaussianView->onGUI();
        ImGui::Begin("Mesh & Controls");
        ImGui::Checkbox("Show Mesh", &_showMesh);
        ImGui::Checkbox("Show Yellow Wireframe", &_wireframeRenderer._showYellowWireframe);
        ImGui::SliderFloat("ExpMap Radius", &_expMapRadius, 0.05f, 5.0f);
        ImGui::Separator();
        ImGui::Checkbox("Show Gaussian Outlines", &_showGaussianOutlines);
        ImGui::Separator();
        {
            bool prevShowTex = _showTexture;
            ImGui::Checkbox("Show Texture", &_showTexture);
            if (_showTexture != prevShowTex && !_lastAllUVs.empty()) {
                _gaussianView->restoreOpacities();
                if (_showTexture) {
                    uploadCommittedPlusStaged();
                } else {
                    // Toggling textures off is "no Gaussian points at a patch" rather
                    // than "there is no texture" -- every patch stays registered, so
                    // turning it back on costs nothing.
                    auto blended = computeBlendedPos(_expMapSolver.GetSurfaceBlend());
                    std::vector<int> none(_gaussianView->getCount(), -1);
                    _gaussianView->setUVsAndTexture(
                        _lastAllUVs, _lastAllDUs, _lastAllDVs, _lastAllSurfDists,
                        blended, none
                    );
                }
            }
        }
        ImGui::Separator();
        if (!_recording360) {
            ImGui::InputInt("360 Frames##360f", &_recordTotalFrames);
            if (_recordTotalFrames < 1) _recordTotalFrames = 1;
            ImGui::SliderFloat("Orbit Zoom##oz", &_orbitZoom, 0.1f, 10.0f, "%.2f");
            ImGui::SliderFloat("Elevation##elev", &_orbitElevation, -89.0f, 89.0f, "%.1f deg");
            if (ImGui::Button("Record 360\xc2\xb0")) {
                // Compute orbit center from mesh bounding box
                if (_mesh && !_mesh->vertices().empty()) {
                    glm::vec3 mn(1e9f), mx(-1e9f);
                    for (const auto& v : _mesh->vertices()) {
                        glm::vec3 gv(v.x(), v.y(), v.z());
                        mn = glm::min(mn, gv);
                        mx = glm::max(mx, gv);
                    }
                    _orbitCenter = (mn + mx) * 0.5f;
                } else {
                    _orbitCenter = {0.f, 0.f, 0.f};
                }
                const sibr::Camera& cam = _camHandler->getCamera();
                glm::vec3 camPos(cam.position().x(), cam.position().y(), cam.position().z());
                glm::vec2 toXZ(_orbitCenter.x - camPos.x, _orbitCenter.z - camPos.z);
                float baseRadius = glm::length(toXZ);
                if (baseRadius < 1e-4f) baseRadius = 1.0f;
                _orbitRadius = baseRadius * _orbitZoom;
                // Base dir = one level above SIBR_viewers (mirrors the old layout
                // where video/ was a sibling of SIBR_viewers). getInstallDirectory()
                // is <exe>/../.. i.e. SIBR_viewers/install, so go up two more.
                _videoDir = sibr::parentDirectory(sibr::parentDirectory(sibr::getInstallDirectory())) + "/video";
                std_fs::create_directories(_videoDir);
                _recordFrame = 0;
                _recording360 = true;
                std::cout << "[Record360] Start " << _recordTotalFrames
                          << " frames, center=(" << _orbitCenter.x << ","
                          << _orbitCenter.y << "," << _orbitCenter.z
                          << "), r=" << _orbitRadius << ", elev=" << _orbitElevation << "\n";
            }
        } else {
            ImGui::Text("Recording %d / %d ...", _recordFrame, _recordTotalFrames);
            if (ImGui::Button("Cancel##360cancel")) _recording360 = false;
        }
        ImGui::Separator();
        if (ImGui::Button("Clear ExpMap")) {
            _gaussianView->restoreOpacities();
            _texGaussians.clear();
            _expMapSolver.ClearRaycastState();
            _lastAllOrigPos.clear();
            std::vector<sibr::Vector2f> empty_uvs(_gaussianView->getCount(), sibr::Vector2f(-1.f, -1.f));
            std::vector<sibr::Vector3f> empty_dUs(_gaussianView->getCount(), sibr::Vector3f(0.f, 0.f, 0.f));
            std::vector<sibr::Vector3f> empty_dVs(_gaussianView->getCount(), sibr::Vector3f(0.f, 0.f, 0.f));
            std::vector<float>          empty_sd(_gaussianView->getCount(), 1e9f);
            std::vector<sibr::Vector3f> empty_sp(_gaussianView->getCount(), sibr::Vector3f(-1.f,-1.f,-1.f));
            _gaussianView->clearTextures();
            _lastAllTexIdx.assign(_gaussianView->getCount(), -1);
            _gaussianView->setUVsAndTexture(empty_uvs, empty_dUs, empty_dVs, empty_sd, empty_sp, _lastAllTexIdx);
            _gaussianView->clearNormalMapTexture();
            _patchNormalMap.clear();
            if (_normalMapGLTex) { glDeleteTextures(1, &_normalMapGLTex); _normalMapGLTex = 0; }
            _hasPending = false; _pendingSlot = -1;
            _texRotationDeg = 0.f;
            _pendUVs.clear(); _pendDUs.clear(); _pendDVs.clear(); _pendSurfDists.clear();
            _previewTexIdx.assign(_gaussianView->getCount(), -1);
        }

        // Confirm the staged patch: fold it into the committed record and reset the
        // workspace so the next pick starts from a blank UV-result window.
        // imgui 1.60 has no BeginDisabled; just don't offer the button when there is
        // nothing staged, and say so instead.
        if (!_hasPending) {
            ImGui::TextDisabled("Pick a region to stage a patch");
        } else if (ImGui::Button("Confirm Patch")) {
            const int n = _gaussianView->getCount();
            for (int i = 0; i < n && i < (int)_pendUVs.size(); ++i) {
                if (_pendUVs[i].x() < 0.f) continue;
                _lastAllUVs[i]       = _pendUVs[i];
                _lastAllDUs[i]       = _pendDUs[i];
                _lastAllDVs[i]       = _pendDVs[i];
                // Bake the staged texture rotation into the committed UVs so it
                // survives the pend arrays being cleared below.
                applyTexRotation(_lastAllUVs[i], _lastAllDUs[i], _lastAllDVs[i]);
                _lastAllSurfDists[i] = _pendSurfDists[i];
                _lastAllTexIdx[i]    = _pendingSlot;   // later patch wins on overlap
            }
            _texRotationDeg = 0.f;
            // The slot is now owned by a committed patch, so the next pick must not
            // reuse it -- clearing _pendingSlot is what makes registerTexture append.
            _hasPending = false; _pendingSlot = -1;
            _pendUVs.clear(); _pendDUs.clear(); _pendDVs.clear(); _pendSurfDists.clear();
            _previewTexIdx = _lastAllTexIdx;

            // Reset the workspace: UV result contents, the normal-map preview, and the
            // background image it was painted over. The patch itself stays on the model.
            _expMapSolver.ClearRaycastState();
            _expMapSolver.GetTextureLoader().Clear();
            _texPtr.reset();
            _texGaussians.clear();
            _patchNormalMap.clear();
            _normalMapW = _normalMapH = 0;
            if (_normalMapGLTex) { glDeleteTextures(1, &_normalMapGLTex); _normalMapGLTex = 0; }

            auto blended = computeBlendedPos(_expMapSolver.GetSurfaceBlend());
            _gaussianView->restoreOpacities();
            _gaussianView->setUVsAndTexture(_lastAllUVs, _lastAllDUs, _lastAllDVs,
                                            _lastAllSurfDists, blended, _lastAllTexIdx);
        }
        if (_hasPending) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.f, 0.8f, 0.2f, 1.f), "staged - not applied yet");

            // Spin the staged texture about the patch centre. Re-bakes the coverage
            // mask + normal map and re-uploads the preview so the 3D view follows
            // the slider live.
            if (ImGui::SliderFloat("Texture Rotation##texrot", &_texRotationDeg, -180.f, 180.f, "%.1f deg")) {
                _expMapSolver.RegenerateCudaTexture(_texRotationDeg);
                reRegisterStagedColorTexture();
                bakePatchNormalMap(_texRotationDeg);
                uploadCommittedPlusStaged();
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset##texrot") && _texRotationDeg != 0.f) {
                _texRotationDeg = 0.f;
                _expMapSolver.RegenerateCudaTexture(0.f);
                reRegisterStagedColorTexture();
                bakePatchNormalMap(0.f);
                uploadCommittedPlusStaged();
            }
        }
        ImGui::End();
        _expMapSolver.RenderUI();

        // Normal map display window
        if (_normalMapGLTex != 0 && !_patchNormalMap.empty()) {
            ImGui::Begin("Normal Map", nullptr, ImGuiWindowFlags_NoScrollbar);
            float avail = ImGui::GetContentRegionAvail().x;
            float aspect = (_normalMapW > 0) ? (float)_normalMapH / (float)_normalMapW : 1.f;
            ImGui::Image((ImTextureID)(intptr_t)_normalMapGLTex, ImVec2(avail, avail * aspect));
            ImGui::End();
        }

        // Surface blend slider changed -> update GPU positions with new blend amount
        if (_expMapSolver.ConsumeSurfaceBlendDirty() && !_lastAllUVs.empty() && !_lastAllOrigPos.empty()) {
            uploadCommittedPlusStaged();
        }
        if (_expMapSolver.ConsumeTextureDirty()) {
            _texPtr = _expMapSolver.GetTextureLoader().getTexture();
            // Fires on a new image load and on a FlipU/FlipV toggle. Either way the
            // GL texture changed, so re-register the staged slot and re-bake the
            // normal map (kept at the current rotation) against the new image size.
            if (_hasPending && _texPtr) {
                // Register the triangle-masked texture (alpha 0 outside the patch),
                // not the raw rectangular image, so renderCUDA can cull fragments
                // that fall past the patch instead of splatting them as a fuzzy halo.
                sibr::Texture2DRGBA::Ptr colorTex = _expMapSolver.GetCudaTexture();
                _pendingSlot = _gaussianView->registerTexture(colorTex ? colorTex : _texPtr, _pendingSlot);
                bakePatchNormalMap(_texRotationDeg);
            }
            if (!_lastAllUVs.empty() || _hasPending)
                uploadCommittedPlusStaged();
        }
    }

private:
    // Push committed patches plus the staged one (if any) to the renderer. Every
    // refresh path goes through here so a staged preview is never silently dropped
    // by an unrelated slider.
    void uploadCommittedPlusStaged() {
        const int n = _gaussianView->getCount();
        if ((int)_lastAllUVs.size() != n) return;
        // Separate declarations: uvs is Vector2f, dus/dvs are Vector3f, so a single
        // auto declarator list cannot deduce one type for all three.
        auto uvs = _lastAllUVs;
        auto dus = _lastAllDUs;
        auto dvs = _lastAllDVs;
        auto sds = _lastAllSurfDists;
        auto idx = _lastAllTexIdx;
        if (_hasPending && (int)_pendUVs.size() == n) {
            for (int i = 0; i < n; ++i) {
                if (_pendUVs[i].x() < 0.f) continue;
                uvs[i] = _pendUVs[i]; dus[i] = _pendDUs[i]; dvs[i] = _pendDVs[i];
                applyTexRotation(uvs[i], dus[i], dvs[i]);
                sds[i] = _pendSurfDists[i]; idx[i] = _pendingSlot;
            }
        }
        _previewTexIdx = idx;
        auto blended = computeBlendedPos(_expMapSolver.GetSurfaceBlend());
        _gaussianView->setUVsAndTexture(uvs, dus, dvs, sds, blended, idx);
    }

    // Rotate one Gaussian's texture mapping about the patch centre (UV 0.5,0.5) by
    // _texRotationDeg. uv rotates directly; dU/dV rotate so that renderCUDA's
    // Gram solve (offset = du*dU + dv*dV) recovers the rotated (du,dv), keeping
    // the whole chart a rigid spin -- no stretch, no reprojection.
    void applyTexRotation(sibr::Vector2f& uv, sibr::Vector3f& dU, sibr::Vector3f& dV) const {
        if (std::abs(_texRotationDeg) < 1e-4f) return;
        constexpr float kDeg2Rad = 3.14159265358979323846f / 180.f;
        const float c = std::cos(_texRotationDeg * kDeg2Rad);
        const float s = std::sin(_texRotationDeg * kDeg2Rad);
        const float x = uv.x() - 0.5f, y = uv.y() - 0.5f;
        uv = sibr::Vector2f(c * x - s * y + 0.5f, s * x + c * y + 0.5f);
        const sibr::Vector3f ndU = c * dU - s * dV;
        const sibr::Vector3f ndV = s * dU + c * dV;
        dU = ndU; dV = ndV;
    }

    // Push the staged patch's (rotation-baked) coverage-masked colour texture back
    // into its existing CUDA slot. Call after ExpMapSolver::RegenerateCudaTexture,
    // i.e. whenever the texture rotation changes.
    void reRegisterStagedColorTexture() {
        if (_pendingSlot < 0) return;
        sibr::Texture2DRGBA::Ptr colorTex = _expMapSolver.GetCudaTexture();
        if (colorTex) _pendingSlot = _gaussianView->registerTexture(colorTex, _pendingSlot);
    }

    // Rasterize the active patch's face normals into the normal map, in the same
    // UV space the colour texture is sampled in -- so it must be re-baked with the
    // current rotation whenever _texRotationDeg changes, or renderCUDA would shade
    // against normals that no longer line up with the (rotated) texture lookup.
    void bakePatchNormalMap(float rotDeg) {
        if (!_texPtr || !_mesh) return;
        const int NM_W = _texPtr->w(), NM_H = _texPtr->h();
        _normalMapW = NM_W; _normalMapH = NM_H;
        _patchNormalMap.assign(NM_W * NM_H * 4, 0);  // alpha=0 -> no shading outside the patch

        constexpr float kDeg2Rad = 3.14159265358979323846f / 180.f;
        const float c = std::cos(rotDeg * kDeg2Rad), s = std::sin(rotDeg * kDeg2Rad);
        auto rot = [&](glm::vec2 uv) {
            const float x = uv.x - 0.5f, y = uv.y - 0.5f;
            return glm::vec2(c * x - s * y + 0.5f, s * x + c * y + 0.5f);
        };

        const auto& uvMap = _expMapSolver.GetDisplayUVs();
        const auto& tris  = _mesh->triangles();
        for (int triID : _expMapSolver.GetActiveTriIndices()) {
            if (triID < 0 || triID >= (int)tris.size() || triID >= (int)_faceNormals.size()) continue;
            const auto& t = tris[triID];
            auto i0 = uvMap.find((int)t[0]), i1 = uvMap.find((int)t[1]), i2 = uvMap.find((int)t[2]);
            if (i0 == uvMap.end() || i1 == uvMap.end() || i2 == uvMap.end()) continue;

            const sibr::Vector3f& fn = _faceNormals[triID];
            uint8_t nr = (uint8_t)std::max(0, std::min(255, (int)(fn.x() * 127.5f + 127.5f)));
            uint8_t ng = (uint8_t)std::max(0, std::min(255, (int)(fn.y() * 127.5f + 127.5f)));
            uint8_t nb = (uint8_t)std::max(0, std::min(255, (int)(fn.z() * 127.5f + 127.5f)));

            glm::vec2 p0 = rot(i0->second), p1 = rot(i1->second), p2 = rot(i2->second);
            p0 = {p0.x * NM_W, p0.y * NM_H};
            p1 = {p1.x * NM_W, p1.y * NM_H};
            p2 = {p2.x * NM_W, p2.y * NM_H};
            int xmin = std::max(0,        (int)std::floor(std::min({p0.x, p1.x, p2.x})));
            int xmax = std::min(NM_W - 1, (int)std::ceil (std::max({p0.x, p1.x, p2.x})));
            int ymin = std::max(0,        (int)std::floor(std::min({p0.y, p1.y, p2.y})));
            int ymax = std::min(NM_H - 1, (int)std::ceil (std::max({p0.y, p1.y, p2.y})));
            for (int py = ymin; py <= ymax; ++py) {
                for (int px = xmin; px <= xmax; ++px) {
                    float qx = px + 0.5f, qy = py + 0.5f;
                    float d0 = (p1.x-p0.x)*(qy-p0.y) - (p1.y-p0.y)*(qx-p0.x);
                    float d1 = (p2.x-p1.x)*(qy-p1.y) - (p2.y-p1.y)*(qx-p1.x);
                    float d2 = (p0.x-p2.x)*(qy-p2.y) - (p0.y-p2.y)*(qx-p2.x);
                    if (!((d0>=0&&d1>=0&&d2>=0)||(d0<=0&&d1<=0&&d2<=0))) continue;
                    int pidx = (py * NM_W + px) * 4;
                    _patchNormalMap[pidx+0] = nr;
                    _patchNormalMap[pidx+1] = ng;
                    _patchNormalMap[pidx+2] = nb;
                    _patchNormalMap[pidx+3] = 255;
                }
            }
        }

        _gaussianView->setNormalMapTexture(_patchNormalMap, NM_W, NM_H, _pendingSlot);

        if (_normalMapGLTex == 0) glGenTextures(1, &_normalMapGLTex);
        glBindTexture(GL_TEXTURE_2D, _normalMapGLTex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, NM_W, NM_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, _patchNormalMap.data());
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // Compute per-Gaussian surface position interpolated by blend (0=original, 1=surface).
    std::vector<sibr::Vector3f> computeBlendedPos(float blend) const {
        const int n = (int)_lastAllOrigPos.size();
        std::vector<sibr::Vector3f> out(n, sibr::Vector3f(-1.f, -1.f, -1.f));
        for (int i = 0; i < n; ++i) {
            if (i < (int)_lastAllUVs.size() && _lastAllUVs[i].x() >= 0.f)
                out[i] = _lastAllOrigPos[i] + blend * (_lastAllSurfPos[i] - _lastAllOrigPos[i]);
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // Gaussian outline rendering (3-D ellipses in the mesh tangent plane)
    // -------------------------------------------------------------------------
    void initGaussianOutlineRenderer() {
        const std::string vsSrc = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
)";
        const std::string fsSrc = R"(
#version 330 core
uniform vec4 uColor;
out vec4 fragColor;
void main() { fragColor = uColor; }
)";
        GLuint vs = detail::compileGLShader(vsSrc, GL_VERTEX_SHADER);
        GLuint fs = detail::compileGLShader(fsSrc, GL_FRAGMENT_SHADER);
        _gaussOutlineShader = detail::linkProgram(vs, fs);

        GLint linkOK = 0;
        glGetProgramiv(_gaussOutlineShader, GL_LINK_STATUS, &linkOK);
        if (!linkOK) {
            GLchar log[512]; glGetProgramInfoLog(_gaussOutlineShader, 512, nullptr, log);
            std::cerr << "[GaussianOutline] Shader link error: " << log << "\n";
            glDeleteProgram(_gaussOutlineShader);
            _gaussOutlineShader = 0;
            return;
        }

        glGenVertexArrays(1, &_gaussOutlineVAO);
        glGenBuffers(1, &_gaussOutlineVBO);
        glBindVertexArray(_gaussOutlineVAO);
        glBindBuffer(GL_ARRAY_BUFFER, _gaussOutlineVBO);
        glBufferData(GL_ARRAY_BUFFER, 34 * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW); // pre-allocate
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        std::cout << "[GaussianOutline] Shader initialized OK. Prog=" << _gaussOutlineShader
                  << " VAO=" << _gaussOutlineVAO << " VBO=" << _gaussOutlineVBO << "\n";
    }

    void renderGaussianOutlines(const sibr::Camera& eye) {
        if (!_gaussOutlineShader || !_gaussOutlineVAO) return;

        const auto& gaussians = _expMapSolver.GetProjectedGaussians();
        if (gaussians.empty()) return;

        static constexpr float kPi = 3.14159265358979323846f;
        static constexpr int   SEGS = 32;

        // Camera position & forward vector in world space (for depth sorting)
        const glm::vec3 camPos(eye.position().x(), eye.position().y(), eye.position().z());
        // Linear depth along camera forward axis (more accurate than squared distance)
        const sibr::Vector3f eyeDir = eye.dir();
        const glm::vec3 camFwd(eyeDir.x(), eyeDir.y(), eyeDir.z());

        // ---------------------------------------------------------------
        // Compute the exact 1-sigma ellipse ring for one Gaussian.
        // The 3D covariance is projected onto the mesh tangent plane (dU/dV),
        // giving the actual footprint shape, size and orientation on the surface.
        // ---------------------------------------------------------------
        const float outlineBlend = _expMapSolver.GetSurfaceBlend();
        auto blendedCenter = [&](const ProjectedGaussian& pg) -> glm::vec3 {
            return pg.originalPos + outlineBlend * (pg.position - pg.originalPos);
        };

        auto makeRing = [&](const ProjectedGaussian& pg) -> std::vector<glm::vec3> {
            // Quaternion layout: rotation.x=w, .y=x, .z=y, .w=z
            const float qw = pg.rotation.x, qx = pg.rotation.y;
            const float qy = pg.rotation.z, qz = pg.rotation.w;

            // Rotation matrix (GLM column-major): R[i] = world direction of axis i
            const glm::mat3 R(
                1.f-2.f*(qy*qy+qz*qz),  2.f*(qx*qy+qw*qz),   2.f*(qx*qz-qw*qy),
                2.f*(qx*qy-qw*qz),    1.f-2.f*(qx*qx+qz*qz),  2.f*(qy*qz+qw*qx),
                2.f*(qx*qz+qw*qy),    2.f*(qy*qz-qw*qx),   1.f-2.f*(qx*qx+qy*qy)
            );

            const float s0 = std::exp(pg.scale.x);
            const float s1 = std::exp(pg.scale.y);
            const float s2 = std::exp(pg.scale.z);

            // Build orthonormal tangent basis (t1, t2) from mesh dU/dV vectors.
            // If dU is degenerate, fall back to drawing using the two largest 3D axes.
            glm::vec3 t1 = pg.dU;
            const float lenU = glm::length(t1);
            if (lenU < 1e-7f) {
                float sArr[3] = {s0, s1, s2};
                int idx[3] = {0, 1, 2};
                if (sArr[idx[0]] < sArr[idx[1]]) std::swap(idx[0], idx[1]);
                if (sArr[idx[0]] < sArr[idx[2]]) std::swap(idx[0], idx[2]);
                if (sArr[idx[1]] < sArr[idx[2]]) std::swap(idx[1], idx[2]);
                std::vector<glm::vec3> ring(SEGS);
                const glm::vec3 ctr0 = blendedCenter(pg);
                for (int i = 0; i < SEGS; ++i) {
                    const float phi = 2.f * kPi * i / SEGS;
                    ring[i] = ctr0 + sArr[idx[0]]*std::cos(phi)*R[idx[0]]
                                   + sArr[idx[1]]*std::sin(phi)*R[idx[1]];
                }
                return ring;
            }
            t1 /= lenU;

            glm::vec3 t2 = pg.dV - glm::dot(pg.dV, t1) * t1;
            const float lenV = glm::length(t2);
            if (lenV < 1e-7f) {
                glm::vec3 arb = (std::abs(t1.x) < 0.9f) ? glm::vec3(1,0,0) : glm::vec3(0,1,0);
                t2 = glm::normalize(glm::cross(t1, arb));
            } else {
                t2 /= lenV;
            }

            // Project the 3D covariance Σ = R diag(s²) R^T onto the mesh tangent plane.
            // This gives the ellipse footprint that lies flat on the triangle surface.
            const glm::mat3 Rt = glm::transpose(R);
            auto SigmaV = [&](const glm::vec3& v) -> glm::vec3 {
                glm::vec3 u = Rt * v;
                return R * glm::vec3(s0*s0*u.x, s1*s1*u.y, s2*s2*u.z);
            };

            const glm::vec3 St1 = SigmaV(t1), St2 = SigmaV(t2);
            const float a2d = glm::dot(t1, St1);
            const float b2d = glm::dot(t1, St2);
            const float c2d = glm::dot(t2, St2);

            const float disc = std::sqrt(std::max(0.f, 0.25f*(a2d-c2d)*(a2d-c2d) + b2d*b2d));
            const float lam1 = std::max(0.f, 0.5f*(a2d+c2d) + disc);
            const float lam2 = std::max(0.f, 0.5f*(a2d+c2d) - disc);

            const float semiA = std::sqrt(lam1);
            const float semiB = std::sqrt(lam2);

            glm::vec2 ev1;
            if (std::abs(b2d) > 1e-8f) {
                ev1 = glm::normalize(glm::vec2(b2d, lam1 - a2d));
            } else {
                ev1 = (a2d >= c2d) ? glm::vec2(1.f, 0.f) : glm::vec2(0.f, 1.f);
            }
            const glm::vec2 ev2(-ev1.y, ev1.x);

            const glm::vec3 axisA = semiA * (ev1.x * t1 + ev1.y * t2);
            const glm::vec3 axisB = semiB * (ev2.x * t1 + ev2.y * t2);

            std::vector<glm::vec3> ring(SEGS);
            const glm::vec3 ctr1 = blendedCenter(pg);
            for (int i = 0; i < SEGS; ++i) {
                const float phi = 2.f * kPi * i / SEGS;
                ring[i] = ctr1 + std::cos(phi)*axisA + std::sin(phi)*axisB;
            }
            return ring;
        };

        static constexpr float kFillAlpha    = 0.11f;
        static constexpr float kOutlineAlpha = 0.90f;

        auto drawOne = [&](const std::vector<glm::vec3>& ring,
                            const glm::vec3& center,
                            GLint colorLoc,
                            float cr, float cg, float cb) {
            std::vector<glm::vec3> fan;
            fan.reserve(SEGS + 2);
            fan.push_back(center);
            fan.insert(fan.end(), ring.begin(), ring.end());
            fan.push_back(ring[0]);

            glUniform4f(colorLoc, cr, cg, cb, kFillAlpha);
            glBufferData(GL_ARRAY_BUFFER,
                         (GLsizeiptr)(fan.size() * sizeof(glm::vec3)),
                         fan.data(), GL_STREAM_DRAW);
            glDrawArrays(GL_TRIANGLE_FAN, 0, (GLsizei)fan.size());

            glUniform4f(colorLoc, cr, cg, cb, kOutlineAlpha);
            glBufferData(GL_ARRAY_BUFFER,
                         (GLsizeiptr)(ring.size() * sizeof(glm::vec3)),
                         ring.data(), GL_STREAM_DRAW);
            glDrawArrays(GL_LINE_LOOP, 0, (GLsizei)ring.size());
        };

        // ---------------------------------------------------------------
        // Collect all selected Gaussians (geo + app) and sort back-to-front
        // so semi-transparent ellipses blend correctly.
        // ---------------------------------------------------------------
        struct DrawItem {
            const ProjectedGaussian* pg;
            float depth;  // squared distance from camera (for sorting)
        };
        std::vector<DrawItem> items;
        items.reserve(gaussians.size());

        const float surfDistMin = _expMapSolver.GetSurfDistMin();
        const float surfDistMax = _expMapSolver.GetSurfDistMax();
        for (const auto& pg : gaussians) {
            float d = glm::length(pg.originalPos - pg.position);
            if (d < surfDistMin || d > surfDistMax) continue;
            float depth = glm::dot(blendedCenter(pg) - camPos, camFwd);
            items.push_back({&pg, depth});
        }

        // Back-to-front (largest depth first -> drawn first, so near ones on top)
        std::sort(items.begin(), items.end(),
                  [](const DrawItem& a, const DrawItem& b){ return a.depth > b.depth; });

        // ---- setup GL state ----
        // Unbind SSBO slot 0 left by the Gaussian copy-renderer
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);

        glm::mat4 mvp = glm::make_mat4(eye.viewproj().data());

        GLboolean depthWas = GL_FALSE, blendWas = GL_FALSE;
        glGetBooleanv(GL_DEPTH_TEST, &depthWas);
        glGetBooleanv(GL_BLEND,      &blendWas);

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glUseProgram(_gaussOutlineShader);
        GLint mvpLoc   = glGetUniformLocation(_gaussOutlineShader, "uMVP");
        GLint colorLoc = glGetUniformLocation(_gaussOutlineShader, "uColor");
        if (mvpLoc == -1 || colorLoc == -1) {
            std::cerr << "[GaussianOutline] Uniform not found: uMVP=" << mvpLoc
                      << " uColor=" << colorLoc << "\n";
        }
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));

        glBindVertexArray(_gaussOutlineVAO);
        glBindBuffer(GL_ARRAY_BUFFER, _gaussOutlineVBO);

        // Draw in depth-sorted order (back -> front)
        for (const auto& item : items)
            drawOne(makeRing(*item.pg), blendedCenter(*item.pg), colorLoc, 1.f, 0.2f, 0.2f);

        // Log once when count changes
        static size_t lastDrawTotal = 0;
        if (items.size() != lastDrawTotal) {
            lastDrawTotal = items.size();
            std::cout << "[GaussianOutline] drew " << items.size() << "\n";
        }

        // ---- restore GL state ----
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glUseProgram(0);
        if (!blendWas)  glDisable(GL_BLEND);
        if (depthWas)   glEnable(GL_DEPTH_TEST);
    }

    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------

    void performRaycast(const sibr::Input& input) {
        if (!_mesh || !_camHandler) return;
        const auto& cam = _camHandler->getCamera();
        sibr::Vector2f mp((float)input.mousePosition().x(), (float)input.mousePosition().y());
        float ndcX = (2.f * mp.x()) / _viewport.finalWidth() - 1.f;
        float ndcY = 1.f - (2.f * mp.y()) / _viewport.finalHeight();
        sibr::Matrix4f invVP = cam.viewproj().inverse();
        sibr::Vector4f nearPt = invVP * sibr::Vector4f(ndcX, ndcY, -1.f, 1.f);
        sibr::Vector4f farPt  = invVP * sibr::Vector4f(ndcX, ndcY,  1.f, 1.f);
        nearPt /= nearPt.w(); farPt /= farPt.w();
        sibr::Vector3f rayDir(farPt.x()-nearPt.x(), farPt.y()-nearPt.y(), farPt.z()-nearPt.z());
        rayDir.normalize();

        float minDist = 1e9f; int hitTriID = -1; sibr::Vector3f hitPos;
        const auto& verts = _mesh->vertices();
        const auto& triangles = _mesh->triangles();
        for (size_t t = 0; t < triangles.size(); ++t) {
            const auto& tri = triangles[t];
            sibr::Vector3f e1 = verts[tri[1]] - verts[tri[0]], e2 = verts[tri[2]] - verts[tri[0]], h = rayDir.cross(e2);
            float a = e1.dot(h); if (std::abs(a) < 1e-6f) continue;
            float f = 1.f/a; sibr::Vector3f s = cam.position() - verts[tri[0]];
            float u = f * s.dot(h); if (u < 0.f || u > 1.f) continue;
            sibr::Vector3f q = s.cross(e1); float v = f * rayDir.dot(q); if (v < 0.f || u + v > 1.f) continue;
            float td = f * e2.dot(q); if (td > 1e-6f && td < minDist) { minDist = td; hitTriID = (int)t; hitPos = cam.position() + rayDir * td; }
        }
        if (hitTriID < 0) return;

        _expMapSolver.OnRaycastHit(hitPos, _expMapRadius, hitTriID);
        _texPtr = _expMapSolver.GetTextureLoader().getTexture();
        // Do NOT early-return on null texture: UV result window still needs Gaussian projection.

        // Restore GPU positions to original before reading, so cpuPos is always
        // the true pre-snap positions (needed for correct blend origin).
        _gaussianView->restoreOpacities();

        const std::set<int>& activeSet = _expMapSolver.GetActiveTriIndices();
        const std::map<int, glm::vec2>& glmUVMap = _expMapSolver.GetDisplayUVs();
        const int nGauss = _gaussianView->getCount();
        const auto& cpuPos = _gaussianView->getCpuPositions();
        std::vector<float> cpuRot, cpuScale, cpuOpacity;
        _gaussianView->downloadGaussianData(cpuRot, cpuScale, cpuOpacity);

        struct TriInfo { glm::vec3 v0, v1, v2, e1, e2, n, dU, dV; float d00, d01, d11; glm::vec2 uv0, uv1, uv2; bool valid; int vi0, vi1, vi2; };
        std::vector<TriInfo> triCache;
        // Also build a global-triID → cache-index map for O(1) fid lookup
        std::unordered_map<int,int> triIdxToCache;
        triIdxToCache.reserve(activeSet.size());
        {
            int cIdx = 0;
            for (int triID : activeSet) {
                triIdxToCache[triID] = cIdx++;
                TriInfo ti; ti.valid = false; const auto& tri = triangles[triID];
                ti.vi0 = tri[0]; ti.vi1 = tri[1]; ti.vi2 = tri[2];
                if (!glmUVMap.count(tri[0]) || !glmUVMap.count(tri[1]) || !glmUVMap.count(tri[2])) { triCache.push_back(ti); continue; }
                ti.v0 = {verts[tri[0]].x(), verts[tri[0]].y(), verts[tri[0]].z()};
                ti.v1 = {verts[tri[1]].x(), verts[tri[1]].y(), verts[tri[1]].z()};
                ti.v2 = {verts[tri[2]].x(), verts[tri[2]].y(), verts[tri[2]].z()};
                ti.e1 = ti.v1 - ti.v0; ti.e2 = ti.v2 - ti.v0;
                glm::vec3 rn = glm::cross(ti.e1, ti.e2); float ln = glm::length(rn);
                if (ln < 1e-9f) { triCache.push_back(ti); continue; }
                ti.n = rn/ln; ti.d00 = glm::dot(ti.e1, ti.e1); ti.d01 = glm::dot(ti.e1, ti.e2); ti.d11 = glm::dot(ti.e2, ti.e2);
                ti.uv0 = glmUVMap.at(tri[0]); ti.uv1 = glmUVMap.at(tri[1]); ti.uv2 = glmUVMap.at(tri[2]);
                float du1 = ti.uv1.x - ti.uv0.x, dv1 = ti.uv1.y - ti.uv0.y, du2 = ti.uv2.x - ti.uv0.x, dv2 = ti.uv2.y - ti.uv0.y, det = du1*dv2 - du2*dv1;
                if (std::abs(det) < 1e-9f) { triCache.push_back(ti); continue; }
                ti.dU = (dv2*ti.e1 - dv1*ti.e2)/det; ti.dV = (-du2*ti.e1 + du1*ti.e2)/det; ti.valid = true; triCache.push_back(ti);
            }
        }

        // Vertex-level membership, so the base layer can be resolved exactly instead
        // of by a distance guess. Base-layer Gaussians (mesh_face_idx < 0) sit exactly
        // on mesh vertices -- one per vertex, verified: 250,059 of them against 250,059
        // vertices, every one within 0.01 mm of its vertex. So "is this Gaussian part
        // of the selection" is answerable outright: yes iff its vertex is a corner of
        // some face in the active set. vertToCache maps such a vertex to the faces it
        // belongs to, and vertGrid finds a Gaussian's vertex from its position.
        std::unordered_map<int, std::vector<int>> vertToCache;
        for (int ci = 0; ci < (int)triCache.size(); ++ci) {
            const TriInfo& ti = triCache[ci];
            if (!ti.valid) continue;
            vertToCache[ti.vi0].push_back(ci);
            vertToCache[ti.vi1].push_back(ci);
            vertToCache[ti.vi2].push_back(ci);
        }
        const float VCELL = 0.01f;                 // grid pitch, >> the match tolerance
        const float VEPS2 = 1e-8f;                 // (0.1 mm)^2 in training units
        auto vkey = [VCELL](float x, float y, float z) {
            return ((int64_t)std::floor(x / VCELL) * 73856093LL)
                 ^ ((int64_t)std::floor(y / VCELL) * 19349663LL)
                 ^ ((int64_t)std::floor(z / VCELL) * 83492791LL);
        };
        std::unordered_map<int64_t, std::vector<int>> vertGrid;
        for (const auto& kv : vertToCache) {
            const auto& V = verts[kv.first];
            vertGrid[vkey(V.x(), V.y(), V.z())].push_back(kv.first);
        }

        // Unified splat format: every Gaussian carries its own mesh_face_idx.
        // Without it (PLY lacking the property) every Gaussian falls back to the
        // nearest-triangle search below.
        const bool canUseAllFids = (int)_allFids.size() == nGauss;

        const float r2 = _expMapRadius * _expMapRadius;
        const glm::vec3 ctr(hitPos.x(), hitPos.y(), hitPos.z());
        std::vector<sibr::Vector2f> all_uvs(nGauss, sibr::Vector2f(-1.f, -1.f));
        std::vector<sibr::Vector3f> all_dUs(nGauss, sibr::Vector3f(0.f, 0.f, 0.f));
        std::vector<sibr::Vector3f> all_dVs(nGauss, sibr::Vector3f(0.f, 0.f, 0.f));
        std::vector<float>          all_surfDists(nGauss, 1e9f);
        std::vector<sibr::Vector3f> all_surfPos(nGauss, sibr::Vector3f(-1.f, -1.f, -1.f));
        _texGaussians.clear();

        // Sanity-check the fid mapping itself: a Gaussian must sit almost on top of
        // the triangle its own fid points at (verified against the PLY: median
        // distance 0.0007, max 0.006).  If the sorted/PLY index spaces are crossed
        // the distances come out ~0.46 instead, i.e. the same as picking a random
        // triangle -- which is exactly what a near-zero "by fid" count looks like.
        {
            int nChk = 0, nBad = 0; double sumD = 0.0;
            const int stride = std::max(1, nGauss / 500);
            for (int k = 0; k < nGauss && nChk < 500; k += stride) {
                int fd = (k < (int)_sortedFids.size()) ? _sortedFids[k] : -1;
                if (fd < 0 || fd >= (int)triangles.size()) continue;
                const auto& tr = triangles[fd];
                glm::vec3 c((verts[tr[0]].x() + verts[tr[1]].x() + verts[tr[2]].x()) / 3.f,
                            (verts[tr[0]].y() + verts[tr[1]].y() + verts[tr[2]].y()) / 3.f,
                            (verts[tr[0]].z() + verts[tr[1]].z() + verts[tr[2]].z()) / 3.f);
                glm::vec3 p(cpuPos[k].x(), cpuPos[k].y(), cpuPos[k].z());
                float d = glm::length(p - c);
                sumD += d; ++nChk;
                if (d > 0.05f) ++nBad;
            }
            std::cout << "[fid check] sorted-space samples=" << nChk
                      << "  meanDist=" << (nChk ? sumD / nChk : -1.0)
                      << "  farOnes=" << nBad << " (expect meanDist<0.01, farOnes~0)"
                      << std::endl;

            // Same check against the raw PLY-order fids, as a control.
            nChk = 0; nBad = 0; sumD = 0.0;
            for (int k = 0; k < nGauss && nChk < 500; k += stride) {
                int fd = (k < (int)_allFids.size()) ? _allFids[k] : -1;
                if (fd < 0 || fd >= (int)triangles.size()) continue;
                const auto& tr = triangles[fd];
                glm::vec3 c((verts[tr[0]].x() + verts[tr[1]].x() + verts[tr[2]].x()) / 3.f,
                            (verts[tr[0]].y() + verts[tr[1]].y() + verts[tr[2]].y()) / 3.f,
                            (verts[tr[0]].z() + verts[tr[1]].z() + verts[tr[2]].z()) / 3.f);
                glm::vec3 p(cpuPos[k].x(), cpuPos[k].y(), cpuPos[k].z());
                float d = glm::length(p - c);
                sumD += d; ++nChk;
                if (d > 0.05f) ++nBad;
            }
            std::cout << "[fid check] ply-space    samples=" << nChk
                      << "  meanDist=" << (nChk ? sumD / nChk : -1.0)
                      << "  farOnes=" << nBad << std::endl;
        }

        // Per-stage counters: when the UV window comes up nearly empty this tells
        // you which filter ate the Gaussians, instead of having to guess.
        int nInSphere = 0, nByFid = 0, nByFallback = 0, nRejectedByFid = 0, nRejectedByVert = 0;
        int nValidTris = 0;
        int nFidNonNeg = 0, nFidInActive = 0;
        for (const auto& ti : triCache) if (ti.valid) ++nValidTris;

        // Where does activeSet actually sit relative to the click, and does it
        // overlap the triangles the Gaussians are anchored to?
        {
            double sumTd = 0.0; int nTd = 0; int minID = INT_MAX, maxID = -1;
            for (int triID : activeSet) {
                if (triID < 0 || triID >= (int)triangles.size()) continue;
                const auto& tr = triangles[triID];
                glm::vec3 c((verts[tr[0]].x() + verts[tr[1]].x() + verts[tr[2]].x()) / 3.f,
                            (verts[tr[0]].y() + verts[tr[1]].y() + verts[tr[2]].y()) / 3.f,
                            (verts[tr[0]].z() + verts[tr[1]].z() + verts[tr[2]].z()) / 3.f);
                sumTd += glm::length(c - ctr); ++nTd;
                minID = std::min(minID, triID); maxID = std::max(maxID, triID);
            }
            std::cout << "[activeSet] tris=" << nTd
                      << "  meanDistToHit=" << (nTd ? sumTd / nTd : -1.0)
                      << "  (radius=" << _expMapRadius << ")"
                      << "  triID range=[" << minID << "," << maxID << "]" << std::endl;
        }

        for (int k = 0; k < nGauss; ++k) {
            glm::vec3 p(cpuPos[k].x(), cpuPos[k].y(), cpuPos[k].z());
            if (glm::dot(p - ctr, p - ctr) > r2) continue;
            ++nInSphere;
            {
                int fdbg = (k < (int)_sortedFids.size()) ? _sortedFids[k] : -1;
                if (fdbg >= 0) { ++nFidNonNeg; if (triIdxToCache.count(fdbg)) ++nFidInActive; }
            }

            glm::vec2 bUV(0.f, 0.f);
            glm::vec3 bDU(0.f), bDV(0.f);
            glm::vec3 bSurfPos(0.f);
            float bUVMaxDelta = 0.0f;
            float surfDist = 1e9f;
            bool found = false;

            // Resolve which face ID to use for this Gaussian: every Gaussian has
            // its own fid via _allFids.
            int fidToUse = -1;
            if (canUseAllFids) {
                // cpuPos/cpuRot/cpuScale/cpuOpacity are all in GPU Morton-sorted order
                // (see GaussianView::getCpuPositions()), but _allFids was loaded straight
                // from the PLY in file order -- indexing it directly by k compared each
                // Gaussian's position against an unrelated Gaussian's face plane, which
                // is what produced the bogus "surface distance" values. _sortedFids
                // (built above from _allFids + getPlyToSorted()) is the same face-ID
                // data re-ordered into the GPU/sorted index space k actually lives in.
                fidToUse = (k < (int)_sortedFids.size()) ? _sortedFids[k] : -1;
            }

            // Fid-based projection: avoids inclined-triangle error for anchored Gaussians.
            if (fidToUse >= 0) {
                int fid = fidToUse;
                auto it = triIdxToCache.find(fid);
                if (it != triIdxToCache.end()) {
                    const TriInfo& ti = triCache[it->second];
                    if (ti.valid) {
                        float planeDist = glm::dot(p - ti.v0, ti.n);
                        glm::vec3 pv = (p - ti.n * planeDist) - ti.v0;
                        float den = ti.d00 * ti.d11 - ti.d01 * ti.d01;
                        // den == |e1 x e2|^2, i.e. (2*area)^2 -- it scales with the
                        // 4th power of edge length, so an absolute cutoff is useless:
                        // this mesh has 500k faces over a ~1 unit scene (edge ~0.004),
                        // giving den ~1.9e-10 and the old 1e-9 threshold discarded
                        // every single one of them.  Compare against d00*d11 instead,
                        // which is scale-free and only rejects genuinely degenerate
                        // (near-collinear) triangles.
                        if (std::abs(den) >= 1e-12f * ti.d00 * ti.d11) {
                            float d20 = glm::dot(pv, ti.e1);
                            float d21 = glm::dot(pv, ti.e2);
                            float bv = (ti.d11 * d20 - ti.d01 * d21) / den;
                            float bw = (ti.d00 * d21 - ti.d01 * d20) / den;
                            float bu = 1.f - bv - bw;
                            bUV      = bu * ti.uv0 + bv * ti.uv1 + bw * ti.uv2;
                            bSurfPos = bu * ti.v0   + bv * ti.v1   + bw * ti.v2;
                            bDU = ti.dU; bDV = ti.dV;
                            glm::vec2 uvMin = glm::min(glm::min(ti.uv0, ti.uv1), ti.uv2);
                            glm::vec2 uvMax = glm::max(glm::max(ti.uv0, ti.uv1), ti.uv2);
                            bUVMaxDelta = glm::length(uvMax - uvMin) * 0.5f;
                            surfDist = std::abs(planeDist);
                            found = true;
                            ++nByFid;
                        }
                    }
                }
            }

            // A Gaussian that carries a valid mesh_face_idx has already told us which
            // face it belongs to. If that face is not in the active set, the Gaussian
            // is not part of this selection -- full stop. Sending it to the
            // nearest-triangle fallback re-bound it to whichever selected face happened
            // to be closest, which is how Gaussians on the underside of the table top
            // ended up marked as part of a pick on the top surface.
            // Only Gaussians with no face of their own (fid < 0 -- the base layer,
            // 250,059 of them here, 10.9%) have nothing to go on and need the fallback.
            if (!found && fidToUse >= 0) {
                ++nRejectedByFid;
                continue;
            }

            // Base layer (fid < 0): identify the vertex it sits on, and keep it only if
            // that vertex belongs to a selected face. Then bind it to one of that
            // vertex's own faces -- never to some other face that merely happens to be
            // near, which is what the old unrestricted search did.
            if (!found) {
                int ownVert = -1;
                float bestVD = VEPS2;
                for (int dz = -1; dz <= 1 && ownVert < 0; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    auto git = vertGrid.find(vkey(p.x + dx * VCELL, p.y + dy * VCELL, p.z + dz * VCELL));
                    if (git == vertGrid.end()) continue;
                    for (int vi : git->second) {
                        const auto& V = verts[vi];
                        glm::vec3 d = p - glm::vec3(V.x(), V.y(), V.z());
                        float d2 = glm::dot(d, d);
                        if (d2 < bestVD) { bestVD = d2; ownVert = vi; }
                    }
                }
                if (ownVert < 0) { ++nRejectedByVert; continue; }

                float best = 1e18f;
                for (int ci : vertToCache[ownVert]) {
                    const TriInfo& ti = triCache[ci];
                    if (!ti.valid) continue;
                    float planeDist = glm::dot(p - ti.v0, ti.n);
                    glm::vec3 pv = (p - ti.n * planeDist) - ti.v0;
                    float den = ti.d00 * ti.d11 - ti.d01 * ti.d01;
                    // Scale-free degeneracy test -- see the note on the fid path above.
                    if (std::abs(den) < 1e-12f * ti.d00 * ti.d11) continue;
                    float d20 = glm::dot(pv, ti.e1);
                    float d21 = glm::dot(pv, ti.e2);
                    float bv = (ti.d11 * d20 - ti.d01 * d21) / den;
                    float bw = (ti.d00 * d21 - ti.d01 * d20) / den;
                    float bu = 1.f - bv - bw;
                    if (bu >= -0.001f && bv >= -0.001f && bw >= -0.001f) {
                        float d2 = planeDist * planeDist;
                        if (d2 < best) {
                            best = d2;
                            bUV      = bu * ti.uv0 + bv * ti.uv1 + bw * ti.uv2;
                            bSurfPos = bu * ti.v0   + bv * ti.v1   + bw * ti.v2;
                            bDU = ti.dU; bDV = ti.dV;
                            glm::vec2 uvMin = glm::min(glm::min(ti.uv0, ti.uv1), ti.uv2);
                            glm::vec2 uvMax = glm::max(glm::max(ti.uv0, ti.uv1), ti.uv2);
                            bUVMaxDelta = glm::length(uvMax - uvMin) * 0.5f;
                            found = true;
                        }
                    }
                }
                if (found) { surfDist = std::sqrt(best); ++nByFallback; }
            }

            if (!found) continue;

            all_uvs[k]       = { bUV.x, bUV.y };
            all_dUs[k]       = { bDU.x, bDU.y, bDU.z };
            all_dVs[k]       = { bDV.x, bDV.y, bDV.z };
            all_surfDists[k] = surfDist;
            all_surfPos[k]   = { bSurfPos.x, bSurfPos.y, bSurfPos.z };

            ProjectedGaussian pg;
            pg.originalIndex = k; pg.position = bSurfPos; pg.originalPos = p;
            pg.uv = bUV; pg.dU = bDU; pg.dV = bDV; pg.uvMaxDelta = bUVMaxDelta;
            float opc = std::max(1e-4f, std::min(1.f - 1e-4f, cpuOpacity[k]));
            pg.opacity = std::log(opc / (1.f - opc));
            pg.scale = { std::log(std::max(1e-9f, cpuScale[3*k])),
                         std::log(std::max(1e-9f, cpuScale[3*k+1])),
                         std::log(std::max(1e-9f, cpuScale[3*k+2])) };
            pg.rotation = { cpuRot[4*k], cpuRot[4*k+1], cpuRot[4*k+2], cpuRot[4*k+3] };
            _texGaussians.push_back(pg);
        }

        std::cout << "[UV pick] radius=" << _expMapRadius
                  << "\n  triCache=" << triCache.size() << " (valid " << nValidTris << ")"
                  << "\n  in sphere=" << nInSphere
                  << "  fid>=0=" << nFidNonNeg
                  << "  fid in activeSet=" << nFidInActive
                  << "\n  by fid=" << nByFid
                  << "  by fallback=" << nByFallback
                  << "  rejected(own fid outside selection)=" << nRejectedByFid
                  << "  rejected(base layer outside selection)=" << nRejectedByVert
                  << "\n  accepted=" << _texGaussians.size() << std::endl;

        // A pick is only a *preview* now -- it is staged here and does not touch the
        // committed state until "Confirm Patch" is pressed. A mis-click therefore
        // costs nothing: the next pick simply replaces the staged patch, reusing its
        // texture slot so repeated attempts do not leak one slot each.
        if ((int)_lastAllUVs.size() != nGauss) {
            _lastAllUVs.assign(nGauss, sibr::Vector2f(-1.f, -1.f));
            _lastAllDUs.assign(nGauss, sibr::Vector3f(0.f, 0.f, 0.f));
            _lastAllDVs.assign(nGauss, sibr::Vector3f(0.f, 0.f, 0.f));
            _lastAllSurfDists.assign(nGauss, 1e9f);
            _lastAllTexIdx.assign(nGauss, -1);
        }
        if ((int)_lastAllTexIdx.size() != nGauss) _lastAllTexIdx.assign(nGauss, -1);

        // Register the triangle-masked texture (alpha 0 outside the patch), not the
        // raw rectangular image: renderCUDA gates on that alpha to cut the decal at
        // the patch boundary instead of letting edge Gaussians splat a fuzzy halo.
        sibr::Texture2DRGBA::Ptr _colorTex = _expMapSolver.GetCudaTexture();
        if (!_colorTex) _colorTex = _texPtr;
        const int _newSlot = _colorTex
            ? _gaussianView->registerTexture(_colorTex, _pendingSlot)
            : -1;
        _pendingSlot = _newSlot;
        _pendUVs = all_uvs; _pendDUs = all_dUs; _pendDVs = all_dVs;
        _pendSurfDists = all_surfDists;
        _hasPending = (_newSlot >= 0);

        // What gets uploaded is committed + staged, so the preview looks exactly like
        // the result would, without the staged patch being part of the record yet.
        all_uvs = _lastAllUVs; all_dUs = _lastAllDUs; all_dVs = _lastAllDVs;
        all_surfDists = _lastAllSurfDists;
        _previewTexIdx = _lastAllTexIdx;
        for (int i = 0; i < nGauss; ++i) {
            if (_pendUVs[i].x() < 0.f) continue;     // not in this pick
            all_uvs[i]        = _pendUVs[i];
            all_dUs[i]        = _pendDUs[i];
            all_dVs[i]        = _pendDVs[i];
            all_surfDists[i]  = _pendSurfDists[i];
            _previewTexIdx[i] = _newSlot;
        }
        _lastAllSurfDists = all_surfDists; _lastAllSurfPos = all_surfPos;
        // Save original positions (cpuPos is already restored to pre-snap state above)
        _lastAllOrigPos.resize(nGauss);
        for (int i = 0; i < nGauss; ++i) _lastAllOrigPos[i] = cpuPos[i];
        _expMapSolver.SetMainGaussiansForNextSave(_texGaussians);
        // Supply the projected Gaussians directly to the UV result window.
        if (canUseAllFids) _expMapSolver.SetProjectedGaussians(_texGaussians);
        // Reset blend to 0: Gaussians start at their original positions
        _expMapSolver.ResetSurfaceBlend();
        // A fresh pick starts unrotated; the "Texture Rotation" slider drives it from here.
        _texRotationDeg = 0.f;
        // Bake the patch's flat normal map into UV (ExpMap) space, same slot as the
        // colour texture so each patch shades against its own normals.
        bakePatchNormalMap(0.f);

        // Upload UV data + texture to 3D view (requires a loaded texture for CUDA rendering).
        if (_texPtr)
            _gaussianView->setUVsAndTexture(all_uvs, all_dUs, all_dVs, all_surfDists, _lastAllOrigPos, _previewTexIdx);
        // record hit position for toggle reuse
        _lastHitPos    = sibr::Vector3f(hitPos.x(), hitPos.y(), hitPos.z());
        _lastExpRadius = _expMapRadius;
        // suppress non-UV gaussians inside texture region to prevent occlusion
        // _gaussianView->suppressGaussiansInRegion(
        //     all_uvs,
        //     _lastHitPos,
        //     _expMapRadius,
        //     all_surfDists
        // );
    }

    GaussianView::Ptr               _gaussianView;
    MeshWireframeRenderer           _wireframeRenderer;
    ExpMapSolverSIBR                _expMapSolver;
    const sibr::Mesh* _mesh;
    sibr::InteractiveCameraHandler::Ptr _camHandler;
    sibr::Viewport                  _viewport;
    sibr::Vector3f _meshColor;
    bool   _showMesh = false;
    float  _expMapRadius = 0.2f;
    // Rigid rotation of the staged patch's texture about the patch centre
    // (UV 0.5,0.5, where Compute() normalizes the chart's bounding box), in
    // degrees. Applied as a post-transform on the staged UV / dU / dV before
    // upload -- equivalent to spinning the ExpMap seed frame about its normal, the
    // way the reference ExpMapDemo does it, but without recomputing the map.
    // Reset to 0 on every new pick / confirm.
    float  _texRotationDeg = 0.f;
    bool   _showGaussianOutlines = false;
    bool   _showTexture = true;
    GLuint _gaussOutlineVAO = 0, _gaussOutlineVBO = 0, _gaussOutlineShader = 0;
    std::vector<ProjectedGaussian>  _texGaussians;
    GLuint _liveSSBO = 0;
    sibr::Texture2DRGBA::Ptr        _texPtr;
    // Per-Gaussian texture index, kept alongside _lastAllUVs. Persisting it across
    // picks is what lets several painted patches coexist: a new pick only overwrites
    // the Gaussians it actually covers, everything else keeps the patch it already had.
    std::vector<int>                _lastAllTexIdx;
    // Staged (previewed but not yet confirmed) patch. _previewTexIdx is what the
    // renderer currently shows: committed patches plus the staged one.
    bool                            _hasPending   = false;
    int                             _pendingSlot  = -1;
    std::vector<sibr::Vector2f>     _pendUVs;
    std::vector<sibr::Vector3f>     _pendDUs, _pendDVs;
    std::vector<float>              _pendSurfDists;
    std::vector<int>                _previewTexIdx;
    std::vector<sibr::Vector2f>     _lastAllUVs;
    std::vector<sibr::Vector3f>     _lastAllDUs, _lastAllDVs;
    std::vector<sibr::Vector3f>     _lastAllSurfPos;
    std::vector<sibr::Vector3f>     _lastAllOrigPos;
    std::vector<float>              _lastAllSurfDists;
    std::vector<int>                _allFids;          // per-Gaussian face IDs (new unified format)
    std::vector<int>                _sortedFids;       // face IDs in GPU Morton-sorted order
    // Per-face world normals, kept for the ExpMap patch normal-map bake. (These also used
    // to drive a per-Gaussian backface cull, removed 2026-08-04: that cull decided
    // visibility from the face normal alone, which flips discretely at silhouettes.)
    std::vector<sibr::Vector3f>     _faceNormals;
    std::vector<uint8_t>            _patchNormalMap;   // RGBA uint8 normal map for current patch
    int                             _normalMapW = 0, _normalMapH = 0;
    GLuint                          _normalMapGLTex = 0;
    sibr::Vector3f                  _lastHitPos   = sibr::Vector3f(0,0,0);
    float                           _lastExpRadius = 0.5f;

    // 360° orbit recording
    bool             _recording360      = false;
    int              _recordFrame       = 0;
    int              _recordTotalFrames = 120;
    float            _orbitZoom         = 2.5f;
    float            _orbitElevation    = 0.0f;
    glm::vec3        _orbitCenter       = {0.f, 0.f, 0.f};
    float            _orbitRadius       = 1.0f;
    std::string      _videoDir;
    cv::VideoWriter  _videoWriter;
};

// =============================================================================
// Utilities
// =============================================================================
// Load mesh_vertex / mesh_face elements from an SG-format PLY file.
// Returns true if a valid mesh was found and loaded into `mesh`.
static bool loadMeshFromSgPly(const std::string& plyPath, sibr::Mesh& mesh)
{
    std::ifstream f(plyPath, std::ios::binary);
    if (!f.good()) return false;

    auto propBytes = [](const std::string& t) -> size_t {
        if (t == "float" || t == "int" || t == "uint") return 4;
        if (t == "double" || t == "int64" || t == "uint64") return 8;
        if (t == "short"  || t == "ushort") return 2;
        return 1;
    };

    struct ElemInfo { std::string name; int count = 0; size_t stride = 0; bool hasList = false; };
    std::vector<ElemInfo> elems;
    ElemInfo* cur = nullptr;

    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line == "end_header") break;
        std::stringstream ss(line);
        std::string tok; ss >> tok;
        if (tok == "element") {
            std::string name; int cnt; ss >> name >> cnt;
            elems.push_back({name, cnt, 0, false});
            cur = &elems.back();
        } else if (tok == "property" && cur) {
            std::string type; ss >> type;
            if (type == "list") cur->hasList = true;
            else cur->stride += propBytes(type);
        }
    }

    ElemInfo* gaussEl = nullptr, *mvEl = nullptr, *mfEl = nullptr;
    for (auto& e : elems) {
        if (e.name == "vertex")      gaussEl = &e;
        if (e.name == "mesh_vertex") mvEl    = &e;
        if (e.name == "mesh_face")   mfEl    = &e;
    }
    if (!mvEl || !mfEl) return false;

    // Skip Gaussian data
    if (gaussEl)
        f.seekg((std::streamoff)gaussEl->count * (std::streamoff)gaussEl->stride, std::ios::cur);

    // Read mesh vertices
    std::vector<sibr::Vector3f> verts(mvEl->count);
    f.read((char*)verts.data(), mvEl->count * 3 * sizeof(float));

    // Read mesh faces (format: uchar count, then count × int indices)
    std::vector<sibr::Vector3u> tris;
    tris.reserve(mfEl->count);
    for (int i = 0; i < mfEl->count; i++) {
        uint8_t n = 0;
        f.read((char*)&n, 1);
        int32_t idx[4] = {0, 0, 0, 0};
        f.read((char*)idx, n * sizeof(int32_t));
        if (n >= 3)
            tris.push_back(sibr::Vector3u((uint)idx[0], (uint)idx[1], (uint)idx[2]));
    }

    if (tris.empty()) return false;
    mesh.vertices(verts);
    mesh.triangles(tris);
    std::cout << "Loaded SG mesh from PLY: " << verts.size() << " verts, "
              << tris.size() << " faces\n";
    return true;
}

// =============================================================================
static std::string findLargestNumberedSubdirectory(const std::string& dirPath) {
    std_fs::path p(dirPath);
    if (!std_fs::exists(p) || !std_fs::is_directory(p)) return "";
    std::regex rx(R"_(iteration_(\d+))_");
    std::string best;
    int bestN = -1;
    for (const auto& entry : std_fs::directory_iterator(p)) {
        if (!std_fs::is_directory(entry)) continue;
        std::string name = entry.path().filename().string();
        std::smatch m;
        if (std::regex_match(name, m, rx)) {
            int n = std::stoi(m[1]);
            if (n > bestN) {
                bestN = n;
                best = name;
            }
        }
    }
    return best;
}

static std::pair<int,int> findArg(const std::string& line, const std::string& name) {
    size_t s = line.find(name, 0);
    s = line.find("=", s) + 1;
    size_t e = line.find_first_of(",)", s);
    return {(int)s, (int)e};
}

static void* User_ReadOpen(ImGuiContext*, ImGuiSettingsHandler*, const char*) {
    return (void*)0x1;
}

static void User_ReadLine(ImGuiContext*, ImGuiSettingsHandler* h, void*, const char* line) {
    int i;
    if (sscanf_s(line, "DontShow=%d", &i) == 1)
        *((bool*)h->UserData) = (i != 0);
}

static void User_WriteAll(ImGuiContext*, ImGuiSettingsHandler* h, ImGuiTextBuffer* buf) {
    buf->reserve(buf->size() + 96);
    buf->appendf("[UserData][UserData]\nDontShow=%d\n\n", *((bool*)h->UserData) ? 1 : 0);
}

// =============================================================================
// main
// =============================================================================
int main(int ac, char** av) {
    CommandLineArgs::parseMainArgs(ac, av);
    GaussianAppArgs myArgs;
    myArgs.displayHelpIfRequired();

    if (!myArgs.modelPath.isInit() && myArgs.modelPathShort.isInit())
        myArgs.modelPath = myArgs.modelPathShort.get();
    if (!myArgs.dataset_path.isInit() && myArgs.pathShort.isInit())
        myArgs.dataset_path = myArgs.pathShort.get();

    sibr::Window window("sibr_3Dgaussian", sibr::Vector2i(50, 50), myArgs);

    bool messageRead = false;
    ImGuiSettingsHandler ini_handler;
    ini_handler.TypeName   = "UserData";
    ini_handler.UserData   = &messageRead;
    ini_handler.TypeHash   = ImHash("UserData", 0);
    ini_handler.ReadOpenFn = User_ReadOpen;
    ini_handler.ReadLineFn = User_ReadLine;
    ini_handler.WriteAllFn = User_WriteAll;
    ImGui::GetCurrentContext()->SettingsHandlers.push_back(ini_handler);
    window.loadSettings();

    std::string cfgLine;
    std::ifstream cfgFile(myArgs.modelPath.get() + "/cfg_args");
    if (!cfgFile.good())
        SIBR_ERR << "Could not find cfg_args at: " << myArgs.modelPath.get();
    std::getline(cfgFile, cfgLine);

    if (!myArgs.dataset_path.isInit()) {
        auto rng = findArg(cfgLine, "source_path");
        myArgs.dataset_path = cfgLine.substr(rng.first + 1, rng.second - rng.first - 2);
    }

    auto rng = findArg(cfgLine, "sh_degree");
    int sh_degree = std::stoi(cfgLine.substr(rng.first, rng.second - rng.first));

    rng = findArg(cfgLine, "white_background");
    bool white_background =
        cfgLine.substr(rng.first, rng.second - rng.first).find("True") != std::string::npos;

    BasicIBRScene::SceneOptions myOpts;
    myOpts.renderTargets = myArgs.loadImages;
    myOpts.mesh    = true;
    myOpts.images  = myArgs.loadImages;
    myOpts.cameras = true;
    myOpts.texture = false;

    BasicIBRScene::Ptr scene;
    try {
        scene.reset(new BasicIBRScene(myArgs, myOpts));
    } catch (...) {
        myArgs.dataset_path = myArgs.modelPath.get();
        scene.reset(new BasicIBRScene(myArgs, myOpts));
    }

    std::string plyBase = myArgs.modelPath.get();
    if (plyBase.back() != '/' && plyBase.back() != '\\')
        plyBase += "/";
    plyBase += "point_cloud";

    std::string iterDir;
    if (!myArgs.iteration.isInit()) {
        iterDir = findLargestNumberedSubdirectory(plyBase);
        std::cout << "Auto-detected iteration: " << iterDir << "\n";
    } else {
        iterDir = "iteration_" + myArgs.iteration.get();
    }

    const std::string plyDir       = plyBase + "/" + iterDir + "/";
    const std::string finalPlyPath = plyDir + "point_cloud.ply";

    // New unified format: splat.ply may live directly in the model root.
    // Use it as a fallback when the standard point_cloud.ply is absent.
    std::string modelRoot = myArgs.modelPath.get();
    if (modelRoot.back() != '/' && modelRoot.back() != '\\') modelRoot += "/";
    const std::string rootSplatPath = modelRoot + "splat.ply";
    const bool useRootSplat = !std_fs::exists(finalPlyPath) && std_fs::exists(rootSplatPath);
    const std::string effectivePlyPath = useRootSplat ? rootSplatPath : finalPlyPath;
    if (useRootSplat)
        std::cout << "Using root splat.ply: " << rootSplatPath << "\n";

    sibr::Mesh::Ptr geoMesh(new sibr::Mesh());
    const sibr::Mesh* meshToRender = nullptr;
    if (loadMeshFromSgPly(effectivePlyPath, *geoMesh)) {
        meshToRender = geoMesh.get();
    } else if (!scene->proxies()->proxy().vertices().empty()) {
        meshToRender = &scene->proxies()->proxy();
        std::cout << "Fallback: proxy mesh\n";
    }

    uint scene_w = scene->cameras()->inputCameras()[0]->w();
    uint scene_h = scene->cameras()->inputCameras()[0]->h();
    float aspect = scene_w * 1.f / scene_h;

    uint rw = myArgs.rendering_size.get()[0];
    uint rh = myArgs.rendering_size.get()[1];
    rw = (rw <= 0) ? std::min(1200U, scene_w) : rw;
    rh = (rh <= 0) ? (uint)(std::min(1200U, scene_w) / aspect) : rh;
    Vector2u usedRes(rw, rh);

    GaussianView::Ptr gaussianView(new GaussianView(
        scene,
        usedRes.x(),
        usedRes.y(),
        effectivePlyPath.c_str(),
        &messageRead,
        sh_degree,
        white_background,
        !myArgs.noInterop,
        myArgs.device));

    sibr::InteractiveCameraHandler::Ptr generalCamera(new InteractiveCameraHandler());
    generalCamera->setup(
        scene->cameras()->inputCameras(),
        Viewport(0, 0, (float)usedRes.x(), (float)usedRes.y()),
        nullptr);

    // Unified splat format: load per-Gaussian mesh_face_idx from the PLY.
    std::string splatPath = "";
    if (std_fs::exists(effectivePlyPath)) {
        splatPath = effectivePlyPath;
        std::cout << "Unified splat format: loading mesh_face_idx from " << splatPath << "\n";
    }

    MeshGaussianView::Ptr meshView(new MeshGaussianView(
        gaussianView, meshToRender, generalCamera, splatPath));

    MultiViewManager multiViewManager(window, false);
    if (myArgs.rendering_mode == 1)
        multiViewManager.renderingMode(IRenderingMode::Ptr(new StereoAnaglyphRdrMode()));

    multiViewManager.addIBRSubView(
        "Point view",
        meshView,
        usedRes,
        ImGuiWindowFlags_ResizeFromAnySide | ImGuiWindowFlags_NoBringToFrontOnFocus);
    multiViewManager.addCameraForView("Point view", generalCamera);

    const auto topView = std::make_shared<sibr::SceneDebugView>(
        scene, generalCamera, myArgs, myArgs.imagesPath.get());
    multiViewManager.addSubView("Top view", topView, usedRes);
    topView->active(false);

    generalCamera->getCameraRecorder().setViewPath(gaussianView, myArgs.dataset_path.get());
    if (myArgs.pathFile.get() != "") {
        generalCamera->getCameraRecorder().loadPath(myArgs.pathFile.get(), usedRes.x(), usedRes.y());
        generalCamera->getCameraRecorder().recordOfflinePath(
            myArgs.outPath,
            multiViewManager.getIBRSubView("Point view"),
            "");
        if (!myArgs.noExit)
            exit(0);
    }

    while (window.isOpened()) {
        sibr::Input::poll();
        window.makeContextCurrent();
        if (sibr::Input::global().key().isPressed(sibr::Key::Escape))
            window.close();
        multiViewManager.onUpdate(sibr::Input::global());
        multiViewManager.onRender(window);
        window.swapBuffer();
    }

    return EXIT_SUCCESS;
}