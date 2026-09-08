#pragma once

// Standard library
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstdint>

// SIBR / Third-party
#include <core/graphics/Image.hpp>
#include <core/graphics/Texture.hpp>
#include <core/graphics/Mesh.hpp>
#include <boost/filesystem.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GL/glew.h>
#include <imgui/imgui.h>

namespace fs = boost::filesystem;

// ============================================================================
// Physical attributes of each Gaussian
// ============================================================================
struct GaussianProps {
    float opacity;
    float scale[3];
    float rot[4];
};

// ============================================================================
// A Gaussian projected into UV space
// ============================================================================
struct ProjectedGaussian {
    int       originalIndex;
    glm::vec2 uv;
    glm::vec3 position;
    glm::vec3 originalPos;
    float     opacity;
    glm::vec3 scale;
    glm::vec4 rotation;
    glm::vec3 dU;
    glm::vec3 dV;
    float     uvMaxDelta = 0.0f; // half-diagonal of triangle UV bbox, used to clamp extrapolation at seams
};

namespace detail {
inline const std::string& meshVertSrc() {
    static const std::string s = R"(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aNormal;
        uniform mat4 uMVP;
        out vec3 vNormal;
        void main() {
            gl_Position = uMVP * vec4(aPos, 1.0);
            vNormal = aNormal;
        }
    )";
    return s;
}

inline const std::string& meshFragSrc() {
    static const std::string s = R"(
        #version 330 core
        in vec3 vNormal;
        out vec4 FragColor;
        uniform vec3  uColor;
        uniform bool  uUseLighting;
        void main() {
            if (uUseLighting) {
                vec3 lightDir = normalize(vec3(0.5, 0.5, 1.0));
                float diff = max(dot(normalize(vNormal), lightDir), 0.3);
                FragColor = vec4(uColor * diff, 1.0);
            } else {
                FragColor = vec4(uColor, 1.0);
            }
        }
    )";
    return s;
}

inline GLuint compileGLShader(const std::string& src, GLenum type) {
    if (src.empty()) return 0;
    GLuint sh = glCreateShader(type);
    const char* s = src.c_str();
    glShaderSource(sh, 1, &s, nullptr);
    glCompileShader(sh);
    GLint ok; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLchar log[512]; glGetShaderInfoLog(sh, 512, nullptr, log);
        std::cerr << "[GL] Shader compile error: " << log << "\n";
    }
    return sh;
}

inline GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}
}

class TextureLoader {
public:
    TextureLoader() = default;

    bool LoadImage(const std::string& path) {
        if (!_originalImage.load(path)) return false;
        _path = path;
        // The GL upload -> ImGui/CUDA sampling chain shows the image left-right
        // mirrored versus the source file. Undo it once here so both the
        // UV-result preview and the decal baked onto the model match the file.
        cv::Mat corrected;
        cv::flip(_originalImage.toOpenCV(), corrected, 0);
        sibr::ImageRGBA img;
        img.fromOpenCV(corrected);
        _texture.reset(new sibr::Texture2DRGBA(img, SIBR_GPU_LINEAR_SAMPLING));
        return true;
    }

    // rotDeg spins the coverage mask about the UV centre (0.5,0.5), matching the
    // rotation renderCUDA applies to the sampled UV (see applyTexRotation /
    // bakePatchNormalMap). The colour content is left axis-aligned -- only the
    // mask must follow the rotated lookup, or a non-round patch's edge drifts.
    sibr::Texture2DRGBA::Ptr generateCudaTexture(const std::vector<sibr::Vector3u>& tris, const std::map<int, glm::vec2>& uvs, float rotDeg = 0.f) {
        if (_originalImage.w() == 0 || _originalImage.h() == 0) return nullptr;
        int W = _originalImage.w();
        int H = _originalImage.h();

        const float rotRad = rotDeg * 0.01745329252f;
        const float rc = std::cos(rotRad), rs = std::sin(rotRad);
        auto rotUV = [rc, rs](glm::vec2 p) -> glm::vec2 {
            const float x = p.x - 0.5f, y = p.y - 0.5f;
            return { rc * x - rs * y + 0.5f, rs * x + rc * y + 0.5f };
        };

        sibr::ImageRGBA maskedImage(W, H);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                maskedImage(x, y) = _originalImage(x, y);
            }
        }

        if (!tris.empty() && !uvs.empty()) {
            std::vector<uint8_t> mask(W * H, 0);

            for (const auto& t : tris) {
                if (!uvs.count(t[0]) || !uvs.count(t[1]) || !uvs.count(t[2])) continue;
                glm::vec2 p0 = rotUV(uvs.at(t[0]));
                glm::vec2 p1 = rotUV(uvs.at(t[1]));
                glm::vec2 p2 = rotUV(uvs.at(t[2]));

                p0.x *= (W - 1); p0.y *= (H - 1);
                p1.x *= (W - 1); p1.y *= (H - 1);
                p2.x *= (W - 1); p2.y *= (H - 1);

                int minX = std::max(0, (int)std::floor(std::min({p0.x, p1.x, p2.x})));
                int maxX = std::min(W - 1, (int)std::ceil(std::max({p0.x, p1.x, p2.x})));
                int minY = std::max(0, (int)std::floor(std::min({p0.y, p1.y, p2.y})));
                int maxY = std::min(H - 1, (int)std::ceil(std::max({p0.y, p1.y, p2.y})));

                auto edge = [](const glm::vec2& a, const glm::vec2& b, const glm::vec2& c) {
                    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
                };

                if (edge(p0, p1, p2) < 0) std::swap(p1, p2);

                for (int y = minY; y <= maxY; ++y) {
                    for (int x = minX; x <= maxX; ++x) {
                        glm::vec2 p(x + 0.5f, y + 0.5f);
                        float w0 = edge(p1, p2, p);
                        float w1 = edge(p2, p0, p);
                        float w2 = edge(p0, p1, p);
                        if (w0 >= -1.0f && w1 >= -1.0f && w2 >= -1.0f) {
                            mask[y * W + x] = 1;
                        }
                    }
                }
            }

            // Texture dilation: spread valid triangle colors into neighboring pixels
            // to prevent bilinear sampling at triangle edges from mixing in wrong colors.
            // Dilated pixels copy RGB from a valid neighbor but keep Alpha = 0.
            const int DILATION_PASSES = 4;
            for (int pass = 0; pass < DILATION_PASSES; ++pass) {
                std::vector<uint8_t> prevMask = mask; // snapshot: source is only the previous pass
                const int dx4[] = {-1, 1,  0, 0};
                const int dy4[] = { 0, 0, -1, 1};
                for (int y = 0; y < H; ++y) {
                    for (int x = 0; x < W; ++x) {
                        if (prevMask[y * W + x] != 0) continue; // already valid, skip
                        for (int d = 0; d < 4; ++d) {
                            int nx = x + dx4[d], ny = y + dy4[d];
                            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                            if (prevMask[ny * W + nx] != 0) {
                                // copy RGB from valid neighbor, leave Alpha unchanged (stays 0)
                                maskedImage(x, y).x() = maskedImage(nx, ny).x();
                                maskedImage(x, y).y() = maskedImage(nx, ny).y();
                                maskedImage(x, y).z() = maskedImage(nx, ny).z();
                                mask[y * W + x] = 2; // mark as "dilated but not inside triangle"
                                break;
                            }
                        }
                    }
                }
            }

            // Set Alpha: inside triangle = opaque, everything else = transparent
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    if (mask[y * W + x] == 0) {
                        // fully empty: zero RGB too to avoid stray color at edges
                        maskedImage(x, y).x() = 0;
                        maskedImage(x, y).y() = 0;
                        maskedImage(x, y).z() = 0;
                        maskedImage(x, y).w() = 0;
                    } else if (mask[y * W + x] == 2) {
                        // dilated pixel: RGB filled, Alpha stays 0
                        maskedImage(x, y).w() = 0;
                    }
                    // mask == 1: original pixel inside triangle, leave unchanged
                }
            }
        }
        return std::make_shared<sibr::Texture2DRGBA>(maskedImage, SIBR_GPU_LINEAR_SAMPLING);
    }

    std::vector<std::string> scanForImages(const std::string& directory) const {
        std::vector<std::string> result;
        fs::path dirPath(directory);
        if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) return result;

        static const std::vector<std::string> kExts = {".png", ".jpg", ".jpeg", ".bmp", ".tga"};
        for (const auto& entry : fs::directory_iterator(dirPath)) {
            if (!fs::is_regular_file(entry.status())) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(kExts.begin(), kExts.end(), ext) != kExts.end())
                result.push_back(entry.path().string());
        }
        return result;
    }

    const sibr::Texture2DRGBA::Ptr& getTexture() const { return _texture; }
    const std::string&              getPath()    const { return _path; }

    // Drop the loaded image. Used after a patch is committed so the UV-result
    // window starts empty for the next pick instead of still showing the image
    // that was just applied.
    void Clear() {
        _texture.reset();
        _originalImage = sibr::ImageRGBA();
        _path.clear();
    }

private:
    sibr::ImageRGBA          _originalImage;
    sibr::Texture2DRGBA::Ptr _texture;
    std::string              _path;
};

class SimplePointRenderer {
public:
    void load(const std::string& path) {
        sibr::Mesh::Ptr mesh(new sibr::Mesh());
        if (!mesh->load(path)) return;
        _count = mesh->vertices().size();
        if (_count == 0) return;

        _rawData.clear();
        _rawData.reserve(_count * 3);
        for (const auto& v : mesh->vertices()) {
            _rawData.push_back(v.x());
            _rawData.push_back(v.y());
            _rawData.push_back(v.z());
        }

        glGenVertexArrays(1, &_vao);
        glGenBuffers(1, &_vbo);
        glBindVertexArray(_vao);
        glBindBuffer(GL_ARRAY_BUFFER, _vbo);
        glBufferData(GL_ARRAY_BUFFER, _rawData.size() * sizeof(float),
                     _rawData.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
        _loaded = true;
    }

    void loadFids(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) return;

        std::string line;
        long vertex_count   = 0;
        int  face_id_byte_offset = -1;
        bool face_id_is_int = false;
        int  byte_offset    = 0;
        bool is_binary      = false;

        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::stringstream ss(line);
            std::string token; ss >> token;

            if (token == "format") {
                ss >> token;
                if (token == "binary_little_endian") is_binary = true;
            } else if (token == "element") {
                ss >> token;
                if (token == "vertex") ss >> vertex_count;
            } else if (token == "property") {
                std::string type, name; ss >> type >> name;
                int sz = 0;
                if      (type == "float"  || type == "int"   || type == "uint")  sz = 4;
                else if (type == "double" || type == "int64" || type == "uint64") sz = 8;
                else if (type == "short"  || type == "ushort") sz = 2;
                else if (type == "char"   || type == "uchar")  sz = 1;
                else sz = 4; // fallback assumption

                if (name == "face_id") {
                    face_id_byte_offset = byte_offset;
                    face_id_is_int = (type == "int" || type == "uint");
                }
                byte_offset += sz;
            } else if (token == "end_header") {
                break;
            }
        }

        if (!is_binary || vertex_count == 0 || face_id_byte_offset == -1) return;

        _fids.clear();
        _fids.reserve(vertex_count);
        const int stride = byte_offset;
        std::vector<char> buf(stride);
        for (long i = 0; i < vertex_count; ++i) {
            file.read(buf.data(), stride);
            if (!file) return;
            int fid;
            if (face_id_is_int) {
                int32_t v; std::memcpy(&v, buf.data() + face_id_byte_offset, sizeof(int32_t));
                fid = v;
            } else {
                float v; std::memcpy(&v, buf.data() + face_id_byte_offset, sizeof(float));
                fid = static_cast<int>(v);
            }
            _fids.push_back(fid);
        }
    }

    void draw() const {
        if (!_loaded) return;
        glBindVertexArray(_vao);
        glDrawArrays(GL_POINTS, 0, (GLsizei)_count);
        glBindVertexArray(0);
    }

    bool                      isLoaded()   const { return _loaded; }
    const std::vector<float>& getRawData() const { return _rawData; }
    const std::vector<int>&   getFids()    const { return _fids; }

    void loadGaussianProps(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) return;

        std::string line;
        long vertex_count = 0;
        bool is_binary    = false;
        std::map<std::string, int> propOffset;
        int prop_count = 0;

        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::stringstream ss(line);
            std::string token; ss >> token;
            if (token == "format") {
                ss >> token;
                if (token == "binary_little_endian") is_binary = true;
            } else if (token == "element") {
                ss >> token;
                if (token == "vertex") ss >> vertex_count;
            } else if (token == "property") {
                std::string type, name; ss >> type >> name;
                if (type == "float") { propOffset[name] = prop_count++; }
            } else if (token == "end_header") {
                break;
            }
        }
        if (!is_binary || vertex_count == 0) return;

        static const std::vector<std::string> kRequired =
            {"opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"};
        for (const auto& r : kRequired)
            if (!propOffset.count(r)) { std::cerr << "[loadGaussianProps] missing: " << r << "\n"; return; }

        _gaussianProps.clear();
        _gaussianProps.reserve(vertex_count);
        const size_t stride = prop_count * sizeof(float);
        std::vector<float> buf(prop_count);

        for (long i = 0; i < vertex_count; ++i) {
            file.read(reinterpret_cast<char*>(buf.data()), stride);
            if (!file) { _gaussianProps.clear(); return; }
            GaussianProps p;
            p.opacity  = buf[propOffset.at("opacity")];
            p.scale[0] = buf[propOffset.at("scale_0")];
            p.scale[1] = buf[propOffset.at("scale_1")];
            p.scale[2] = buf[propOffset.at("scale_2")];
            p.rot[0]   = buf[propOffset.at("rot_0")];
            p.rot[1]   = buf[propOffset.at("rot_1")];
            p.rot[2]   = buf[propOffset.at("rot_2")];
            p.rot[3]   = buf[propOffset.at("rot_3")];
            _gaussianProps.push_back(p);
        }
    }

    const std::vector<GaussianProps>& getGaussianProps() const { return _gaussianProps; }

private:
    GLuint _vao    = 0;
    GLuint _vbo    = 0;
    size_t _count  = 0;
    bool   _loaded = false;
    std::vector<float>        _rawData;
    std::vector<int>          _fids;
    std::vector<GaussianProps> _gaussianProps;
};

class PointCloudRenderer {
public:
    PointCloudRenderer()  { init(); }
    ~PointCloudRenderer() { if (_progID) glDeleteProgram(_progID); }

    void render(
        const glm::mat4&           mvp,
        float                      pointSize,
        const SimplePointRenderer& geo,      const sibr::Vector3f& geoColor, bool showGeo,
        const SimplePointRenderer& app,      const sibr::Vector3f& appColor, bool showApp
    ) {
        glUseProgram(_progID);
        glUniformMatrix4fv(_locMVP, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform1i(_locUseLighting, 0);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_POLYGON_OFFSET_POINT);
        glPolygonOffset(0.0f, -10.0f);
        glPointSize(pointSize);

        if (showGeo && geo.isLoaded()) {
            glUniform3f(_locColor, geoColor.x(), geoColor.y(), geoColor.z());
            geo.draw();
        }
        if (showApp && app.isLoaded()) {
            glUniform3f(_locColor, appColor.x(), appColor.y(), appColor.z());
            app.draw();
        }

        glDisable(GL_POLYGON_OFFSET_POINT);
        glUseProgram(0);
    }

private:
    void init() {
        GLuint vs = detail::compileGLShader(detail::meshVertSrc(), GL_VERTEX_SHADER);
        GLuint fs = detail::compileGLShader(detail::meshFragSrc(), GL_FRAGMENT_SHADER);
        _progID = detail::linkProgram(vs, fs);
        _locMVP         = glGetUniformLocation(_progID, "uMVP");
        _locColor       = glGetUniformLocation(_progID, "uColor");
        _locUseLighting = glGetUniformLocation(_progID, "uUseLighting");
    }

    GLuint _progID         = 0;
    GLint  _locMVP         = -1;
    GLint  _locColor       = -1;
    GLint  _locUseLighting = -1;
};

class MeshWireframeRenderer {
public:
    MeshWireframeRenderer()  { init(); }
    ~MeshWireframeRenderer() {
        if (_progID) glDeleteProgram(_progID);
        if (_staticVao) {
            glDeleteBuffers(1, &_staticVbo);
            glDeleteVertexArrays(1, &_staticVao);
        }
        if (_activeVao) {
            glDeleteBuffers(1, &_activeVbo);
            glDeleteVertexArrays(1, &_activeVao);
        }
    }

    void uploadMesh(const sibr::Mesh* mesh) {
        if (!mesh || mesh->vertices().empty()) return;

        const auto& verts   = mesh->vertices();
        const auto& normals = mesh->normals();
        const auto& tris    = mesh->triangles();

        std::vector<float> lines;
        lines.reserve(tris.size() * 6 * 6);
        for (const auto& tri : tris) {
            pushEdges(tri, verts, normals, lines);
        }

        if (_staticVao) {
            glDeleteBuffers(1, &_staticVbo);
            glDeleteVertexArrays(1, &_staticVao);
        }

        glGenVertexArrays(1, &_staticVao);
        glGenBuffers(1, &_staticVbo);
        glBindVertexArray(_staticVao);
        glBindBuffer(GL_ARRAY_BUFFER, _staticVbo);
        glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(float), lines.data(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);
        _staticCount = (GLsizei)(lines.size() / 6);
    }

    // `generation` identifies the current contents of activeIndices: pass a
    // value that changes whenever the selection changes, so the highlight
    // buffer can be rebuilt only then instead of every frame.
    void render(
        const sibr::Mesh* mesh,
        const std::set<int>&    activeIndices,
        const glm::mat4&        mvp,
        bool                    showMesh,
        uint64_t                generation
    ) {
        if (!mesh || mesh->vertices().empty()) return;

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glUseProgram(_progID);
        glUniformMatrix4fv(_locMVP, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform1i(_locUseLighting, 0);

        if (showMesh && _staticVao) {
            glLineWidth(1.0f);
            glUniform3f(_locColor, 0.f, 0.f, 0.f);
            glBindVertexArray(_staticVao);
            glDrawArrays(GL_LINES, 0, _staticCount);
            glBindVertexArray(0);
        }

        if (_showYellowWireframe && !activeIndices.empty()) {
            // The selected triangle set only changes on a new raycast, but this
            // used to walk the std::set, rebuild the whole line buffer and
            // create/upload/destroy a VAO+VBO pair every single frame. Rebuild
            // only when the caller's generation counter moves; steady-state
            // frames are now one draw call against a persistent buffer.
            if (generation != _activeGen || _activeCount == 0) {
                const auto& verts   = mesh->vertices();
                const auto& normals = mesh->normals();
                const auto& tris    = mesh->triangles();

                std::vector<float> lines;
                lines.reserve(activeIndices.size() * 6 * 6);
                for (int idx : activeIndices)
                    if (idx >= 0 && idx < (int)tris.size())
                        pushEdges(tris[idx], verts, normals, lines);

                uploadActive(lines);
                _activeGen = generation;
            }

            if (_activeCount > 0) {
                // Always draw on top of everything (3DGS, mesh, point clouds).
                glDepthFunc(GL_ALWAYS);
                glLineWidth(2.5f);
                glUniform3f(_locColor, 1.f, 1.f, 0.f);
                glBindVertexArray(_activeVao);
                glDrawArrays(GL_LINES, 0, _activeCount);
                glBindVertexArray(0);
                glDepthFunc(GL_LEQUAL); // restore for subsequent draw calls
            }
        }

        glUseProgram(0);
    }

    bool _showYellowWireframe = true;

private:
    void pushEdges(
        const sibr::Vector3u&              tri,
        const std::vector<sibr::Vector3f>& verts,
        const std::vector<sibr::Vector3f>& normals,
        std::vector<float>&                out
    ) {
        const sibr::Vector3f kDefaultN(0.f, 1.f, 0.f);
        auto push = [&](unsigned int vid) {
            const auto& v = verts[vid];
            const auto& n = (vid < normals.size()) ? normals[vid] : kDefaultN;
            out.push_back(v.x()); out.push_back(v.y()); out.push_back(v.z());
            out.push_back(n.x()); out.push_back(n.y()); out.push_back(n.z());
        };
        push(tri[0]); push(tri[1]);
        push(tri[1]); push(tri[2]);
        push(tri[2]); push(tri[0]);
    }

    // Upload the highlighted-selection lines into a persistent VAO/VBO, created
    // once and then reused (and re-specified) across selections.
    void uploadActive(const std::vector<float>& data) {
        if (!_activeVao) {
            glGenVertexArrays(1, &_activeVao);
            glGenBuffers(1, &_activeVbo);
            glBindVertexArray(_activeVao);
            glBindBuffer(GL_ARRAY_BUFFER, _activeVbo);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);
            glBindVertexArray(0);
        }
        glBindBuffer(GL_ARRAY_BUFFER, _activeVbo);
        glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        _activeCount = (GLsizei)(data.size() / 6);
    }

    void init() {
        GLuint vs = detail::compileGLShader(detail::meshVertSrc(), GL_VERTEX_SHADER);
        GLuint fs = detail::compileGLShader(detail::meshFragSrc(), GL_FRAGMENT_SHADER);
        _progID = detail::linkProgram(vs, fs);
        _locMVP         = glGetUniformLocation(_progID, "uMVP");
        _locColor       = glGetUniformLocation(_progID, "uColor");
        _locUseLighting = glGetUniformLocation(_progID, "uUseLighting");
    }

    GLuint _progID         = 0;
    GLint  _locMVP         = -1;
    GLint  _locColor       = -1;
    GLint  _locUseLighting = -1;

    GLuint  _staticVao     = 0;
    GLuint  _staticVbo     = 0;
    GLsizei _staticCount   = 0;

    GLuint   _activeVao    = 0;
    GLuint   _activeVbo    = 0;
    GLsizei  _activeCount  = 0;
    uint64_t _activeGen    = (uint64_t)-1;  // no selection uploaded yet
};

struct TextureSlotData {
    std::vector<sibr::Vector3u>      triangles;
    std::map<int, glm::vec2>         uvMap;
    sibr::Texture2DRGBA::Ptr         texture;
    std::string                      texturePath;
    std::string                      name;
    float                            alpha   = 1.f;
    bool                             visible = true;
    std::vector<ProjectedGaussian>   geoPoints;
    std::vector<ProjectedGaussian>   appPoints;

    GLuint                           ssbo = 0;
    bool                             isDirty = true;
};

class TextureProjector {
public:
    TextureProjector()  { init(); }
    ~TextureProjector() { if (_progID) glDeleteProgram(_progID); }

    void renderSlot(
        const sibr::Mesh* mesh,
        const std::vector<sibr::Vector3u>& triangles,
        const std::map<int, glm::vec2>&    uvMap,
        const sibr::Texture2DRGBA::Ptr&    texture,
        const glm::mat4&                   mvp,
        const glm::vec3&                   camPos,
        float                              alpha
    ) {
        if (!mesh || mesh->vertices().empty()) return;
        if (triangles.empty() || !texture || texture->handle() == 0) return;

        const auto& vertices = mesh->vertices();
        const auto& normals  = mesh->normals();
        const sibr::Vector3f kDefaultN(0.f, 1.f, 0.f);

        std::vector<float> vboData;
        vboData.reserve(triangles.size() * 3 * 7);
        for (const auto& tri : triangles) {
            if (!uvMap.count(tri[0]) || !uvMap.count(tri[1]) || !uvMap.count(tri[2])) continue;

            std::vector<float> triVerts;
            bool allValid = true;
            for (int k = 0; k < 3; ++k) {
                int vid = tri[k];
                const auto& v  = vertices[vid];
                glm::vec4 clip = mvp * glm::vec4(v.x(), v.y(), v.z(), 1.f);
                if (clip.w < 1e-6f) { allValid = false; break; }

                glm::vec2 ndc(clip.x / clip.w, clip.y / clip.w);
                const glm::vec2& uv = uvMap.at(vid);
                const auto& n = (vid < (int)normals.size()) ? normals[vid] : kDefaultN;

                triVerts.push_back(ndc.x);
                triVerts.push_back(ndc.y);
                triVerts.push_back(uv.x);
                triVerts.push_back(1.f - uv.y);
                triVerts.push_back(n.x());
                triVerts.push_back(n.y());
                triVerts.push_back(n.z());
            }
            if (allValid)
                for (float f : triVerts) vboData.push_back(f);
        }
        if (vboData.empty()) return;

        GLuint vao, vbo;
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vboData.size() * sizeof(float), vboData.data(), GL_DYNAMIC_DRAW);

        constexpr GLsizei stride = 7 * sizeof(float);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, (void*)(4 * sizeof(float)));
        glEnableVertexAttribArray(2);

        glUseProgram(_progID);
        glUniform1f(_locAlpha, alpha);
        glUniform1i(_locUseLighting, _useLighting ? 1 : 0);

        if (_useLighting) {
            glm::vec3 camDir  = glm::length(camPos) > 1e-4f ? glm::normalize(camPos) : glm::vec3(0, 0, 1);
            glm::vec3 lightDir = glm::normalize(camDir + glm::vec3(0.f, 0.4f, 0.f));
            glUniform3f(_locLightDir,         lightDir.x, lightDir.y, lightDir.z);
            glUniform1f(_locAmbient,          kAmbient);
            glUniform1f(_locDiffuseStrength,  kDiffuseStrength);
            glUniform1f(_locSpecularStrength, kSpecularStrength);
            glUniform1f(_locShininess,        kShininess);
        }

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture->handle());

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_FALSE);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(vboData.size() / 7));

        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glDepthMask(GL_TRUE);
        glBindTexture(GL_TEXTURE_2D, 0);
        glUseProgram(0);
        glBindVertexArray(0);
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }

    void render(
        const sibr::Mesh* mesh,
        const std::vector<sibr::Vector3u>& triangles,
        const std::map<int, glm::vec2>&    uvMap,
        const sibr::Texture2DRGBA::Ptr&    texture,
        const glm::mat4&                   mvp,
        const glm::vec3&                   camPos
    ) {
        if (!_enabled) return;
        renderSlot(mesh, triangles, uvMap, texture, mvp, camPos, _alpha);
    }

    void renderAll(
        const sibr::Mesh* mesh,
        const std::vector<TextureSlotData>&     slots,
        const glm::mat4&                        mvp,
        const glm::vec3&                        camPos
    ) {
        if (!_enabled) return;
        for (const auto& slot : slots) {
            if (!slot.visible) continue;
            renderSlot(mesh, slot.triangles, slot.uvMap, slot.texture, mvp, camPos, slot.alpha);
        }
    }

    void renderGUI() {
        ImGui::Checkbox("Show Texture Projection", &_enabled);
        if (_enabled) {
            ImGui::SameLine();
            ImGui::PushItemWidth(100.f);
            ImGui::SliderFloat("Alpha##proj", &_alpha, 0.f, 1.f);
            ImGui::PopItemWidth();
            ImGui::SameLine();
            ImGui::Checkbox("Lighting##proj", &_useLighting);
        }
    }

    bool  isEnabled() const { return _enabled; }
    void  setEnabled(bool e) { _enabled = e; }
    float getAlpha()  const { return _alpha; }
    void  setAlpha(float a) { _alpha = a; }

private:
    void init() {
        const std::string vertSrc = R"(
            #version 330 core
            layout(location = 0) in vec2 aNDC;
            layout(location = 1) in vec2 aTexCoord;
            layout(location = 2) in vec3 aNormal;
            out vec2 vTexCoord;
            out vec3 vNormal;
            void main() {
                gl_Position = vec4(aNDC, 0.0, 1.0);
                vTexCoord   = aTexCoord;
                vNormal     = aNormal;
            }
        )";
        const std::string fragSrc = R"(
            #version 330 core
            in vec2 vTexCoord;
            in vec3 vNormal;
            out vec4 FragColor;
            uniform sampler2D uTexture;
            uniform float     uAlpha;
            uniform bool      uUseLighting;
            uniform vec3      uLightDir;
            uniform float     uAmbient;
            uniform float     uDiffuseStrength;
            uniform float     uSpecularStrength;
            uniform float     uShininess;
            void main() {
                vec4 texCol = texture(uTexture, vTexCoord);
                vec3 rgb    = texCol.rgb;
                if (uUseLighting) {
                    vec3  N    = normalize(vNormal);
                    vec3  L    = normalize(uLightDir);
                    float diff = max(dot(N, L), 0.0);
                    vec3  H    = normalize(L + vec3(0.0, 0.0, 1.0));
                    float spec = pow(max(dot(N, H), 0.0), uShininess);
                    float light = uAmbient
                                + uDiffuseStrength  * diff
                                + uSpecularStrength * spec;
                    rgb = clamp(rgb * light, 0.0, 1.0);
                }
                FragColor = vec4(rgb, texCol.a * uAlpha);
            }
        )";
        GLuint vs = detail::compileGLShader(vertSrc, GL_VERTEX_SHADER);
        GLuint fs = detail::compileGLShader(fragSrc, GL_FRAGMENT_SHADER);
        _progID = detail::linkProgram(vs, fs);
        _locAlpha            = glGetUniformLocation(_progID, "uAlpha");
        _locUseLighting      = glGetUniformLocation(_progID, "uUseLighting");
        _locLightDir         = glGetUniformLocation(_progID, "uLightDir");
        _locAmbient          = glGetUniformLocation(_progID, "uAmbient");
        _locDiffuseStrength  = glGetUniformLocation(_progID, "uDiffuseStrength");
        _locSpecularStrength = glGetUniformLocation(_progID, "uSpecularStrength");
        _locShininess        = glGetUniformLocation(_progID, "uShininess");
        glUseProgram(_progID);
        glUniform1i(glGetUniformLocation(_progID, "uTexture"), 0);
        glUseProgram(0);
    }

    GLuint _progID              = 0;
    GLint  _locAlpha            = -1;
    GLint  _locUseLighting      = -1;
    GLint  _locLightDir         = -1;
    GLint  _locAmbient          = -1;
    GLint  _locDiffuseStrength  = -1;
    GLint  _locSpecularStrength = -1;
    GLint  _locShininess        = -1;
    float  _alpha               = 1.f;
    bool   _enabled             = true;
    bool   _useLighting         = true;

    static constexpr float kAmbient          = 0.35f;
    static constexpr float kDiffuseStrength  = 0.65f;
    static constexpr float kSpecularStrength = 0.15f;
    static constexpr float kShininess        = 32.f;
};

#pragma pack(push, 1)
struct GaussianAttr {
    glm::vec4 pos_opacity;
    glm::vec4 rot;
    glm::vec4 scale_uvx;
    glm::vec4 du_uvy;
    glm::vec4 dv;  // dv.xyz = dV tangent, dv.w = uvMaxDelta (seam clamp limit)
};
#pragma pack(pop)

// ============================================================================
// MeshDepthStencilRenderer
// ============================================================================
class MeshDepthStencilRenderer {
public:
    MeshDepthStencilRenderer()  { init(); }
    ~MeshDepthStencilRenderer() {
        if (_progID) glDeleteProgram(_progID);
        if (_vao)    glDeleteVertexArrays(1, &_vao);
        if (_vbo)    glDeleteBuffers(1, &_vbo);
        if (_ebo)    glDeleteBuffers(1, &_ebo);
    }

    void render(const sibr::Mesh* mesh, const glm::mat4& mvp) {
        if (!mesh || mesh->vertices().empty() || mesh->triangles().empty()) return;

        if (mesh != _lastMesh) {
            uploadMesh(mesh);
            _lastMesh = mesh;
        }

        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);

        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_ALWAYS, 1, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
        glStencilMask(0xFF);

        glUseProgram(_progID);
        glUniformMatrix4fv(_locMVP, 1, GL_FALSE, glm::value_ptr(mvp));

        glBindVertexArray(_vao);
        glDrawElements(GL_TRIANGLES, _indexCount, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        glUseProgram(0);

        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);
    }

private:
    void init() {
        const std::string vs = R"(
            #version 330 core
            layout(location = 0) in vec3 aPos;
            uniform mat4 uMVP;
            void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
        )";
        const std::string fs = R"(
            #version 330 core
            void main() {}
        )";
        GLuint sv = detail::compileGLShader(vs, GL_VERTEX_SHADER);
        GLuint sf = detail::compileGLShader(fs, GL_FRAGMENT_SHADER);
        _progID = detail::linkProgram(sv, sf);
        _locMVP = glGetUniformLocation(_progID, "uMVP");

        glGenVertexArrays(1, &_vao);
        glGenBuffers(1, &_vbo);
        glGenBuffers(1, &_ebo);
    }

    void uploadMesh(const sibr::Mesh* mesh) {
        const auto& verts = mesh->vertices();
        const auto& tris  = mesh->triangles();

        std::vector<float> vdata;
        vdata.reserve(verts.size() * 3);
        for (const auto& v : verts) {
            vdata.push_back(v.x());
            vdata.push_back(v.y());
            vdata.push_back(v.z());
        }

        std::vector<uint32_t> idata;
        idata.reserve(tris.size() * 3);
        for (const auto& t : tris) {
            idata.push_back(t[0]);
            idata.push_back(t[1]);
            idata.push_back(t[2]);
        }
        _indexCount = (GLsizei)idata.size();

        glBindVertexArray(_vao);
        glBindBuffer(GL_ARRAY_BUFFER, _vbo);
        glBufferData(GL_ARRAY_BUFFER, vdata.size() * sizeof(float), vdata.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idata.size() * sizeof(uint32_t), idata.data(), GL_STATIC_DRAW);

        glBindVertexArray(0);
    }

    GLuint _progID     = 0;
    GLuint _vao        = 0;
    GLuint _vbo        = 0;
    GLuint _ebo        = 0;
    GLint  _locMVP     = -1;
    GLsizei _indexCount = 0;
    const sibr::Mesh* _lastMesh = nullptr;
};

class GaussianSplatRenderer {
public:
    GaussianSplatRenderer()  { init(); }
    ~GaussianSplatRenderer() {
        if (_progID)  glDeleteProgram(_progID);
        if (_quadVBO) glDeleteBuffers(1, &_quadVBO);
    }

    void render(
        const std::vector<ProjectedGaussian>& geoPoints,
        const std::vector<ProjectedGaussian>& appPoints,
        GLuint& ssbo,
        bool& isDirty,
        const sibr::Texture2DRGBA::Ptr&       texture,
        const glm::mat4& view,
        const glm::mat4& proj,
        float vpW, float vpH
    ) {
        if (!_enabled) return;
        renderFlat(geoPoints, appPoints, ssbo, isDirty, texture, view, proj, vpW, vpH, _alpha);
    }

    void renderAll(
        std::vector<TextureSlotData>& slots,
        const glm::mat4& view,
        const glm::mat4& proj,
        float vpW, float vpH
    ) {
        if (!_enabled) return;
        for (auto& slot : slots) {
            if (!slot.visible || !slot.texture) continue;
            renderFlat(slot.geoPoints, slot.appPoints, slot.ssbo, slot.isDirty, slot.texture, view, proj, vpW, vpH, slot.alpha);
        }
    }

    void renderGUI() {
        ImGui::Checkbox("Show Texture GS (splats)", &_enabled);
        if (_enabled) {
            ImGui::SameLine();
            ImGui::PushItemWidth(100.f);
            ImGui::SliderFloat("Alpha##gs", &_alpha, 0.f, 1.f);
            ImGui::PopItemWidth();
        }
    }

    bool  isEnabled() const { return _enabled; }

private:
    void renderFlat(
        const std::vector<ProjectedGaussian>& geoPoints,
        const std::vector<ProjectedGaussian>& appPoints,
        GLuint& ssbo,
        bool& isDirty,
        const sibr::Texture2DRGBA::Ptr& texture,
        const glm::mat4& view,
        const glm::mat4& proj,
        float vpW, float vpH,
        float alpha
    ) {
        if (!texture || texture->handle() == 0) return;

        int totalSize = (int)(geoPoints.size() + appPoints.size());
        if (totalSize == 0) return;

        if (isDirty) {
            if (ssbo == 0) { glGenBuffers(1, &ssbo); }

            std::vector<GaussianAttr> packed(totalSize);
            for (int i = 0; i < totalSize; ++i) {
                const ProjectedGaussian* p = (i < (int)geoPoints.size())
                    ? &geoPoints[i]
                    : &appPoints[i - (int)geoPoints.size()];
                packed[i].pos_opacity = glm::vec4(p->position, p->opacity);
                packed[i].rot         = p->rotation;
                packed[i].scale_uvx   = glm::vec4(p->scale, p->uv.x);
                packed[i].du_uvy      = glm::vec4(p->dU, p->uv.y);
                packed[i].dv          = glm::vec4(p->dV, p->uvMaxDelta);
            }

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
            glBufferData(GL_SHADER_STORAGE_BUFFER, packed.size() * sizeof(GaussianAttr), packed.data(), GL_STATIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

            isDirty = false;
        }

        std::vector<int> indices(totalSize);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            const ProjectedGaussian* pA = (a < (int)geoPoints.size())
                ? &geoPoints[a] : &appPoints[a - (int)geoPoints.size()];
            const ProjectedGaussian* pB = (b < (int)geoPoints.size())
                ? &geoPoints[b] : &appPoints[b - (int)geoPoints.size()];
            float za = view[0][2]*pA->position.x + view[1][2]*pA->position.y + view[2][2]*pA->position.z + view[3][2];
            float zb = view[0][2]*pB->position.x + view[1][2]*pB->position.y + view[2][2]*pB->position.z + view[3][2];
            return za < zb;
        });

        GLuint indexVBO;
        glGenBuffers(1, &indexVBO);
        glBindBuffer(GL_ARRAY_BUFFER, indexVBO);
        glBufferData(GL_ARRAY_BUFFER, indices.size() * sizeof(int), indices.data(), GL_DYNAMIC_DRAW);

        GLuint vao;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glBindBuffer(GL_ARRAY_BUFFER, _quadVBO);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, indexVBO);
        glVertexAttribIPointer(1, 1, GL_INT, sizeof(int), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribDivisor(1, 1);

        glBindVertexArray(0);

        float focalX = proj[0][0] * vpW * 0.5f;
        float focalY = proj[1][1] * vpH * 0.5f;

        glm::mat4 invView = glm::inverse(view);
        glm::mat4 invProj = glm::inverse(proj);
        glm::vec3 camPos  = glm::vec3(invView[3]);

        glUseProgram(_progID);
        glUniformMatrix4fv(_locView,    1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(_locProj,    1, GL_FALSE, glm::value_ptr(proj));
        glUniformMatrix4fv(_locInvView, 1, GL_FALSE, glm::value_ptr(invView));
        glUniformMatrix4fv(_locInvProj, 1, GL_FALSE, glm::value_ptr(invProj));
        glUniform3fv(_locCamPos,        1, glm::value_ptr(camPos));
        glUniform1f(_locFocalX, focalX);
        glUniform1f(_locFocalY, focalY);
        glUniform1f(_locW,      vpW);
        glUniform1f(_locH,      vpH);
        glUniform1f(_locAlpha,  alpha);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture->handle());

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);

        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_EQUAL, 1, 0xFF); 
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        glStencilMask(0x00);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glBindVertexArray(vao);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, (GLsizei)totalSize);
        glBindVertexArray(0);

        glDisable(GL_BLEND);
        glDisable(GL_STENCIL_TEST);
        glStencilMask(0xFF);
        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);

        glBindTexture(GL_TEXTURE_2D, 0);
        glUseProgram(0);

        glDeleteVertexArrays(1, &vao);
        glDeleteBuffers(1, &indexVBO);
    }

    void init() {
        const float quad[] = { -1.f,-1.f,  1.f,-1.f,  -1.f,1.f,  1.f,1.f };
        glGenBuffers(1, &_quadVBO);
        glBindBuffer(GL_ARRAY_BUFFER, _quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        const std::string vs = R"(
            #version 430 core
            layout(location=0) in vec2 aQuad;
            layout(location=1) in int  aIndex;

            struct GaussianAttr {
                vec4 pos_opacity;
                vec4 rot;
                vec4 scale_uvx;
                vec4 du_uvy;
                vec4 dv;
            };

            layout(std430, binding = 0) buffer GaussianData {
                GaussianAttr gData[];
            };

            uniform mat4  uView;
            uniform mat4  uProj;
            uniform float uFocalX;
            uniform float uFocalY;
            uniform float uW;
            uniform float uH;

            out vec2  vDelta;
            out vec3  vConic;
            out float vOpacity;
            out vec2  vNDC;

            flat out vec3  vCenter;
            flat out vec2  vUV_center;
            flat out vec3  v_dU;
            flat out vec3  v_dV;
            flat out vec3  vNormal;
            flat out float vUVMaxDelta;

            void main() {
                GaussianAttr attr = gData[aIndex];
                vec3 aPos     = attr.pos_opacity.xyz;
                float aOpacity = attr.pos_opacity.w;
                vec4 aRot     = attr.rot;
                vec3 aScale   = attr.scale_uvx.xyz;
                vec2 aUV      = vec2(attr.scale_uvx.w, attr.du_uvy.w);
                vec3 a_dU     = attr.du_uvy.xyz;
                vec3 a_dV     = attr.dv.xyz;

                vCenter      = aPos;
                vUV_center   = aUV;
                v_dU         = a_dU;
                v_dV         = a_dV;
                vUVMaxDelta  = attr.dv.w;

                vec3 N_mesh = cross(a_dU, a_dV);
                float lenN  = length(N_mesh);
                vec3 n      = (lenN > 1e-8) ? (N_mesh / lenN) : vec3(0.0, 1.0, 0.0);
                vNormal     = n;
                vOpacity    = aOpacity;

                vec4 posV = uView * vec4(aPos, 1.0);
                float tz  = posV.z;
                if (tz >= -0.1) { gl_Position = vec4(0,0,2,1); return; }
                float tx = posV.x, ty = posV.y, tz2 = tz*tz;

                vec3  sc = exp(aScale);
                if (sc.x > 0.05 || sc.y > 0.05) { gl_Position = vec4(0,0,2,1); return; }
                
                // Force rotation to be surface-aligned:
                // Col 0 = tangent X (normalized dU)
                // Col 2 = normal
                // Col 1 = tangent Y (n x tX)
                vec3 tX = normalize(a_dU);
                vec3 tY = normalize(cross(n, tX));
                mat3 R  = mat3(tX, tY, n);

                // Force scale Z to be extremely small to ensure flatness
                float flatZ = min(sc.x, sc.y) * 0.001; 
                mat3 S    = mat3(sc.x,0,0, 0,sc.y,0, 0,0,flatZ);
                mat3 M_cov = R * S;
                mat3 Sig  = M_cov * transpose(M_cov);

                mat3 J = mat3(
                    vec3(uFocalX/tz,  0.0, 0.0),
                    vec3(0.0, uFocalY/tz,  0.0),
                    vec3(-uFocalX*tx/tz2, -uFocalY*ty/tz2, 0.0)
                );
                mat3 W    = mat3(uView);
                mat3 T    = J * W;
                mat3 cov2 = T * Sig * transpose(T);

                float a = cov2[0][0] + 0.3;
                float b = cov2[0][1];
                float c = cov2[1][1] + 0.3;
                float det = a*c - b*b;
                if (det <= 0.0) { gl_Position = vec4(0,0,2,1); return; }

                vConic = vec3(c/det, -b/det, a/det);

                float radX = ceil(3.0 * sqrt(a));
                float radY = ceil(3.0 * sqrt(c));

                vec4  posC      = uProj * posV;
                vec2  centerNDC = posC.xy / posC.w;
                vec2  delta     = aQuad * vec2(radX, radY);
                vNDC = centerNDC + delta*(2.0/vec2(uW,uH));

                gl_Position = vec4(vNDC, posC.z/posC.w, 1.0);
                vDelta = delta;
            }
        )";

        const std::string fs = R"(
            #version 330 core
            in vec2  vDelta;
            in vec3  vConic;
            in float vOpacity;
            in vec2  vNDC;

            flat in vec3  vCenter;
            flat in vec2  vUV_center;
            flat in vec3  v_dU;
            flat in vec3  v_dV;
            flat in vec3  vNormal;
            flat in float vUVMaxDelta;

            out vec4 FragColor;

            uniform sampler2D uTexture;
            uniform float     uAlpha;
            uniform mat4      uInvView;
            uniform mat4      uInvProj;
            uniform vec3      uCamPos;

            void main() {
                float dx = vDelta.x, dy = vDelta.y;
                float power = -0.5*(vConic.x*dx*dx + 2.0*vConic.y*dx*dy + vConic.z*dy*dy);
                if (power > 0.0) discard;

                float alpha = min(0.99, (1.0/(1.0+exp(-vOpacity))) * exp(power));
                if (alpha < 0.1) discard; 

                float denom = dot(vNormal, normalize((uInvView * vec4(
                    normalize((uInvProj * vec4(vNDC.x, vNDC.y, 1.0, 1.0)).xyz), 0.0)).xyz));
                if (denom >= -1e-4) discard;

                vec4 clipH = uInvProj * vec4(vNDC.x, vNDC.y, 1.0, 1.0);
                clipH /= clipH.w;
                vec3 rayDir = normalize((uInvView * vec4(normalize(clipH.xyz), 0.0)).xyz);
                float t_hit = dot(vCenter - uCamPos, vNormal) / denom;
                vec3 hitPos  = uCamPos + t_hit * rayDir;
                vec3 offset  = hitPos - vCenter;

                // Solve the 2x2 Gram system to get correct UV offsets even when
                // dU and dV are not orthogonal (e.g. skewed UV parameterization).
                // offset = deltaU * dU + deltaV * dV  →  Cramer's rule:
                float dUU = dot(v_dU, v_dU);
                float dVV = dot(v_dV, v_dV);
                float dUV = dot(v_dU, v_dV);
                float D   = dUU * dVV - dUV * dUV;
                float deltaU, deltaV;
                if (D > 1e-12) {
                    float bU = dot(offset, v_dU);
                    float bV = dot(offset, v_dV);
                    deltaU = (bU * dVV - bV * dUV) / D;
                    deltaV = (bV * dUU - bU * dUV) / D;
                } else {
                    deltaU = 0.0;
                    deltaV = 0.0;
                }

                // Clamp UV extrapolation to the assigned triangle's UV extent.
                // Prevents tangent-frame discontinuities at UV seams from
                // mapping pixels to the wrong texture region.
                float maxDelta = max(vUVMaxDelta, 0.05); // Increased floor from 0.005 to 0.05
                deltaU = clamp(deltaU, -maxDelta, maxDelta);
                deltaV = clamp(deltaV, -maxDelta, maxDelta);

                vec2 exactUV = vUV_center + vec2(deltaU, deltaV);

                vec2 uv_dx = dFdx(exactUV);
                vec2 uv_dy = dFdy(exactUV);

                if (exactUV.x < 0.0 || exactUV.x > 1.0 || exactUV.y < 0.0 || exactUV.y > 1.0) discard;

                // Calculate proper fragment depth from the plane intersection point
                vec4 p_clip = uProj * uView * vec4(hitPos, 1.0);
                gl_FragDepth = (p_clip.z / p_clip.w) * 0.5 + 0.5;
                // Add a tiny offset to ensure it stays in front of the mesh
                gl_FragDepth -= 0.00001;

                vec4 c = textureGrad(uTexture, exactUV, uv_dx, uv_dy);
                if (c.a < 0.01) discard;

                FragColor = vec4(c.rgb, c.a * alpha * uAlpha);
            }
        )";

        GLuint sv = detail::compileGLShader(vs, GL_VERTEX_SHADER);
        GLuint sf = detail::compileGLShader(fs, GL_FRAGMENT_SHADER);
        _progID = detail::linkProgram(sv, sf);

        _locView     = glGetUniformLocation(_progID, "uView");
        _locProj     = glGetUniformLocation(_progID, "uProj");
        _locInvView  = glGetUniformLocation(_progID, "uInvView");
        _locInvProj  = glGetUniformLocation(_progID, "uInvProj");
        _locCamPos   = glGetUniformLocation(_progID, "uCamPos");
        _locFocalX   = glGetUniformLocation(_progID, "uFocalX");
        _locFocalY   = glGetUniformLocation(_progID, "uFocalY");
        _locW        = glGetUniformLocation(_progID, "uW");
        _locH        = glGetUniformLocation(_progID, "uH");
        _locAlpha    = glGetUniformLocation(_progID, "uAlpha");

        glUseProgram(_progID);
        glUniform1i(glGetUniformLocation(_progID, "uTexture"), 0);
        glUseProgram(0);
    }

    GLuint _progID  = 0;
    GLuint _quadVBO = 0;

    GLint  _locView     = -1;
    GLint  _locProj     = -1;
    GLint  _locInvView  = -1;
    GLint  _locInvProj  = -1;
    GLint  _locCamPos   = -1;
    GLint  _locFocalX   = -1;
    GLint  _locFocalY   = -1;
    GLint  _locW        = -1;
    GLint  _locH        = -1;
    GLint  _locAlpha    = -1;

    float _alpha   = 1.f;
    bool  _enabled = true;
};