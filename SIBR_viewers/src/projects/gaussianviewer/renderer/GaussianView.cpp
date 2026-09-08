/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */

#include <projects/gaussianviewer/renderer/GaussianView.hpp>
#include <core/graphics/GUI.hpp>
#include <thread>
#include <boost/asio.hpp>
#include <rasterizer.h>
#include <imgui_internal.h>
#include <cmath>
#include <algorithm>

typedef sibr::Vector3f Pos;
template<int D>
struct SHs { float shs[(D+1)*(D+1)*3]; };
struct Scale { float scale[3]; };
struct Rot   { float rot[4]; };
template<int D>
struct RichPoint { Pos pos; float n[3]; SHs<D> shs; float opacity; Scale scale; Rot rot; };

float sigmoid(const float m1)         { return 1.0f / (1.0f + exp(-m1)); }
float inverse_sigmoid(const float m1) { return log(m1 / (1.0f - m1)); }

# define CUDA_SAFE_CALL_ALWAYS(A) \
A; \
cudaDeviceSynchronize(); \
if (cudaPeekAtLastError() != cudaSuccess) \
SIBR_ERR << cudaGetErrorString(cudaGetLastError());

// Release builds drop the per-call cudaDeviceSynchronize(): in the render loop
// CUDA_SAFE_CALL wraps several memcpys plus the rasterizer forward every frame,
// and forcing a full device stall after each one is a large FPS hit. Errors are
// still caught by the explicit cudaPeekAtLastError() at the end of onRenderIBR().
// Debug builds keep the synchronizing version for pinpointing failures.
// (Matches the upstream INRIA definition.)
#if defined(_DEBUG) || defined(DEBUG)
# define CUDA_SAFE_CALL(A) CUDA_SAFE_CALL_ALWAYS(A)
#else
# define CUDA_SAFE_CALL(A) A
#endif

template<int D>
int loadPly(const char* filename,
	std::vector<Pos>& pos, std::vector<SHs<3>>& shs,
	std::vector<float>& opacities, std::vector<Scale>& scales,
	std::vector<Rot>& rot,
	sibr::Vector3f& minn, sibr::Vector3f& maxx,
	std::vector<int>& plyToSorted)
{
	std::ifstream infile(filename, std::ios_base::binary);
	if (!infile.good())
		SIBR_ERR << "Unable to find model's PLY file, attempted:\n" << filename << std::endl;

	// Parse header to get vertex count and bytes-per-vertex (handles extra fields like mesh_face_idx)
	auto propTypeBytes = [](const std::string& t) -> size_t {
		if (t == "float" || t == "int" || t == "uint") return 4;
		if (t == "double" || t == "int64" || t == "uint64") return 8;
		if (t == "short"  || t == "ushort") return 2;
		return 1; // char, uchar
	};

	int count = 0;
	size_t bytes_per_vertex = 0;
	bool in_vertex = false;
	std::string buff;

	while (std::getline(infile, buff)) {
		while (!buff.empty() && (buff.back() == '\r' || buff.back() == '\n')) buff.pop_back();
		if (buff == "end_header") break;
		std::stringstream ss(buff);
		std::string tok; ss >> tok;
		if (tok == "element") {
			std::string name; int cnt; ss >> name >> cnt;
			in_vertex = (name == "vertex");
			if (in_vertex) { count = cnt; bytes_per_vertex = 0; }
		} else if (tok == "property" && in_vertex) {
			std::string type; ss >> type;
			if (type != "list") bytes_per_vertex += propTypeBytes(type);
		}
	}

	SIBR_LOG << "Loading " << count << " Gaussian splats" << std::endl;

	const size_t std_bytes   = sizeof(RichPoint<D>);
	const size_t extra_bytes = (bytes_per_vertex > std_bytes) ? (bytes_per_vertex - std_bytes) : 0;

	std::vector<RichPoint<D>> points(count);
	if (extra_bytes == 0) {
		infile.read((char*)points.data(), count * std_bytes);
	} else {
		// Extended format (e.g. SG PLY with mesh_face_idx): read standard fields, skip extras
		std::vector<char> skip_buf(extra_bytes);
		for (int i = 0; i < count; i++) {
			infile.read((char*)&points[i], std_bytes);
			infile.read(skip_buf.data(), extra_bytes);
		}
	}

	pos.resize(count); shs.resize(count); scales.resize(count);
	rot.resize(count); opacities.resize(count);

	minn = sibr::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	maxx = -minn;
	for (int i = 0; i < count; i++) {
		maxx = maxx.cwiseMax(points[i].pos);
		minn = minn.cwiseMin(points[i].pos);
	}

	std::vector<std::pair<uint64_t,int>> mapp(count);
	for (int i = 0; i < count; i++) {
		sibr::Vector3f rel    = (points[i].pos - minn).array() / (maxx - minn).array();
		sibr::Vector3f scaled = ((float((1 << 21) - 1)) * rel);
		sibr::Vector3i xyz    = scaled.cast<int>();
		uint64_t code = 0;
		for (int j = 0; j < 21; j++) {
			code |= ((uint64_t(xyz.x() & (1 << j))) << (2*j + 0));
			code |= ((uint64_t(xyz.y() & (1 << j))) << (2*j + 1));
			code |= ((uint64_t(xyz.z() & (1 << j))) << (2*j + 2));
		}
		mapp[i] = {code, i};
	}
	std::sort(mapp.begin(), mapp.end(),
		[](const std::pair<uint64_t,int>& a, const std::pair<uint64_t,int>& b){ return a.first < b.first; });

	plyToSorted.resize(count);
	for (int k = 0; k < count; k++) plyToSorted[mapp[k].second] = k;

	int SH_N = (D+1)*(D+1);
	for (int k = 0; k < count; k++) {
		int i = mapp[k].second;
		pos[k] = points[i].pos;
		float len2 = 0;
		for (int j = 0; j < 4; j++) len2 += points[i].rot.rot[j] * points[i].rot.rot[j];
		float len = sqrt(len2);
		for (int j = 0; j < 4; j++) rot[k].rot[j] = points[i].rot.rot[j] / len;
		for (int j = 0; j < 3; j++) scales[k].scale[j] = exp(points[i].scale.scale[j]);
		opacities[k] = sigmoid(points[i].opacity);
		shs[k].shs[0] = points[i].shs.shs[0];
		shs[k].shs[1] = points[i].shs.shs[1];
		shs[k].shs[2] = points[i].shs.shs[2];
		for (int j = 1; j < SH_N; j++) {
			shs[k].shs[j*3+0] = points[i].shs.shs[(j-1)+3];
			shs[k].shs[j*3+1] = points[i].shs.shs[(j-1)+SH_N+2];
			shs[k].shs[j*3+2] = points[i].shs.shs[(j-1)+2*SH_N+1];
		}
	}
	return count;
}

void savePly(const char* filename,
	const std::vector<Pos>& pos, const std::vector<SHs<3>>& shs,
	const std::vector<float>& opacities, const std::vector<Scale>& scales,
	const std::vector<Rot>& rot,
	const sibr::Vector3f& minn, const sibr::Vector3f& maxx)
{
	int count = 0;
	for (int i = 0; i < (int)pos.size(); i++) {
		if (pos[i].x()<minn.x()||pos[i].y()<minn.y()||pos[i].z()<minn.z()||
		    pos[i].x()>maxx.x()||pos[i].y()>maxx.y()||pos[i].z()>maxx.z()) continue;
		count++;
	}
	std::vector<RichPoint<3>> points(count);
	SIBR_LOG << "Saving " << count << " Gaussian splats" << std::endl;

	std::ofstream outfile(filename, std::ios_base::binary);
	outfile << "ply\nformat binary_little_endian 1.0\nelement vertex " << count << "\n";
	for (auto s : {"x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2"})
		outfile << "property float " << s << "\n";
	for (int i = 0; i < 45; i++) outfile << "property float f_rest_" << i << "\n";
	for (auto s : {"opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"})
		outfile << "property float " << s << "\n";
	outfile << "end_header\n";

	count = 0;
	for (int i = 0; i < (int)pos.size(); i++) {
		if (pos[i].x()<minn.x()||pos[i].y()<minn.y()||pos[i].z()<minn.z()||
		    pos[i].x()>maxx.x()||pos[i].y()>maxx.y()||pos[i].z()>maxx.z()) continue;
		points[count].pos = pos[i];
		points[count].rot = rot[i];
		for (int j = 0; j < 3; j++) points[count].scale.scale[j] = log(scales[i].scale[j]);
		points[count].opacity = inverse_sigmoid(opacities[i]);
		points[count].shs.shs[0]=shs[i].shs[0];
		points[count].shs.shs[1]=shs[i].shs[1];
		points[count].shs.shs[2]=shs[i].shs[2];
		for (int j = 1; j < 16; j++) {
			points[count].shs.shs[(j-1)+3]  = shs[i].shs[j*3+0];
			points[count].shs.shs[(j-1)+18] = shs[i].shs[j*3+1];
			points[count].shs.shs[(j-1)+33] = shs[i].shs[j*3+2];
		}
		count++;
	}
	outfile.write((char*)points.data(), sizeof(RichPoint<3>) * points.size());
}

namespace sibr
{
	class BufferCopyRenderer
	{
	public:
		BufferCopyRenderer() {
			_shader.init("CopyShader",
				sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.vert"),
				sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.frag"));
			_flip.init(_shader, "flip");
			_width.init(_shader, "width");
			_height.init(_shader, "height");
		}
		void process(uint bufferID, IRenderTarget& dst, int width, int height, bool disableTest = true) {
			if (disableTest) glDisable(GL_DEPTH_TEST); else glEnable(GL_DEPTH_TEST);
			_shader.begin(); _flip.send(); _width.send(); _height.send();
			dst.clear(); dst.bind();
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID);
			sibr::RenderUtility::renderScreenQuad();
			dst.unbind(); _shader.end();
		}
		bool& flip()   { return _flip.get(); }
		int&  width()  { return _width.get(); }
		int&  height() { return _height.get(); }
	private:
		GLShader         _shader;
		GLuniform<bool>  _flip   = false;
		GLuniform<int>   _width  = 1000;
		GLuniform<int>   _height = 800;
	};
}

std::function<char*(size_t N)> resizeFunctional(void** ptr, size_t& S) {
	auto lambda = [ptr, &S](size_t N) {
		if (N > S) {
			if (*ptr) CUDA_SAFE_CALL(cudaFree(*ptr));
			CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
			S = 2 * N;
		}
		return reinterpret_cast<char*>(*ptr);
	};
	return lambda;
}

sibr::GaussianView::GaussianView(const sibr::BasicIBRScene::Ptr& ibrScene,
	uint render_w, uint render_h, const char* file, bool* messageRead,
	int sh_degree, bool white_bg, bool useInterop, int device)
	: _scene(ibrScene), _dontshow(messageRead), _sh_degree(sh_degree),
	  sibr::ViewBase(render_w, render_h)
{
	int num_devices;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceCount(&num_devices));
	_device = device;
	if (device >= num_devices) {
		if (num_devices == 0) SIBR_ERR << "No CUDA devices detected!";
		else SIBR_ERR << "Provided device index exceeds number of available CUDA devices!";
	}
	CUDA_SAFE_CALL_ALWAYS(cudaSetDevice(device));
	cudaDeviceProp prop;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceProperties(&prop, device));
	if (prop.major < 7) SIBR_ERR << "Sorry, need at least compute capability 7.0+!";

	_pointbasedrenderer.reset(new PointBasedRenderer());
	_copyRenderer = new BufferCopyRenderer();
	_copyRenderer->flip() = true;
	_copyRenderer->width() = render_w;
	_copyRenderer->height() = render_h;

	std::vector<uint> imgs_ulr;
	const auto& cams = ibrScene->cameras()->inputCameras();
	for (size_t cid = 0; cid < cams.size(); ++cid)
		if (cams[cid]->isActive()) imgs_ulr.push_back(uint(cid));
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);

	std::vector<Pos> pos; std::vector<Rot> rot;
	std::vector<Scale> scale; std::vector<float> opacity; std::vector<SHs<3>> shs;

	if      (sh_degree == 0) count = loadPly<0>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax, _plyToSorted);
	else if (sh_degree == 1) count = loadPly<1>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax, _plyToSorted);
	else if (sh_degree == 2) count = loadPly<2>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax, _plyToSorted);
	else                     count = loadPly<3>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax, _plyToSorted);

	_boxmin = _scenemin;
	_boxmax = _scenemax;
	int P = count;

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&pos_cuda,     sizeof(Pos)    * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda,     pos.data(),     sizeof(Pos)    * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rot_cuda,     sizeof(Rot)    * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_cuda,     rot.data(),     sizeof(Rot)    * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&shs_cuda,     sizeof(SHs<3>) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_cuda,     shs.data(),     sizeof(SHs<3>) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&opacity_cuda, sizeof(float)  * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_cuda, opacity.data(), sizeof(float)  * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&scale_cuda,   sizeof(Scale)  * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_cuda,   scale.data(),   sizeof(Scale)  * P, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&view_cuda,       sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&proj_cuda,       sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&cam_pos_cuda,    3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rect_cuda,       2 * P * sizeof(int)));

	float bg[3] = {
		white_bg ? 1.f : 0.f,
		white_bg ? 1.f : 0.f,
		white_bg ? 1.f : 0.f
	};
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

	gData = new GaussianData(P, (float*)pos.data(), (float*)rot.data(),
	                         (float*)scale.data(), opacity.data(), (float*)shs.data());
	_gaussianRenderer = new GaussianSurfaceRenderer();

	glCreateBuffers(1, &imageBuffer);
	glNamedBufferStorage(imageBuffer, render_w * render_h * 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

	if (useInterop) {
		if (cudaPeekAtLastError() != cudaSuccess)
			SIBR_ERR << "A CUDA error occurred in setup: " << cudaGetErrorString(cudaGetLastError());
		cudaGraphicsGLRegisterBuffer(&imageBufferCuda, imageBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
		useInterop &= (cudaGetLastError() == cudaSuccess);
	}
	if (!useInterop) {
		fallback_bytes.resize(render_w * render_h * 3 * sizeof(float));
		cudaMalloc(&fallbackBufferCuda, fallback_bytes.size());
		_interop_failed = true;
	}

	geomBufferFunc    = resizeFunctional(&geomPtr,    allocdGeom);
	binningBufferFunc = resizeFunctional(&binningPtr, allocdBinning);
	imgBufferFunc     = resizeFunctional(&imgPtr,     allocdImg);
}

void sibr::GaussianView::setScene(const sibr::BasicIBRScene::Ptr& newScene) {
	_scene = newScene;
	std::vector<uint> imgs_ulr;
	const auto& cams = newScene->cameras()->inputCameras();
	for (size_t cid = 0; cid < cams.size(); ++cid)
		if (cams[cid]->isActive()) imgs_ulr.push_back(uint(cid));
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);
}

void sibr::GaussianView::onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye)
{
	if (currMode == "Ellipsoids") {
		_gaussianRenderer->process(count, *gData, eye, dst, 0.2f);
	} else if (currMode == "Initial Points") {
		_pointbasedrenderer->process(_scene->proxies()->proxy(), eye, dst);
	} else {
		auto view_mat = eye.view();
		auto proj_mat = eye.viewproj();
		view_mat.row(1) *= -1;
		view_mat.row(2) *= -1;
		proj_mat.row(1) *= -1;
		float tan_fovy = tan(eye.fovy() * 0.5f);
		float tan_fovx = tan_fovy * eye.aspect();

		CUDA_SAFE_CALL(cudaMemcpy(view_cuda,    view_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(proj_cuda,    proj_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(cam_pos_cuda, &eye.position(), sizeof(float) * 3,      cudaMemcpyHostToDevice));

		float* image_cuda = nullptr;
		if (!_interop_failed) {
			size_t bytes = 0;
			CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &imageBufferCuda));
			CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&image_cuda, &bytes, imageBufferCuda));
		} else {
			image_cuda = fallbackBufferCuda;
		}

		int* rects  = _fastCulling ? rect_cuda       : nullptr;
		float* boxmin = _cropping   ? (float*)&_boxmin : nullptr;
		float* boxmax = _cropping   ? (float*)&_boxmax : nullptr;

		// Compute a single unified view-space depth for all UV Gaussians.
		// By forcing every UV Gaussian to the same depth during the radix sort,
		// we eliminate the parallax effect that makes the texture appear to protrude
		// when viewed from the side.
		float uvFixedDepth = 0.f;
		if (_hasUVSurface) {
			// Transform world-space centroid into the same view space used by the CUDA rasterizer
			// (row 1 and 2 negated above, so view-space z is positive for objects in front).
			sibr::Vector4f p(_uvSurfacePos.x(), _uvSurfacePos.y(), _uvSurfacePos.z(), 1.f);
			sibr::Vector4f pv = view_mat * p;
			// Only use the depth if the surface is in front of the camera (z > 0.2 matches frustum cull threshold)
			if (pv.z() > 0.2f)
				uvFixedDepth = pv.z();
		}

		CudaRasterizer::Rasterizer::forward(
			geomBufferFunc, binningBufferFunc, imgBufferFunc,
			count, _sh_degree, 16,
			background_cuda, _resolution.x(), _resolution.y(),
			pos_cuda, shs_cuda, nullptr, opacity_cuda, scale_cuda,
			_scalingModifier, rot_cuda, nullptr,
			view_cuda, proj_cuda, cam_pos_cuda,
			tan_fovx, tan_fovy,
			false, image_cuda, _antialiasing,
			nullptr, rects, boxmin, boxmax,
			uvs_cuda, dU_cuda, dV_cuda, surfaceDist_cuda, _texObjs_cuda, texIdx_cuda,
			uvFixedDepth,
			_normalObjs_cuda,
			false  // debug
		);

		if (!_interop_failed) {
			CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &imageBufferCuda));
		} else {
			CUDA_SAFE_CALL(cudaMemcpy(fallback_bytes.data(), fallbackBufferCuda, fallback_bytes.size(), cudaMemcpyDeviceToHost));
			glNamedBufferSubData(imageBuffer, 0, fallback_bytes.size(), fallback_bytes.data());
		}
		_copyRenderer->process(imageBuffer, dst, _resolution.x(), _resolution.y());
	}

	if (cudaPeekAtLastError() != cudaSuccess)
		SIBR_ERR << "A CUDA error occurred during rendering: " << cudaGetErrorString(cudaGetLastError());
}

void sibr::GaussianView::onUpdate(Input& input) {}

void sibr::GaussianView::onGUI()
{
	const std::string guiName = "3D Gaussians";
	if (ImGui::Begin(guiName.c_str())) {
		if (ImGui::BeginCombo("Render Mode", currMode.c_str())) {
			if (ImGui::Selectable("Splats"))         currMode = "Splats";
			if (ImGui::Selectable("Initial Points")) currMode = "Initial Points";
			if (ImGui::Selectable("Ellipsoids"))     currMode = "Ellipsoids";
			ImGui::EndCombo();
		}
	}
	if (currMode == "Splats")
		ImGui::SliderFloat("Scaling Modifier", &_scalingModifier, 0.001f, 1.0f);
	ImGui::Checkbox("Fast culling", &_fastCulling);
	ImGui::Checkbox("Antialiasing", &_antialiasing);
	ImGui::Checkbox("Crop Box",     &_cropping);
	if (_cropping) {
		ImGui::SliderFloat("Box Min X", &_boxmin.x(), _scenemin.x(), _scenemax.x());
		ImGui::SliderFloat("Box Min Y", &_boxmin.y(), _scenemin.y(), _scenemax.y());
		ImGui::SliderFloat("Box Min Z", &_boxmin.z(), _scenemin.z(), _scenemax.z());
		ImGui::SliderFloat("Box Max X", &_boxmax.x(), _scenemin.x(), _scenemax.x());
		ImGui::SliderFloat("Box Max Y", &_boxmax.y(), _scenemin.y(), _scenemax.y());
		ImGui::SliderFloat("Box Max Z", &_boxmax.z(), _scenemin.z(), _scenemax.z());
		ImGui::InputText("File", _buff, 512);
		if (ImGui::Button("Save")) {
			std::vector<Pos>    pos(count);
			std::vector<Rot>    rot(count);
			std::vector<float>  opacity(count);
			std::vector<SHs<3>> shs(count);
			std::vector<Scale>  scale(count);
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos.data(),     pos_cuda,     sizeof(Pos)    * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot.data(),     rot_cuda,     sizeof(Rot)    * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity.data(), opacity_cuda, sizeof(float)  * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs.data(),     shs_cuda,     sizeof(SHs<3>) * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale.data(),   scale_cuda,   sizeof(Scale)  * count, cudaMemcpyDeviceToHost));
			savePly(_buff, pos, shs, opacity, scale, rot, _boxmin, _boxmax);
		}
	}
	ImGui::End();

	if (!*_dontshow && !accepted && _interop_failed)
		ImGui::OpenPopup("Error Using Interop");
	if (!*_dontshow && !accepted && _interop_failed &&
		ImGui::BeginPopupModal("Error Using Interop", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
		ImGui::SetItemDefaultFocus();
		ImGui::SetWindowFontScale(2.0f);
		ImGui::Text("This application tries to use CUDA/OpenGL interop.\n"
			" It did NOT work for your current configuration.\n"
			" For highest performance, OpenGL and CUDA must run on the same\n"
			" GPU on an OS that supports interop.You can try to pass a\n"
			" non-zero index via --device on a multi-GPU system, and/or try\n"
			" attaching the monitors to the main CUDA card.\n"
			" On a laptop with one integrated and one dedicated GPU, you can try\n"
			" to set the preferred GPU via your operating system.\n\n"
			" FALLING BACK TO SLOWER RENDERING WITH CPU ROUNDTRIP\n");
		ImGui::Separator();
		if (ImGui::Button("  OK  ")) {
			ImGui::CloseCurrentPopup();
			accepted = true;
		}
		ImGui::SameLine();
		ImGui::Checkbox("Don't show this message again", _dontshow);
		ImGui::EndPopup();
	}
}

// =============================================================================
// getCpuPositions
// =============================================================================
const std::vector<sibr::Vector3f>& sibr::GaussianView::getCpuPositions()
{
	if ((int)_cpuPos.size() != count) {
		_cpuPos.resize(count);
		cudaMemcpy(_cpuPos.data(), pos_cuda,
		           sizeof(sibr::Vector3f) * count, cudaMemcpyDeviceToHost);
		SIBR_LOG << "[GaussianView] Downloaded " << count << " positions." << std::endl;
	}
	return _cpuPos;
}

// =============================================================================
// downloadGaussianData
// =============================================================================
void sibr::GaussianView::downloadGaussianData(
	std::vector<float>& outRot,
	std::vector<float>& outScale,
	std::vector<float>& outOpacity)
{
	outRot.resize((size_t)count * 4);
	outScale.resize((size_t)count * 3);
	outOpacity.resize((size_t)count);
	cudaMemcpy(outRot.data(),     rot_cuda,     sizeof(float) * count * 4, cudaMemcpyDeviceToHost);
	cudaMemcpy(outScale.data(),   scale_cuda,   sizeof(float) * count * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(outOpacity.data(), opacity_cuda, sizeof(float) * count,     cudaMemcpyDeviceToHost);
}

void sibr::GaussianView::setOpacityArray(const std::vector<float>& opacities)
{
    if ((int)opacities.size() != count) return;
    cudaMemcpy(opacity_cuda, opacities.data(), sizeof(float) * count, cudaMemcpyHostToDevice);
}

void sibr::GaussianView::setNormalMapTexture(const std::vector<uint8_t>& pixels, int w, int h, int slot)
{
    if (slot < 0) return;

    // Grow to cover this slot so _normalObjs stays index-compatible with _texObjs.
    // A patch that never baked a normal map keeps a 0 here, which the kernel reads
    // as "no normal map" for that patch alone.
    if ((int)_normalObjs.size() <= slot) {
        _normalObjs.resize(slot + 1, 0);
        _normalArrays.resize(slot + 1, nullptr);
    }
    if (_normalObjs[slot])   { cudaDestroyTextureObject(_normalObjs[slot]); _normalObjs[slot] = 0; }
    if (_normalArrays[slot]) { cudaFreeArray(_normalArrays[slot]); _normalArrays[slot] = nullptr; }

    cudaArray_t         _normal_array = nullptr;
    cudaTextureObject_t normal_tex_cuda = 0;
    if (pixels.empty() || w <= 0 || h <= 0) return;

    std::vector<float> fp(w * h * 4);
    for (int i = 0; i < w * h * 4; ++i)
        fp[i] = pixels[i] / 255.0f;

    cudaChannelFormatDesc cd = cudaCreateChannelDesc(32,32,32,32, cudaChannelFormatKindFloat);
    CUDA_SAFE_CALL_ALWAYS(cudaMallocArray(&_normal_array, &cd, w, h));
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy2DToArray(
        _normal_array, 0, 0, fp.data(),
        w * 4 * sizeof(float), w * 4 * sizeof(float), h,
        cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = _normal_array;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0]      = cudaAddressModeClamp;
    texDesc.addressMode[1]      = cudaAddressModeClamp;
    texDesc.filterMode          = cudaFilterModeLinear;
    texDesc.readMode            = cudaReadModeElementType;
    texDesc.normalizedCoords    = 1;
    texDesc.maxAnisotropy       = 1;
    texDesc.mipmapFilterMode    = cudaFilterModePoint;
    texDesc.maxMipmapLevelClamp = 0;

    CUDA_SAFE_CALL_ALWAYS(cudaCreateTextureObject(&normal_tex_cuda, &resDesc, &texDesc, NULL));

    _normalObjs[slot]   = normal_tex_cuda;
    _normalArrays[slot] = _normal_array;

    if (_normalObjs.size() > _normalObjsCapacity) {
        if (_normalObjs_cuda) cudaFree(_normalObjs_cuda);
        _normalObjsCapacity = _normalObjs.size() * 2;
        CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&_normalObjs_cuda,
            sizeof(cudaTextureObject_t) * _normalObjsCapacity));
    }
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_normalObjs_cuda, _normalObjs.data(),
        sizeof(cudaTextureObject_t) * _normalObjs.size(), cudaMemcpyHostToDevice));
}

void sibr::GaussianView::clearNormalMapTexture()
{
    for (auto t : _normalObjs)   if (t) cudaDestroyTextureObject(t);
    for (auto a : _normalArrays) if (a) cudaFreeArray(a);
    _normalObjs.clear();
    _normalArrays.clear();
    if (_normalObjs_cuda) { cudaFree(_normalObjs_cuda); _normalObjs_cuda = nullptr; }
    _normalObjsCapacity = 0;
}

void sibr::GaussianView::suppressGaussiansInRegion(
    const std::vector<sibr::Vector2f>& uvs,
    const sibr::Vector3f& center,
    float suppressRadius,
    const std::vector<float>& surfaceDists)
{
    if ((int)uvs.size() != count) return;
    getCpuPositions();
    if ((int)_cpuOpacityCache.size() != count) {
        _cpuOpacityCache.resize(count);
        cudaMemcpy(_cpuOpacityCache.data(), opacity_cuda, sizeof(float) * count, cudaMemcpyDeviceToHost);
    }

    const bool hasSurfDist = ((int)surfaceDists.size() == count);
    std::vector<float> modOpc(_cpuOpacityCache);

    // Cache original scale values for restoration by restoreOpacities().
    // (setUVsAndTexture no longer modifies scale, so we cache it here on first call.)
    if (_cpuScaleCache.empty()) {
        _cpuScaleCache.resize(count * 3);
        cudaMemcpy(_cpuScaleCache.data(), scale_cuda, sizeof(float) * count * 3, cudaMemcpyDeviceToHost);
    }
    // Always start from the cached originals so re-calls don't accumulate suppression.
    std::vector<float> modScale(_cpuScaleCache);

    const float suppressRadius2 = suppressRadius * suppressRadius;

    for (int i = 0; i < count; ++i) {
        if (uvs[i].x() >= 0.f) continue; // has UV -> keep original opacity and scale

        const sibr::Vector3f& p = _cpuPos[i];
        float dx = p.x() - center.x(), dy = p.y() - center.y(), dz = p.z() - center.z();
        if (dx*dx + dy*dy + dz*dz > suppressRadius2) continue;

        float factor;
        if (hasSurfDist && surfaceDists[i] < suppressRadius) {
            // dist=0 (on surface) -> factor=0 (fully suppressed)
            // dist=suppressRadius -> factor≈0.95 (mostly untouched)
            float t = surfaceDists[i] / suppressRadius;
            factor = 1.0f - std::exp(-3.0f * t * t);
            modOpc[i] *= factor;
        } else {
            factor = 0.f;
            modOpc[i] = 0.f;
        }

        // For heavily suppressed Gaussians: also zero scale so cov3D→0.
        // The preprocessCUDA opacity early-exit (opacities < 1/255 → return) will
        // then remove these entirely from the tile list, preventing any T consumption.
        if (factor < 0.1f) {
            modScale[3*i+0] = 1e-6f;
            modScale[3*i+1] = 1e-6f;
            modScale[3*i+2] = 1e-6f;
        }
    }
    cudaMemcpy(opacity_cuda, modOpc.data(), sizeof(float) * count, cudaMemcpyHostToDevice);
    cudaMemcpy(scale_cuda,   modScale.data(), sizeof(float) * count * 3, cudaMemcpyHostToDevice);
}

void sibr::GaussianView::restoreOpacities() {
    if ((int)_cpuOpacityCache.size() != count) return;
    cudaMemcpy(opacity_cuda, _cpuOpacityCache.data(), sizeof(float) * count, cudaMemcpyHostToDevice);
    _cpuOpacityCache.clear();

    if (!_cpuPosCache.empty()) {
        cudaMemcpy(pos_cuda, _cpuPosCache.data(),
                   sizeof(float) * count * 3, cudaMemcpyHostToDevice);
        _cpuPosCache.clear();
        _cpuPos.clear();
    }
    if (!_cpuRotCache.empty()) {
        cudaMemcpy(rot_cuda, _cpuRotCache.data(),
                   sizeof(float) * count * 4, cudaMemcpyHostToDevice);
        _cpuRotCache.clear();
    }
    if (!_cpuScaleCache.empty()) {
        cudaMemcpy(scale_cuda, _cpuScaleCache.data(),
                   sizeof(float) * count * 3, cudaMemcpyHostToDevice);
        _cpuScaleCache.clear();
    }
}

void sibr::GaussianView::updateGeometry(
    const std::vector<sibr::Vector3f>& positions,
    const std::vector<sibr::Vector4f>& rotations,
    const std::vector<sibr::Vector3f>& scales)
{
    if ((int)positions.size() == count)
        cudaMemcpy(pos_cuda,   positions.data(), sizeof(float) * count * 3, cudaMemcpyHostToDevice);
    if ((int)rotations.size() == count)
        cudaMemcpy(rot_cuda,   rotations.data(), sizeof(float) * count * 4, cudaMemcpyHostToDevice);
    if ((int)scales.size() == count)
        cudaMemcpy(scale_cuda, scales.data(),    sizeof(float) * count * 3, cudaMemcpyHostToDevice);
}

// =============================================================================
// setUVsAndTexture
// Uploads UV/tangent data and texture to GPU. Gaussian position, scale, and
// rotation are intentionally left unchanged — each Gaussian keeps its original
// geometry. Texture UV is computed in renderCUDA via ray-tangent-plane
// intersection using the per-Gaussian dU/dV vectors.
// =============================================================================
void sibr::GaussianView::setUVsAndTexture(
	const std::vector<sibr::Vector2f>& uvs,
	const std::vector<sibr::Vector3f>& dUs,
	const std::vector<sibr::Vector3f>& dVs,
	const std::vector<float>&          surfaceDists,
	const std::vector<sibr::Vector3f>& surfacePositions,
	const std::vector<int>& texIdx)
{
	if ((int)uvs.size() != count || (int)dUs.size() != count ||
	    (int)dVs.size() != count || (int)surfaceDists.size() != count) {
		SIBR_ERR << "[GaussianView::setUVsAndTexture] size mismatch: "
		         << "uvs=" << uvs.size() << " dUs=" << dUs.size()
		         << " dVs=" << dVs.size() << " surfaceDists=" << surfaceDists.size()
		         << " count=" << count;
		return;
	}

	if (!uvs_cuda)
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&uvs_cuda, sizeof(float) * count * 2));
	if (!dU_cuda)
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&dU_cuda, sizeof(float) * count * 3));
	if (!dV_cuda)
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&dV_cuda, sizeof(float) * count * 3));
	if (!surfaceDist_cuda)
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&surfaceDist_cuda, sizeof(float) * count));
	if (!texIdx_cuda)
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&texIdx_cuda, sizeof(int) * count));

	{
		// texIdx may legitimately be empty (callers that only refresh UVs); treat that
		// as "nothing textured" rather than leaving stale indices on the device.
		std::vector<int> idx = texIdx;
		if ((int)idx.size() != count) idx.assign(count, -1);
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(texIdx_cuda, idx.data(), sizeof(int) * count, cudaMemcpyHostToDevice));
	}

	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(uvs_cuda,         uvs.data(),          sizeof(float) * count * 2, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(dU_cuda,          dUs.data(),          sizeof(float) * count * 3, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(dV_cuda,          dVs.data(),          sizeof(float) * count * 3, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(surfaceDist_cuda, surfaceDists.data(), sizeof(float) * count,     cudaMemcpyHostToDevice));

	// [DISABLED] Snap UV Gaussians onto the mesh surface.
	// if ((int)surfacePositions.size() == count) {
	// 	if (_cpuPosCache.empty()) {
	// 		_cpuPosCache.resize(count);
	// 		cudaMemcpy(_cpuPosCache.data(), pos_cuda,
	// 		           sizeof(float) * count * 3, cudaMemcpyDeviceToHost);
	// 	}
	// 	std::vector<float> snappedPos(count * 3);
	// 	cudaMemcpy(snappedPos.data(), pos_cuda, sizeof(float) * count * 3, cudaMemcpyDeviceToHost);
	// 	for (int i = 0; i < count; ++i) {
	// 		if (uvs[i].x() < 0.f) continue;
	// 		snappedPos[3*i+0] = surfacePositions[i].x();
	// 		snappedPos[3*i+1] = surfacePositions[i].y();
	// 		snappedPos[3*i+2] = surfacePositions[i].z();
	// 	}
	// 	cudaMemcpy(pos_cuda, snappedPos.data(), sizeof(float) * count * 3, cudaMemcpyHostToDevice);
	// 	_cpuPos.clear();
	// }

	_hasUVSurface = false; // uvFixedDepth path disabled; depth bias in preprocessCUDA handles ordering
}

void sibr::GaussianView::clearTextures()
{
	for (auto t : _texObjs)   if (t) cudaDestroyTextureObject(t);
	for (auto a : _texArrays) if (a) cudaFreeArray(a);
	_texObjs.clear();
	_texArrays.clear();
	_texRefs.clear();
	if (_texObjs_cuda) { cudaFree(_texObjs_cuda); _texObjs_cuda = nullptr; }
	_texObjsCapacity = 0;
	// Slots are shared between the two lists -- dropping one without the other
	// would leave every remaining normal map attached to the wrong patch.
	clearNormalMapTexture();
}

int sibr::GaussianView::registerTexture(sibr::Texture2DRGBA::Ptr texPtr, int reuseSlot)
{
	if (!texPtr || texPtr->handle() == 0) return -1;

	// Deliberately no de-duplication by texture pointer. A slot is one *pick*, not
	// one image: the normal map stored alongside it is baked in that pick's own
	// ExpMap UV space, so painting a second region with the same image file must
	// still get its own slot. Sharing the slot let the second pick's normal map
	// overwrite the first pick's, which is what made every patch shade against the
	// most recently baked normals. Cost is one extra upload of an image already in
	// VRAM -- cheap next to getting the shading wrong.

	cudaArray_t         arr = nullptr;
	cudaTextureObject_t obj = 0;
	{
			int w = texPtr->w();
			int h = texPtr->h();

			std::vector<sibr::Vector4ub> pixels(w * h);
			glBindTexture(GL_TEXTURE_2D, texPtr->handle());
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
			glBindTexture(GL_TEXTURE_2D, 0);

			std::vector<float> fp(w * h * 4);
			for (int i = 0; i < w * h * 4; ++i)
				fp[i] = ((unsigned char*)pixels.data())[i] / 255.0f;

			cudaChannelFormatDesc cd = cudaCreateChannelDesc(32,32,32,32, cudaChannelFormatKindFloat);
			CUDA_SAFE_CALL_ALWAYS(cudaMallocArray(&arr, &cd, w, h));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy2DToArray(
				arr, 0, 0, fp.data(),
				w * 4 * sizeof(float), w * 4 * sizeof(float), h,
				cudaMemcpyHostToDevice));

			cudaResourceDesc resDesc{};
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = arr;

			cudaTextureDesc texDesc{};
			texDesc.addressMode[0]      = cudaAddressModeBorder;
			texDesc.addressMode[1]      = cudaAddressModeBorder;
			texDesc.borderColor[0]      = 0.f;
			texDesc.borderColor[1]      = 0.f;
			texDesc.borderColor[2]      = 0.f;
			texDesc.borderColor[3]      = 0.f;
			texDesc.filterMode          = cudaFilterModeLinear;
			texDesc.readMode            = cudaReadModeElementType;
			texDesc.normalizedCoords    = 1;
			texDesc.maxAnisotropy       = 16;
			texDesc.mipmapFilterMode    = cudaFilterModePoint;
			texDesc.maxMipmapLevelClamp = 0;

			CUDA_SAFE_CALL_ALWAYS(cudaCreateTextureObject(&obj, &resDesc, &texDesc, NULL));
	}

	int slot;
	if (reuseSlot >= 0 && reuseSlot < (int)_texObjs.size()) {
		// Overwrite in place: the caller is re-previewing the same pending patch.
		if (_texObjs[reuseSlot])   cudaDestroyTextureObject(_texObjs[reuseSlot]);
		if (_texArrays[reuseSlot]) cudaFreeArray(_texArrays[reuseSlot]);
		_texObjs[reuseSlot]   = obj;
		_texArrays[reuseSlot] = arr;
		_texRefs[reuseSlot]   = texPtr;
		slot = reuseSlot;
	} else {
		_texObjs.push_back(obj);
		_texArrays.push_back(arr);
		_texRefs.push_back(texPtr);
		slot = (int)_texObjs.size() - 1;
	}

	// Grow the device-side list. Realloc rather than a fixed cap: the number of
	// simultaneous patches is unbounded by design.
	if (_texObjs.size() > _texObjsCapacity) {
		if (_texObjs_cuda) cudaFree(_texObjs_cuda);
		_texObjsCapacity = _texObjs.size() * 2;
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&_texObjs_cuda,
			sizeof(cudaTextureObject_t) * _texObjsCapacity));
	}
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(_texObjs_cuda, _texObjs.data(),
		sizeof(cudaTextureObject_t) * _texObjs.size(), cudaMemcpyHostToDevice));

	return slot;
}

sibr::GaussianView::~GaussianView()
{
	cudaFree(pos_cuda);
	cudaFree(rot_cuda);
	cudaFree(scale_cuda);
	cudaFree(opacity_cuda);
	cudaFree(shs_cuda);
	cudaFree(view_cuda);
	cudaFree(proj_cuda);
	cudaFree(cam_pos_cuda);
	cudaFree(background_cuda);
	cudaFree(rect_cuda);

	if (uvs_cuda)          cudaFree(uvs_cuda);
	if (dU_cuda)           cudaFree(dU_cuda);
	if (dV_cuda)           cudaFree(dV_cuda);
	if (surfaceDist_cuda)  cudaFree(surfaceDist_cuda);

	if (texIdx_cuda)   cudaFree(texIdx_cuda);
	for (auto t : _texObjs)   if (t) cudaDestroyTextureObject(t);
	for (auto a : _texArrays) if (a) cudaFreeArray(a);
	if (_texObjs_cuda) cudaFree(_texObjs_cuda);
	for (auto t : _normalObjs)   if (t) cudaDestroyTextureObject(t);
	for (auto a : _normalArrays) if (a) cudaFreeArray(a);
	if (_normalObjs_cuda) cudaFree(_normalObjs_cuda);

	if (!_interop_failed) cudaGraphicsUnregisterResource(imageBufferCuda);
	else cudaFree(fallbackBufferCuda);

	glDeleteBuffers(1, &imageBuffer);

	if (geomPtr)    cudaFree(geomPtr);
	if (binningPtr) cudaFree(binningPtr);
	if (imgPtr)     cudaFree(imgPtr);

	delete _copyRenderer;
}