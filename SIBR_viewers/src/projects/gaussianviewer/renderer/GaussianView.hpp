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
#pragma once

# include "Config.hpp"
# include <core/renderer/RenderMaskHolder.hpp>
# include <core/scene/BasicIBRScene.hpp>
# include <core/system/SimpleTimer.hpp>
# include <core/system/Config.hpp>
# include <core/graphics/Mesh.hpp>
# include <core/view/ViewBase.hpp>
# include <core/renderer/CopyRenderer.hpp>
# include <core/renderer/PointBasedRenderer.hpp>
# include <core/graphics/Image.hpp>
# include <memory>
# include <core/graphics/Texture.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <functional>
#include <unordered_map>
#include <map>
#include <array>
# include "GaussianSurfaceRenderer.hpp"

namespace CudaRasterizer
{
	class Rasterizer;
}

namespace sibr {

	class BufferCopyRenderer;
	class BufferCopyRenderer2;

	class SIBR_EXP_ULR_EXPORT GaussianView : public sibr::ViewBase
	{
		SIBR_CLASS_PTR(GaussianView);

	public:

		GaussianView(const sibr::BasicIBRScene::Ptr& ibrScene, uint render_w, uint render_h,
		             const char* file, bool* message_read, int sh_degree,
		             bool white_bg = false, bool useInterop = true, int device = 0);

		void setScene(const sibr::BasicIBRScene::Ptr& newScene);

		void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override;
		void onUpdate(Input& input) override;
		void onGUI() override;

		int getCount() const { return count; }

		const std::vector<sibr::Vector3f>& getCpuPositions();

		void downloadGaussianData(
			std::vector<float>& outRot,
			std::vector<float>& outScale,
			std::vector<float>& outOpacity);

		// Upload a new per-Gaussian opacity array (sigmoid-domain, same order as GPU buffers).
		void setOpacityArray(const std::vector<float>& opacities);

		// Mapping from PLY-file index to Morton-sorted GPU index (set during loadPly).
		const std::vector<int>& getPlyToSorted() const { return _plyToSorted; }

		// Upload per-Gaussian UV + dU + dV + surfaceDist + surfacePositions + per-Gaussian
		// texture index. texIdx[i] < 0 means Gaussian i samples no texture; otherwise it
		// indexes the patch list (see registerTexture).
		void setUVsAndTexture(
			const std::vector<sibr::Vector2f>& uvs,
			const std::vector<sibr::Vector3f>& dUs,
			const std::vector<sibr::Vector3f>& dVs,
			const std::vector<float>&          surfaceDists,
			const std::vector<sibr::Vector3f>& surfacePositions,  // NEW: projected point on mesh surface
			const std::vector<int>&            texIdx);

		// Uploads a texture and returns the index to store in texIdx for the Gaussians
		// that should sample it. Textures accumulate; call clearTextures() to drop them.
		// reuseSlot >= 0 overwrites that slot instead of appending -- used while a
		// patch is still being previewed, so re-picking before confirming does not
		// leak a slot per attempt.
		int  registerTexture(sibr::Texture2DRGBA::Ptr texPtr, int reuseSlot = -1);
		void clearTextures();
		int  textureCount() const { return (int)_texObjs.size(); }

		void setNormalMapTexture(const std::vector<uint8_t>& pixels, int w, int h, int slot);
		void clearNormalMapTexture();

		void suppressGaussiansInRegion(
			const std::vector<sibr::Vector2f>& uvs,
			const sibr::Vector3f& center,
			float suppressRadius,
			const std::vector<float>& surfaceDists = {});

		void restoreOpacities();

		void updateGeometry(
			const std::vector<sibr::Vector3f>& positions,
			const std::vector<sibr::Vector4f>& rotations,
			const std::vector<sibr::Vector3f>& scales);


		const std::shared_ptr<sibr::BasicIBRScene>& getScene() const { return _scene; }

		virtual ~GaussianView() override;

		bool* _dontshow;

	protected:

		std::string currMode = "Splats";
		// Default ON: the flat mesh-bound splats this viewer targets shimmer
		// heavily at silhouettes without the AA convolution filter (orbit
		// flicker roughly halves with it, measured 2026-07-13).
		bool _antialiasing = true;
		bool _cropping = false;
		sibr::Vector3f _boxmin, _boxmax, _scenemin, _scenemax;
		char _buff[512] = "cropped.ply";

		bool _fastCulling = true;
		int _device = 0;
		int _sh_degree = 3;

		int count;
		float* pos_cuda;
		float* rot_cuda;
		float* scale_cuda;
		float* opacity_cuda;
		float* shs_cuda;
		int* rect_cuda;

		float* uvs_cuda = nullptr;
		float* dU_cuda = nullptr;
		float* dV_cuda = nullptr;
		float* surfaceDist_cuda = nullptr;

		// Multiple textures can be live at once: each painted patch keeps its own,
		// and every Gaussian carries the index of the one it samples (-1 = none).
		// A Gaussian belongs to at most one patch -- a later patch that claims it
		// simply overwrites the index. There is no cap on how many patches exist;
		// _texObjs_cuda is reallocated whenever the list grows.
		std::vector<cudaTextureObject_t> _texObjs;      // host-side, one per patch
		std::vector<cudaArray_t>         _texArrays;    // backing storage, freed with them
		std::vector<sibr::Texture2DRGBA::Ptr> _texRefs; // keeps the GL textures alive
		cudaTextureObject_t*             _texObjs_cuda = nullptr;  // device copy of _texObjs
		size_t                           _texObjsCapacity = 0;
		int*                             texIdx_cuda = nullptr;    // per-Gaussian patch index

		// Normal maps are per-patch too, kept in lockstep with _texObjs so the same
		// per-Gaussian index selects both. Without this a new patch's normal map
		// replaced the single shared one and every earlier patch started shading
		// against it.
		std::vector<cudaTextureObject_t> _normalObjs;
		std::vector<cudaArray_t>         _normalArrays;
		cudaTextureObject_t*             _normalObjs_cuda = nullptr;
		size_t                           _normalObjsCapacity = 0;

		sibr::Vector3f _uvSurfacePos = sibr::Vector3f(0.f, 0.f, 0.f);
		bool _hasUVSurface = false;

		std::vector<int> _plyToSorted;

		std::vector<sibr::Vector3f> _cpuPos;
		std::vector<float> _cpuOpacityCache;
		std::vector<sibr::Vector3f> _cpuPosCache;  // original position snapshot
		std::vector<float> _cpuRotCache;            // original rotation snapshot [w,x,y,z] * count
		std::vector<float> _cpuScaleCache;          // original scale snapshot [sx,sy,sz] * count

		GLuint imageBuffer;
		cudaGraphicsResource_t imageBufferCuda;

		size_t allocdGeom = 0, allocdBinning = 0, allocdImg = 0;
		void* geomPtr = nullptr, * binningPtr = nullptr, * imgPtr = nullptr;
		std::function<char* (size_t N)> geomBufferFunc, binningBufferFunc, imgBufferFunc;

		float* view_cuda;
		float* proj_cuda;
		float* cam_pos_cuda;
		float* background_cuda;

		float _scalingModifier = 1.0f;
		GaussianData* gData;

		bool _interop_failed = false;
		std::vector<char> fallback_bytes;
		float* fallbackBufferCuda = nullptr;
		bool accepted = false;

		std::shared_ptr<sibr::BasicIBRScene> _scene;
		PointBasedRenderer::Ptr _pointbasedrenderer;
		BufferCopyRenderer* _copyRenderer;
		GaussianSurfaceRenderer* _gaussianRenderer;
	};

} /*namespace sibr*/