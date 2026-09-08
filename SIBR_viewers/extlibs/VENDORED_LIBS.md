# Vendored extlibs

一般情況下 `extlibs/` 底下所有相依套件都由 CMake 在設定階段自動下載
（`sibr_addlibrary` 抓預編譯包、`sibr_gitlibrary` 從 git clone），因此不進版控。

**例外：** 以下兩個函式庫因為有「本地修改」，直接併入本 repo，不再由 CMake 重新下載：

| 目錄 | 上游 | 基準 commit | 本地修改 |
|---|---|---|---|
| `imgui/imgui/` | https://gitlab.inria.fr/sibr/libs/imgui.git | `e7f0fa31b9fa3ee4ecd2620b9951f131b4e377c6` | `imconfig.h` |
| `CudaRasterizer/CudaRasterizer/` | https://github.com/graphdeco-inria/diff-gaussian-rasterization.git | `255c532d5fd0c3d83afec0e8e49574b65ae381bd` | `cuda_rasterizer/forward.cu`、`forward.h`、`rasterizer.h`、`rasterizer_impl.cu` |

`CudaRasterizer/CudaRasterizer/third_party/glm/` 為 glm submodule 的內容
（上游 `https://github.com/g-truc/glm.git` @ `5c46b9c0`），只保留標頭，已移除 `doc/`、`test/`。

## 運作方式

`cmake/{windows,linux}/sibr_library.cmake` 的 `sibr_gitlibrary` 已加入判斷：
若 `extlibs/<root>/<source>/CMakeLists.txt` 已存在（即已隨 repo 帶入），
就直接使用本地版本，`FetchContent` 不會 clone 覆蓋，本地修改得以保留。

## 若要還原成上游乾淨版本

刪掉對應目錄後重新執行 CMake 設定即可，CMake 會依 `GIT_TAG` 重新 clone。
