"""
render_views.py — 轉盤式渲染 armadillo.obj，取景/順序比照 data/armadillo1/train
採樣策略：固定仰角分層，每層繞方位角一圈（跟 train/ 裡連續張之間平順旋轉
的拍攝順序同一種風格），單一固定距離（不再有近距特寫層）。
輸出: data/armadillo1/renders/*.png + transforms.json (NeRF 相容格式)

安裝依賴:
    pip install pyrender trimesh pillow numpy
"""

import os, math, json
import numpy as np
from PIL import Image

import trimesh
import pyrender

# ── 設定 ────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
# 先在 Blender 裡把 armadillo.obj 染色/貼材質，存成這個檔名（連同 .mtl + 貼圖一起匯出）。
# 這個腳本只負責拍多視角照片，不再自己決定顏色。
OBJ_PATH    = os.path.join(_HERE, "armadillo_colored.obj")
OUT_DIR     = os.path.join(_HERE, "renders")
IMG_W       = 1024
IMG_H       = 1024
# 只在 mesh 沒有材質/貼圖/頂點顏色時才會用到（保底，不應該實際被觸發）。
FALLBACK_COLOR = [180/255, 100/255, 40/255, 1.0]
BG_COLOR    = [0.0, 0.0, 0.0, 0.0]               # 透明背景（alpha=0，訓練正確區分前景/背景）

# 單一固定距離：train/ 的取景（模型佔畫面比例）比對後換算約落在 radius≈2.3
# （用同一顆 FOV=60° 相機，量 train/cam000.png 前景 bbox 佔畫面高度約68%，
# 跟本腳本在不同 radius 下量出來的佔比內插得出）。不再有近距特寫層。
RADIUS = 2.3

# 仰角分層：每層固定仰角、方位角繞一圈 n_shots 張，跟 train/ 裡連續影格
# 之間平順旋轉（不是隨機跳視角）的拍攝順序同一種風格。第一層固定是 0°
# （水平視角），因為第一張畫面要對到 train/cam000.png 那個正面對稱姿勢；
# 之後往上下交錯擴展。10 層 × 60 張／層（每張約轉 6°，跟 train/ 觀察到
# 的旋轉速度接近）＝ 600 張。
ELEVATION_LAYERS = [
    (elev, 60) for elev in [0, 15, -15, 30, -30, 45, -45, 60, -60, 75]
]
# 方位角起始偏移：實測 elevation=0°、azimuth=90° 這個方向渲染出來的畫面
# 跟 train/cam000.png 的姿勢（殼頂朝鏡頭、雙臂對稱張開、尾巴垂在胯下）
# 吻合，所以每一層都從 90° 開始繞，而不是從 0° 開始。
AZIMUTH_OFFSET_DEG = 90.0
# ────────────────────────────────────────────────────────────────────────────


def look_at_pose(eye: np.ndarray,
                 target: np.ndarray = np.zeros(3),
                 world_up: np.ndarray = np.array([0, 1, 0])) -> np.ndarray:
    """建立 4×4 相機到世界的姿態矩陣（OpenGL 慣例：相機看向 -Z）。"""
    eye    = np.asarray(eye,    float)
    target = np.asarray(target, float)
    up     = np.asarray(world_up, float)

    z = eye - target
    z /= np.linalg.norm(z)

    if abs(float(np.dot(z, up))) > 0.99:   # 接近極點，換備用 up
        up = np.array([1.0, 0.0, 0.0])

    x = np.cross(up, z);  x /= np.linalg.norm(x)
    y = np.cross(z, x)

    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = eye
    return pose


def layer_viewpoints(elev_deg: float, n_shots: int, az_offset_deg: float = 0.0):
    """
    在指定仰角 elev_deg 繞一圈，產生 n_shots 個相機位置。
    座標系：Y 軸朝上。
    仰角 90° = 正上方，-90° = 正下方。
    az_offset_deg：方位角起始偏移，j=0 時的方位角 = az_offset_deg。
    """
    elev_rad = math.radians(elev_deg)
    az_offset_rad = math.radians(az_offset_deg)
    pts = []
    for j in range(n_shots):
        az_rad = az_offset_rad + 2 * math.pi * j / n_shots   # 方位角均勻繞一圈
        x = math.cos(elev_rad) * math.cos(az_rad)
        y = math.sin(elev_rad)               # Y 是上
        z = math.cos(elev_rad) * math.sin(az_rad)
        pts.append(np.array([x, y, z]))
    return pts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 載入並正規化 mesh ────────────────────────────────────────────────────
    print(f"載入模型: {OBJ_PATH}")
    mesh = trimesh.load(OBJ_PATH, force='mesh')

    # ── mesh 清理：退化面/重複面/翻轉法線的面，在打光計算時 n·l 為負，
    #    會被渲染成散布在表面上的黑點 ──────────────────────────────────────
    n_faces_before = len(mesh.faces)
    mesh.merge_vertices()
    if hasattr(mesh, "nondegenerate_faces"):   # trimesh >= 4.x
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
    else:                                       # trimesh 3.x
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)   # 統一面的朝向，修正翻轉的法線
    if len(mesh.faces) != n_faces_before:
        print(f"mesh 清理：移除 {n_faces_before - len(mesh.faces)} 個壞面")

    mesh.vertices -= mesh.bounding_box.centroid
    mesh.vertices /= mesh.bounding_sphere.primitive.radius
    print(f"Mesh 頂點數: {len(mesh.vertices):,}  面數: {len(mesh.faces):,}")

    # ── 建立 pyrender 場景 ───────────────────────────────────────────────────
    # 顏色來自 mesh 自己的貼圖/材質（Blender 染色後匯出的 .mtl+貼圖），
    # 不再用固定色蓋掉全部面。pyrender 會自動從 mesh.visual 讀出貼圖/UV。
    #
    # 注意：pyrender 對 baseColorTexture 會做 sRGB→線性 的色彩轉換，但對
    # baseColorFactor 常數不會。這張貼圖其實整張都是同一個顏色（Blender
    # 那邊就是拿純色貼圖染色的），透過材質貼圖路徑渲染出來的顏色會比同一
    # 個數值直接當 baseColorFactor 暗上將近一倍（實測：貼圖路徑 RGB≈
    # (106,62,35)，同色但走 baseColorFactor 是 (193,148,99)，跟舊版
    # renders_old_flatcolor_backup 的顏色吻合）。既然貼圖本身是純色，直接
    # 取貼圖的顏色值改設成 baseColorFactor、不透過貼圖採樣，繞開這個色偏。
    # 如果之後貼圖換成真的有花紋/漸層的圖，這裡就要改回真的用貼圖採樣，
    # 並另外處理 gamma。
    py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    for prim in py_mesh.primitives:
        has_texture_or_vcolor = prim.material.baseColorTexture is not None or prim.color_0 is not None
        if not has_texture_or_vcolor:
            print(f"警告：{OBJ_PATH} 沒有偵測到貼圖/頂點顏色，改用 FALLBACK_COLOR。"
                  f"確認 Blender 匯出時有正確帶出貼圖(.mtl+圖片)或頂點顏色。")
            prim.material.baseColorFactor = FALLBACK_COLOR
        elif prim.material.baseColorTexture is not None:
            tex_img = prim.material.baseColorTexture.source[..., :3].astype(np.float64) / 255.0
            mean_color = tex_img.reshape(-1, 3).mean(axis=0)
            prim.material.baseColorTexture = None
            prim.material.baseColorFactor = [*mean_color.tolist(), 1.0]
        prim.material.metallicFactor  = 0.0
        prim.material.roughnessFactor = 1.0   # 純漫反射：消除鏡面高光。高光是視角相關的，
                                               # 低階球諧無法表達，會被烤成漂浮亮斑並造成閃爍
        prim.material.doubleSided     = True  # 雙面受光：即使仍有翻轉法線的面也不會變黑

    # 固定方向光源（不跟著相機走）：跟 train/ 參考圖的光影是同一種風格
    # ——那批圖的亮暗區域是固定在物體本身（例如殼頂/上半身常亮、下半身/
    # 側面常暗），不是頭燈那種「永遠正面打光」的均勻效果。用 cam180.png
    # （接近俯視角度、整體均勻偏亮）和 cam200.png（側面較低角度、明顯
    # 暗上不少）比對確認：亮暗會隨「物體朝向」而非「相機朝向」變化，
    # 代表光源方向是固定在世界座標，大致從上方照下來。
    # 對比度校準：抽樣 train/*.png 算「陰影/亮部」比值(p10/p95)平均約 0.54。
    #
    # 只用單一固定光源時，仰角接近 ±75°（幾乎從物體背光面拍）的視角會
    # 完全拿不到直射光，整張圖攤平成近乎純剪影——但 train/ 裡最暗的一批
    # 圖（如 cam190.png）雖然暗，仍然看得出殼的紋理、肌肉線條等表面細節，
    # 從沒攤平過。這代表 train/ 的光源不只一個：加一盞方向大致相反、
    # 強度較弱的補光（fill light），背光側才會保留可見的陰影層次，不會
    # 死黑一片。
    KEY_LIGHT_DIR   = np.array([0.0, -1.0, 0.15])   # 主光，大致向下略偏前
    FILL_LIGHT_DIR  = np.array([0.0,  1.0, -0.15])  # 補光，大致與主光相反、強度較弱
    LIGHT_AMBIENT       = 0.4
    KEY_LIGHT_INTENSITY  = 3.0
    FILL_LIGHT_INTENSITY = 1.5

    def _rotation_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """回傳把單位向量 a 轉到單位向量 b 的旋轉矩陣。"""
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        v = np.cross(a, b)
        s = np.linalg.norm(v)
        c = np.dot(a, b)
        if s < 1e-8:
            return np.eye(3)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

    scene = pyrender.Scene(
        bg_color      = BG_COLOR,
        ambient_light = [LIGHT_AMBIENT] * 3,
    )
    scene.add(py_mesh)

    # DirectionalLight 沿著它 local -Z 方向照射，把 -Z 轉到目標世界方向。
    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=KEY_LIGHT_INTENSITY)
    key_pose = np.eye(4)
    key_pose[:3, :3] = _rotation_from_to(np.array([0.0, 0.0, -1.0]), KEY_LIGHT_DIR)
    scene.add(key_light, pose=key_pose)  # 固定 pose，不在迴圈內更新

    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=FILL_LIGHT_INTENSITY)
    fill_pose = np.eye(4)
    fill_pose[:3, :3] = _rotation_from_to(np.array([0.0, 0.0, -1.0]), FILL_LIGHT_DIR)
    scene.add(fill_light, pose=fill_pose)  # 固定 pose，不在迴圈內更新

    fov_y  = math.pi / 3.0
    camera = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=IMG_W / IMG_H)
    cam_node = scene.add(camera, pose=np.eye(4))
    renderer = pyrender.OffscreenRenderer(IMG_W, IMG_H)

    fl = 0.5 * IMG_H / math.tan(fov_y / 2)
    cx, cy = IMG_W / 2.0, IMG_H / 2.0

    # ── 轉盤式渲染：固定仰角分層，每層方位角繞一圈 ──────────────────────────
    frames = []
    total  = sum(n for _, n in ELEVATION_LAYERS)

    print(f"轉盤式採樣計畫，radius={RADIUS}（合計 {total} 張）：")
    for elev, n in ELEVATION_LAYERS:
        print(f"  elevation={elev:.1f}°  →  {n} 張")

    idx = 0
    for elev, n_shots in ELEVATION_LAYERS:
        viewpoints = layer_viewpoints(elev, n_shots, az_offset_deg=AZIMUTH_OFFSET_DEG)
        print(f"\n渲染 elevation={elev:.1f}°，共 {n_shots} 張 ...")
        for pt in viewpoints:
            eye  = pt * RADIUS
            pose = look_at_pose(eye)

            scene.set_pose(cam_node, pose)  # 只有相機動，光源固定在世界座標
            color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

            fname = f"{idx:04d}.png"
            Image.fromarray(color).save(os.path.join(OUT_DIR, fname))

            frames.append({
                "file_path": f"./renders/{fname}",
                "transform_matrix": pose.tolist(),
            })
            idx += 1

            if idx % 100 == 0:
                print(f"  {idx}/{total} 完成")

    renderer.delete()

    # ── 儲存 transforms.json ─────────────────────────────────────────────────
    transforms = {
        "fl_x": fl,
        "fl_y": fl,
        "cx":   cx,
        "cy":   cy,
        "w":    IMG_W,
        "h":    IMG_H,
        "camera_angle_x": 2 * math.atan(IMG_W / (2 * fl)),
        "camera_angle_y": fov_y,
        "frames": frames,
    }
    json_path = os.path.join(os.path.dirname(OUT_DIR), "transforms.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)

    print(f"\n完成！")
    print(f"  圖片: {OUT_DIR}/  ({len(frames)} 張 PNG)")
    print(f"  相機姿態: {json_path}")


if __name__ == "__main__":
    main()
