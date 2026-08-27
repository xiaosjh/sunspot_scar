# GUI Read FITS

这是一个面向本地太阳 FITS 序列的第一版小工具。目标是逐步做成类似 JHelioviewer 的轻量浏览器。

## 运行

在 PowerShell 里进入本目录：

```powershell
cd C:\Learning\PHD2nd\sunspotscar\program\GUI_read_fits
python app.py
```

如果缺少依赖，可以安装：

```powershell
python -m pip install -r requirements.txt
```

其中 MP4 导出依赖 `imageio-ffmpeg`；如果暂时没安装，也可以先导出 GIF。

## 当前功能

- 默认读取 `C:\Learning\PHD2nd\sunspotscar\data\X`
- 自动列出事件文件夹
- 自动列出事件下的波段/数据目录
- `hmi.B_720s` 下的 `Br_sub.fits` 和 `Br.fits` 会作为单帧数据源单独列出
- 默认不勾选通道，可以用 `Sub maps`、`All`、`Clear` 快速选择
- `Load selected` 放在左侧 Channels 区域，选择通道后可以就近加载
- 多面板无缝拼接显示多个波段
- AIA 131/211/304 优先使用 SunPy 的 AIA 色表
- HMI continuum 和 magnetogram 使用灰度显示
- HMI continuum (`hmi.Ic_*`) 显示时会旋转 180 度，以匹配其 `CROTA2` 接近 180 度的 FITS 姿态
- HMI continuum 使用真实 min/max 显示范围，不做百分位裁剪
- `Display` 面板可以按 layer 调整 JHelioviewer 风格的 Levels：两个把手都在 `-100 - 1000` 范围内，左把手始终小于右把手；AIA 默认 `-100 / 500`，HMI 白光默认 `-100 / 100`
- Levels 是一条更宽的轨道上的两个把手，右侧显示当前 low/high 百分比；拖动把手只移动固定基准范围内的端点，不会每次重新计算百分位
- 每个已加载波段都有独立 Levels；先在 `Layer` 下拉框选波段，再调整 Levels
- AIA 默认使用 `asinh` 拉伸，接近 SunPy AIA Map 的默认显示方式；也可以切到 `log` 或 `linear`
- Display 面板会按当前 layer 类型显示相关控件：AIA 显示 AIA stretch，Br 图显示 Br min/max，普通 HMI 灰度图不显示这些无关项
- AIA 亮区过曝时可以把当前 layer 的右侧 `Levels` 把手继续往 `1000` 调，相当于提高白点、压住亮区
- 拖动 Levels 把手时会直接重映射当前帧缓存图像，不重新读取 FITS
- `Display` 面板可以手动设置 `Br.fits` / `Br_sub.fits` 的显示上下限，单位按 FITS 数据本身的高斯值处理，默认 `-800, 800`
- 时间和波段信息叠加在每张图左下角
- 用底部滑条快速浏览时间序列
- 底部 timeline 固定显示，不会被主画布挤掉
- 加载预览时底部进度条显示 `n/total`
- 拖动 timeline 时优先只换预览缓存；松开滑条后才为当前帧补缩放高清缓存
- 调整 Levels 后，后续拖动 timeline 会检查每帧 RGB 缓存对应的显示参数，不匹配就用 scalar 缓存立即重映射
- 上边栏和左侧导出面板都可以设置 FPS，播放和导出共用同一个数值
- 可以用键盘左/右方向键逐帧后退/前进
- `Play/Pause` 按 FPS 播放
- 可设置 start/end/fps 并导出 MP4 或 GIF
- 导出视频会使用当前每个 layer 的显示设置、当前缩放/平移视野，并包含已显示的 annotation 线段
- 加载时会建立显示缓存，播放和拖动时优先使用缓存，`Cache px` 控制缓存图像最长边像素数，默认 `4096`
- 如果 `Cache px >= Zoom px`，放大和平移也直接复用加载时建立的高清缓存，不会再逐帧现场补高清；只有 `Cache px` 小于 `Zoom px` 时，放大后当前帧才会按 `Zoom px` 补高清缓存
- 预览生成会先降采样再套色表，整张 4096 图加载会明显快于旧版
- `Workers` 控制并行生成预览的 worker 数，默认最多 12，可以手动调到 16 或更高试验
- `Processes` 默认开启，用多进程生成预览，更容易吃到多核心 CPU
- `Disk cache` 会把当前 Event 的预览图保存到 `.preview_cache`，同一 Event 内重新选择波段时可以复用；切换 Event 或正常关闭软件时会自动清空
- 鼠标滚轮缩放、左键拖拽平移、双击或 `Reset view` 恢复全图
- 顶部 `Draw` 可以选择 `Line`（直线）或 `Curve`（沿鼠标轨迹的曲线），默认是 `Line`；按住 `Shift` 并按住左键拖动进行绘制，松开左键或 `Shift` 都会立即结束当前批注。批注会同步显示在所有波段中，并跟随缩放/平移移动
- `Shift + Backspace` 或 `Shift + Delete` 会删除上一条 annotation 批注
- 上边栏 `Annotation` 控制批注显示/隐藏，`Clear lines` 清空批注
- 左侧 `Annotation` 面板可以调整线宽和颜色，默认线宽 `0.5`
- 左侧边栏可以滚动，也可以拖动中间分隔条调整宽度，避免窗口较小时遮挡下面的控制项
- 拖拽平移用屏幕像素位移计算，并只更新现有坐标轴范围，不重建整张图，交互更顺滑
- 缩放后如果已有缓存不够清晰，才会按 `Zoom px` 给当前帧补高清缓存，不需要重载整个时间序列
- `Cache px` 回车或失焦后会重建当前事件预览缓存；`Zoom px` 回车或失焦后会清掉高清缓存并对当前帧立即生效
- 图像保持原始长宽比，不再被强制拉伸成面板比例
- 多面板显示会按图像宽高比紧凑居中，减少面板之间的大黑缝
- 同一 Event 内已经加载过的波段会保留在内存缓存中，取消勾选后再勾回可直接复用；切换 Event 时清空
- 加载时除 RGB 预览外还保留降采样后的 scalar 数据，用于快速调整 levels

## 常用修改位置

- 默认勾选通道：现在默认读取 `131sub_map`、`304sub_map`、`Br_sub.fits`、`hmi.Ic_45s` 子图；如需修改，可改 `app.py` 顶部的 `DEFAULT_CHANNEL_HINTS`，如果要改变自动匹配规则，再改 `is_default_selected_channel()`。
- Br 默认显示范围：修改 `app.py` 顶部的 `DEFAULT_BR_MIN_G` 和 `DEFAULT_BR_MAX_G`。
- 默认批注模式：修改 `app.py` 顶部的 `DEFAULT_ANNOTATION_MODE`，可选 `"Line"` 或 `"Curve"`。
- AIA 色表：修改 `channel_cmap()`；当前优先使用 SunPy，缺少 SunPy 时使用内置 fallback 色表。
- HMI continuum 方向：修改 `orient_image_for_display()`。
- 显示范围策略：修改 `base_display_range()`、`display_limits()` 和 `apply_stretch()`；当前 AIA 先估计基准范围，再用 GUI Levels 移动 low/high 端点，最后做 `asinh`/`log`/`linear` 拉伸。
- 显示参数缓存：RGB 磁盘缓存会带 `DisplaySettings.cache_key()`，内存里同时保留 scalar 缓存，方便不同 layer 快速重映射。
- 快速 levels 重映射：修改 `recolor_current_frame()` 和 `colorize_preview_data()`。
- Annotation 直线/曲线：修改 `AnnotationStroke`、`redraw_annotations()` 和鼠标事件处理函数。
- 左下角文字：修改 `overlay_text()` 和 `channel_display_name()`。
- 播放流畅度和清晰度：调整界面里的 `Cache px`。数值越大越清楚，但加载更慢、占用内存更多。
- 缩放清晰度：调整 `Zoom px`。默认 `4096`，最清楚但更慢。
- 加载速度：调大 `Workers` 可利用更多 CPU/磁盘并发；如果磁盘压力大或电脑变卡，可以调小，或取消 `Processes` 改用线程。
- 磁盘缓存：存放在 `GUI_read_fits/.preview_cache`，仅保留当前 Event；切换 Event 或正常关闭软件时由 `clear_event_cache(clear_disk=True)` 自动清空。
- 内存缓存：同一 Event 内由 `preview_cache` 保留，切换 Event 或关闭软件时同步清空。

## 下一步建议

1. 增加亮度范围调节和 log 显示。
2. 叠加经纬网或太阳坐标网格。
3. 增加事件标注、截图、批量导出。
4. 增加真正的局部 tile 缓存，只为放大区域读取/渲染高清图；当前版本仍是当前帧整图高清预览缓存。
