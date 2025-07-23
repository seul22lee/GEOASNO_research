import matplotlib.pyplot as plt
import numpy as np

data = np.random.randint(0, 2, size=(100, 100), dtype=np.uint8) * 255

fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(10, 5))  # 이 줄 중요!

img_main = ax_main.imshow(data, cmap="gray", interpolation="nearest", origin="lower")
img_zoom = ax_zoom.imshow(np.zeros((20, 20)), cmap="gray", interpolation="nearest", origin="lower")

ax_main.set_title("Main View")
ax_zoom.set_title("Zoom Region")
ax_zoom.axis("off")

# 마우스 이벤트 핸들러
def on_mouse_move(event):
    if event.inaxes != ax_main:
        return
    if event.xdata is None or event.ydata is None:
        return

    cx, cy = int(event.xdata), int(event.ydata)
    half = 10

    x1 = max(0, cx - half)
    x2 = min(data.shape[1], cx + half)
    y1 = max(0, cy - half)
    y2 = min(data.shape[0], cy + half)

    zoomed = data[y1:y2, x1:x2]
    img_zoom.set_data(zoomed)
    ax_zoom.set_title(f"Zoom @ ({cx}, {cy})")
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
plt.tight_layout()
plt.show()
