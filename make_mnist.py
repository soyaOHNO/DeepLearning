import cv2
import numpy as np
import os

def order_points(pts):
    """4つの点を左上、右上、右下、左下の順に並び替える"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

class PerspectiveAdjuster:
    """マウス操作で紙の四隅（射影補正の頂点）を調整するクラス"""
    def __init__(self, img, initial_pts):
        self.img = img
        self.disp_img = img.copy()
        self.pts = initial_pts.tolist() if isinstance(initial_pts, np.ndarray) else initial_pts
        self.dragging_idx = None
        self.drag_thresh = 30 # クリック判定のピクセル閾値

    def draw(self):
        self.disp_img = self.img.copy()
        pts_arr = np.array(self.pts, dtype=np.int32)
        
        # 枠線を描画（緑）
        cv2.polylines(self.disp_img, [pts_arr], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # 四隅の頂点に丸を描画（赤）
        for pt in self.pts:
            cv2.circle(self.disp_img, tuple(map(int, pt)), 8, (0, 0, 255), -1)

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 最も近い頂点を探す
            dists = [np.sqrt((px - x)**2 + (py - y)**2) for px, py in self.pts]
            min_idx = np.argmin(dists)
            if dists[min_idx] < self.drag_thresh:
                self.dragging_idx = min_idx

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_idx is not None:
                self.pts[self.dragging_idx] = [x, y]
                self.draw()
                cv2.imshow(param, self.disp_img)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_idx = None
            self.pts = order_points(np.array(self.pts)).tolist()
            self.draw()
            cv2.imshow(param, self.disp_img)

class GridAdjuster:
    """マウス操作でグリッド線を調整するクラス"""
    def __init__(self, img, rows=10, cols=10):
        self.img = img
        self.disp_img = img.copy()
        self.h, self.w = img.shape[:2]
        
        # 初期グリッド線
        self.x_lines = [int(x) for x in np.linspace(0, self.w, cols + 1)]
        self.y_lines = [int(y) for y in np.linspace(0, self.h, rows + 1)]
        
        self.dragging_line = None
        self.drag_thresh = 15

    def draw(self):
        self.disp_img = self.img.copy()
        for x in self.x_lines:
            cv2.line(self.disp_img, (x, 0), (x, self.h), (0, 0, 255), 1)
        for y in self.y_lines:
            cv2.line(self.disp_img, (0, y), (self.w, y), (0, 0, 255), 1)

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            min_dist_x = min([(abs(lx - x), i) for i, lx in enumerate(self.x_lines)])
            min_dist_y = min([(abs(ly - y), i) for i, ly in enumerate(self.y_lines)])

            if min_dist_x[0] < self.drag_thresh and min_dist_x[0] <= min_dist_y[0]:
                self.dragging_line = ('x', min_dist_x[1])
            elif min_dist_y[0] < self.drag_thresh:
                self.dragging_line = ('y', min_dist_y[1])

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_line:
                axis, idx = self.dragging_line
                if axis == 'x':
                    self.x_lines[idx] = max(0, min(x, self.w - 1))
                else:
                    self.y_lines[idx] = max(0, min(y, self.h - 1))
                self.draw()
                cv2.imshow(param, self.disp_img)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_line = None
            self.x_lines.sort()
            self.y_lines.sort()
            self.draw()
            cv2.imshow(param, self.disp_img)

# ==========================================
# 1. 画像読み込みとサイズ縮小・初期輪郭の自動推測
# ==========================================
img = cv2.imread("HandWrite.JPEG")
if img is None:
    print("Error: 画像が見つかりません。")
    exit()

# ★ここで画面に収まるサイズにリサイズする（例：高さを800pxに固定）
max_display_height = 800
h, w = img.shape[:2]
if h > max_display_height:
    scale = max_display_height / h
    img = cv2.resize(img, (int(w * scale), max_display_height))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 輪郭が見つかった場合はそれを初期値に、見つからなければ画像全体の少し内側を初期値にする
if contours:
    paper = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(paper)
    initial_pts = cv2.boxPoints(rect).astype(np.float32)
else:
    h, w = img.shape[:2]
    initial_pts = np.array([[w*0.1, h*0.1], [w*0.9, h*0.1], [w*0.9, h*0.9], [w*0.1, h*0.9]], dtype=np.float32)

initial_pts = order_points(initial_pts)

# ==========================================
# 2. ステップ1: 射影補正（四隅）の調整UI
# ==========================================
print("【ステップ1】 紙の四隅にある赤い丸をドラッグして位置を合わせてください。")
print("調整が終わったら ENTERキー を押してください。")

win_perspective = "Step 1: Adjust Corners (ENTER to confirm)"
cv2.namedWindow(win_perspective)

persp_adjuster = PerspectiveAdjuster(img, initial_pts)
persp_adjuster.draw()
cv2.setMouseCallback(win_perspective, persp_adjuster.mouse_event, param=win_perspective)

while True:
    cv2.imshow(win_perspective, persp_adjuster.disp_img)
    if cv2.waitKey(1) == 13: # ENTER key
        break
cv2.destroyWindow(win_perspective)

# 確定した四隅の点で射影変換を実行
final_pts = np.array(persp_adjuster.pts, dtype="float32")
(tl, tr, br, bl) = final_pts

widthA = np.linalg.norm(br - bl)
widthB = np.linalg.norm(tr - tl)
heightA = np.linalg.norm(tr - br)
heightB = np.linalg.norm(tl - bl)

maxW = int(max(widthA, widthB))
maxH = int(max(heightA, heightB))

dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
M = cv2.getPerspectiveTransform(final_pts, dst)
warp_color = cv2.warpPerspective(img, M, (maxW, maxH)) # ← img (カラー画像) を使う！

# ==========================================
# 3. ステップ2: グリッド調整UI
# ==========================================
print("【ステップ2】 赤いグリッド線をドラッグしてマスの境界に合わせてください。")
print("調整が終わったら ENTERキー を押してください。")

rows, cols = 10, 10

win_grid = "Step 2: Adjust Grid (ENTER to confirm)"
cv2.namedWindow(win_grid)

grid_adjuster = GridAdjuster(warp_color, rows=rows, cols=cols)
grid_adjuster.draw()
cv2.setMouseCallback(win_grid, grid_adjuster.mouse_event, param=win_grid)

while True:
    cv2.imshow(win_grid, grid_adjuster.disp_img)
    if cv2.waitKey(1) == 13: 
        break
cv2.destroyWindow(win_grid)

# ==========================================
# 4. 確定したグリッドでのセル分割・MNIST生成
# ==========================================
x_lines = grid_adjuster.x_lines
y_lines = grid_adjuster.y_lines

for i in range(10):
    os.makedirs(f"dataset/{i}", exist_ok=True)

count = [0] * 10

# マージン（セルの内側何ピクセルを切り取るか。少し残った境界線を避けるため）
margin = 5 

for r in range(rows):
    for c in range(cols):
        y1, y2 = y_lines[r], y_lines[r+1]
        x1, x2 = x_lines[c], x_lines[c+1]

        # セルが小さすぎる・つぶれている場合はスキップ
        if y2 - y1 <= margin*2 or x2 - x1 <= margin*2:
            continue

        # カラー画像のまま切り抜く（少し内側）
        cell_color = warp_color[y1+margin : y2-margin, x1+margin : x2-margin]

        # 【改善ポイント】B（青）チャンネルだけを取り出す（OpenCVはBGRの順なので0番目）
        b_channel = cell_color[:, :, 0]
        cv2.imshow("B Channel", b_channel)
        cv2.waitKey(100) # 100msだけ表示して確認

        # 二値化
        _, th = cv2.threshold(b_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 輪郭検出
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 小さいノイズ輪郭を除去
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]

        if len(contours) == 0:
            continue

        # 最大の輪郭（文字）を見つける
        digit_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(digit_cnt)
        digit = th[y:y+h, x:x+w]

        # 正方形パディング
        size = max(w, h)
        canvas = np.zeros((size, size), np.uint8)
        canvas[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = digit

        # MNISTサイズ（20x20にリサイズ後、28x28の中央に配置）
        digit_resized = cv2.resize(canvas, (20, 20))
        final = np.zeros((28, 28), np.uint8)
        final[4:24, 4:24] = digit_resized

        # 重心中央寄せ
        m = cv2.moments(final)
        if m["m00"] != 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            shiftx = 14 - cx
            shifty = 14 - cy
            M_affine = np.float32([[1, 0, shiftx], [0, 1, shifty]])
            final = cv2.warpAffine(final, M_affine, (28, 28))

        # 列番号(c)をラベルにする
        label = c % 10 
        
        count[label] += 1
        cv2.imwrite(f"dataset/{label}/{label}_{count[label]}.png", final)

print("Finished: データセットの生成が完了しました。")