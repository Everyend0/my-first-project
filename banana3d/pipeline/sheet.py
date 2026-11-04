"""Step 4 - presentation sheets: 8-view turnaround + drawing-vs-3D comparison."""
import glob
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
LABELS = {
    0: "정면 FRONT 0°",
    45: "대각선 앞-오른쪽 45°",
    90: "옆 오른쪽 SIDE 90°",
    135: "대각선 뒤-오른쪽 135°",
    180: "뒤 BACK 180°",
    225: "대각선 뒤-왼쪽 225°",
    270: "옆 왼쪽 SIDE 270°",
    315: "대각선 앞-왼쪽 315°",
}
BG = (232, 232, 232)


def font(sz):
    return ImageFont.truetype(FONT, sz)


def turnaround(name):
    rd = os.path.join(ROOT, "output", name, "renders")
    cell, pad, top = 640, 18, 64
    W = 4 * cell + 5 * pad
    H = 2 * (cell + top) + 3 * pad + 90
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)
    d.text((pad, 24), f"{name}  -  3D turnaround (8 views)  |  shadows · shading · specular highlights",
           fill=(30, 30, 30), font=font(38))
    for i, ang in enumerate(sorted(LABELS)):
        f = glob.glob(os.path.join(rd, f"{name}_{ang:03d}_*.png"))
        if not f:
            continue
        im = Image.open(f[0]).convert("RGB").resize((cell, cell), Image.LANCZOS)
        r, c = divmod(i, 4)
        x = pad + c * (cell + pad)
        y = 90 + pad + r * (cell + top + pad)
        d.text((x + 6, y + 10), LABELS[ang], fill=(20, 20, 20), font=font(32))
        sheet.paste(im, (x, y + top))
    out = os.path.join(ROOT, "output", name, f"{name}_turnaround_sheet.png")
    sheet.save(out, optimize=True)
    print(out)


def comparison(name, metrics):
    src = Image.open(os.path.join(ROOT, "src", f"{name}.png")).convert("RGB")
    ren = Image.open(os.path.join(ROOT, "output", name, "renders", f"{name}_front_ortho_match.png")).convert("RGB")
    ov = Image.open(os.path.join(ROOT, "output", name, f"{name}_overlay_check.png")).convert("RGB")
    w, h = src.size
    pad, top, bottom = 20, 70, 120
    sheet = Image.new("RGB", (3 * w + 4 * pad, h + top + bottom), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    titles = ["원본 그림 (drawing)", "3D 모델 정면 렌더 (orthographic)", "겹침 검사: 검정=일치  빨강=그림에만  파랑=3D에만"]
    for i, (im, t) in enumerate(zip([src, ren, ov], titles)):
        x = pad + i * (w + pad)
        d.text((x, 18), t, fill=(20, 20, 20), font=font(34))
        sheet.paste(im, (x, top))
    m = metrics
    txt = (f"실루엣 일치율(IoU) {m['silhouette_iou'] * 100:.2f}%   ·   외곽선 위치 평균 오차 {m['silhouette_edge_mean_px']:.2f}px "
           f"(95% 지점 {m['silhouette_edge_p95_px']:.1f}px)   ·   몸통 경계 평균 오차 {m['body_edge_mean_px']:.2f}px   ·   "
           f"잉크 일치율 {m['ink_iou'] * 100:.1f}% (검정 부분의 하이라이트 반사 포함)")
    d.text((pad, top + h + 30), txt, fill=(20, 20, 20), font=font(30))
    out = os.path.join(ROOT, "output", name, f"{name}_drawing_vs_3d.png")
    sheet.save(out, optimize=True)
    print(out)


if __name__ == "__main__":
    name = sys.argv[1]
    metrics = json.load(open(os.path.join(ROOT, "output", name, f"{name}_fidelity.json")))
    turnaround(name)
    comparison(name, metrics)
