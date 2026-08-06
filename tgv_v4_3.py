import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import pandas as pd
import json, io, os, re, ctypes

# ══════════════════════════════════════════════════════════════════════
#  디자인 시스템
# ══════════════════════════════════════════════════════════════════════
BG0   = "#0d0f14"
BG1   = "#13161e"
BG2   = "#1c2030"
BG3   = "#242840"
BG4   = "#2d3350"
FG    = "#d4daf0"
FG2   = "#606880"
FG3   = "#8899bb"
AC    = "#3d8ef0"
AC2   = "#1a6fd8"
GR    = "#1fba6e"
GR2   = "#178f53"
OR    = "#e8602a"
OR2   = "#c04d20"
RD    = "#d94060"
BD    = "#252c40"
BD2   = "#1e2438"
EXCL_COLOR = (0, 80, 255)   # 수동 제외 영역 표시색 (BGR)
HOLE_COLOR = (220, 80, 0)   # 검출된 홀 표시색 (BGR)
EDGE_HOLE_COLOR = (0, 180, 255)  # 가장자리 반원 표시색 (BGR)

_F    = "Segoe UI"
F_XS  = (_F, 8)
F_SM  = (_F, 9)
F_SMB = (_F, 9,  "bold")
F_MD  = (_F, 10)
F_MDB = (_F, 10, "bold")
F_LG  = (_F, 11, "bold")
F_XL  = (_F, 13, "bold")

# ══════════════════════════════════════════════════════════════════════
#  1. 데이터 I/O
# ══════════════════════════════════════════════════════════════════════
def save_recipe(path, d):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=4, ensure_ascii=False)

def load_recipe(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ══════════════════════════════════════════════════════════════════════
#  2. 비전 처리 엔진
# ══════════════════════════════════════════════════════════════════════
class Vision:
    def __init__(self):
        self.px_per_mm = 2600.0
        self.px_per_um = 2.6

    def update(self, p):
        px_per_mm = p.get('px_per_mm', 0)
        if px_per_mm > 0:
            self.px_per_mm = px_per_mm
            self.px_per_um = px_per_mm / 1000.0

    # ── 홀 검출 ─────────────────────────────────────────────────────
    @staticmethod
    def _fit_circle(pts):
        """최소제곱법 대수적 원 피팅: 엣지 픽셀 좌표 → (cx, cy, r)"""
        x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)
        A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
        b = x ** 2 + y ** 2
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, d = sol
        r = float(np.sqrt(abs(d + cx ** 2 + cy ** 2)))
        return float(cx), float(cy), r

    def find_all_holes(self, img, rim_thr=120, min_area=1500, min_r=50, max_r=300):
        """
        HoughCircles로 홀 중심/반경 검출.
        - 이미지를 패딩해서 중심이 이미지 밖에 있는 반원·부분 홀도 감지
        - 가장자리 홀: Canny 엣지 + 최소제곱 원 피팅으로 정밀도 향상
        반환: [{'center':(x,y), 'r':int, 'is_edge':bool}, ...]
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # ── 패딩: 이미지 밖에 중심이 있는 홀도 HoughCircles가 탐색할 수 있도록
        pad = min(max_r, 180)
        bg_val = int(np.median(blurred))
        padded = cv2.copyMakeBorder(blurred, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=bg_val)

        raw = None
        for _p2 in [30, 25, 20, 15, 12, 10]:
            raw = cv2.HoughCircles(
                padded, cv2.HOUGH_GRADIENT, dp=1,
                minDist=int(min_r * 1.8),
                param1=60, param2=_p2,
                minRadius=min_r, maxRadius=max_r,
            )
            if raw is not None:
                break

        if raw is None:
            return []

        # Canny 엣지 (가장자리 홀 원 정밀 피팅용)
        canny = cv2.Canny(blurred, 30, 80)
        ey, ex = np.where(canny > 0)
        edge_pts = np.column_stack([ex, ey]) if len(ey) >= 10 else None

        # 타이트 이진화 (내부 홀 반경 재계산용)
        _tight_thr = max(50, int(rim_thr * 0.65))
        _, _m_tight = cv2.threshold(gray, _tight_thr, 255, cv2.THRESH_BINARY_INV)

        holes = []
        for (xp, yp, r_raw) in np.round(raw[0]).astype(int):
            # 패딩 좌표 → 원본 좌표
            x = int(xp - pad)
            y = int(yp - pad)

            if r_raw < min_r or r_raw > max_r:
                continue
            # 원이 이미지와 전혀 겹치지 않으면 제외
            if x + r_raw < 0 or x - r_raw > w or y + r_raw < 0 or y - r_raw > h:
                continue

            is_edge = (x - r_raw < 0 or x + r_raw > w or
                       y - r_raw < 0 or y + r_raw > h)

            r_use = float(r_raw)
            cx_use, cy_use = float(x), float(y)

            if is_edge and edge_pts is not None:
                # 가장자리 홀: Canny 엣지에서 원 주변 픽셀만 추출 → 최소제곱 원 피팅
                dists = np.sqrt((edge_pts[:, 0] - x) ** 2 +
                                (edge_pts[:, 1] - y) ** 2)
                tol = max(18, r_raw * 0.13)
                rim_pts = edge_pts[np.abs(dists - r_raw) < tol]
                if len(rim_pts) >= 12:
                    try:
                        fx, fy, fr = self._fit_circle(rim_pts)
                        if min_r * 0.7 <= fr <= max_r * 1.3:
                            cx_use, cy_use, r_use = fx, fy, fr
                    except Exception:
                        pass

            elif not is_edge:
                # 내부 홀: tight 마스크로 반경 재계산
                _roi = np.zeros(gray.shape, np.uint8)
                cv2.circle(_roi, (int(x), int(y)), int(r_raw), 255, -1)
                _m_roi = cv2.bitwise_and(_m_tight, _m_tight, mask=_roi)
                _tc, _ = cv2.findContours(_m_roi, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
                if _tc:
                    _best = max(_tc, key=cv2.contourArea)
                    if cv2.contourArea(_best) > 200:
                        _, _tr = cv2.minEnclosingCircle(_best)
                        if _tr > min_r * 0.5:
                            r_use = _tr

            holes.append({'center': (int(round(cx_use)), int(round(cy_use))),
                          'r': int(round(r_use)),
                          'is_edge': is_edge})

        # NMS: 중복 감지 제거
        if len(holes) > 1:
            holes.sort(key=lambda hh: hh['r'], reverse=True)
            kept = []
            for hole in holes:
                cx, cy = hole['center']
                dup = False
                for k in kept:
                    kx, ky = k['center']
                    dist = np.sqrt((cx - kx) ** 2 + (cy - ky) ** 2)
                    if dist < (hole['r'] + k['r']) * 0.6:
                        dup = True
                        break
                if not dup:
                    kept.append(hole)
            holes = kept
        return holes

    def make_hole_mask(self, shape, holes, margin, exclusions=None):
        """
        홀 마스크 생성: 검출된 홀(r + margin) + 수동 제외 영역을 0으로 채움.
        v4.3: 로컬 잔차법이 헤일로를 자동 처리하므로 고정 margin만 사용.
        반환: 255=검사영역, 0=제외
        """
        m = np.ones(shape, np.uint8) * 255
        for hole in (holes or []):
            cx, cy = hole['center']
            r_excl = max(0, hole['r'] + margin)
            cv2.circle(m, (cx, cy), r_excl, 0, -1)
        for exc in (exclusions or []):
            cv2.circle(m, exc['center'], exc['r'], 0, -1)
        return m

    # ── 방사형 halo 보정 ────────────────────────────────────────────
    def _radial_halo_correct(self, gray, holes, halo_px=150):
        """
        각 홀 중심에서의 방사 거리별 평균 밝기를 계산하고
        halo 구간(림 바깥 ~ halo_px)을 균일하게 보정.
        B 영역을 제외하지 않고 배경 편차만 제거.
        """
        out = gray.astype(np.float32)
        h, w = gray.shape

        for hole in holes:
            cx, cy = hole['center']
            r_rim = hole['r']
            r_in  = r_rim + 3
            # 가장자리 홀도 보정되도록 이미지 경계 클립 제거
            # zone_mask는 이미지 내부 픽셀만 참조하므로 경계 초과분은 자동 무시됨
            r_out = int(r_rim + halo_px)
            if r_out <= r_in + 10:
                continue

            # 픽셀별 방사 거리 (정수 반경으로 bin 분류)
            y_idx, x_idx = np.ogrid[:h, :w]
            dist_i = np.sqrt((x_idx - cx) ** 2 +
                             (y_idx - cy) ** 2).astype(np.int32)

            zone_mask = (dist_i >= r_in) & (dist_i < r_out)
            d_flat = dist_i.ravel()
            g_flat = gray.ravel().astype(np.float64)
            z_flat = zone_mask.ravel()

            n_bins = r_out + 1
            sums = np.bincount(d_flat[z_flat],
                               weights=g_flat[z_flat], minlength=n_bins)
            cnts = np.bincount(d_flat[z_flat], minlength=n_bins)

            ri = np.arange(r_in, r_out)
            valid = cnts[ri] > 0
            means = np.where(valid,
                             sums[ri] / np.maximum(cnts[ri], 1),
                             np.nan)

            # NaN 보간
            nans = np.isnan(means)
            if nans.any():
                xi = np.where(~nans)[0]
                if len(xi) < 2:
                    continue
                means[nans] = np.interp(np.where(nans)[0], xi, means[xi])

            # A 영역(먼 쪽 35%) 평균을 균일 배경 기준으로 사용
            far0 = max(0, int(len(means) * 0.65))
            bg_ref = float(np.nanmean(means[far0:]))

            # 반경 → 보정량 배열
            corr_lut = np.zeros(n_bins, dtype=np.float32)
            corr_lut[ri] = (bg_ref - means).astype(np.float32)

            # 픽셀에 적용
            out_flat = out.ravel()
            idx_zone = np.where(z_flat)[0]
            out_flat[idx_zone] += corr_lut[d_flat[idx_zone]]

        return np.clip(out, 0, 255).astype(np.uint8)

    # ── 전처리 ──────────────────────────────────────────────────────
    def preview(self, img, holes, p, exclusions=None):
        self.update(p)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        br, ct = p.get('br', 0), p.get('ct', 1.0)
        if br != 0 or ct != 1.0:
            g = cv2.convertScaleAbs(g, alpha=ct, beta=br)

        bl = p.get('bl', 0)
        if bl > 0:
            g = cv2.GaussianBlur(g, (bl | 1, bl | 1), 0)

        # ── 로컬 잔차 검출 ────────────────────────────────────────────
        # 큰 가우시안으로 로컬 배경 추정 → 잔차(배경 - 원본)가 양수인 픽셀 = 어두운 점
        # 효과: 헤일로/조명 불균일이 배경 추정에 포함되어 자동 제거
        bg_k = p.get('bg_kernel', 71)
        bg_k = max(11, bg_k | 1)          # 항상 홀수, 최소 11
        local_bg = cv2.GaussianBlur(g.astype(np.float32), (bg_k, bg_k), bg_k / 3.0)
        residual = local_bg - g.astype(np.float32)
        residual_u8 = np.clip(residual, 0, 255).astype(np.uint8)
        res_th = p.get('res_th', 25)
        _, b = cv2.threshold(residual_u8, res_th, 255, cv2.THRESH_BINARY)

        # 가장자리 여백 마스크
        margin_edge = p.get('margin', 0)
        if margin_edge > 0:
            ih, iw = b.shape
            edge_mask = np.zeros((ih, iw), np.uint8)
            edge_mask[margin_edge:ih - margin_edge,
                      margin_edge:iw - margin_edge] = 255
            b = cv2.bitwise_and(b, b, mask=edge_mask)

        # 홀 + 수동 제외 영역 마스크 (고정 margin)
        hole_margin = p.get('hole_margin', 30)
        hm = self.make_hole_mask(b.shape, holes, hole_margin, exclusions=exclusions)
        b = cv2.bitwise_and(b, b, mask=hm)

        return b

    # ── 분석 ────────────────────────────────────────────────────────
    def analyze(self, img, binary, p, holes=None, exclusions=None):
        self.update(p)
        cs, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        out = img.copy()
        rows, total, idx = [], 0, 1
        min_area = p.get('min_area', 1.0)
        hole_margin = p.get('hole_margin', 5)
        CLR_P = (0, 220, 60)

        for c in cs:
            area = float(cv2.contourArea(c))
            if area < min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            total += 1
            cv2.drawContours(out, [c], -1, CLR_P, 1)
            rows.append({
                'no': idx,
                'x_px': cx, 'y_px': cy,
                'x_mm': cx / self.px_per_mm,
                'y_mm': cy / self.px_per_mm,
                'cnt': 1,
            })
            idx += 1

        # 홀 표시: 내원(검출 림) + 외원(실제 마스크 경계)
        for hole in (holes or []):
            clr = EDGE_HOLE_COLOR if hole['is_edge'] else HOLE_COLOR
            # 검출된 림 경계 (얇은 선)
            cv2.circle(out, hole['center'], hole['r'], clr, 1)
            # 실제 마스크 경계 (굵은 선) — 사용자가 보는 제외 경계
            cv2.circle(out, hole['center'], hole['r'] + hole_margin, clr, 2)
            if hole['is_edge']:
                cv2.putText(out, "E",
                            (hole['center'][0] - 8, hole['center'][1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, EDGE_HOLE_COLOR, 1)

        # 수동 제외 영역 표시
        for exc in (exclusions or []):
            cv2.circle(out, exc['center'], exc['r'], EXCL_COLOR, 2)
            cv2.line(out,
                     (exc['center'][0] - exc['r'], exc['center'][1]),
                     (exc['center'][0] + exc['r'], exc['center'][1]),
                     EXCL_COLOR, 1)

        return out, rows, total


# ══════════════════════════════════════════════════════════════════════
#  3. UI 구성
# ══════════════════════════════════════════════════════════════════════
class UI:
    def __init__(self, root, cb):
        self.root = root
        self.cb   = cb
        self._cache    = {'lbl_1': (None, False),
                          'lbl_2': (None, True),
                          'lbl_3': (None, False)}
        self.ratios    = {'lbl_1': 1.0, 'lbl_2': 1.0, 'lbl_3': 1.0}
        self.offsets   = {'lbl_1': (0, 0), 'lbl_2': (0, 0), 'lbl_3': (0, 0)}
        self.tab_lbs   = {}
        self._ctx_key  = None

        # 변수
        self.v_mag_sel   = tk.StringVar(value="")
        self.v_px_per_mm = tk.DoubleVar(value=0.0)
        self.v_pdiam     = tk.StringVar(value="1.5")
        self.v_br        = tk.IntVar(value=0)
        self.v_ct        = tk.DoubleVar(value=1.0)
        self.v_bl        = tk.IntVar(value=0)
        self.v_bg_kernel = tk.IntVar(value=71)     # 로컬 배경 추정 커널 (항상 홀수)
        self.v_res_th    = tk.IntVar(value=25)    # 잔차 임계값
        self.v_margin    = tk.IntVar(value=0)
        self.v_hole_thr  = tk.IntVar(value=120)   # 홀 감지 림 임계값
        self.v_hole_mg   = tk.IntVar(value=30)    # 홀 마스크 여유(px)
        self.v_excl_r    = tk.IntVar(value=80)    # 수동 제외 원 반경
        self.v_draw_mode = tk.BooleanVar(value=False)
        self.lbl_th_hint = None   # 임계값 힌트 레이블
        self._mag_btns   = {}
        self.lbl_min_size = None
        self.lbl_excl_count = None
        self.btn_draw_mode  = None

        self._ctx = tk.Menu(root, tearoff=0, bg=BG2, fg=FG,
                            activebackground=AC, activeforeground="white")
        self._ctx.add_command(label="클립보드에 이미지 복사",
                              command=self._copy_clip)
        self._build()

    # ── 헬퍼 ──────────────────────────────────────────────────────────
    def _frame(self, parent, bg=BG1, **kw):
        return tk.Frame(parent, bg=bg, **kw)

    def _label(self, parent, text, font=F_SM, fg=FG3, bg=BG1, **kw):
        return tk.Label(parent, text=text, font=font, fg=fg, bg=bg, **kw)

    def _btn(self, parent, text, cmd, bg=BG3, fg=FG, font=F_SMB, pad=(12, 6)):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg=fg, font=font,
                         relief="flat", bd=0,
                         padx=pad[0], pady=pad[1],
                         cursor="hand2",
                         activebackground=BG4, activeforeground=FG)

    def _sep(self, parent, orient="v"):
        if orient == "v":
            tk.Frame(parent, bg=BD2, width=1).pack(
                side=tk.LEFT, fill=tk.Y, padx=10)
        else:
            tk.Frame(parent, bg=BD2, height=1).pack(fill=tk.X, pady=6)

    def _slider(self, parent, label, var, lo, hi, res, cmd, width=120):
        wrap = self._frame(parent, bg=BG1)
        wrap.pack(side=tk.LEFT, padx=8, pady=2)
        hdr = self._frame(wrap, BG1)
        hdr.pack(fill=tk.X)
        self._label(hdr, label, F_XS, FG2, BG1).pack(side=tk.LEFT)
        tk.Label(hdr, textvariable=var, font=F_XS,
                 fg=AC, bg=BG1, width=6, anchor="e").pack(side=tk.RIGHT)
        row = self._frame(wrap, BG1)
        row.pack()
        btn_cfg = dict(font=(_F, 8), bg=BG3, fg=FG3, relief="flat",
                       padx=4, pady=1, cursor="hand2",
                       activebackground=BG4, activeforeground=AC)
        tk.Button(row, text="‹",
                  command=lambda: self._step(var, res, -1, lo, hi, cmd),
                  **btn_cfg).pack(side=tk.LEFT)
        tk.Scale(row, from_=lo, to=hi, resolution=res, variable=var,
                 orient=tk.HORIZONTAL, length=width, showvalue=False,
                 bg=BG1, fg=FG, troughcolor=BG3,
                 activebackground=AC,
                 highlightthickness=0, bd=0,
                 command=lambda x: cmd()).pack(side=tk.LEFT)
        tk.Button(row, text="›",
                  command=lambda: self._step(var, res, +1, lo, hi, cmd),
                  **btn_cfg).pack(side=tk.LEFT)
        return wrap

    @staticmethod
    def _step(var, res, d, lo, hi, cmd):
        v = round(var.get() + res * d, 4)
        var.set(max(lo, min(hi, v)))
        cmd()

    # ── 메인 빌드 ─────────────────────────────────────────────────────
    def _build(self):
        self.root.configure(bg=BG0)
        s = ttk.Style()
        s.theme_use('clam')
        s.configure("App.TNotebook",
                    background=BG1, borderwidth=0, tabmargins=0)
        s.configure("App.TNotebook.Tab",
                    background=BG2, foreground=FG2, font=F_SM, padding=[12, 6])
        s.map("App.TNotebook.Tab",
              background=[("selected", AC2)], foreground=[("selected", "white")])
        s.configure("App.TCombobox",
                    fieldbackground=BG3, background=BG3,
                    foreground=FG, arrowcolor=FG3, borderwidth=0, padding=4)
        s.map("App.TCombobox",
              fieldbackground=[("readonly", BG3)], foreground=[("readonly", FG)])
        s.configure("App.Treeview",
                    background=BG2, fieldbackground=BG2,
                    foreground=FG, font=F_SM, rowheight=26, borderwidth=0)
        s.configure("App.Treeview.Heading",
                    background=BG0, foreground=FG3, font=F_SMB,
                    relief="flat", padding=6)
        s.map("App.Treeview",
              background=[("selected", AC2)], foreground=[("selected", "white")])

        # ── 툴바 ──────────────────────────────────────────────────────
        tb = tk.Frame(self.root, bg=BG1, height=48)
        tb.pack(side=tk.TOP, fill=tk.X)
        tb.pack_propagate(False)

        tk.Label(tb, text="TGV PSL Inspector",
                 font=(_F, 12, "bold"), bg=BG1, fg=FG, padx=16).pack(
            side=tk.LEFT, fill=tk.Y)
        tk.Label(tb, text="v4.3",
                 font=(_F, 8), bg=BG1, fg=FG2).pack(side=tk.LEFT, pady=(22, 0))

        def tb_grp(btns, padx_l=10):
            grp = tk.Frame(tb, bg=BG1)
            grp.pack(side=tk.LEFT, fill=tk.Y, padx=(padx_l, 0), pady=9)
            for text, cmd, bg, fg in btns:
                tk.Button(grp, text=text, command=cmd,
                          bg=bg, fg=fg, font=F_SMB,
                          relief="flat", bd=0, padx=14, pady=0, height=1,
                          cursor="hand2",
                          activebackground=BG4,
                          activeforeground=FG).pack(side=tk.LEFT, fill=tk.Y, padx=1)
            return grp

        tk.Frame(tb, bg=BD, width=1).pack(side=tk.LEFT, fill=tk.Y, pady=8, padx=8)
        tb_grp([("폴더 로드", self.cb['load_folder'], OR, "white")], 0)
        tk.Frame(tb, bg=BD, width=1).pack(side=tk.LEFT, fill=tk.Y, pady=8, padx=8)
        tb_grp([("Setting Manager", self.cb['open_settings'], BG3, FG3)], 0)

        self.lbl_status = tk.Label(tb, text="폴더를 로드하세요",
                                   font=F_SM, bg=BG1, fg=FG2)
        self.lbl_status.pack(side=tk.RIGHT, fill=tk.Y, padx=20)
        tk.Frame(self.root, bg=BD, height=1).pack(fill=tk.X)

        # ── 중단: 파일탐색기 + 이미지 3패널 ─────────────────────────
        mid = tk.Frame(self.root, bg=BG0)
        mid.pack(expand=True, fill=tk.BOTH)

        nav = tk.Frame(mid, bg=BG1, width=220)
        nav.pack(side=tk.LEFT, fill=tk.Y)
        nav.pack_propagate(False)
        tk.Label(nav, text="파일 목록", font=F_SMB,
                 bg=BG1, fg=FG3, padx=12, pady=8).pack(anchor="w")
        tk.Frame(nav, bg=BD, height=1).pack(fill=tk.X)
        self.nb = ttk.Notebook(nav, style="App.TNotebook")
        self.nb.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_change)

        tk.Frame(mid, bg=BD, width=1).pack(side=tk.LEFT, fill=tk.Y)

        img_area = tk.Frame(mid, bg=BG0)
        img_area.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        for i in range(3):
            img_area.columnconfigure(i, weight=1)
        img_area.rowconfigure(0, weight=1)

        self.lbl_1 = self._img_pane(img_area, "① 원본 이미지", 0, zoomable=True)
        self.lbl_2 = self._img_pane(img_area, "② 이진화 프리뷰", 1,
                                     zoomable=True, saveable=True)
        self.lbl_3 = self._img_pane(img_area, "③ 파티클 검출 결과", 2,
                                     click=True, zoomable=True, saveable=True)

        tk.Frame(self.root, bg=BD, height=1).pack(fill=tk.X)

        # ── 하단 컨트롤 (고정 높이 285px) ────────────────────────────
        bot = tk.Frame(self.root, bg=BG1, height=285)
        bot.pack(side=tk.BOTTOM, fill=tk.X)
        bot.pack_propagate(False)

        def panel(parent, title, accent=FG3, side=tk.LEFT,
                  expand=False, padx=(0, 1), w=None):
            outer = tk.Frame(parent, bg=BD, padx=1, pady=1)
            outer.pack(side=side, fill=tk.BOTH, expand=expand, padx=padx)
            if w:
                outer.configure(width=w)
            inner = tk.Frame(outer, bg=BG1)
            inner.pack(fill=tk.BOTH, expand=True)
            hdr = tk.Frame(inner, bg=BG2, height=28)
            hdr.pack(fill=tk.X)
            hdr.pack_propagate(False)
            tk.Label(hdr, text=title, font=F_SMB,
                     bg=BG2, fg=accent, padx=12).pack(side=tk.LEFT, fill=tk.Y)
            body = tk.Frame(inner, bg=BG1, padx=8, pady=6)
            body.pack(fill=tk.BOTH, expand=True)
            return body

        c1 = panel(bot, "배율별 레시피", accent=AC, w=230)
        self._build_c1(c1)
        c2 = panel(bot, "전처리 파라미터", accent=FG3, expand=True)
        self._build_c2(c2)
        c3 = panel(bot, "검사 범위 & 결과", accent=FG3, side=tk.RIGHT, w=620)
        self._build_c3(c3)

    # ── C1: 배율 선택 ────────────────────────────────────────────────
    def _build_c1(self, p):
        MAGS = ["x50", "x100", "x200", "x500", "x1000"]
        tk.Label(p, text="배율 선택", font=F_XS,
                 bg=BG1, fg=FG2).pack(anchor="w", pady=(0, 4))
        btn_f = tk.Frame(p, bg=BG1)
        btn_f.pack(fill=tk.X, pady=(0, 6))

        def select_mag(mag):
            for m, b in self._mag_btns.items():
                b.config(bg=AC if m == mag else BG3,
                         fg="white" if m == mag else FG3)
            self.v_mag_sel.set(mag)
            self.cb['on_mag_select'](mag)

        for mag in MAGS:
            b = tk.Button(btn_f, text=mag, width=5,
                          command=lambda m=mag: select_mag(m),
                          bg=BG3, fg=FG3, font=F_SMB,
                          relief="flat", bd=0, padx=4, pady=5,
                          cursor="hand2",
                          activebackground=AC2, activeforeground="white")
            b.pack(side=tk.LEFT, padx=2)
            self._mag_btns[mag] = b

        tk.Frame(p, bg=BD2, height=1).pack(fill=tk.X, pady=(4, 6))
        self.lbl_calib_info = tk.Label(p, text="배율을 선택하세요",
                                        font=F_XS, bg=BG1, fg=FG2,
                                        anchor="w", justify="left")
        self.lbl_calib_info.pack(anchor="w")
        tk.Frame(p, bg=BD2, height=1).pack(fill=tk.X, pady=(6, 4))

        pd_f = tk.Frame(p, bg=BG1)
        pd_f.pack(fill=tk.X)
        tk.Label(pd_f, text="목표 파티클 크기",
                 font=F_SMB, bg=BG1, fg=FG).pack(anchor="w")
        inp_f = tk.Frame(pd_f, bg=BG1)
        inp_f.pack(anchor="w", pady=(3, 0))
        e = tk.Entry(inp_f, textvariable=self.v_pdiam, width=7,
                     font=F_MD, bg=BG3, fg=FG, insertbackground=AC,
                     relief="flat", highlightthickness=1,
                     highlightbackground=BD, highlightcolor=AC)
        e.pack(side=tk.LEFT)
        tk.Label(inp_f, text=" μm", font=F_XS, bg=BG1, fg=FG2).pack(side=tk.LEFT)
        # 입력 후 0.6초 대기 → 자동 갱신 (타이핑 중 과도한 재분석 방지)
        _pdiam_after = [None]
        def _pdiam_delayed(*_):
            if _pdiam_after[0]:
                e.after_cancel(_pdiam_after[0])
            _pdiam_after[0] = e.after(600, self.cb['update_all'])
        self.v_pdiam.trace_add('write', _pdiam_delayed)
        e.bind("<Return>",   lambda ev: (e.after_cancel(_pdiam_after[0]) if _pdiam_after[0] else None,
                                         self.cb['update_all']()))
        e.bind("<FocusOut>", lambda ev: self.cb['update_all']())
        tk.Label(p, text="※ Setting Manager에서\n   배율별 Calibration 필요",
                 font=F_XS, bg=BG1, fg=FG2, justify="left").pack(anchor="w", pady=(6, 0))

    def update_calib_display(self, mag, px_per_mm):
        if px_per_mm > 0:
            um_per_px = 1000.0 / px_per_mm
            self.lbl_calib_info.config(
                text=f"{mag}  |  {px_per_mm:.1f} px/mm  ({um_per_px:.3f} ㎛/px)", fg=GR)
        else:
            self.lbl_calib_info.config(text=f"{mag}  |  Calibration 미완료", fg=OR)
        self.update_min_size_display()

    def update_min_size_display(self):
        if self.lbl_min_size is None:
            return
        px_per_mm = self.v_px_per_mm.get()
        try:
            pdiam = float(self.v_pdiam.get())
        except (ValueError, tk.TclError):
            pdiam = 1.5
        if px_per_mm <= 0:
            self.lbl_min_size.config(text="필터 비활성 (배율 미선택)", fg=FG2)
            return
        px_per_um   = px_per_mm / 1000.0
        radius_px   = (pdiam * px_per_um) / 2.0
        min_area_px = round(np.pi * radius_px ** 2, 1)
        self.lbl_min_size.config(
            text=f"≥ {pdiam:.1f} μm  ({min_area_px:.1f} px²)", fg=AC)

    def highlight_mag_btn(self, mag):
        for m, b in self._mag_btns.items():
            b.config(bg=AC if m == mag else BG3,
                     fg="white" if m == mag else FG3)

    # ── C2: 전처리 파라미터 ──────────────────────────────────────────
    def _build_c2(self, p):
        # ── 2컬럼 레이아웃 (스크롤 없음) ────────────────────────────────
        # 좌: 이미지 보정 + 파티클 임계값 + 조명 불균일 보정
        # 우: TGV 홀 자동 마스킹
        left  = tk.Frame(p, bg=BG1)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 1))
        tk.Frame(p, bg=BD2, width=1).pack(side=tk.LEFT, fill=tk.Y, pady=4)
        right = tk.Frame(p, bg=BG1, width=300)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(1, 0))
        right.pack_propagate(False)

        def row_hdr(parent, text, accent=FG3):
            f = tk.Frame(parent, bg=BG1)
            f.pack(fill=tk.X, padx=4, pady=(4, 2))
            tk.Frame(f, bg=accent, width=3).pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
            tk.Label(f, text=text, font=F_SMB, bg=BG1, fg=FG).pack(side=tk.LEFT)

        # ── 좌 컬럼 ──────────────────────────────────────────────────
        # 행 1: 이미지 보정
        row_hdr(left, "이미지 보정", FG3)
        r1 = self._frame(left, BG1); r1.pack(fill=tk.X, padx=4, pady=(0, 2))
        self._slider(r1, "밝기",   self.v_br, -100, 100, 1, self.cb['update_all'])
        self._slider(r1, "대비",   self.v_ct, 0.5, 3.0, 0.05, self.cb['update_all'])
        self._slider(r1, "블러",   self.v_bl, 0, 15, 1, self.cb['update_all'])

        tk.Frame(left, bg=BD2, height=1).pack(fill=tk.X, padx=8, pady=1)

        # 행 2: 로컬 잔차 파티클 검출 (v4.3 핵심)
        row_hdr(left, "파티클 검출  —  로컬 잔차법 (Local Residual)", AC)

        res_outer = tk.Frame(left, bg=BG2, bd=0)
        res_outer.pack(fill=tk.X, padx=4, pady=(0, 3))
        tk.Frame(res_outer, bg=AC, width=3).pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        res_inner = tk.Frame(res_outer, bg=BG2)
        res_inner.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=4)

        # 배경 추정 커널
        r_bgk = tk.Frame(res_inner, bg=BG2); r_bgk.pack(fill=tk.X, pady=(0, 2))
        self._slider(r_bgk, "배경 추정 커널",
                     self.v_bg_kernel, 11, 201, 2, self.cb['update_all'], 160)
        tk.Label(r_bgk, text="클수록 헤일로·조명 편차 자동 제거\n(기본 71, 항상 홀수 처리)",
                 font=F_XS, bg=BG2, fg=FG2, justify="left").pack(side=tk.LEFT, padx=8)

        # 잔차 임계값
        r_res = tk.Frame(res_inner, bg=BG2); r_res.pack(fill=tk.X, pady=(2, 0))
        self._slider(r_res, "잔차 임계값  ",
                     self.v_res_th, 3, 100, 1, self.cb['update_all'], 160)
        info_f = tk.Frame(r_res, bg=BG2); info_f.pack(side=tk.LEFT, padx=8)
        tk.Label(info_f, text="최소 검출 크기", font=F_XS, bg=BG2, fg=FG2).pack(anchor="w")
        self.lbl_min_size = tk.Label(info_f, text="— (배율 미선택)",
                                      font=F_SMB, bg=BG2, fg=FG2)
        self.lbl_min_size.pack(anchor="w")
        tk.Label(info_f, text="낮을수록 민감 / 높을수록 엄격 (기본 25)",
                 font=F_XS, bg=BG2, fg=FG2).pack(anchor="w")
        self.lbl_th_hint = None

        # ── 우 컬럼: TGV 홀 자동 마스킹 ─────────────────────────────
        row_hdr(right, "TGV 홀 자동 마스킹  [앞/뒷면 공통]", RD)
        r3 = self._frame(right, BG1); r3.pack(fill=tk.X, padx=4, pady=(0, 3))
        self._slider(r3, "홀 감지 임계값 (기본 120)",
                     self.v_hole_thr, 60, 180, 2, self.cb['update_all'], 140)
        tk.Label(r3, text="낮을수록 림만/\n높을수록 그림자 포함",
                 font=F_XS, bg=BG1, fg=FG2, justify="left").pack(side=tk.LEFT, padx=6)
        tk.Frame(right, bg=BD2, height=1).pack(fill=tk.X, padx=8, pady=2)
        r3b = self._frame(right, BG1); r3b.pack(fill=tk.X, padx=4, pady=(0, 2))
        self._slider(r3b, "홀 마스크 여유 (px)",
                     self.v_hole_mg, 0, 60, 1, self.cb['update_all'], 120)
        tk.Label(r3b, text="림 반경 외부\n추가 제외",
                 font=F_XS, bg=BG1, fg=FG2, justify="left").pack(side=tk.LEFT, padx=6)

    # ── C3: 검사 범위 & 결과 ─────────────────────────────────────────
    def _build_c3(self, p):
        # 좌: 가장자리 여백 + 수동 제외
        left = tk.Frame(p, bg=BG1)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        # 가장자리 여백
        tk.Frame(left, bg=RD, height=2).pack(fill=tk.X, pady=(0, 6))
        tk.Label(left, text="가장자리 여백", font=F_SMB, bg=BG1, fg=FG).pack(anchor="w")
        tk.Label(left, text="0 = 전체 이미지 검사",
                 font=F_XS, bg=BG1, fg=FG2).pack(anchor="w", pady=(2, 4))
        r_mg = self._frame(left, BG1); r_mg.pack(anchor="w")
        self._slider(r_mg, "여백 (px)", self.v_margin, 0, 300, 5,
                     self.cb['update_all'], 120)
        self.lbl_margin_info = tk.Label(left, text="전체 이미지 검사 중",
                                         font=F_XS, bg=BG1, fg=FG2)
        self.lbl_margin_info.pack(anchor="w", pady=(2, 0))

        def _update_mg_info(*a):
            m = self.v_margin.get()
            self.lbl_margin_info.config(
                text="전체 이미지 검사 중" if m == 0 else f"상·하·좌·우 {m}px 제외",
                fg=FG2 if m == 0 else OR)

        self.v_margin.trace_add("write", _update_mg_info)

        tk.Frame(left, bg=BD2, height=1).pack(fill=tk.X, pady=6)

        # 수동 제외 영역
        tk.Label(left, text="수동 제외 영역 (반원 등)", font=F_SMB,
                 bg=BG1, fg=FG).pack(anchor="w")
        tk.Label(left,
                 text="③ 이미지 우클릭 → 제외 영역 추가\n기존 영역 위 우클릭 → 삭제  |  편집 모드: 좌클릭으로도 추가",
                 font=F_XS, bg=BG1, fg=FG2, justify="left").pack(anchor="w", pady=(2, 4))

        r_excl = self._frame(left, BG1); r_excl.pack(anchor="w")
        self._slider(r_excl, "제외 반경(px)", self.v_excl_r, 20, 300, 5,
                     lambda: None, 110)

        btn_row = self._frame(left, BG1); btn_row.pack(fill=tk.X, pady=(4, 0))
        self.btn_draw_mode = tk.Button(btn_row, text="편집 모드 OFF",
                                        command=self.cb['toggle_draw_mode'],
                                        bg=BG3, fg=FG2, font=F_SMB,
                                        relief="flat", padx=10, pady=3,
                                        cursor="hand2",
                                        activebackground=RD,
                                        activeforeground="white")
        self.btn_draw_mode.pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(btn_row, text="초기화",
                  command=self.cb['clear_exclusions'],
                  bg=BG3, fg=FG2, font=F_SMB,
                  relief="flat", padx=8, pady=3,
                  cursor="hand2").pack(side=tk.LEFT)

        self.lbl_excl_count = tk.Label(left, text="제외 영역: 0개",
                                        font=F_XS, bg=BG1, fg=FG2)
        self.lbl_excl_count.pack(anchor="w", pady=(4, 0))

        # 구분선
        tk.Frame(p, bg=BD2, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        # 우: 결과 테이블
        right = tk.Frame(p, bg=BG1)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Frame(right, bg=AC, height=2).pack(fill=tk.X, pady=(0, 4))
        tk.Label(right, text="검출 결과", font=F_SMB, bg=BG1, fg=FG).pack(
            anchor="w", pady=(0, 4))

        tree_f = tk.Frame(right, bg=BG1)
        tree_f.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(tree_f, columns=("no", "x", "y", "cnt"),
                                  show="headings", height=4,
                                  style="App.Treeview")
        for col, txt, w in [("no", "#", 32), ("x", "X (mm)", 80),
                              ("y", "Y (mm)", 80), ("cnt", "개수", 50)]:
            self.tree.heading(col, text=txt)
            self.tree.column(col, width=w, anchor="center")
        tsb = ttk.Scrollbar(tree_f, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self.cb['on_tree_select'])

        btn_f = tk.Frame(right, bg=BG1)
        btn_f.pack(fill=tk.X, pady=(6, 0))
        for text, cmd, bg in [
            ("Excel 저장",    self.cb['save_excel'],  GR2),
            ("폴더 일괄 검사", self.cb['run_batch'],   OR2),
        ]:
            tk.Button(btn_f, text=text, command=cmd,
                      bg=bg, fg="white", font=F_SMB,
                      relief="flat", bd=0, padx=14, pady=5,
                      cursor="hand2",
                      activebackground=BG4, activeforeground=FG
                      ).pack(side=tk.LEFT, padx=(0, 4))

    def update_draw_mode_btn(self, active):
        if self.btn_draw_mode:
            if active:
                self.btn_draw_mode.config(text="편집 모드 ON", bg=RD, fg="white")
            else:
                self.btn_draw_mode.config(text="편집 모드 OFF", bg=BG3, fg=FG2)

    def update_excl_count(self, n):
        if self.lbl_excl_count:
            self.lbl_excl_count.config(
                text=f"제외 영역: {n}개",
                fg=OR if n > 0 else FG2)

    def _on_tab_change(self, e=None):
        try:
            cur = self.nb.tab(self.nb.select(), "text").split()[0]
            for k, lb in self.tab_lbs.items():
                if k != cur:
                    lb.selection_clear(0, tk.END)
        except:
            pass

    # ── 이미지 패널 ───────────────────────────────────────────────────
    def _img_pane(self, parent, title, col, click=False,
                  zoomable=False, saveable=False):
        outer = tk.Frame(parent, bg=BD)
        outer.grid(row=0, column=col, sticky="nsew", padx=1, pady=1)
        key = f"lbl_{col + 1}"

        title_bar = tk.Frame(outer, bg=BG2, height=28)
        title_bar.pack(side=tk.TOP, fill=tk.X)
        title_bar.pack_propagate(False)
        tk.Label(title_bar, text=title, font=F_SMB,
                 bg=BG2, fg=FG3, padx=10).pack(side=tk.LEFT, fill=tk.Y)

        if zoomable:
            tk.Button(title_bar, text="확대",
                      command=lambda k=key: self._open_zoom_window(k),
                      bg=BG3, fg=AC, font=F_XS, relief="flat", bd=0,
                      padx=10, pady=0, cursor="hand2",
                      activebackground=AC2, activeforeground="white"
                      ).pack(side=tk.RIGHT, fill=tk.Y, padx=(2, 4), pady=4)
        if saveable:
            tk.Button(title_bar, text="저장",
                      command=lambda k=key: self._save_image(k),
                      bg=BG3, fg=GR, font=F_XS, relief="flat", bd=0,
                      padx=10, pady=0, cursor="hand2",
                      activebackground=GR2, activeforeground="white"
                      ).pack(side=tk.RIGHT, fill=tk.Y, padx=2, pady=4)

        lbl = tk.Label(outer, bg="#06080e")
        lbl.pack(expand=True, fill=tk.BOTH)
        lbl.bind("<Configure>", lambda e, k=key: self._resize(k))
        lbl.bind("<Button-3>",  lambda e, k=key: self._ctx_show(e, k))
        if click:
            lbl.bind("<Button-1>", self.cb['on_image_click'])
            lbl.bind("<Button-3>", self.cb['on_image_rclick'])
        return lbl

    def _save_image(self, key):
        img, gray = self._cache[key]
        if img is None:
            messagebox.showwarning("알림", "저장할 이미지가 없습니다.")
            return
        default = {'lbl_2': "binary_preview", 'lbl_3': "particle_result"}.get(key, "image")
        path = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile=default,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"),
                       ("BMP", "*.bmp"), ("TIFF", "*.tif")])
        if not path:
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            fmt = {'.jpg': '.jpg', '.jpeg': '.jpg', '.png': '.png',
                   '.bmp': '.bmp', '.tif': '.tif', '.tiff': '.tif'}.get(ext, '.png')
            success, buf = cv2.imencode(fmt, img)
            if success:
                buf.tofile(path)
                self.status(f"저장 완료: {os.path.basename(path)}", GR)
        except Exception as ex:
            messagebox.showerror("저장 오류", str(ex))

    def _open_zoom_window(self, key='lbl_3'):
        img, gray = self._cache[key]
        if img is None:
            return
        titles = {
            'lbl_1': "원본 이미지 — 확대 보기",
            'lbl_2': "이진화 프리뷰 — 확대 보기",
            'lbl_3': "파티클 검출 결과 — 확대 보기",
        }
        win = tk.Toplevel(self.root)
        win.title(titles.get(key, "확대 보기"))
        win.configure(bg=BG0)
        win.geometry("1100x800")

        ctrl = tk.Frame(win, bg=BG1, pady=6)
        ctrl.pack(side=tk.TOP, fill=tk.X)
        zoom_var = tk.DoubleVar(value=1.0)
        zoom_lbl = tk.Label(ctrl, text="배율: 100%", font=F_SMB, bg=BG1, fg=FG, width=12)
        zoom_lbl.pack(side=tk.LEFT, padx=12)

        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if gray else cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]

        canvas_frame = tk.Frame(win, bg=BG0)
        canvas_frame.pack(expand=True, fill=tk.BOTH, padx=4, pady=4)
        h_sb = ttk.Scrollbar(canvas_frame, orient="horizontal")
        v_sb = ttk.Scrollbar(canvas_frame, orient="vertical")
        canvas = tk.Canvas(canvas_frame, bg="#080a10",
                            xscrollcommand=h_sb.set, yscrollcommand=v_sb.set,
                            highlightthickness=0)
        h_sb.config(command=canvas.xview)
        v_sb.config(command=canvas.yview)
        h_sb.pack(side=tk.BOTTOM, fill=tk.X)
        v_sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(expand=True, fill=tk.BOTH)
        _ref = [None]

        def render(zoom):
            nw = max(1, int(orig_w * zoom))
            nh = max(1, int(orig_h * zoom))
            interp = cv2.INTER_LINEAR if zoom >= 1 else cv2.INTER_AREA
            tk_img = ImageTk.PhotoImage(
                Image.fromarray(cv2.resize(rgb, (nw, nh), interpolation=interp)))
            _ref[0] = tk_img
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=tk_img)
            canvas.configure(scrollregion=(0, 0, nw, nh))
            zoom_lbl.config(text=f"배율: {int(zoom * 100)}%")

        for lbl_text, z in [("50%", .5), ("100%", 1.), ("150%", 1.5),
                             ("200%", 2.), ("300%", 3.)]:
            tk.Button(ctrl, text=lbl_text, command=lambda z=z: render(z),
                      bg=BG2, fg=FG, font=F_XS, relief="flat",
                      padx=10, pady=3, cursor="hand2",
                      activebackground=AC, activeforeground="white"
                      ).pack(side=tk.LEFT, padx=3)

        tk.Frame(ctrl, bg=BD, width=1, height=22).pack(side=tk.LEFT, padx=8, fill=tk.Y)
        tk.Label(ctrl, text="슬라이더:", font=F_XS, bg=BG1, fg=FG2).pack(side=tk.LEFT)
        tk.Scale(ctrl, from_=0.2, to=4.0, resolution=0.1,
                 variable=zoom_var, orient=tk.HORIZONTAL, length=160,
                 bg=BG1, fg=FG, troughcolor=BG3,
                 highlightthickness=0, bd=0, showvalue=False,
                 command=lambda v: render(float(v))).pack(side=tk.LEFT, padx=6)

        def on_wheel(event):
            z = round(zoom_var.get() + (0.1 if event.delta > 0 else -0.1), 1)
            z = max(0.2, min(4.0, z)); zoom_var.set(z); render(z)

        canvas.bind("<MouseWheel>", on_wheel)
        tk.Frame(ctrl, bg=BD, width=1, height=22).pack(side=tk.RIGHT, padx=8, fill=tk.Y)
        self._btn(ctrl, "✕  닫기", win.destroy,
                  bg="#55263a", fg="white", font=F_XS, pad=(10, 3)
                  ).pack(side=tk.RIGHT, padx=8)
        render(1.0)
        win.focus_set()

    def rebuild_tabs(self, fd):
        for t in self.nb.tabs():
            self.nb.forget(t)
        self.tab_lbs.clear()
        for key in sorted(fd, key=lambda x: (0, int(x[1:])) if x[1:].isdigit() else (1, x)):
            fl = fd[key]
            if not fl:
                continue
            tf = tk.Frame(self.nb, bg=BG1)
            self.nb.add(tf, text=f" {key} ({len(fl)}) ")
            sb = ttk.Scrollbar(tf)
            lb = tk.Listbox(tf, yscrollcommand=sb.set, width=26, font=F_SM,
                             exportselection=False,
                             bg=BG2, fg=FG, selectbackground=AC,
                             selectforeground="white",
                             relief="flat", highlightthickness=0)
            sb.config(command=lb.yview)
            lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            for f in fl:
                lb.insert(tk.END, f)
            lb.bind("<<ListboxSelect>>", self.cb['on_file_select'])
            self.tab_lbs[key] = lb

    # ── 파라미터 get/set ──────────────────────────────────────────────
    def get_params(self):
        px_per_mm = self.v_px_per_mm.get()
        try:
            pdiam = float(self.v_pdiam.get())
        except (ValueError, tk.TclError):
            pdiam = 1.5
        if px_per_mm > 0:
            r_px     = (pdiam * (px_per_mm / 1000.0)) / 2.0
            min_area = max(1.0, np.pi * r_px ** 2)
        else:
            min_area = 1.0
        return dict(
            mag=self.v_mag_sel.get(),
            px_per_mm=px_per_mm, pdiam=pdiam, min_area=min_area,
            br=self.v_br.get(), ct=self.v_ct.get(), bl=self.v_bl.get(),
            bg_kernel=self.v_bg_kernel.get(),
            res_th=self.v_res_th.get(),
            margin=self.v_margin.get(),
            hole_thr=self.v_hole_thr.get(),
            hole_margin=self.v_hole_mg.get(),
        )

    def set_params(self, p):
        if 'pdiam'       in p: self.v_pdiam.set(str(p['pdiam']))
        if 'br'         in p: self.v_br.set(p['br'])
        if 'ct'         in p: self.v_ct.set(p['ct'])
        if 'bl'         in p: self.v_bl.set(p['bl'])
        if 'bg_kernel'  in p: self.v_bg_kernel.set(p.get('bg_kernel', 71))
        if 'res_th'     in p: self.v_res_th.set(p.get('res_th', 25))
        if 'margin'     in p: self.v_margin.set(p['margin'])
        if 'hole_thr'   in p: self.v_hole_thr.set(p.get('hole_thr', 120))
        if 'hole_margin'in p: self.v_hole_mg.set(p.get('hole_margin', 30))
        self.update_min_size_display()

    def update_table(self, rows):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for r in rows:
            r['tid'] = self.tree.insert("", tk.END,
                values=(r['no'], f"{r['x_mm']:.4f}", f"{r['y_mm']:.4f}", r['cnt']))

    def status(self, text, fg=FG2):
        self.lbl_status.config(text=text, fg=fg)

    def show(self, img, lbl, gray=False):
        if img is None:
            return
        key = {self.lbl_1: 'lbl_1', self.lbl_2: 'lbl_2', self.lbl_3: 'lbl_3'}.get(lbl)
        if key:
            self._cache[key] = (img.copy(), gray)
            self._resize(key)

    def _resize(self, key):
        lbl = getattr(self, key, None)
        if not lbl:
            return
        img, gray = self._cache[key]
        if img is None:
            return
        w, h = lbl.winfo_width(), lbl.winfo_height()
        if w < 10 or h < 10:
            return
        rgb  = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if gray else cv2.COLOR_BGR2RGB)
        ih, iw = rgb.shape[:2]
        r = min(w / iw, h / ih)
        self.ratios[key] = r
        nw, nh = max(1, int(iw * r)), max(1, int(ih * r))
        self.offsets[key] = ((w - nw) // 2, (h - nh) // 2)
        tk_img = ImageTk.PhotoImage(Image.fromarray(cv2.resize(rgb, (nw, nh))))
        lbl.config(image=tk_img)
        lbl.image = tk_img

    def _ctx_show(self, e, key):
        if self._cache[key][0] is not None:
            self._ctx_key = key
            self._ctx.post(e.x_root, e.y_root)

    def _copy_clip(self):
        if not self._ctx_key:
            return
        img, gray = self._cache[self._ctx_key]
        if img is None:
            return
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if gray else cv2.COLOR_BGR2RGB)
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, "BMP")
            data = buf.getvalue()[14:]
            buf.close()
            k32, u32 = ctypes.windll.kernel32, ctypes.windll.user32
            k32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
            k32.GlobalAlloc.restype  = ctypes.c_void_p
            k32.GlobalLock.argtypes  = [ctypes.c_void_p]
            k32.GlobalLock.restype   = ctypes.c_void_p
            k32.GlobalUnlock.argtypes = [ctypes.c_void_p]
            u32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
            if u32.OpenClipboard(None):
                u32.EmptyClipboard()
                h = k32.GlobalAlloc(0x0002, len(data))
                ptr = k32.GlobalLock(h)
                if ptr:
                    ctypes.memmove(ptr, data, len(data))
                    k32.GlobalUnlock(h)
                    u32.SetClipboardData(8, h)
                u32.CloseClipboard()
                self.status("클립보드 복사 완료!", AC)
        except Exception as ex:
            messagebox.showerror("오류", str(ex))


# ══════════════════════════════════════════════════════════════════════
#  4. Setting Manager (v4에서 유지)
# ══════════════════════════════════════════════════════════════════════
class SettingManager:
    MAGS = ["x50", "x100", "x200", "x500", "x1000"]
    SAVE_PATH = os.path.join(
        os.environ.get("APPDATA", os.path.expanduser("~")),
        "TGV_PSL_Inspector", "settings_v43.json"
    )

    def __init__(self, parent_root, ui, on_apply):
        self.parent   = parent_root
        self.ui       = ui
        self.on_apply = on_apply
        _defaults = {
            "x50":   {"bg_kernel": 51, "res_th": 18, "hole_margin": 35, "pdiam": 1.5},
            "x100":  {"bg_kernel": 71, "res_th": 25, "hole_margin": 35, "pdiam": 1.5},
            "x200":  {"bg_kernel": 51, "res_th": 20, "hole_margin": 30, "pdiam": 1.5},
            "x500":  {"bg_kernel": 31, "res_th": 20, "hole_margin": 25, "pdiam": 1.5},
            "x1000": {"bg_kernel": 41, "res_th": 25, "hole_margin": 20, "pdiam": 1.5},
        }
        self.data = {}
        for mag in self.MAGS:
            d = _defaults.get(mag, {})
            self.data[mag] = {
                "px_per_mm": 0.0, "calibrated": False,
                "pdiam":      d.get("pdiam", 1.5),
                "bg_kernel":  d.get("bg_kernel", 71),
                "res_th":     d.get("res_th", 25),
                "margin": 0,
                "hole_thr": 120,
                "hole_margin": d.get("hole_margin", 30),
                "br": 0, "ct": 1.0, "bl": 0,
            }
        self._load()
        self._cal_img    = None
        self._cal_pts    = []
        self._cal_mag    = None
        self._tk_img_ref = None

    def _save(self):
        os.makedirs(os.path.dirname(self.SAVE_PATH), exist_ok=True)
        with open(self.SAVE_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def _load(self):
        if os.path.exists(self.SAVE_PATH):
            try:
                with open(self.SAVE_PATH, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                for mag in self.MAGS:
                    if mag in saved:
                        self.data[mag].update(saved[mag])
            except:
                pass

    def get_mag_data(self, mag):
        return self.data.get(mag, {})

    def save_recipe_for_mag(self, mag, params):
        d = self.data[mag]
        for k in ('pdiam', 'bg_kernel', 'res_th', 'margin',
                   'hole_thr', 'hole_margin', 'br', 'ct', 'bl'):
            if k in params:
                d[k] = params[k]
        self._save()

    def open(self):
        win = tk.Toplevel(self.parent)
        win.title("Setting Manager")
        win.geometry("1100x720")
        win.configure(bg=BG0)
        win.grab_set()

        top = tk.Frame(win, bg=BG1, height=46)
        top.pack(fill=tk.X)
        top.pack_propagate(False)
        tk.Label(top, text="Setting Manager", font=F_LG,
                 bg=BG1, fg=FG, padx=16).pack(side=tk.LEFT, fill=tk.Y)
        tk.Frame(top, bg=BD, width=1).pack(side=tk.LEFT, fill=tk.Y, pady=8, padx=8)

        self._mag_tab_btns = {}
        tab_f = tk.Frame(top, bg=BG1)
        tab_f.pack(side=tk.LEFT, fill=tk.Y, pady=8)

        content = tk.Frame(win, bg=BG0)
        content.pack(expand=True, fill=tk.BOTH)
        left  = tk.Frame(content, bg=BG1, width=640)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left.pack_propagate(False)
        tk.Frame(content, bg=BD, width=1).pack(side=tk.LEFT, fill=tk.Y)
        right = tk.Frame(content, bg=BG1, width=420)
        right.pack(side=tk.RIGHT, fill=tk.BOTH)

        self._build_calib_panel(left)
        self._build_recipe_panel(right)

        def switch_mag(mag):
            for m, b in self._mag_tab_btns.items():
                b.config(bg=AC2 if m == mag else BG3,
                         fg="white" if m == mag else FG3)
            self._cal_mag = mag
            self._cal_pts = []
            self._refresh_calib_panel(mag)
            self._refresh_recipe_panel(mag)

        for mag in self.MAGS:
            b = tk.Button(tab_f, text=mag, width=6,
                          command=lambda m=mag: switch_mag(m),
                          bg=BG3, fg=FG3, font=F_SMB,
                          relief="flat", bd=0, padx=8, pady=4,
                          cursor="hand2",
                          activebackground=AC2, activeforeground="white")
            b.pack(side=tk.LEFT, padx=2)
            self._mag_tab_btns[mag] = b

        bot = tk.Frame(win, bg=BG2, height=48)
        bot.pack(side=tk.BOTTOM, fill=tk.X)
        bot.pack_propagate(False)
        tk.Frame(bot, bg=BD, height=1).pack(fill=tk.X)
        tk.Button(bot, text="레시피 저장 (현재 배율)",
                  command=lambda: self._save_recipe_clicked(win),
                  bg=GR2, fg="white", font=F_SMB,
                  relief="flat", padx=16, pady=0, height=1,
                  cursor="hand2").pack(side=tk.LEFT, padx=12, pady=10)
        tk.Button(bot, text="닫기", command=win.destroy,
                  bg=BG3, fg=FG3, font=F_SMB,
                  relief="flat", padx=16, pady=0, height=1,
                  cursor="hand2").pack(side=tk.RIGHT, padx=12, pady=10)

        switch_mag(self.MAGS[0])

    def _build_calib_panel(self, parent):
        hdr = tk.Frame(parent, bg=BG2, height=32)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="Calibration — 두 점 클릭 후 실제 거리 입력",
                 font=F_SMB, bg=BG2, fg=FG3, padx=12).pack(side=tk.LEFT, fill=tk.Y)

        guide = tk.Frame(parent, bg=BG1, pady=6)
        guide.pack(fill=tk.X, padx=12)
        for txt in ["① 스케일 눈금 이미지를 불러오세요",
                    "② 이미지에서 두 점을 클릭하세요",
                    "③ 두 점 사이의 실제 거리를 입력하고 Calibration 완료를 누르세요"]:
            tk.Label(guide, text=txt, font=F_XS, bg=BG1, fg=FG2).pack(anchor="w")

        self._cal_canvas = tk.Canvas(parent, bg="#06080e",
                                      highlightthickness=0, cursor="crosshair")
        self._cal_canvas.pack(expand=True, fill=tk.BOTH, padx=8, pady=4)
        self._cal_canvas.bind("<Button-1>", self._on_calib_click)

        ctrl = tk.Frame(parent, bg=BG1, pady=6)
        ctrl.pack(fill=tk.X, padx=8)
        tk.Button(ctrl, text="이미지 불러오기",
                  command=self._load_calib_image,
                  bg=OR, fg="white", font=F_SMB,
                  relief="flat", padx=12, pady=4,
                  cursor="hand2").pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(ctrl, text="두 점 사이 실제 거리:",
                 font=F_XS, bg=BG1, fg=FG2).pack(side=tk.LEFT)
        self._v_dist_val  = tk.DoubleVar(value=1.0)
        self._v_dist_unit = tk.StringVar(value="mm")
        tk.Entry(ctrl, textvariable=self._v_dist_val, width=7,
                 font=F_SM, bg=BG3, fg=FG, insertbackground=AC,
                 relief="flat", highlightthickness=1,
                 highlightbackground=BD, highlightcolor=AC).pack(side=tk.LEFT, padx=4)
        ttk.Combobox(ctrl, textvariable=self._v_dist_unit,
                     values=["mm", "μm"], state="readonly",
                     width=4, style="App.TCombobox").pack(side=tk.LEFT, padx=4)
        self._btn_do_calib = tk.Button(ctrl, text="Calibration 완료",
                                        command=self._do_calibration,
                                        bg=AC2, fg="white", font=F_SMB,
                                        relief="flat", padx=12, pady=4,
                                        cursor="hand2", state="disabled")
        self._btn_do_calib.pack(side=tk.LEFT, padx=8)
        tk.Button(ctrl, text="초기화", command=self._reset_calib,
                  bg=BG3, fg=FG3, font=F_SMB,
                  relief="flat", padx=10, pady=4, cursor="hand2").pack(side=tk.LEFT)

        self._lbl_calib_status = tk.Label(parent, text="이미지를 불러오세요",
                                           font=F_XS, bg=BG1, fg=FG2)
        self._lbl_calib_status.pack(pady=(0, 6))

    def _refresh_calib_panel(self, mag):
        d = self.data[mag]
        self._cal_img = None
        self._cal_pts = []
        self._cal_canvas.delete("all")
        self._btn_do_calib.config(state="disabled")
        if d['calibrated'] and d['px_per_mm'] > 0:
            um = 1000.0 / d['px_per_mm']
            self._lbl_calib_status.config(
                text=f"✓  {mag} 완료  |  {d['px_per_mm']:.2f} px/mm  ({um:.4f} ㎛/px)",
                fg=GR)
        else:
            self._lbl_calib_status.config(
                text=f"{mag}  Calibration 미완료", fg=OR)

    def _load_calib_image(self):
        path = filedialog.askopenfilename(
            title="Calibration 이미지 선택",
            filetypes=[("이미지", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if not path:
            return
        arr = np.fromfile(path, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("오류", "이미지를 읽을 수 없습니다.")
            return
        self._cal_img = img
        self._cal_pts = []
        self._draw_calib_image()
        self._lbl_calib_status.config(text="이미지 로드 완료 — 두 점을 클릭하세요", fg=AC)

    def _draw_calib_image(self):
        if self._cal_img is None:
            return
        canvas = self._cal_canvas
        cw = canvas.winfo_width() or 600
        ch = canvas.winfo_height() or 400
        ih, iw = self._cal_img.shape[:2]
        r = min(cw / iw, ch / ih)
        self._cal_ratio = r
        nw, nh = max(1, int(iw * r)), max(1, int(ih * r))
        rgb = cv2.cvtColor(self._cal_img, cv2.COLOR_BGR2RGB)
        tk_img = ImageTk.PhotoImage(Image.fromarray(cv2.resize(rgb, (nw, nh))))
        self._tk_img_ref = tk_img
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=tk_img)
        for i, (x, y) in enumerate(self._cal_pts):
            cx, cy = int(x * r), int(y * r)
            canvas.create_oval(cx - 6, cy - 6, cx + 6, cy + 6,
                                fill=AC, outline="white", width=2)
            canvas.create_text(cx + 12, cy, text=f"P{i + 1}",
                                fill="white", font=F_SMB)
        if len(self._cal_pts) == 2:
            p1, p2 = self._cal_pts
            canvas.create_line(int(p1[0]*r), int(p1[1]*r),
                                int(p2[0]*r), int(p2[1]*r),
                                fill=AC, width=2, dash=(6, 3))
            dist_px = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            mx = int((p1[0] + p2[0]) / 2 * r)
            my = int((p1[1] + p2[1]) / 2 * r) - 14
            canvas.create_text(mx, my, text=f"{dist_px:.1f} px",
                                fill=AC, font=F_SMB)

    def _on_calib_click(self, e):
        if self._cal_img is None:
            return
        if len(self._cal_pts) >= 2:
            self._cal_pts = []
        r  = getattr(self, '_cal_ratio', 1.0)
        self._cal_pts.append((e.x / r, e.y / r))
        self._draw_calib_image()
        if len(self._cal_pts) == 2:
            self._btn_do_calib.config(state="normal")
            dist = np.hypot(self._cal_pts[1][0] - self._cal_pts[0][0],
                            self._cal_pts[1][1] - self._cal_pts[0][1])
            self._lbl_calib_status.config(
                text=f"두 점 거리: {dist:.1f} px  —  실제 거리를 입력 후 완료",
                fg=AC)
        else:
            self._btn_do_calib.config(state="disabled")
            self._lbl_calib_status.config(text="두 번째 점을 클릭하세요", fg=FG2)

    def _do_calibration(self):
        if len(self._cal_pts) != 2 or self._cal_mag is None:
            return
        dist_px  = np.hypot(self._cal_pts[1][0] - self._cal_pts[0][0],
                             self._cal_pts[1][1] - self._cal_pts[0][1])
        dist_val  = self._v_dist_val.get()
        dist_unit = self._v_dist_unit.get()
        if dist_px < 1 or dist_val <= 0:
            messagebox.showerror("오류", "유효하지 않은 값입니다.")
            return
        dist_mm   = dist_val if dist_unit == "mm" else dist_val / 1000.0
        px_per_mm = dist_px / dist_mm
        d = self.data[self._cal_mag]
        d['px_per_mm']  = round(px_per_mm, 4)
        d['calibrated'] = True
        self._save()
        um = 1000.0 / px_per_mm
        self._lbl_calib_status.config(
            text=f"✓  완료  |  {px_per_mm:.2f} px/mm  ({um:.4f} ㎛/px)", fg=GR)
        self.on_apply(self._cal_mag, px_per_mm)
        messagebox.showinfo("완료",
                            f"{self._cal_mag} Calibration 완료\n"
                            f"· {px_per_mm:.2f} px/mm\n· {um:.4f} ㎛/px")

    def _reset_calib(self):
        self._cal_pts = []
        self._draw_calib_image()
        self._btn_do_calib.config(state="disabled")
        self._lbl_calib_status.config(text="초기화됨 — 두 점을 다시 클릭하세요", fg=FG2)

    def _build_recipe_panel(self, parent):
        hdr = tk.Frame(parent, bg=BG2, height=32)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="배율별 레시피  (직접 편집 가능)", font=F_SMB, bg=BG2, fg=FG3,
                 padx=12).pack(side=tk.LEFT, fill=tk.Y)
        self._recipe_frame = tk.Frame(parent, bg=BG1)
        self._recipe_frame.pack(expand=True, fill=tk.BOTH, padx=12, pady=8)
        self._recipe_vars = {}   # 입력 위젯 변수 저장

    def _refresh_recipe_panel(self, mag):
        for w in self._recipe_frame.winfo_children():
            w.destroy()
        self._recipe_vars = {}
        d = self.data[mag]
        px_per_mm = d['px_per_mm']

        def entry_row(parent, label, key, val, unit="", tip=""):
            bg = BG2 if len(self._recipe_vars) % 2 == 0 else BG1
            row = tk.Frame(parent, bg=bg)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=label, font=F_XS, bg=bg, fg=FG2,
                     width=20, anchor="w", padx=8, pady=5).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(val))
            e = tk.Entry(row, textvariable=var, width=7,
                         font=F_SMB, bg=BG3, fg=FG, insertbackground=AC,
                         relief="flat", highlightthickness=1,
                         highlightbackground=BD, highlightcolor=AC)
            e.pack(side=tk.LEFT, padx=4)
            if unit:
                tk.Label(row, text=unit, font=F_XS, bg=bg, fg=FG2).pack(side=tk.LEFT)
            if tip:
                tk.Label(row, text=tip, font=F_XS, bg=bg, fg=FG2,
                         padx=6).pack(side=tk.LEFT)
            self._recipe_vars[key] = var
            return var

        def readonly_row(label, val, color=FG):
            bg = BG2 if len(self._recipe_vars) % 2 == 0 else BG1
            row = tk.Frame(self._recipe_frame, bg=bg)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=label, font=F_XS, bg=bg, fg=FG2,
                     width=20, anchor="w", padx=8, pady=5).pack(side=tk.LEFT)
            tk.Label(row, text=val, font=F_SMB, bg=bg, fg=color,
                     anchor="w", padx=8).pack(side=tk.LEFT)

        # Calibration 상태 (읽기 전용)
        calib_str = f"{px_per_mm:.2f} px/mm" if d['calibrated'] else "미완료 — 이미지 불러와 두 점 클릭"
        readonly_row("px/mm (Calibration)", calib_str, GR if d['calibrated'] else OR)

        tk.Frame(self._recipe_frame, bg=BD2, height=1).pack(fill=tk.X, pady=(4, 2))
        tk.Label(self._recipe_frame, text="아래 값을 직접 수정 후 저장하세요",
                 font=F_XS, bg=BG1, fg=FG2).pack(anchor="w", pady=(0, 4))

        # 편집 가능 파라미터
        entry_row(self._recipe_frame, "목표 파티클 크기",   "pdiam",       d.get('pdiam', 1.5),      "μm")
        entry_row(self._recipe_frame, "BG 커널 크기",      "bg_kernel",   d.get('bg_kernel', 71),   "px", "홀수, 최소 11")
        entry_row(self._recipe_frame, "잔차 임계값",       "res_th",      d.get('res_th', 25),      "",   "권장 15~40")
        entry_row(self._recipe_frame, "홀 감지 임계값",    "hole_thr",    d.get('hole_thr', 120),   "",   "기본 120")
        entry_row(self._recipe_frame, "홀 마스크 여유",    "hole_margin", d.get('hole_margin', 30), "px", "기본 30")
        entry_row(self._recipe_frame, "가장자리 여백",     "margin",      d.get('margin', 0),       "px")
        entry_row(self._recipe_frame, "블러",             "bl",          d.get('bl', 0),           "",   "0~15")

        # 현재 메인화면에서 불러오기 버튼
        tk.Frame(self._recipe_frame, bg=BD2, height=1).pack(fill=tk.X, pady=(8, 4))
        btn_f = tk.Frame(self._recipe_frame, bg=BG1)
        btn_f.pack(fill=tk.X)
        tk.Button(btn_f, text="메인화면 현재값 불러오기",
                  command=self._load_from_main,
                  bg=BG3, fg=FG2, font=F_XS,
                  relief="flat", padx=10, pady=4,
                  cursor="hand2").pack(side=tk.LEFT)
        tk.Label(btn_f, text="← 슬라이더 값을 여기로 복사",
                 font=F_XS, bg=BG1, fg=FG2, padx=6).pack(side=tk.LEFT)

        if not d['calibrated']:
            tk.Label(self._recipe_frame,
                     text="⚠  Calibration 미완료 — 저장 가능하나 파티클 크기 필터 비활성",
                     font=F_XS, bg=BG1, fg=OR,
                     justify="left").pack(anchor="w", pady=6)

    def _load_from_main(self):
        """메인화면 현재 슬라이더 값을 레시피 패널에 복사"""
        p = self.ui.get_params()
        mapping = {
            'pdiam':       str(p.get('pdiam', 1.5)),
            'bg_kernel':   str(p.get('bg_kernel', 71)),
            'res_th':      str(p.get('res_th', 25)),
            'hole_thr':    str(p.get('hole_thr', 120)),
            'hole_margin': str(p.get('hole_margin', 30)),
            'margin':      str(p.get('margin', 0)),
            'bl':          str(p.get('bl', 0)),
        }
        for k, v in mapping.items():
            if k in self._recipe_vars:
                self._recipe_vars[k].set(v)

    def _save_recipe_clicked(self, win):
        if self._cal_mag is None:
            return
        d = self.data[self._cal_mag]
        def _flt(key, default):
            try: return float(self._recipe_vars[key].get())
            except: return default
        def _int(key, default):
            try: return int(float(self._recipe_vars[key].get()))
            except: return default
        d['pdiam']       = _flt('pdiam',       1.5)
        d['bg_kernel']   = _int('bg_kernel',    71)
        d['res_th']      = _int('res_th',        25)
        d['hole_thr']    = _int('hole_thr',     120)
        d['hole_margin'] = _int('hole_margin',   30)
        d['margin']      = _int('margin',         0)
        d['bl']          = _int('bl',             0)
        self._save()
        self.on_apply(self._cal_mag, d['px_per_mm'], save_recipe=False)
        self._refresh_recipe_panel(self._cal_mag)
        messagebox.showinfo("저장 완료", f"{self._cal_mag} 레시피가 저장되었습니다.")


# ══════════════════════════════════════════════════════════════════════
#  5. 메인 컨트롤러
# ══════════════════════════════════════════════════════════════════════
class App:
    def __init__(self, root):
        self.root = root
        root.title("TGV PSL Inspector  v4.3")
        root.geometry("1920x1000")

        self.dir       = None
        self.files     = {}
        self.img_orig  = self.img_bin = self.img_out = None
        self.holes     = []          # 자동 검출된 홀 목록
        self.exclusions = []         # 수동 제외 영역
        self.draw_mode = False       # 제외 영역 편집 모드
        self.particles = []
        self.vision    = Vision()
        self._cur_mag  = None

        self.ui = UI(root, dict(
            load_folder      = self.load_folder,
            on_file_select   = self.on_file_select,
            run_analysis     = self.run_analysis,
            update_all       = self.update_all,
            save_excel       = self.do_save_excel,
            on_image_click   = self.on_img_click,
            on_image_rclick  = self.on_img_rclick,
            on_tree_select   = self.on_tree_select,
            run_batch        = self.run_batch,
            open_settings    = self.open_settings,
            on_mag_select    = self.on_mag_select,
            toggle_draw_mode = self.toggle_draw_mode,
            clear_exclusions = self.clear_exclusions,
        ))

        self.setting_mgr = SettingManager(root, self.ui, self._on_calib_apply)
        self._refresh_mag_buttons()

    def _refresh_mag_buttons(self):
        for mag in SettingManager.MAGS:
            d   = self.setting_mgr.data[mag]
            btn = self.ui._mag_btns.get(mag)
            if btn:
                btn.config(fg=GR if d['calibrated'] else FG2)

    def _on_calib_apply(self, mag, px_per_mm, save_recipe=False):
        if save_recipe:
            self.setting_mgr.save_recipe_for_mag(mag, self.ui.get_params())
        if mag == self._cur_mag:
            self.on_mag_select(mag)
        self._refresh_mag_buttons()

    def open_settings(self):
        self.setting_mgr.open()

    def on_mag_select(self, mag):
        self._cur_mag = mag
        d = self.setting_mgr.data[mag]
        self.ui.v_px_per_mm.set(d.get('px_per_mm', 0.0))
        self.ui.set_params(d)
        self.ui.highlight_mag_btn(mag)
        self.ui.update_calib_display(mag, d.get('px_per_mm', 0.0))
        if not d['calibrated']:
            self.ui.status(f"{mag} — Calibration 필요 (Setting Manager)", OR)
        else:
            self.ui.status(f"{mag} 선택됨  |  {d['px_per_mm']:.1f} px/mm", GR)
        self.update_all()

    def _mag(self, name):
        m = re.search(r'_?x(\d+)', name, re.I)
        return f"x{m.group(1)}" if m else "기타"

    def load_folder(self):
        d = filedialog.askdirectory()
        if not d:
            return
        self.dir = d
        self.files = {}
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        try:
            fl = sorted([f for f in os.listdir(d) if f.lower().endswith(exts)])
            if not fl:
                messagebox.showwarning("알림", "이미지가 없습니다.")
                return
            for f in fl:
                self.files.setdefault(self._mag(f), []).append(f)
            self.ui.rebuild_tabs(self.files)
            self.ui.status(f"{len(fl)}개 이미지 로드", AC)
            lb = list(self.ui.tab_lbs.values())[0]
            lb.selection_set(0)
            self.on_file_select(None, init=True)
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def on_file_select(self, e, init=False):
        if not self.dir:
            return
        try:
            cur = self.ui.nb.tab(self.ui.nb.select(), "text").split()[0]
            lb  = self.ui.tab_lbs.get(cur)
            if not lb:
                return
            sel = lb.curselection()
            if not sel:
                return
            fname = lb.get(sel[0])
            arr   = np.fromfile(os.path.join(self.dir, fname), np.uint8)
            self.img_orig = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if self.img_orig is None:
                return
            # 파일 변경 시 제외 영역 초기화
            self.exclusions = []
            self.ui.update_excl_count(0)
            self._detect_holes()
            self.ui.show(self.img_orig, self.ui.lbl_1)
            self.update_all()
        except Exception as ex:
            messagebox.showerror("오류", str(ex))

    def _detect_holes(self):
        """이미지에서 홀을 감지하고 self.holes 갱신"""
        if self.img_orig is None:
            self.holes = []
            return
        p = self.ui.get_params()
        rim_thr = p.get('hole_thr', 120)
        self.holes = self.vision.find_all_holes(self.img_orig, rim_thr=rim_thr)

    def update_all(self, *a):
        if self.img_orig is None:
            return
        try:
            p = self.ui.get_params()
            self.ui.update_min_size_display()
            # 홀 임계값 변경 시 재검출
            self._detect_holes()
            self.img_bin = self.vision.preview(
                self.img_orig, self.holes, p, self.exclusions)
            self.ui.show(self.img_bin, self.ui.lbl_2, gray=True)
            self.run_analysis()
        except tk.TclError:
            pass

    def run_analysis(self, *a):
        if self.img_bin is None:
            return
        p = self.ui.get_params()
        out, self.particles, total = self.vision.analyze(
            self.img_orig, self.img_bin, p,
            holes=self.holes, exclusions=self.exclusions)
        self.img_out = out.copy()
        self.ui.show(out, self.ui.lbl_3)
        self.ui.update_table(self.particles)
        try:
            cur = self.ui.nb.tab(self.ui.nb.select(), "text").split()[0]
            lb  = self.ui.tab_lbs.get(cur)
            fn  = lb.get(lb.curselection()[0]) if lb and lb.curselection() else ""
            n_holes = len(self.holes)
            n_edge  = sum(1 for h in self.holes if h['is_edge'])
            self.ui.status(
                f"{fn}  |  파티클: {total} ea  |  홀: {n_holes}개 "
                f"(가장자리 반원: {n_edge}개 제외)", FG)
        except:
            pass

    # ── 수동 제외 영역 ────────────────────────────────────────────────
    def toggle_draw_mode(self):
        self.draw_mode = not self.draw_mode
        self.ui.update_draw_mode_btn(self.draw_mode)

    def clear_exclusions(self):
        self.exclusions = []
        self.ui.update_excl_count(0)
        self.update_all()

    def on_img_click(self, e):
        if self.draw_mode:
            # 제외 영역 추가
            if self.img_out is None:
                return
            r  = self.ui.ratios['lbl_3']
            ox, oy = self.ui.offsets['lbl_3']
            rx, ry = int((e.x - ox) / r), int((e.y - oy) / r)
            er = self.ui.v_excl_r.get()
            self.exclusions.append({'center': (rx, ry), 'r': er})
            self.ui.update_excl_count(len(self.exclusions))
            self.update_all()
        else:
            # 가장 가까운 파티클 하이라이트
            if not self.particles or self.img_out is None:
                return
            r  = self.ui.ratios['lbl_3']
            ox, oy = self.ui.offsets['lbl_3']
            rx, ry = (e.x - ox) / r, (e.y - oy) / r
            near = min(self.particles,
                       key=lambda p: (p['x_px'] - rx) ** 2 + (p['y_px'] - ry) ** 2)
            self.ui.tree.selection_set(near['tid'])
            self.ui.tree.see(near['tid'])
            self._highlight(near)

    def on_img_rclick(self, e):
        if self.img_orig is None:
            return
        r  = self.ui.ratios['lbl_3']
        ox, oy = self.ui.offsets['lbl_3']
        rx = int((e.x - ox) / r)
        ry = int((e.y - oy) / r)

        # 기존 제외 영역 안을 클릭했으면 삭제, 바깥이면 추가
        if self.exclusions:
            nearest = min(self.exclusions,
                          key=lambda ex: (ex['center'][0] - rx) ** 2 +
                                         (ex['center'][1] - ry) ** 2)
            dist = ((nearest['center'][0] - rx) ** 2 +
                    (nearest['center'][1] - ry) ** 2) ** 0.5
            if dist <= nearest['r']:
                self.exclusions.remove(nearest)
                self.ui.update_excl_count(len(self.exclusions))
                self.update_all()
                return

        er = self.ui.v_excl_r.get()
        self.exclusions.append({'center': (rx, ry), 'r': er})
        self.ui.update_excl_count(len(self.exclusions))
        self.update_all()

    def _highlight(self, pt):
        tmp = self.img_out.copy()
        r = max(8, int((np.pi * 10) ** 0.5) + 12)
        cv2.circle(tmp, (int(pt['x_px']), int(pt['y_px'])), r, (0, 255, 255), 3)
        self.ui.show(tmp, self.ui.lbl_3)

    def on_tree_select(self, e):
        sel = self.ui.tree.selection()
        if not sel or self.img_out is None:
            return
        pt = next((r for r in self.particles if r.get('tid') == sel[0]), None)
        if pt:
            self._highlight(pt)

    def run_batch(self):
        if not self.dir or not self.files:
            messagebox.showwarning("경고", "폴더를 먼저 로드하세요.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                             filetypes=[("Excel", "*.xlsx")])
        if not path:
            return
        p = self.ui.get_params()
        rim_thr = p.get('hole_thr', 120)
        summary, sheets = [], {}
        self.ui.status("일괄 검사 중...", OR)
        self.root.update()
        for tag, fl in self.files.items():
            for fname in fl:
                try:
                    arr = np.fromfile(os.path.join(self.dir, fname), np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    holes = self.vision.find_all_holes(img, rim_thr=rim_thr)
                    b     = self.vision.preview(img, holes, p)
                    _, rows, total = self.vision.analyze(img, b, p, holes=holes)
                    n_edge = sum(1 for h in holes if h['is_edge'])
                    summary.append({
                        "파일명": fname, "배율": tag,
                        "파티클 수": total,
                        "검출 홀 수": len(holes),
                        "가장자리 반원": n_edge,
                    })
                    if rows:
                        df = pd.DataFrame(rows)[['no', 'x_px', 'y_px', 'x_mm', 'y_mm', 'cnt']]
                        df.columns = ['순번', 'X픽셀', 'Y픽셀', 'X(mm)', 'Y(mm)', '개수']
                        sheets[re.sub(r'[\\/*?:\[\]]', '', fname)[:25]] = df
                except:
                    continue
        try:
            with pd.ExcelWriter(path, engine='openpyxl') as w:
                pd.DataFrame(summary).to_excel(w, sheet_name="전체 요약", index=False)
                for sn, df in sheets.items():
                    df.to_excel(w, sheet_name=sn, index=False)
            self.ui.status(f"✅  {len(summary)}장 완료 — 보고서 저장됨", GR)
            messagebox.showinfo("완료", f"{len(summary)}장 검사 완료.")
            os.startfile(os.path.abspath(path))
        except Exception as ex:
            messagebox.showerror("오류", str(ex))

    def do_save_excel(self):
        if not self.particles:
            return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                             filetypes=[("Excel", "*.xlsx")])
        if path:
            pd.DataFrame(self.particles)[
                ['no', 'x_px', 'y_px', 'x_mm', 'y_mm', 'cnt']
            ].to_excel(path, index=False)
            messagebox.showinfo("완료", "저장 완료.")
            os.startfile(os.path.abspath(path))


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
