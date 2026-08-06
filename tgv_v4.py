import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import pandas as pd
import json, io, os, re, ctypes

# ══════════════════════════════════════════════════════════════════════
#  디자인 시스템 — Modern Precision Dark
#  컨셉: 반도체 장비 소프트웨어 / 의료기기 UI에서 영감
#        차갑고 정밀한 느낌 + 절제된 강조색
# ══════════════════════════════════════════════════════════════════════

# 색상 팔레트
BG0   = "#0d0f14"    # 최외곽 (거의 검정)
BG1   = "#13161e"    # 메인 패널
BG2   = "#1c2030"    # 카드/섹션
BG3   = "#242840"    # 입력/트로프
BG4   = "#2d3350"    # hover 상태
FG    = "#d4daf0"    # 기본 텍스트
FG2   = "#606880"    # 비활성 텍스트
FG3   = "#8899bb"    # 보조 텍스트
AC    = "#3d8ef0"    # 파랑 강조 (메인)
AC2   = "#1a6fd8"    # 파랑 darker
GR    = "#1fba6e"    # 초록 (분석 실행)
GR2   = "#178f53"    # 초록 darker
OR    = "#e8602a"    # 오렌지 (로드/경고)
OR2   = "#c04d20"    # 오렌지 darker
RD    = "#d94060"    # 빨강 (경계/제한)
BD    = "#252c40"    # 테두리
BD2   = "#1e2438"    # 얇은 구분선

# 폰트 시스템 — Segoe UI (Windows 기본 현대 폰트)
# 크기 일관성: 9(본문) / 10(강조) / 11(섹션) / 13(타이틀)
_F = "Segoe UI"
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
    with open(path,'w') as f: json.dump(d, f, indent=4)

def load_recipe(path):
    with open(path,'r') as f: return json.load(f)

# ══════════════════════════════════════════════════════════════════════
#  2. 비전 처리 엔진
# ══════════════════════════════════════════════════════════════════════
class Vision:
    def __init__(self):
        self.px_per_um = 2.6
        self.px_per_mm = 2600.0
        self.single_area = 12.0

    def update(self, p):
        px_per_mm = p.get('px_per_mm', 0)
        if px_per_mm > 0:
            self.px_per_mm = px_per_mm
            self.px_per_um = px_per_mm / 1000.0
        else:
            sc_um = p.get('sc_um', 50.0)
            sc_px = p.get('sc_px', 130.0)
            self.px_per_um = sc_px / sc_um
            self.px_per_mm = self.px_per_um * 1000.0

    def find_tgv(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, m = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            c = max(cs, key=cv2.contourArea)
            if cv2.contourArea(c) > 2000:
                (x,y), r = cv2.minEnclosingCircle(c)
                return (int(x),int(y)), int(r)
        return None, 0

    def roi_mask(self, shape, center, r_in, r_out):
        m = np.zeros(shape, np.uint8)
        if center:
            cv2.circle(m, center, r_out, 255, -1)
            if r_in > 0: cv2.circle(m, center, r_in, 0, -1)
        else:
            m.fill(255)
        return m

    def preview(self, img, center, p):
        """
        전처리 파이프라인:
        1. 그레이스케일 변환
        2. 밝기/대비 보정 (선택)
        3. 블러 (선택)
        4. Manual 이진화 (BINARY_INV: 임계값보다 어두운 점 = 흰색)
        5. ROI 마스크 적용 (선택)
        """
        self.update(p)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 밝기/대비 보정
        br, ct = p.get('br', 0), p.get('ct', 1.0)
        if br != 0 or ct != 1.0:
            g = cv2.convertScaleAbs(g, alpha=ct, beta=br)

        # 블러 (노이즈 제거)
        bl = p.get('bl', 0)
        if bl > 0:
            g = cv2.GaussianBlur(g, (bl|1, bl|1), 0)

        # 이진화: 임계값보다 어두운 픽셀 = 파티클(흰색)
        th = p.get('th', 140)
        _, b = cv2.threshold(g, th, 255, cv2.THRESH_BINARY_INV)

        # 가장자리 여백 마스크 적용
        margin = p.get('margin', 0)
        if margin > 0:
            h, w = b.shape
            mask = np.zeros((h, w), np.uint8)
            mask[margin:h-margin, margin:w-margin] = 255
            b = cv2.bitwise_and(b, b, mask=mask)

        return b

    def analyze(self, img, binary, p):
        """
        검출 로직:
        - 이진화 결과에서 윤곽선 추출
        - 크기(hole_area) + 원형도(hole_circ) 이상 → TGV 홀 → 제외
        - min_area(목표 파티클 크기 기반) 미만 → 노이즈 → 제외
        - 나머지 → 파티클
        """
        self.update(p)
        cs, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = img.copy()
        rows, total, idx = [], 0, 1

        hole_min_area = p.get('hole_area', 500)
        hole_min_circ = p.get('hole_circ', 0.7)
        min_area      = p.get('min_area', 1.0)   # ★ 목표 파티클 크기 기반

        CLR_PARTICLE = (0, 220, 60)    # 초록: 파티클
        CLR_HOLE     = (0, 60, 220)    # 파랑: TGV 홀 (제외)

        for c in cs:
            area = float(cv2.contourArea(c))
            if area < 1: continue

            peri = float(cv2.arcLength(c, True))
            circ = float(4 * np.pi * area / (peri**2 + 1e-5))

            # TGV 홀 판별
            if area >= hole_min_area and circ >= hole_min_circ:
                cv2.drawContours(out, [c], -1, CLR_HOLE, 2)
                continue

            # ★ 최소 크기 미만 → 노이즈 제거
            if area < min_area:
                continue

            # 파티클
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            total += 1

            cv2.drawContours(out, [c], -1, CLR_PARTICLE, 1)
            rows.append({
                'no': idx,
                'x_px': cx, 'y_px': cy,
                'x_mm': cx / self.px_per_mm,
                'y_mm': cy / self.px_per_mm,
                'cnt': 1
            })
            idx += 1

        return out, rows, total
        return out, rows, total

# ══════════════════════════════════════════════════════════════════════
#  3. UI 구성
# ══════════════════════════════════════════════════════════════════════
class UI:
    def __init__(self, root, cb):
        self.root = root
        self.cb = cb
        self._cache = {'lbl_1':(None,False),'lbl_2':(None,True),'lbl_3':(None,False)}
        self.ratios  = {'lbl_1':1.0,'lbl_2':1.0,'lbl_3':1.0}
        self.tab_lbs = {}
        self._ctx_key = None

        # ── 변수 ──────────────────────────────────────────────────────
        self.v_mag_sel   = tk.StringVar(value="")
        self.v_px_per_mm = tk.DoubleVar(value=0.0)
        self.v_pdiam     = tk.DoubleVar(value=1.5)   # 목표 파티클 크기 (μm)
        self.v_br    = tk.IntVar(value=0)
        self.v_ct    = tk.DoubleVar(value=1.0)
        self.v_bl    = tk.IntVar(value=0)
        self.v_th    = tk.IntVar(value=140)
        self.v_margin    = tk.IntVar(value=0)
        self.v_hole_area = tk.IntVar(value=500)
        self.v_hole_circ = tk.DoubleVar(value=0.7)
        self._mag_btns   = {}
        self.lbl_min_size = None   # 임계값 옆 최소 크기 표시 라벨

        self._ctx = tk.Menu(root, tearoff=0, bg=BG2, fg=FG,
                            activebackground=AC, activeforeground="white")
        self._ctx.add_command(label="📋 클립보드에 이미지 복사",
                              command=self._copy_clip)
        self._build()

    # ── 공통 위젯 헬퍼 ────────────────────────────────────────────────
    def _frame(self, parent, bg=BG1, **kw):
        return tk.Frame(parent, bg=bg, **kw)

    def _label(self, parent, text, font=F_SM, fg=FG3, bg=BG1, **kw):
        return tk.Label(parent, text=text, font=font, fg=fg, bg=bg, **kw)

    def _btn(self, parent, text, cmd, bg=BG3, fg=FG, font=F_SMB, pad=(12, 6)):
        b = tk.Button(parent, text=text, command=cmd,
                      bg=bg, fg=fg, font=font,
                      relief="flat", bd=0,
                      padx=pad[0], pady=pad[1],
                      cursor="hand2",
                      activebackground=BG4,
                      activeforeground=FG)
        return b

    def _sep(self, parent, orient="v"):
        if orient == "v":
            tk.Frame(parent, bg=BD2, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        else:
            tk.Frame(parent, bg=BD2, height=1).pack(fill=tk.X, pady=6)

    def _entry(self, parent, var, w=10):
        return tk.Entry(parent, textvariable=var, width=w,
                        font=F_MD,
                        bg=BG3, fg=FG,
                        insertbackground=AC,
                        relief="flat", bd=0,
                        highlightthickness=1,
                        highlightbackground=BD,
                        highlightcolor=AC)

    def _chk(self, parent, text, var, cmd, fg=FG):
        return tk.Checkbutton(parent, text=text, variable=var, command=cmd,
                              font=F_SM, bg=BG1, fg=fg,
                              activebackground=BG1, activeforeground=AC,
                              selectcolor=BG3,
                              relief="flat", bd=0)

    def _combobox(self, parent, var, vals, w=18):
        return ttk.Combobox(parent, textvariable=var, values=vals,
                            state="readonly", width=w, style="App.TCombobox")

    def _section(self, parent, title, fg_t=AC, side=tk.LEFT, padx=4, pady=2):
        f = tk.LabelFrame(parent, text=f"  {title}  ",
                          font=F_SMB, bg=BG1, fg=fg_t,
                          relief="flat",
                          highlightbackground=BD,
                          highlightthickness=1,
                          padx=8, pady=8)
        f.pack(side=side, fill=tk.Y, padx=padx, pady=pady)
        return f

    def _tag(self, parent, text, fg=AC, bg=BG3):
        """작은 태그 라벨 — 섹션 설명용"""
        return tk.Label(parent, text=f" {text} ",
                        font=F_XS, fg=fg, bg=bg,
                        relief="flat", padx=4, pady=2)

    # ── 슬라이더 — 라벨 위, 현재값 표시, ◀▶ 버튼 ────────────────
    def _slider(self, parent, label, var, lo, hi, res, cmd, width=120):
        wrap = self._frame(parent, bg=BG1)
        wrap.pack(side=tk.LEFT, padx=8, pady=2)

        # 라벨 + 현재값 한 줄
        hdr = self._frame(wrap, BG1)
        hdr.pack(fill=tk.X)
        self._label(hdr, label, F_XS, FG2, BG1).pack(side=tk.LEFT)
        tk.Label(hdr, textvariable=var, font=F_XS,
                 fg=AC, bg=BG1, width=6, anchor="e").pack(side=tk.RIGHT)

        # ‹ 스케일 ›
        row = self._frame(wrap, BG1)
        row.pack()

        btn_cfg = dict(font=(_F, 8), bg=BG3, fg=FG3,
                       relief="flat", padx=4, pady=1,
                       cursor="hand2",
                       activebackground=BG4, activeforeground=AC)

        tk.Button(row, text="‹", command=lambda: self._step(var,res,-1,lo,hi,cmd),
                  **btn_cfg).pack(side=tk.LEFT)
        tk.Scale(row, from_=lo, to=hi, resolution=res, variable=var,
                 orient=tk.HORIZONTAL, length=width, showvalue=False,
                 bg=BG1, fg=FG, troughcolor=BG3,
                 activebackground=AC,
                 highlightthickness=0, bd=0,
                 command=lambda x: cmd()).pack(side=tk.LEFT)
        tk.Button(row, text="›", command=lambda: self._step(var,res,+1,lo,hi,cmd),
                  **btn_cfg).pack(side=tk.LEFT)
        return wrap

    @staticmethod
    def _step(var, res, d, lo, hi, cmd):
        v = round(var.get() + res*d, 4)
        var.set(max(lo, min(hi, v))); cmd()

    # ══════════════════════════════════════════════════════════════════
    #  메인 UI 빌드
    # ══════════════════════════════════════════════════════════════════
    def _build(self):
        self.root.configure(bg=BG0)

        # ── ttk 스타일 ──────────────────────────────────────────────────
        s = ttk.Style(); s.theme_use('clam')

        s.configure("App.TNotebook",
                    background=BG1, borderwidth=0, tabmargins=0)
        s.configure("App.TNotebook.Tab",
                    background=BG2, foreground=FG2,
                    font=F_SM, padding=[12, 6])
        s.map("App.TNotebook.Tab",
              background=[("selected", AC2)],
              foreground=[("selected", "white")])

        s.configure("App.TCombobox",
                    fieldbackground=BG3, background=BG3,
                    foreground=FG, arrowcolor=FG3,
                    borderwidth=0, relief="flat",
                    padding=4)
        s.map("App.TCombobox",
              fieldbackground=[("readonly", BG3)],
              foreground=[("readonly", FG)])

        s.configure("App.Treeview",
                    background=BG2, fieldbackground=BG2,
                    foreground=FG, font=F_SM,
                    rowheight=26, borderwidth=0)
        s.configure("App.Treeview.Heading",
                    background=BG0, foreground=FG3,
                    font=F_SMB, relief="flat", padding=6)
        s.map("App.Treeview",
              background=[("selected", AC2)],
              foreground=[("selected", "white")])

        # ══ A. 최상단 툴바 ════════════════════════════════════════════
        tb = tk.Frame(self.root, bg=BG1, height=48)
        tb.pack(side=tk.TOP, fill=tk.X)
        tb.pack_propagate(False)

        # 좌측: 앱 타이틀
        tk.Label(tb, text="TGV PSL Inspector",
                 font=(_F, 12, "bold"), bg=BG1, fg=FG,
                 padx=16).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(tb, text="v4.0",
                 font=(_F, 8), bg=BG1, fg=FG2,
                 padx=0).pack(side=tk.LEFT, pady=(22, 0))

        # 버튼 그룹 공통 스타일
        BTN_H = 30  # 모든 버튼 높이 고정

        def tb_group(btns, padx_left=10):
            """버튼 그룹: 동일 높이 박스 버튼들을 묶음"""
            grp = tk.Frame(tb, bg=BG1)
            grp.pack(side=tk.LEFT, fill=tk.Y,
                     padx=(padx_left, 0), pady=9)
            for text, cmd, bg, fg in btns:
                b = tk.Button(grp, text=text, command=cmd,
                              bg=bg, fg=fg,
                              font=F_SMB,
                              relief="flat", bd=0,
                              padx=14, pady=0,
                              height=1,
                              cursor="hand2",
                              activebackground=BG4,
                              activeforeground=FG)
                b.pack(side=tk.LEFT, fill=tk.Y, padx=1)
            return grp

        # 그룹 1: 파일
        tk.Frame(tb, bg=BD, width=1).pack(side=tk.LEFT,
                                           fill=tk.Y, pady=8, padx=8)
        tb_group([
            ("폴더 로드", self.cb['load_folder'], OR, "white"),
        ], padx_left=0)

        # 그룹 2: 설정
        tk.Frame(tb, bg=BD, width=1).pack(side=tk.LEFT,
                                           fill=tk.Y, pady=8, padx=8)
        tb_group([
            ("Setting Manager", self.cb['open_settings'], BG3, FG3),
        ], padx_left=0)

        # 우측: 상태
        self.lbl_status = tk.Label(tb,
                                   text="폴더를 로드하세요",
                                   font=F_SM, bg=BG1, fg=FG2)
        self.lbl_status.pack(side=tk.RIGHT, fill=tk.Y, padx=20)

        # 툴바 하단 구분선
        tk.Frame(self.root, bg=BD, height=1).pack(fill=tk.X)

        # ══ B. 중단: 파일탐색기 + 이미지 뷰 ═════════════════════════
        mid = tk.Frame(self.root, bg=BG0)
        mid.pack(expand=True, fill=tk.BOTH, padx=0, pady=0)

        # 파일 탐색기
        nav = tk.Frame(mid, bg=BG1, width=220)
        nav.pack(side=tk.LEFT, fill=tk.Y)
        nav.pack_propagate(False)

        tk.Label(nav, text="파일 목록", font=F_SMB,
                 bg=BG1, fg=FG3, padx=12, pady=8).pack(anchor="w")
        tk.Frame(nav, bg=BD, height=1).pack(fill=tk.X)

        self.nb = ttk.Notebook(nav, style="App.TNotebook")
        self.nb.pack(fill=tk.BOTH, expand=True, pady=(4,0))
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_change)

        # 세로 구분선
        tk.Frame(mid, bg=BD, width=1).pack(side=tk.LEFT, fill=tk.Y)

        # 이미지 3패널
        img_area = tk.Frame(mid, bg=BG0)
        img_area.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        for i in range(3): img_area.columnconfigure(i, weight=1)
        img_area.rowconfigure(0, weight=1)

        self.lbl_1 = self._img_pane(img_area, "① 원본 이미지", 0, zoomable=True)
        self.lbl_2 = self._img_pane(img_area, "② 이진화 프리뷰", 1, zoomable=True, saveable=True)
        self.lbl_3 = self._img_pane(img_area, "③ 파티클 검출 결과", 2, click=True, zoomable=True, saveable=True)

        # 하단 구분선
        tk.Frame(self.root, bg=BD, height=1).pack(fill=tk.X)

        # ── 하단 구분선
        tk.Frame(self.root, bg=BD, height=1).pack(fill=tk.X)

        # ══ C. 하단 컨트롤 패널 (고정 높이 270px) ════════════════════
        bot = tk.Frame(self.root, bg=BG1, height=270)
        bot.pack(side=tk.BOTTOM, fill=tk.X)
        bot.pack_propagate(False)

        def panel(parent, title, accent=FG3, side=tk.LEFT,
                  expand=False, padx=(0,1), pady=0, w=None):
            outer = tk.Frame(parent, bg=BD, padx=1, pady=1)
            outer.pack(side=side, fill=tk.BOTH, expand=expand,
                       padx=padx, pady=pady)
            if w: outer.configure(width=w)
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

        # C1: 배율별 레시피 (좌측 고정폭)
        c1 = panel(bot, "배율별 레시피", accent=AC, w=230)
        self._build_c1(c1)

        # C2: 전처리 파라미터 (중앙 확장)
        c2 = panel(bot, "전처리 파라미터", accent=FG3, expand=True)
        self._build_c2(c2)

        # C3: 검사 범위 & 결과 (우측 — 가로 배치)
        c3 = panel(bot, "검사 범위 & 결과", accent=FG3, side=tk.RIGHT, w=580)
        self._build_c3(c3)

    # ── C1: 현미경 배율 선택 ──────────────────────────────────────────
    def _build_c1(self, p):
        MAGS = ["x50", "x100", "x200", "x500", "x1000"]

        tk.Label(p, text="배율 선택", font=F_XS,
                 bg=BG1, fg=FG2).pack(anchor="w", pady=(0,4))

        btn_f = tk.Frame(p, bg=BG1)
        btn_f.pack(fill=tk.X, pady=(0,6))

        def select_mag(mag):
            # 버튼 색상 업데이트
            for m, b in self._mag_btns.items():
                b.config(bg=AC if m==mag else BG3,
                         fg="white" if m==mag else FG3)
            self.v_mag_sel.set(mag)
            self.cb['on_mag_select'](mag)

        for mag in MAGS:
            b = tk.Button(btn_f, text=mag, width=5,
                          command=lambda m=mag: select_mag(m),
                          bg=BG3, fg=FG3, font=F_SMB,
                          relief="flat", bd=0,
                          padx=4, pady=5, cursor="hand2",
                          activebackground=AC2,
                          activeforeground="white")
            b.pack(side=tk.LEFT, padx=2)
            self._mag_btns[mag] = b

        # px/mm 표시
        tk.Frame(p, bg=BD2, height=1).pack(fill=tk.X, pady=(4,6))

        self.lbl_calib_info = tk.Label(p,
            text="배율을 선택하세요",
            font=F_XS, bg=BG1, fg=FG2, anchor="w", justify="left")
        self.lbl_calib_info.pack(anchor="w")

        tk.Frame(p, bg=BD2, height=1).pack(fill=tk.X, pady=(6,4))

        # 목표 파티클 크기 입력
        pd_f = tk.Frame(p, bg=BG1)
        pd_f.pack(fill=tk.X, pady=(0,2))
        tk.Label(pd_f, text="목표 파티클 크기",
                 font=F_SMB, bg=BG1, fg=FG).pack(anchor="w")
        inp_f = tk.Frame(pd_f, bg=BG1)
        inp_f.pack(anchor="w", pady=(3,0))
        e = tk.Entry(inp_f, textvariable=self.v_pdiam, width=7,
                     font=F_MD, bg=BG3, fg=FG,
                     insertbackground=AC, relief="flat",
                     highlightthickness=1,
                     highlightbackground=BD, highlightcolor=AC)
        e.pack(side=tk.LEFT)
        tk.Label(inp_f, text=" μm  (최소 검출 기준)",
                 font=F_XS, bg=BG1, fg=FG2).pack(side=tk.LEFT)
        e.bind("<FocusOut>", lambda ev: self.cb['update_all']())
        e.bind("<Return>",   lambda ev: self.cb['update_all']())

        tk.Label(p, text="※ Setting Manager에서\n   배율별 Calibration 필요",
                 font=F_XS, bg=BG1, fg=FG2,
                 justify="left").pack(anchor="w", pady=(6,0))

    def update_calib_display(self, mag, px_per_mm):
        """Calibration 정보 + 최소 검출 크기 표시 업데이트"""
        if px_per_mm > 0:
            um_per_px = 1000.0 / px_per_mm
            self.lbl_calib_info.config(
                text=f"{mag}  |  {px_per_mm:.1f} px/mm  ({um_per_px:.3f} ㎛/px)",
                fg=GR)
        else:
            self.lbl_calib_info.config(
                text=f"{mag}  |  Calibration 미완료", fg=OR)
        self.update_min_size_display()

    def update_min_size_display(self):
        """임계값 옆 최소 검출 크기 실시간 계산 및 표시"""
        if self.lbl_min_size is None: return
        px_per_mm = self.v_px_per_mm.get()
        pdiam     = self.v_pdiam.get()
        if px_per_mm <= 0:
            self.lbl_min_size.config(
                text="필터 비활성 (배율 미선택)", fg=FG2)
            return
        px_per_um   = px_per_mm / 1000.0
        radius_px   = (pdiam * px_per_um) / 2.0
        min_area_px = round(np.pi * radius_px ** 2, 1)
        self.lbl_min_size.config(
            text=f"≥ {pdiam:.1f} μm  ({min_area_px:.1f} px² 기준)",
            fg=AC)

    def highlight_mag_btn(self, mag):
        for m, b in self._mag_btns.items():
            b.config(bg=AC if m==mag else BG3,
                     fg="white" if m==mag else FG3)

    # ── C2: 전처리 파라미터 ────────────────────────────────────────────
    def _build_c2(self, p):
        def row_header(parent, text, accent=FG3):
            f = tk.Frame(parent, bg=BG1)
            f.pack(fill=tk.X, padx=4, pady=(4, 2))
            tk.Frame(f, bg=accent, width=3).pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
            tk.Label(f, text=text, font=F_SMB,
                     bg=BG1, fg=FG).pack(side=tk.LEFT)
            return f

        # 행 1: 이미지 보정
        row_header(p, "이미지 보정", FG3)
        r1 = self._frame(p, BG1)
        r1.pack(fill=tk.X, padx=4, pady=(0, 2))
        self._slider(r1, "밝기", self.v_br, -100, 100, 1, self.cb['update_all'])
        self._slider(r1, "대비", self.v_ct, 0.5, 3.0, 0.05, self.cb['update_all'])
        self._slider(r1, "블러", self.v_bl, 0, 15, 1, self.cb['update_all'])

        tk.Frame(p, bg=BD2, height=1).pack(fill=tk.X, padx=8, pady=1)

        # 행 2: 파티클 검출 임계값 + 최소 검출 크기 표시
        row_header(p, "파티클 검출 임계값", AC)
        r2 = self._frame(p, BG1)
        r2.pack(fill=tk.X, padx=4, pady=(0, 2))
        self._slider(r2, "임계값  (낮을수록 엄격  /  높을수록 민감)",
                     self.v_th, 50, 254, 1, self.cb['update_all'], width=280)

        # 최소 검출 크기 실시간 표시
        info_f = tk.Frame(r2, bg=BG1)
        info_f.pack(side=tk.LEFT, padx=16)
        tk.Label(info_f, text="최소 검출 크기", font=F_XS,
                 bg=BG1, fg=FG2).pack(anchor="w")
        self.lbl_min_size = tk.Label(info_f,
            text="— (배율 미선택)",
            font=F_SMB, bg=BG1, fg=FG2)
        self.lbl_min_size.pack(anchor="w")

        tk.Frame(p, bg=BD2, height=1).pack(fill=tk.X, padx=8, pady=1)

        # 행 3: TGV 홀 제외
        row_header(p, "TGV 홀 자동 제외", RD)
        r3 = self._frame(p, BG1)
        r3.pack(fill=tk.X, padx=4, pady=(0, 2))
        self._slider(r3, "홀 최소 면적 (px²)", self.v_hole_area,
                     100, 2000, 50, self.cb['update_all'])
        self._slider(r3, "홀 원형도  (0=불규칙 / 1=완전한 원)", self.v_hole_circ,
                     0.3, 1.0, 0.05, self.cb['update_all'])
        tk.Label(r3, text="두 조건 동시 충족 시 홀로 판단하여 제외",
                 font=F_XS, bg=BG1, fg=FG2).pack(side=tk.LEFT, padx=12)

    # ── C3: 가장자리 여백 + 결과 테이블 (가로 배치) ─────────────────
    def _build_c3(self, p):
        # 좌: 가장자리 여백 설정
        left = tk.Frame(p, bg=BG1)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        tk.Frame(left, bg=RD, height=2).pack(fill=tk.X, pady=(0,6))
        tk.Label(left, text="가장자리 여백", font=F_SMB,
                 bg=BG1, fg=FG).pack(anchor="w")
        tk.Label(left, text="0 = 전체 이미지 검사",
                 font=F_XS, bg=BG1, fg=FG2).pack(anchor="w", pady=(2,4))

        r1 = self._frame(left, BG1)
        r1.pack(anchor="w")
        self._slider(r1, "여백 (px)", self.v_margin,
                     0, 300, 5, self.cb['update_all'], 130)

        self.lbl_margin_info = tk.Label(left,
            text="전체 이미지 검사 중",
            font=F_XS, bg=BG1, fg=FG2)
        self.lbl_margin_info.pack(anchor="w", pady=(4,0))

        def update_margin_info(*a):
            m = self.v_margin.get()
            if m == 0:
                self.lbl_margin_info.config(text="전체 이미지 검사 중", fg=FG2)
            else:
                self.lbl_margin_info.config(text=f"상·하·좌·우  {m}px 제외", fg=OR)
        self.v_margin.trace_add("write", update_margin_info)

        # 세로 구분선
        tk.Frame(p, bg=BD2, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=(0,8))

        # 우: 결과 테이블 + 액션 버튼
        right = tk.Frame(p, bg=BG1)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Frame(right, bg=AC, height=2).pack(fill=tk.X, pady=(0,4))
        tk.Label(right, text="검출 결과", font=F_SMB,
                 bg=BG1, fg=FG).pack(anchor="w", pady=(0,4))

        tree_f = tk.Frame(right, bg=BG1)
        tree_f.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(tree_f, columns=("no","x","y","cnt"),
                                  show="headings", height=4,
                                  style="App.Treeview")
        for col, txt, w in [("no","#",32),("x","X (mm)",80),
                             ("y","Y (mm)",80),("cnt","개수",50)]:
            self.tree.heading(col, text=txt)
            self.tree.column(col, width=w, anchor="center")
        tsb = ttk.Scrollbar(tree_f, orient="vertical",
                             command=self.tree.yview)
        self.tree.configure(yscrollcommand=tsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self.cb['on_tree_select'])

        # 액션 버튼 (가로 배치)
        btn_f = tk.Frame(right, bg=BG1)
        btn_f.pack(fill=tk.X, pady=(6,0))

        for text, cmd, bg in [
            ("Excel 저장",    self.cb['save_excel'], GR2),
            ("폴더 일괄 검사", self.cb['run_batch'],  OR2),
        ]:
            tk.Button(btn_f, text=text, command=cmd,
                      bg=bg, fg="white", font=F_SMB,
                      relief="flat", bd=0,
                      padx=14, pady=5,
                      cursor="hand2",
                      activebackground=BG4,
                      activeforeground=FG
                      ).pack(side=tk.LEFT, padx=(0,4))

    def _on_mag(self, e=None):
        d={"10x 프리셋":(100.0,200.0),"100x 프리셋":(50.0,260.0),"200x 프리셋":(50.0,520.0)}
        v = self.v_mag.get()
        if v in d: self.v_scum.set(d[v][0]); self.v_scpx.set(d[v][1])
        self.cb['update_all']()

    def _on_tab_change(self, e=None):
        try:
            cur = self.nb.tab(self.nb.select(),"text").split()[0]
            for k,lb in self.tab_lbs.items():
                if k!=cur: lb.selection_clear(0,tk.END)
        except: pass

    # ── 이미지 패널 ───────────────────────────────────────────────────
    def _img_pane(self, parent, title, col, click=False, zoomable=False, saveable=False):
        outer = tk.Frame(parent, bg=BD)
        outer.grid(row=0, column=col, sticky="nsew", padx=1, pady=1)

        key = f"lbl_{col+1}"

        # 타이틀 바 — 헤더 높이 고정 28px
        title_bar = tk.Frame(outer, bg=BG2, height=28)
        title_bar.pack(side=tk.TOP, fill=tk.X)
        title_bar.pack_propagate(False)

        tk.Label(title_bar, text=title, font=F_SMB,
                 bg=BG2, fg=FG3, padx=10).pack(side=tk.LEFT, fill=tk.Y)

        # 버튼들은 RIGHT로 쌓이므로 역순으로 pack
        if zoomable:
            tk.Button(title_bar, text="확대",
                      command=lambda k=key: self._open_zoom_window(k),
                      bg=BG3, fg=AC, font=F_XS,
                      relief="flat", bd=0, padx=10, pady=0,
                      cursor="hand2",
                      activebackground=AC2,
                      activeforeground="white"
                      ).pack(side=tk.RIGHT, fill=tk.Y, padx=(2,4), pady=4)

        if saveable:
            tk.Button(title_bar, text="저장",
                      command=lambda k=key: self._save_image(k),
                      bg=BG3, fg=GR, font=F_XS,
                      relief="flat", bd=0, padx=10, pady=0,
                      cursor="hand2",
                      activebackground=GR2,
                      activeforeground="white"
                      ).pack(side=tk.RIGHT, fill=tk.Y, padx=2, pady=4)

        # 이미지 영역
        lbl = tk.Label(outer, bg="#06080e")
        lbl.pack(expand=True, fill=tk.BOTH)
        lbl.bind("<Configure>", lambda e, k=key: self._resize(k))
        lbl.bind("<Button-3>",  lambda e, k=key: self._ctx_show(e, k))
        if click: lbl.bind("<Button-1>", self.cb['on_image_click'])
        return lbl

    def _save_image(self, key):
        """이미지를 파일로 저장 — 다른 이름으로 저장 다이얼로그"""
        img, gray = self._cache[key]
        if img is None:
            messagebox.showwarning("알림", "저장할 이미지가 없습니다.\n먼저 이미지를 불러오세요.")
            return

        # 기본 파일명 제안
        default_names = {
            'lbl_2': "binary_preview",
            'lbl_3': "particle_result",
        }
        default = default_names.get(key, "image")

        path = filedialog.asksaveasfilename(
            title="다른 이름으로 저장",
            defaultextension=".png",
            initialfile=default,
            filetypes=[
                ("PNG 이미지",  "*.png"),
                ("JPEG 이미지", "*.jpg"),
                ("BMP 이미지",  "*.bmp"),
                ("TIFF 이미지", "*.tif"),
            ]
        )
        if not path:
            return

        try:
            # BGR → 저장 (gray면 grayscale로)
            if gray:
                save_img = img  # 이미 grayscale
            else:
                save_img = img  # BGR 그대로 imwrite

            # 한글 경로 지원
            ext = os.path.splitext(path)[1].lower()
            encode_map = {
                '.jpg': '.jpg', '.jpeg': '.jpg',
                '.png': '.png', '.bmp': '.bmp',
                '.tif': '.tif', '.tiff': '.tif'
            }
            fmt = encode_map.get(ext, '.png')
            success, buf = cv2.imencode(fmt, save_img)
            if success:
                buf.tofile(path)
                self.status(f"저장 완료:  {os.path.basename(path)}", GR)
            else:
                messagebox.showerror("오류", "이미지 인코딩에 실패했습니다.")
        except Exception as ex:
            messagebox.showerror("저장 오류", str(ex))

    def _open_zoom_window(self, key='lbl_3'):
        """선택한 패널 이미지를 별도 창으로 띄워 확대/축소 지원"""
        img, gray = self._cache[key]
        if img is None:
            return

        titles = {
            'lbl_1': "🔍 원본 이미지 — 확대 보기",
            'lbl_2': "🔍 이진화 프리뷰 — 확대 보기",
            'lbl_3': "🔍 파티클 검출 결과 — 확대 보기",
        }
        win_title = titles.get(key, "🔍 확대 보기")

        win = tk.Toplevel(self.root)
        win.title(win_title)
        win.configure(bg=BG0)
        win.geometry("1100x800")

        # 상단 컨트롤 바
        ctrl = tk.Frame(win, bg=BG1, pady=6)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        zoom_var = tk.DoubleVar(value=1.0)
        zoom_lbl = tk.Label(ctrl, text="배율: 100%", font=F_SMB, bg=BG1, fg=FG, width=12)
        zoom_lbl.pack(side=tk.LEFT, padx=12)

        # 원본 이미지 (BGR → RGB)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if gray else cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]

        # 캔버스 (스크롤 지원)
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
        v_sb.pack(side=tk.RIGHT,  fill=tk.Y)
        canvas.pack(expand=True, fill=tk.BOTH)

        # 이미지 렌더 함수
        _tk_img_ref = [None]

        def render(zoom):
            nw = max(1, int(orig_w * zoom))
            nh = max(1, int(orig_h * zoom))
            resized = cv2.resize(rgb, (nw, nh),
                                 interpolation=cv2.INTER_LINEAR if zoom >= 1
                                 else cv2.INTER_AREA)
            tk_img = ImageTk.PhotoImage(Image.fromarray(resized))
            _tk_img_ref[0] = tk_img
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=tk_img)
            canvas.configure(scrollregion=(0, 0, nw, nh))
            zoom_lbl.config(text=f"배율: {int(zoom*100)}%")

        # 버튼들
        def set_zoom(z):
            zoom_var.set(z)
            render(z)

        for label, z in [("50%", 0.5), ("100%", 1.0), ("150%", 1.5),
                         ("200%", 2.0), ("300%", 3.0)]:
            tk.Button(ctrl, text=label, command=lambda z=z: set_zoom(z),
                      bg=BG2, fg=FG, font=F_XS, relief="flat",
                      padx=10, pady=3, cursor="hand2",
                      activebackground=AC, activeforeground="white"
                      ).pack(side=tk.LEFT, padx=3)

        tk.Frame(ctrl, bg=BD, width=1, height=22).pack(side=tk.LEFT, padx=8, fill=tk.Y)

        # 슬라이더로 연속 조절
        tk.Label(ctrl, text="슬라이더:", font=F_XS, bg=BG1, fg=FG2).pack(side=tk.LEFT)
        def on_slider(val):
            render(float(val))
        sl = tk.Scale(ctrl, from_=0.2, to=4.0, resolution=0.1,
                      variable=zoom_var, orient=tk.HORIZONTAL, length=160,
                      bg=BG1, fg=FG, troughcolor=BG3,
                      highlightthickness=0, bd=0, showvalue=False,
                      command=on_slider)
        sl.pack(side=tk.LEFT, padx=6)

        # 마우스 휠로 확대/축소
        def on_wheel(event):
            z = zoom_var.get()
            z = round(z + (0.1 if event.delta > 0 else -0.1), 1)
            z = max(0.2, min(4.0, z))
            zoom_var.set(z)
            render(z)
        canvas.bind("<MouseWheel>", on_wheel)

        # 닫기 버튼
        tk.Frame(ctrl, bg=BD, width=1, height=22).pack(side=tk.RIGHT, padx=8, fill=tk.Y)
        self._btn(ctrl, "✕  닫기", win.destroy,
                  bg="#55263a", fg="white", font=F_XS, pad=(10,3)).pack(side=tk.RIGHT, padx=8)

        # 초기 렌더
        render(1.0)
        win.focus_set()

    def rebuild_tabs(self, fd):
        for t in self.nb.tabs(): self.nb.forget(t)
        self.tab_lbs.clear()
        for key in sorted(fd, key=lambda x:(0,int(x[1:])) if x[1:].isdigit() else (1,x)):
            fl = fd[key]
            if not fl: continue
            tf = tk.Frame(self.nb, bg=BG1)
            self.nb.add(tf, text=f" {key} ({len(fl)}) ")
            sb = ttk.Scrollbar(tf, )
            lb = tk.Listbox(tf, yscrollcommand=sb.set, width=26, font=F_SM,
                            exportselection=False,
                            bg=BG2, fg=FG, selectbackground=AC,
                            selectforeground="white",
                            relief="flat", highlightthickness=0)
            sb.config(command=lb.yview)
            lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            for f in fl: lb.insert(tk.END, f)
            lb.bind("<<ListboxSelect>>", self.cb['on_file_select'])
            self.tab_lbs[key] = lb

    # ── 파라미터 get/set ──────────────────────────────────────────────
    def get_params(self):
        px_per_mm = self.v_px_per_mm.get()
        pdiam     = self.v_pdiam.get()
        # 캘리브레이션 완료된 경우에만 최소 면적 필터 적용
        if px_per_mm > 0:
            px_per_um  = px_per_mm / 1000.0
            radius_px  = (pdiam * px_per_um) / 2.0
            min_area   = max(1.0, np.pi * radius_px ** 2)
        else:
            min_area = 1.0   # 캘리브레이션 미완료 → 사실상 필터 없음 (1px² 이상 전부)
        return dict(
            mag=self.v_mag_sel.get(),
            px_per_mm=px_per_mm,
            pdiam=pdiam,
            min_area=min_area,
            br=self.v_br.get(), ct=self.v_ct.get(), bl=self.v_bl.get(),
            th=self.v_th.get(),
            margin=self.v_margin.get(),
            hole_area=self.v_hole_area.get(),
            hole_circ=self.v_hole_circ.get(),
            sc_um=50.0,
            sc_px=px_per_mm * 0.05 if px_per_mm > 0 else 130.0,
        )

    def set_params(self, p):
        if 'pdiam' in p: self.v_pdiam.set(p['pdiam'])
        self.v_br.set(p.get("br", 0))
        self.v_ct.set(p.get("ct", 1.0))
        self.v_bl.set(p.get("bl", 0))
        self.v_th.set(p.get("th", 140))
        self.v_margin.set(p.get("margin", 0))
        self.v_hole_area.set(p.get("hole_area", 500))
        self.v_hole_circ.set(p.get("hole_circ", 0.7))
        self.update_min_size_display()

    def update_table(self, rows):
        for i in self.tree.get_children(): self.tree.delete(i)
        for r in rows:
            r['tid'] = self.tree.insert("", tk.END,
                values=(r['no'], f"{r['x_mm']:.4f}", f"{r['y_mm']:.4f}", r['cnt']))

    def status(self, text, fg=FG2):
        self.lbl_status.config(text=text, fg=fg)

    # ── 이미지 표시 ───────────────────────────────────────────────────
    def show(self, img, lbl, gray=False):
        if img is None: return
        key = {self.lbl_1:'lbl_1',self.lbl_2:'lbl_2',self.lbl_3:'lbl_3'}.get(lbl)
        if key:
            self._cache[key] = (img.copy(), gray)
            self._resize(key)

    def _resize(self, key):
        lbl = getattr(self, key, None)
        if not lbl: return
        img, gray = self._cache[key]
        if img is None: return
        w,h = lbl.winfo_width(), lbl.winfo_height()
        if w<10 or h<10: return
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if gray else cv2.COLOR_BGR2RGB)
        ih,iw = rgb.shape[:2]
        r = min(w/iw, h/ih)
        self.ratios[key] = r
        nw,nh = max(1,int(iw*r)), max(1,int(ih*r))
        tk_img = ImageTk.PhotoImage(Image.fromarray(cv2.resize(rgb,(nw,nh))))
        lbl.config(image=tk_img); lbl.image = tk_img

    def _ctx_show(self, e, key):
        if self._cache[key][0] is not None:
            self._ctx_key = key
            self._ctx.post(e.x_root, e.y_root)

    def _copy_clip(self):
        if not self._ctx_key: return
        img, gray = self._cache[self._ctx_key]
        if img is None: return
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if gray else cv2.COLOR_BGR2RGB)
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf,"BMP")
            data = buf.getvalue()[14:]; buf.close()
            k32,u32 = ctypes.windll.kernel32, ctypes.windll.user32
            k32.GlobalAlloc.argtypes=[ctypes.c_uint,ctypes.c_size_t]
            k32.GlobalAlloc.restype=ctypes.c_void_p
            k32.GlobalLock.argtypes=[ctypes.c_void_p]
            k32.GlobalLock.restype=ctypes.c_void_p
            k32.GlobalUnlock.argtypes=[ctypes.c_void_p]
            u32.SetClipboardData.argtypes=[ctypes.c_uint,ctypes.c_void_p]
            if u32.OpenClipboard(None):
                u32.EmptyClipboard()
                h=k32.GlobalAlloc(0x0002,len(data))
                ptr=k32.GlobalLock(h)
                if ptr: ctypes.memmove(ptr,data,len(data)); k32.GlobalUnlock(h); u32.SetClipboardData(8,h)
                u32.CloseClipboard()
                self.status("📋 클립보드 복사 완료!", AC)
        except Exception as ex:
            messagebox.showerror("오류",str(ex))

# ══════════════════════════════════════════════════════════════════════
#  5. Setting Manager
# ══════════════════════════════════════════════════════════════════════
class SettingManager:
    """
    배율별 Calibration + 레시피 관리
    저장 위치: %APPDATA%/TGV_PSL_Inspector/settings.json
    """
    MAGS = ["x50", "x100", "x200", "x500", "x1000"]
    SAVE_PATH = os.path.join(
        os.environ.get("APPDATA", os.path.expanduser("~")),
        "TGV_PSL_Inspector", "settings.json"
    )

    def __init__(self, parent_root, ui, on_apply):
        self.parent = parent_root
        self.ui     = ui
        self.on_apply = on_apply   # 배율 적용 콜백

        # 설정 데이터 구조
        self.data = {mag: {
            "px_per_mm": 0.0,
            "calibrated": False,
            "pdiam": 1.5,        # ★ 목표 파티클 크기 (μm)
            "th": 140,
            "hole_area": 500,
            "hole_circ": 0.7,
            "margin": 0,
            "br": 0, "ct": 1.0, "bl": 0,
        } for mag in self.MAGS}

        self._load()

        # Calibration 작업 상태
        self._cal_img     = None   # 원본 BGR
        self._cal_pts     = []     # 클릭한 두 점
        self._cal_mag     = None   # 현재 작업 배율
        self._tk_img_ref  = None

    # ── 저장/불러오기 ────────────────────────────────────────────────
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
            except: pass

    def get_mag_data(self, mag):
        return self.data.get(mag, {})

    def save_recipe_for_mag(self, mag, params):
        """현재 분석 파라미터를 해당 배율 레시피로 저장"""
        d = self.data[mag]
        d['pdiam']     = params.get('pdiam', 1.5)
        d['th']        = params.get('th', 140)
        d['hole_area'] = params.get('hole_area', 500)
        d['hole_circ'] = params.get('hole_circ', 0.7)
        d['margin']    = params.get('margin', 0)
        d['br']        = params.get('br', 0)
        d['ct']        = params.get('ct', 1.0)
        d['bl']        = params.get('bl', 0)
        self._save()

    # ── Setting Manager 창 열기 ──────────────────────────────────────
    def open(self):
        win = tk.Toplevel(self.parent)
        win.title("Setting Manager")
        win.geometry("1100x720")
        win.configure(bg=BG0)
        win.grab_set()

        # ── 상단 탭: 배율별 ──
        top = tk.Frame(win, bg=BG1, height=46)
        top.pack(fill=tk.X)
        top.pack_propagate(False)
        tk.Label(top, text="Setting Manager", font=F_LG,
                 bg=BG1, fg=FG, padx=16).pack(side=tk.LEFT, fill=tk.Y)
        tk.Frame(top, bg=BD, width=1).pack(side=tk.LEFT, fill=tk.Y, pady=8, padx=8)

        self._mag_tab_btns = {}
        tab_f = tk.Frame(top, bg=BG1)
        tab_f.pack(side=tk.LEFT, fill=tk.Y, pady=8)

        # 메인 컨텐츠 영역
        content = tk.Frame(win, bg=BG0)
        content.pack(expand=True, fill=tk.BOTH, padx=0, pady=0)

        # 좌: Calibration 패널 / 우: 레시피 패널
        left  = tk.Frame(content, bg=BG1, width=640)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left.pack_propagate(False)
        tk.Frame(content, bg=BD, width=1).pack(side=tk.LEFT, fill=tk.Y)
        right = tk.Frame(content, bg=BG1, width=420)
        right.pack(side=tk.RIGHT, fill=tk.BOTH)

        self._build_calib_panel(left, win)
        self._build_recipe_panel(right, win)

        # 배율 탭 버튼
        def switch_mag(mag):
            for m, b in self._mag_tab_btns.items():
                b.config(bg=AC2 if m==mag else BG3,
                         fg="white" if m==mag else FG3)
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

        # 하단 버튼
        bot = tk.Frame(win, bg=BG2, height=48)
        bot.pack(side=tk.BOTTOM, fill=tk.X)
        bot.pack_propagate(False)
        tk.Frame(bot, bg=BD, height=1).pack(fill=tk.X)

        tk.Button(bot, text="레시피 저장  (현재 배율)",
                  command=lambda: self._save_recipe_clicked(win),
                  bg=GR2, fg="white", font=F_SMB,
                  relief="flat", padx=16, pady=0, height=1,
                  cursor="hand2").pack(side=tk.LEFT, padx=12, pady=10)

        tk.Button(bot, text="닫기",
                  command=win.destroy,
                  bg=BG3, fg=FG3, font=F_SMB,
                  relief="flat", padx=16, pady=0, height=1,
                  cursor="hand2").pack(side=tk.RIGHT, padx=12, pady=10)

        # 첫 배율 자동 선택
        switch_mag(self.MAGS[0])

    # ── Calibration 패널 ─────────────────────────────────────────────
    def _build_calib_panel(self, parent, win):
        # 헤더
        hdr = tk.Frame(parent, bg=BG2, height=32)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="Calibration — 두 점 클릭 후 실제 거리 입력",
                 font=F_SMB, bg=BG2, fg=FG3, padx=12).pack(side=tk.LEFT, fill=tk.Y)

        # 안내 텍스트
        guide = tk.Frame(parent, bg=BG1, pady=6)
        guide.pack(fill=tk.X, padx=12)
        for i, txt in enumerate([
            "① 스케일 눈금 이미지를 불러오세요",
            "② 이미지에서 두 점을 클릭하세요",
            "③ 두 점 사이의 실제 거리를 입력하고 Calibration 완료를 누르세요",
        ]):
            tk.Label(guide, text=txt, font=F_XS, bg=BG1, fg=FG2).pack(anchor="w")

        # 이미지 캔버스
        self._cal_canvas = tk.Canvas(parent, bg="#06080e",
                                      highlightthickness=0, cursor="crosshair")
        self._cal_canvas.pack(expand=True, fill=tk.BOTH, padx=8, pady=4)
        self._cal_canvas.bind("<Button-1>", self._on_calib_click)

        # 컨트롤 바
        ctrl = tk.Frame(parent, bg=BG1, pady=6)
        ctrl.pack(fill=tk.X, padx=8)

        tk.Button(ctrl, text="이미지 불러오기",
                  command=self._load_calib_image,
                  bg=OR, fg="white", font=F_SMB,
                  relief="flat", padx=12, pady=4,
                  cursor="hand2").pack(side=tk.LEFT, padx=(0,8))

        tk.Label(ctrl, text="두 점 사이 실제 거리:",
                 font=F_XS, bg=BG1, fg=FG2).pack(side=tk.LEFT)

        self._v_dist_val  = tk.DoubleVar(value=1.0)
        self._v_dist_unit = tk.StringVar(value="mm")

        tk.Entry(ctrl, textvariable=self._v_dist_val, width=7,
                 font=F_SM, bg=BG3, fg=FG,
                 insertbackground=AC, relief="flat",
                 highlightthickness=1, highlightbackground=BD,
                 highlightcolor=AC).pack(side=tk.LEFT, padx=4)

        ttk.Combobox(ctrl, textvariable=self._v_dist_unit,
                     values=["mm", "μm"], state="readonly",
                     width=4, style="App.TCombobox").pack(side=tk.LEFT, padx=4)

        self._btn_do_calib = tk.Button(ctrl, text="Calibration 완료",
                                        command=self._do_calibration,
                                        bg=AC2, fg="white", font=F_SMB,
                                        relief="flat", padx=12, pady=4,
                                        cursor="hand2", state="disabled")
        self._btn_do_calib.pack(side=tk.LEFT, padx=8)

        tk.Button(ctrl, text="초기화",
                  command=self._reset_calib,
                  bg=BG3, fg=FG3, font=F_SMB,
                  relief="flat", padx=10, pady=4,
                  cursor="hand2").pack(side=tk.LEFT)

        # 상태 라벨
        self._lbl_calib_status = tk.Label(parent,
            text="이미지를 불러오세요",
            font=F_XS, bg=BG1, fg=FG2)
        self._lbl_calib_status.pack(pady=(0,6))

    def _refresh_calib_panel(self, mag):
        d = self.data[mag]
        self._cal_img = None
        self._cal_pts = []
        self._cal_canvas.delete("all")
        self._btn_do_calib.config(state="disabled")

        if d['calibrated'] and d['px_per_mm'] > 0:
            um_per_px = 1000.0 / d['px_per_mm']
            self._lbl_calib_status.config(
                text=f"✓  {mag} Calibration 완료  |  "
                     f"{d['px_per_mm']:.2f} px/mm  ({um_per_px:.4f} ㎛/px)",
                fg=GR)
        else:
            self._lbl_calib_status.config(
                text=f"{mag}  Calibration 미완료 — 이미지를 불러오세요", fg=OR)

    def _load_calib_image(self):
        path = filedialog.askopenfilename(
            title="Calibration 이미지 선택",
            filetypes=[("이미지 파일","*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if not path: return
        arr = np.fromfile(path, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("오류","이미지를 읽을 수 없습니다."); return
        self._cal_img = img
        self._cal_pts = []
        self._draw_calib_image()
        self._lbl_calib_status.config(
            text="이미지 로드 완료 — 두 점을 클릭하세요", fg=AC)

    def _draw_calib_image(self):
        if self._cal_img is None: return
        canvas = self._cal_canvas
        cw = canvas.winfo_width()  or 600
        ch = canvas.winfo_height() or 400
        ih, iw = self._cal_img.shape[:2]
        r = min(cw/iw, ch/ih)
        self._cal_ratio = r
        nw, nh = max(1, int(iw*r)), max(1, int(ih*r))
        rgb = cv2.cvtColor(self._cal_img, cv2.COLOR_BGR2RGB)
        tk_img = ImageTk.PhotoImage(Image.fromarray(cv2.resize(rgb, (nw, nh))))
        self._tk_img_ref = tk_img
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=tk_img)

        # 클릭 포인트 다시 그리기
        for i, (x, y) in enumerate(self._cal_pts):
            cx, cy = int(x * r), int(y * r)
            canvas.create_oval(cx-6, cy-6, cx+6, cy+6,
                               fill=AC, outline="white", width=2)
            canvas.create_text(cx+12, cy, text=f"P{i+1}",
                               fill="white", font=F_SMB)

        # 선 그리기
        if len(self._cal_pts) == 2:
            p1, p2 = self._cal_pts
            canvas.create_line(
                int(p1[0]*r), int(p1[1]*r),
                int(p2[0]*r), int(p2[1]*r),
                fill=AC, width=2, dash=(6,3))
            # 픽셀 거리 표시
            dist_px = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
            mx = int((p1[0]+p2[0])/2 * r)
            my = int((p1[1]+p2[1])/2 * r) - 14
            canvas.create_text(mx, my,
                text=f"{dist_px:.1f} px",
                fill=AC, font=F_SMB)

    def _on_calib_click(self, e):
        if self._cal_img is None: return
        if len(self._cal_pts) >= 2:
            self._cal_pts = []
        r = getattr(self, '_cal_ratio', 1.0)
        rx, ry = e.x / r, e.y / r
        self._cal_pts.append((rx, ry))
        self._draw_calib_image()

        if len(self._cal_pts) == 2:
            self._btn_do_calib.config(state="normal")
            dist_px = np.hypot(
                self._cal_pts[1][0]-self._cal_pts[0][0],
                self._cal_pts[1][1]-self._cal_pts[0][1])
            self._lbl_calib_status.config(
                text=f"두 점 거리: {dist_px:.1f} px  —  실제 거리를 입력 후 완료를 누르세요",
                fg=AC)
        else:
            self._btn_do_calib.config(state="disabled")
            self._lbl_calib_status.config(
                text="두 번째 점을 클릭하세요", fg=FG2)

    def _do_calibration(self):
        if len(self._cal_pts) != 2 or self._cal_mag is None: return
        dist_px = np.hypot(
            self._cal_pts[1][0]-self._cal_pts[0][0],
            self._cal_pts[1][1]-self._cal_pts[0][1])
        dist_val  = self._v_dist_val.get()
        dist_unit = self._v_dist_unit.get()

        if dist_px < 1 or dist_val <= 0:
            messagebox.showerror("오류","유효하지 않은 값입니다."); return

        # px/mm 계산
        dist_mm = dist_val if dist_unit == "mm" else dist_val / 1000.0
        px_per_mm = dist_px / dist_mm

        d = self.data[self._cal_mag]
        d['px_per_mm']  = round(px_per_mm, 4)
        d['calibrated'] = True
        self._save()

        um_per_px = 1000.0 / px_per_mm
        self._lbl_calib_status.config(
            text=f"✓  Calibration 완료  |  {px_per_mm:.2f} px/mm  ({um_per_px:.4f} ㎛/px)",
            fg=GR)

        # 메인 UI 업데이트
        self.on_apply(self._cal_mag, px_per_mm)
        messagebox.showinfo("완료",
            f"{self._cal_mag} Calibration 완료\n\n"
            f"· {px_per_mm:.2f} px/mm\n"
            f"· {um_per_px:.4f} ㎛/px")

    def _reset_calib(self):
        self._cal_pts = []
        self._draw_calib_image()
        self._btn_do_calib.config(state="disabled")
        self._lbl_calib_status.config(text="초기화됨 — 두 점을 다시 클릭하세요", fg=FG2)

    # ── 레시피 패널 ──────────────────────────────────────────────────
    def _build_recipe_panel(self, parent, win):
        hdr = tk.Frame(parent, bg=BG2, height=32)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="배율별 레시피",
                 font=F_SMB, bg=BG2, fg=FG3, padx=12).pack(side=tk.LEFT, fill=tk.Y)

        self._recipe_frame = tk.Frame(parent, bg=BG1)
        self._recipe_frame.pack(expand=True, fill=tk.BOTH, padx=12, pady=8)

    def _refresh_recipe_panel(self, mag):
        for w in self._recipe_frame.winfo_children():
            w.destroy()
        d = self.data[mag]

        # 최소 면적 계산
        px_per_mm = d['px_per_mm']
        pdiam     = d.get('pdiam', 1.5)
        if d['calibrated'] and px_per_mm > 0:
            px_per_um = px_per_mm / 1000.0
            r_px      = (pdiam * px_per_um) / 2.0
            min_area  = round(np.pi * r_px ** 2, 1)
            pdiam_str = f"{pdiam:.1f} μm  →  {min_area} px²"
        else:
            pdiam_str = f"{pdiam:.1f} μm  (Calibration 필요)"

        rows = [
            ("px/mm (Calibration)",   f"{px_per_mm:.2f}" if d['calibrated'] else "미완료",  "calib"),
            ("목표 파티클 크기",        pdiam_str,   "pdiam"),
            ("파티클 검출 임계값",      str(d['th']),        ""),
            ("홀 최소 면적 (px²)",     str(d['hole_area']), ""),
            ("홀 원형도",              f"{d['hole_circ']:.2f}", ""),
            ("가장자리 여백 (px)",     str(d['margin']),    ""),
            ("밝기 / 대비 / 블러",     f"{d['br']} / {d['ct']} / {d['bl']}", ""),
        ]

        for i, (label, val, tag) in enumerate(rows):
            bg  = BG2 if i % 2 == 0 else BG1
            row = tk.Frame(self._recipe_frame, bg=bg)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=label, font=F_XS, bg=bg, fg=FG2,
                     width=22, anchor="w", padx=8, pady=6).pack(side=tk.LEFT)
            clr = GR  if tag == "calib" and d['calibrated'] \
                 else AC if tag == "pdiam" \
                 else FG
            tk.Label(row, text=val, font=F_SMB, bg=bg, fg=clr,
                     anchor="w", padx=8).pack(side=tk.LEFT)

        if not d['calibrated']:
            tk.Label(self._recipe_frame,
                text="⚠  이 배율은 Calibration이 완료되지 않았습니다.\n"
                     "    좌측에서 Calibration을 먼저 진행하세요.",
                font=F_XS, bg=BG1, fg=OR,
                justify="left").pack(anchor="w", pady=12)

    def _save_recipe_clicked(self, win):
        if self._cal_mag is None: return
        self.on_apply(self._cal_mag,
                      self.data[self._cal_mag]['px_per_mm'],
                      save_recipe=True)
        self._refresh_recipe_panel(self._cal_mag)
        messagebox.showinfo("저장 완료",
            f"{self._cal_mag} 레시피가 저장되었습니다.")


# ══════════════════════════════════════════════════════════════════════
#  4. 메인 컨트롤러
# ══════════════════════════════════════════════════════════════════════
class App:
    def __init__(self, root):
        self.root = root
        root.title("TGV PSL Inspector  v4.0")
        root.geometry("1920x1000")

        self.dir = None
        self.files = {}
        self.img_orig = self.img_bin = self.img_out = None
        self.tgv_center = None
        self.particles = []
        self.vision = Vision()
        self._cur_mag = None

        self.ui = UI(root, dict(
            load_folder    = self.load_folder,
            on_file_select = self.on_file_select,
            save_recipe    = self.do_save_recipe,
            load_recipe    = self.do_load_recipe,
            run_analysis   = self.run_analysis,
            update_all     = self.update_all,
            save_excel     = self.do_save_excel,
            on_image_click = self.on_img_click,
            on_tree_select = self.on_tree_select,
            run_batch      = self.run_batch,
            open_settings  = self.open_settings,
            on_mag_select  = self.on_mag_select,
        ))

        # Setting Manager 초기화
        self.setting_mgr = SettingManager(
            parent_root = root,
            ui          = self.ui,
            on_apply    = self._on_calib_apply,
        )

        # 저장된 calibration 정보로 배율 버튼 상태 반영
        self._refresh_mag_buttons()

    def _refresh_mag_buttons(self):
        """저장된 calibration 상태를 배율 버튼에 반영"""
        for mag in SettingManager.MAGS:
            d = self.setting_mgr.data[mag]
            btn = self.ui._mag_btns.get(mag)
            if btn:
                if d['calibrated']:
                    btn.config(fg=GR)   # 초록: calibration 완료
                else:
                    btn.config(fg=FG2)  # 회색: 미완료

    def _on_calib_apply(self, mag, px_per_mm, save_recipe=False):
        """Calibration 완료 또는 레시피 저장 콜백"""
        if save_recipe:
            # 현재 파라미터를 해당 배율 레시피로 저장
            p = self.ui.get_params()
            self.setting_mgr.save_recipe_for_mag(mag, p)

        # 해당 배율이 현재 선택된 배율이면 즉시 적용
        if mag == self._cur_mag:
            self.on_mag_select(mag)

        self._refresh_mag_buttons()

    def open_settings(self):
        self.setting_mgr.open()

    def on_mag_select(self, mag):
        """배율 버튼 클릭 → 해당 배율 calibration & 레시피 적용"""
        self._cur_mag = mag
        d = self.setting_mgr.data[mag]

        # px_per_mm 적용
        px_per_mm = d.get('px_per_mm', 0.0)
        self.ui.v_px_per_mm.set(px_per_mm)

        # 레시피 파라미터 적용
        self.ui.set_params(d)

        # 버튼 하이라이트
        self.ui.highlight_mag_btn(mag)

        # calibration 정보 표시
        self.ui.update_calib_display(mag, px_per_mm)

        if not d['calibrated']:
            self.ui.status(f"{mag} — Calibration 필요 (Setting Manager 열기)", OR)
        else:
            self.ui.status(f"{mag} 선택됨  |  {px_per_mm:.1f} px/mm", GR)

        self.update_all()

    def _mag(self, name):
        m = re.search(r'_?x(\d+)', name, re.I)
        return f"x{m.group(1)}" if m else "기타"

    def load_folder(self):
        d = filedialog.askdirectory()
        if not d: return
        self.dir = d; self.files = {}
        exts = ('.png','.jpg','.jpeg','.bmp','.tif','.tiff')
        try:
            fl = sorted([f for f in os.listdir(d) if f.lower().endswith(exts)])
            if not fl: messagebox.showwarning("알림","이미지가 없습니다."); return
            for f in fl: self.files.setdefault(self._mag(f),[]).append(f)
            self.ui.rebuild_tabs(self.files)
            self.ui.status(f"📁 {len(fl)}개 이미지 로드", AC)
            lb = list(self.ui.tab_lbs.values())[0]
            lb.selection_set(0)
            self.on_file_select(None, init=True)
        except Exception as e: messagebox.showerror("오류",str(e))

    def on_file_select(self, e, init=False):
        if not self.dir: return
        try:
            cur = self.ui.nb.tab(self.ui.nb.select(),"text").split()[0]
            lb = self.ui.tab_lbs.get(cur)
            if not lb: return
            sel = lb.curselection()
            if not sel: return
            fname = lb.get(sel[0])
            arr = np.fromfile(os.path.join(self.dir,fname), np.uint8)
            self.img_orig = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if self.img_orig is None: return
            center, r = self.vision.find_tgv(self.img_orig)
            self.tgv_center = center
            if center and init:
                p = self.ui.get_params()
                p['roi_in']  = max(0, r-5)
                p['roi_out'] = r+300
                self.ui.set_params(p)
            self.ui.show(self.img_orig, self.ui.lbl_1)
            self.update_all()
        except Exception as ex: messagebox.showerror("오류",str(ex))

    def update_all(self, *a):
        if self.img_orig is None: return
        try:
            p = self.ui.get_params()
            self.ui.update_min_size_display()   # ★ 최소 검출 크기 실시간 갱신
            self.img_bin = self.vision.preview(self.img_orig, self.tgv_center, p)
            self.ui.show(self.img_bin, self.ui.lbl_2, gray=True)
            self.run_analysis()
        except tk.TclError: pass

    def run_analysis(self, *a):
        if self.img_bin is None: return
        p = self.ui.get_params()
        out, self.particles, total = self.vision.analyze(self.img_orig, self.img_bin, p)
        if self.tgv_center:
            cv2.circle(out, self.tgv_center, p['roi_out'], (0,60,220), 2)
            if p['roi_in']>0:
                cv2.circle(out, self.tgv_center, p['roi_in'], (200,0,180), 2)
        self.img_out = out.copy()
        self.ui.show(out, self.ui.lbl_3)
        self.ui.update_table(self.particles)
        try:
            cur = self.ui.nb.tab(self.ui.nb.select(),"text").split()[0]
            lb = self.ui.tab_lbs.get(cur)
            fn = lb.get(lb.curselection()[0]) if lb and lb.curselection() else ""
            self.ui.status(f"📄 {fn}  |  검출: {total} ea", FG)
        except: pass

    def run_batch(self):
        if not self.dir or not self.files:
            messagebox.showwarning("경고","폴더를 먼저 로드하세요."); return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                            filetypes=[("Excel","*.xlsx")])
        if not path: return
        p = self.ui.get_params()
        summary, sheets = [], {}
        self.ui.status("⚡ 일괄 검사 중...", OR); self.root.update()
        for tag,fl in self.files.items():
            for fname in fl:
                try:
                    arr = np.fromfile(os.path.join(self.dir,fname),np.uint8)
                    img = cv2.imdecode(arr,cv2.IMREAD_COLOR)
                    if img is None: continue
                    c,_ = self.vision.find_tgv(img)
                    b   = self.vision.preview(img,c,p)
                    _,rows,total = self.vision.analyze(img,b,p)
                    summary.append({"파일명":fname,"배율":tag,"파티클 수":total})
                    if rows:
                        df=pd.DataFrame(rows)[['no','x_px','y_px','x_mm','y_mm','cnt']]
                        df.columns=['순번','X픽셀','Y픽셀','X(mm)','Y(mm)','개수']
                        sheets[re.sub(r'[\\/*?:\[\]]','',fname)[:25]]=df
                except: continue
        try:
            with pd.ExcelWriter(path,engine='openpyxl') as w:
                pd.DataFrame(summary).to_excel(w,sheet_name="전체 요약",index=False)
                for sn,df in sheets.items(): df.to_excel(w,sheet_name=sn,index=False)
            self.ui.status(f"✅ {len(summary)}장 완료 — 보고서 저장됨", GR)
            messagebox.showinfo("완료",f"{len(summary)}장 검사 완료.")
            os.startfile(os.path.abspath(path))
        except Exception as ex: messagebox.showerror("오류",str(ex))

    def _highlight(self, pt):
        tmp = self.img_out.copy()
        cv2.circle(tmp,(int(pt['x_px']),int(pt['y_px'])),
                   int(self.vision.single_area**0.5)+12,(0,255,255),3)
        self.ui.show(tmp, self.ui.lbl_3)

    def on_img_click(self, e):
        if not self.particles or self.img_out is None: return
        r = self.ui.ratios['lbl_3']
        rx,ry = e.x/r, e.y/r
        near = min(self.particles, key=lambda p:(p['x_px']-rx)**2+(p['y_px']-ry)**2)
        self.ui.tree.selection_set(near['tid']); self.ui.tree.see(near['tid'])
        self._highlight(near)

    def on_tree_select(self, e):
        sel = self.ui.tree.selection()
        if not sel or self.img_out is None: return
        pt = next((r for r in self.particles if r.get('tid')==sel[0]),None)
        if pt: self._highlight(pt)

    def do_save_recipe(self):
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON","*.json")])
        if path: save_recipe(path, self.ui.get_params()); messagebox.showinfo("완료","저장 완료.")

    def do_load_recipe(self):
        path = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if path:
            try: self.ui.set_params(load_recipe(path)); self.update_all()
            except Exception as ex: messagebox.showerror("오류",str(ex))

    def do_save_excel(self):
        if not self.particles: return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                            filetypes=[("Excel","*.xlsx")])
        if path:
            pd.DataFrame(self.particles)[['no','x_px','y_px','x_mm','y_mm','cnt']].to_excel(path,index=False)
            messagebox.showinfo("완료","저장 완료.")
            os.startfile(os.path.abspath(path))

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
