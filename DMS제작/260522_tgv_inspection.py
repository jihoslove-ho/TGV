import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import pandas as pd
import json
import io
import os
import re
import ctypes  # Windows API 호출용

# ==============================================================================
# 1. Data I/O Module
# ==============================================================================
def save_recipe_to_file(filepath, recipe_dict):
    with open(filepath, 'w') as f:
        json.dump(recipe_dict, f, indent=4)

def load_recipe_from_file(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_particles_to_excel(filepath, particle_list):
    if not particle_list:
        return False
    df = pd.DataFrame(particle_list)[['no', 'x_px', 'y_px', 'cnt']]
    df.to_excel(filepath, index=False)
    return True

# ==============================================================================
# 2. Vision Core Module
# ==============================================================================
class VisionProcessor:
    def __init__(self):
        self.PX_PER_UM = 2.6
        self.PX_PER_MM = 2600.0
        self.PSL_RADIUS_PX = 1.95
        self.SINGLE_A = 11.94

    def update_resolution_parameters(self, params):
        scale_px = params.get('scale_px', 130.0)
        scale_um = params.get('scale_um', 50.0)
        particle_diam_um = params.get('part_diam', 1.5)

        self.PX_PER_UM = scale_px / scale_um
        self.PX_PER_MM = self.PX_PER_UM * 1000.0

        psl_diam_px = particle_diam_um * self.PX_PER_UM
        self.PSL_RADIUS_PX = psl_diam_px / 2.0
        self.SINGLE_A = np.pi * (self.PSL_RADIUS_PX ** 2)

    def find_tgv_center(self, img_orig):
        gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_cnt) > 2000:
                (x, y), radius = cv2.minEnclosingCircle(largest_cnt)
                return (int(x), int(y)), int(radius)
        return None, 0

    def create_roi_mask(self, shape, center, inner_r, outer_r):
        mask = np.zeros(shape, dtype=np.uint8)
        if center:
            cv2.circle(mask, center, outer_r, 255, -1)
            if inner_r > 0:
                cv2.circle(mask, center, inner_r, 0, -1)
        else:
            mask.fill(255)
        return mask

    def process_preview(self, img_orig, center, params):
        self.update_resolution_parameters(params)

        gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=params['ct'], beta=params['br'])

        if params['cl']:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)

        if params['tp']:
            k_size = params['tk']
            if k_size % 2 == 0: k_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        b_k = params['bl']
        if b_k > 0: 
            if b_k % 2 == 0: b_k += 1
            gray = cv2.GaussianBlur(gray, (b_k, b_k), 0)

        th_mode = params.get('tm', 'Manual (수동)')
        
        if th_mode == 'Otsu (자동-전역)':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        elif th_mode == 'Adaptive (자동-국부)':
            block_size = params.get('ab', 15)
            if block_size % 2 == 0: block_size += 1
            c_val = params.get('ac', -5)
            
            th_type = cv2.THRESH_BINARY_INV if params.get('ai', False) else cv2.THRESH_BINARY
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, th_type, block_size, c_val)
        else:
            _, binary = cv2.threshold(gray, params['th'], 255, cv2.THRESH_BINARY)

        mask = self.create_roi_mask(gray.shape, center, params['roi_in'], params['roi_out'])
        binary = cv2.bitwise_and(binary, binary, mask=mask)
        
        if center:
            tgv_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(tgv_mask, center, params['roi_in'] + 5, 255, -1) 
            binary = cv2.bitwise_and(binary, binary, mask=cv2.bitwise_not(tgv_mask))
            
        return binary

    def run_analysis(self, img_orig, img_binary, params):
        self.update_resolution_parameters(params)
        
        contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = img_orig.copy() 
        
        particle_list = []
        idx, total_ea = 1, 0
        
        min_area_thresh = self.SINGLE_A * params['ma'] 
        GREEN = (0, 255, 0)
        is_cluster_mode = params.get('cm', False)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_thresh: continue 

            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

            if is_cluster_mode:
                count = 1
            else:
                count = max(1, int(round(area / self.SINGLE_A)))

            if count > 0:
                total_ea += count
                cv2.drawContours(out, [cnt], -1, GREEN, 2) 

                if is_cluster_mode:
                    x, y, w, h = cv2.boundingRect(cnt)
                    text = "1"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    font_thickness = 2 
                    cv2.putText(out, text, (x, y - 7), font, font_scale, (0, 0, 0), font_thickness + 2)
                    cv2.putText(out, text, (x, y - 7), font, font_scale, GREEN, font_thickness)
                else:
                    if count != 1:
                        x, y, w, h = cv2.boundingRect(cnt)
                        text = f"{count}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        font_thickness = 2 
                        cv2.putText(out, text, (x, y - 7), font, font_scale, (0, 0, 0), font_thickness + 2)
                        cv2.putText(out, text, (x, y - 7), font, font_scale, GREEN, font_thickness)

                x_mm, y_mm = cx / self.PX_PER_MM, cy / self.PX_PER_MM
                particle_list.append({'no': idx, 'x_px': cx, 'y_px': cy, 'x_mm': x_mm, 'y_mm': y_mm, 'cnt': count})
                idx += 1
                
        return out, particle_list, total_ea


# ==============================================================================
# 3. UI Manager Module
# ==============================================================================
class UIManager:
    def __init__(self, root, callbacks):
        self.root = root
        self.callbacks = callbacks
        
        self.image_cache = { 'lbl_1': (None, False), 'lbl_2': (None, True), 'lbl_3': (None, False) }
        self.display_ratios = { 'lbl_1': 1.0, 'lbl_2': 1.0, 'lbl_3': 1.0 }

        self.v_mag_selection = tk.StringVar(value="사용자 정의")
        self.v_scale_px = tk.DoubleVar(value=130.0)
        self.v_scale_um = tk.DoubleVar(value=50.0)
        self.v_particle_diam = tk.DoubleVar(value=1.5)
        
        self.v_thresh_mode = tk.StringVar(value="Otsu (자동-전역)") 
        self.v_bright = tk.IntVar(value=0)
        self.v_contrast = tk.DoubleVar(value=1.0)
        self.v_blur = tk.IntVar(value=0) 
        self.v_thresh = tk.IntVar(value=150)
        
        self.v_adapt_block = tk.IntVar(value=15)
        self.v_adapt_c = tk.IntVar(value=-5)
        self.v_adapt_inv = tk.BooleanVar(value=False)
        
        self.v_clahe = tk.BooleanVar(value=False)
        self.v_tophat = tk.BooleanVar(value=True)
        self.v_tophat_k = tk.IntVar(value=35)
        self.v_min_area = tk.DoubleVar(value=0.5)
        self.v_circularity = tk.DoubleVar(value=0.7)
        self.v_roi_outer = tk.IntVar(value=800) 
        self.v_roi_inner = tk.IntVar(value=0)   
        
        self.v_cluster_mode = tk.BooleanVar(value=False) 

        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.active_context_key = None  
        self.context_menu.add_command(label="📋 클립보드에 이미지 복사", command=self.copy_image_to_clipboard)

        self.tab_listboxes = {} 

        self.setup_ui()

    def add_scale(self, parent, label, var, _from, _to, res, cmd):
        f = tk.Frame(parent)
        f.pack(side=tk.TOP, pady=2, anchor="w")
        tk.Label(f, text=label, font=("Arial", 8), width=18, anchor="w").pack(side=tk.LEFT, padx=2)
        
        def step_val(direction):
            val = var.get() + (res * direction)
            val = round(val, 4)
            if val < _from: val = _from
            if val > _to: val = _to
            var.set(val)
            cmd() 

        tk.Button(f, text="◀", command=lambda: step_val(-1), padx=4, pady=0).pack(side=tk.LEFT)
        tk.Scale(f, from_=_from, to=_to, resolution=res, variable=var, orient=tk.HORIZONTAL, length=130, command=lambda x: cmd()).pack(side=tk.LEFT)
        tk.Button(f, text="▶", command=lambda: step_val(1), padx=4, pady=0).pack(side=tk.LEFT)

    def get_recipe_params(self):
        return {
            "mag_sel": self.v_mag_selection.get(),
            "scale_px": self.v_scale_px.get(),
            "scale_um": self.v_scale_um.get(),
            "part_diam": self.v_particle_diam.get(),
            "tm": self.v_thresh_mode.get(),
            "br": self.v_bright.get(), "ct": self.v_contrast.get(),
            "bl": self.v_blur.get(), "th": self.v_thresh.get(),
            "ab": self.v_adapt_block.get(), "ac": self.v_adapt_c.get(), 
            "ai": self.v_adapt_inv.get(),
            "cl": self.v_clahe.get(), "tp": self.v_tophat.get(), "tk": self.v_tophat_k.get(),
            "ma": self.v_min_area.get(), "ci": self.v_circularity.get(),
            "roi_out": self.v_roi_outer.get(), "roi_in": self.v_roi_inner.get(),
            "cm": self.v_cluster_mode.get() 
        }

    def set_recipe_params(self, params):
        self.v_mag_selection.set(params.get("mag_sel", "사용자 정의"))
        self.v_scale_px.set(params.get("scale_px", 130.0))
        self.v_scale_um.set(params.get("scale_um", 50.0))
        self.v_particle_diam.set(params.get("part_diam", 1.5))

        self.v_thresh_mode.set(params.get("tm", "Otsu (자동-전역)"))
        self.v_bright.set(params.get("br", 0))
        self.v_contrast.set(params.get("ct", 1.0))
        self.v_blur.set(params.get("bl", 0))
        self.v_thresh.set(params.get("th", 150))
        
        self.v_adapt_block.set(params.get("ab", 15))
        self.v_adapt_c.set(params.get("ac", -5))
        self.v_adapt_inv.set(params.get("ai", False))
        
        self.v_clahe.set(params.get("cl", False))
        self.v_tophat.set(params.get("tp", True))
        self.v_tophat_k.set(params.get("tk", 35))
        self.v_min_area.set(params.get("ma", 0.5))
        self.v_circularity.set(params.get("ci", 0.7))
        self.v_roi_outer.set(params.get("roi_out", 800))
        self.v_roi_inner.set(params.get("roi_in", 0))
        self.v_cluster_mode.set(params.get("cm", False)) 
        
        self.render_thresh_params()

    def setup_ui(self):
        top_f = tk.Frame(self.root, pady=10, bg="#f8f9fa"); top_f.pack(side=tk.TOP, fill=tk.X)
        tk.Button(top_f, text="폴더 로드", command=self.callbacks['load_folder'], bg="#e67e22", fg="white", font=("bold")).pack(side=tk.LEFT, padx=10)
        tk.Button(top_f, text="레시피 저장", command=self.callbacks['save_recipe']).pack(side=tk.LEFT, padx=5)
        tk.Button(top_f, text="레시피 열기", command=self.callbacks['load_recipe']).pack(side=tk.LEFT, padx=5)
        tk.Button(top_f, text="▶ 분석 실행", bg="#007bff", fg="white", font=("bold"), command=self.callbacks['run_analysis']).pack(side=tk.LEFT, padx=30)
        self.res_lbl = tk.Label(top_f, text="상태: 대기 중", font=("bold", 12), bg="#f8f9fa"); self.res_lbl.pack(side=tk.RIGHT, padx=20)

        main_body = tk.Frame(self.root)
        main_body.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)

        self.tabs_frame = tk.LabelFrame(main_body, text="📁 배율별 파일 분류 탐색기", padx=5, pady=5)
        self.tabs_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        self.notebook = ttk.Notebook(self.tabs_frame, width=240)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        img_frame = tk.Frame(main_body)
        img_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        img_frame.columnconfigure(0, weight=1)
        img_frame.columnconfigure(1, weight=1)
        img_frame.columnconfigure(2, weight=1)
        img_frame.rowconfigure(0, weight=1)

        self.lbl_1 = self.create_img_pane(img_frame, "[1. 원본]", 0)
        self.lbl_2 = self.create_img_pane(img_frame, "[2. 튜닝 프리뷰 (마스크 적용)]", 1)
        self.lbl_3 = self.create_img_pane(img_frame, "[3. 분석 결과]", 2, is_clickable=True)

        bottom_f = tk.Frame(self.root, height=280)
        bottom_f.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        bottom_f.pack_propagate(False) 
        
        mag_f = tk.LabelFrame(bottom_f, text="현미경 배율 및 파티클 설정", padx=10, pady=5, fg="blue")
        mag_f.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        tk.Label(mag_f, text="자주 쓰는 배율 프리셋:", font=("Arial", 8, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)
        mag_cb = ttk.Combobox(mag_f, textvariable=self.v_mag_selection, values=["사용자 정의", "10x 프리셋", "100x 프리셋", "200x 프리셋"], state="readonly", width=18)
        mag_cb.grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
        mag_cb.bind("<<ComboboxSelected>>", self.on_mag_preset_change)

        tk.Label(mag_f, text="스케일 바 물리 크기 (um):", font=("Arial", 8)).grid(row=2, column=0, sticky="w", pady=4)
        self.ent_scale_um = tk.Entry(mag_f, textvariable=self.v_scale_um, width=10)
        self.ent_scale_um.grid(row=2, column=1, sticky="w", padx=5)
        
        tk.Label(mag_f, text="스케일 바 픽셀 크기 (px):", font=("Arial", 8)).grid(row=3, column=0, sticky="w", pady=4)
        self.ent_scale_px = tk.Entry(mag_f, textvariable=self.v_scale_px, width=10)
        self.ent_scale_px.grid(row=3, column=1, sticky="w", padx=5)

        tk.Label(mag_f, text="목표 파티클 직경 (um):", font=("Arial", 8, "bold"), fg="#2c3e50").grid(row=4, column=0, sticky="w", pady=8)
        self.ent_part_diam = tk.Entry(mag_f, textvariable=self.v_particle_diam, width=10, bg="#ecf0f1")
        self.ent_part_diam.grid(row=4, column=1, sticky="w", padx=5)

        self.ent_scale_um.bind("<FocusOut>", lambda e: self.callbacks['update_all']())
        self.ent_scale_um.bind("<Return>", lambda e: self.callbacks['update_all']())
        self.ent_scale_px.bind("<FocusOut>", lambda e: self.callbacks['update_all']())
        self.ent_scale_px.bind("<Return>", lambda e: self.callbacks['update_all']())
        self.ent_part_diam.bind("<FocusOut>", lambda e: self.callbacks['update_all']())
        self.ent_part_diam.bind("<Return>", lambda e: self.callbacks['update_all']())
        
        recipe_container = tk.LabelFrame(bottom_f, text="전처리 레시피 (스크롤 가능)", padx=2, pady=2)
        recipe_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        canvas = tk.Canvas(recipe_container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(recipe_container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(canvas.find_withtag('all')[0], width=e.width))
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        row1 = tk.Frame(self.scrollable_frame); row1.pack(fill=tk.X, pady=5)
        tk.Checkbutton(row1, text="CLAHE", variable=self.v_clahe, command=self.callbacks['update_all'], font=("bold")).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(row1, text="Top-Hat 필터", variable=self.v_tophat, command=self.callbacks['update_all'], font=("bold"), fg="blue").pack(side=tk.LEFT, padx=5)
        self.add_scale(row1, "TopHat 크기", self.v_tophat_k, 15, 100, 2, self.callbacks['update_all'])
        
        tk.Label(row1, text=" | ", fg="gray").pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(row1, text="클러스터 그룹핑 (덩어리 1개 취급)", variable=self.v_cluster_mode, command=self.callbacks['run_analysis'], font=("bold"), fg="red").pack(side=tk.LEFT, padx=5)

        row2 = tk.Frame(self.scrollable_frame); row2.pack(fill=tk.X, pady=2)
        self.add_scale(row2, "밝기", self.v_bright, -100, 100, 1, self.callbacks['update_all'])
        self.add_scale(row2, "대비", self.v_contrast, 0.1, 3.0, 0.01, self.callbacks['update_all'])
        self.add_scale(row2, "블러", self.v_blur, 0, 15, 1, self.callbacks['update_all'])
        
        th_f = tk.Frame(self.scrollable_frame); th_f.pack(fill=tk.X, pady=4)
        tk.Label(th_f, text="이진화 방식", font=("Arial", 8, "bold"), width=18, anchor="w", fg="blue").pack(side=tk.LEFT, padx=2)
        cb = ttk.Combobox(th_f, textvariable=self.v_thresh_mode, values=["Manual (수동)", "Otsu (자동-전역)", "Adaptive (자동-국부)"], state="readonly", width=18)
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>", self.on_thresh_mode_change)
        
        self.th_param_f = tk.Frame(self.scrollable_frame)
        self.th_param_f.pack(fill=tk.X, pady=2)
        self.render_thresh_params() 

        row3 = tk.Frame(self.scrollable_frame); row3.pack(fill=tk.X, pady=2)
        self.add_scale(row3, "최소 감지 면적 비율", self.v_min_area, 0.005, 5.0, 0.005, self.callbacks['update_all'])
        self.add_scale(row3, "단일 판단 원형도", self.v_circularity, 0.4, 1.0, 0.05, self.callbacks['update_all'])

        roi_f = tk.LabelFrame(bottom_f, text="ROI 도넛 영역", padx=10, pady=5, fg="red")
        roi_f.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.add_scale(roi_f, "외곽선", self.v_roi_outer, 100, 1500, 5, self.callbacks['update_all'])
        self.add_scale(roi_f, "내곽선", self.v_roi_inner, 0, 500, 5, self.callbacks['update_all'])

        table_f = tk.Frame(bottom_f, width=350); table_f.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        self.tree = ttk.Treeview(table_f, columns=("no", "x", "y", "cnt"), show="headings", height=8)
        self.tree.heading("no", text="순번"); self.tree.heading("x", text="X (mm)"); self.tree.heading("y", text="Y (mm)"); self.tree.heading("cnt", text="개수")
        self.tree.column("no", width=40, anchor="center"); self.tree.column("x", width=80, anchor="center")
        self.tree.column("y", width=80, anchor="center"); self.tree.column("cnt", width=60, anchor="center")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH)
        self.tree.bind("<<TreeviewSelect>>", self.callbacks['on_tree_select'])
        
        btn_box = tk.Frame(table_f)
        btn_box.pack(fill=tk.X, pady=2)
        tk.Button(btn_box, text="Excel 데이터 저장", command=self.callbacks['save_excel'], bg="#1D6F42", fg="white", height=1).pack(fill=tk.X, side=tk.TOP, pady=1)
        tk.Button(btn_box, text="⚡ 폴더 일괄 자동 검사 및 출력", command=self.callbacks['run_batch_processing'], bg="#d35400", fg="white", font=("bold", 9), height=1).pack(fill=tk.X, side=tk.TOP, pady=1)

    def rebuild_tab_notebook(self, separated_files_dict):
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)
        self.tab_listboxes.clear()

        for mag_key in sorted(separated_files_dict.keys(), key=lambda x: (0, int(x[1:])) if x.startswith('x') and x[1:].isdigit() else (1, x)):
            files_list = separated_files_dict[mag_key]
            if not files_list: continue 

            tab_frame = tk.Frame(self.notebook)
            self.notebook.add(tab_frame, text=f" {mag_key} ({len(files_list)}) ")

            scroll = ttk.Scrollbar(tab_frame, orient="vertical")
            lbox = tk.Listbox(tab_frame, yscrollcommand=scroll.set, width=30, font=("Malgun Gothic", 9), exportselection=False)
            scroll.config(command=lbox.yview)
            
            lbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scroll.pack(side=tk.RIGHT, fill=tk.Y)

            for f in files_list:
                lbox.insert(tk.END, f)

            lbox.bind("<<ListboxSelect>>", self.callbacks['on_file_select'])
            self.tab_listboxes[mag_key] = lbox

    def on_tab_changed(self, event=None):
        try:
            current_tab_idx = self.notebook.index(self.notebook.select())
            current_tab_text = self.notebook.tab(current_tab_idx, "text").split()[0]
            
            for mag_key, lbox in self.tab_listboxes.items():
                if mag_key != current_tab_text:
                    lbox.selection_clear(0, tk.END)
        except Exception:
            pass

    def on_mag_preset_change(self, event=None):
        sel = self.v_mag_selection.get()
        if sel == "10x 프리셋":
            self.v_scale_um.set(100.0)
            self.v_scale_px.set(200.0)
        elif sel == "100x 프리셋":
            self.v_scale_um.set(50.0)
            self.v_scale_px.set(260.0)
        elif sel == "200x 프리셋":
            self.v_scale_um.set(50.0)
            self.v_scale_px.set(520.0)
        self.callbacks['update_all']()

    def render_thresh_params(self):
        for widget in self.th_param_f.winfo_children():
            widget.destroy()
            
        mode = self.v_thresh_mode.get()
        if mode == "Manual (수동)":
            self.add_scale(self.th_param_f, "수동 임계값", self.v_thresh, 0, 255, 1, self.callbacks['update_all'])
        elif mode == "Adaptive (자동-국부)":
            self.add_scale(self.th_param_f, "적응형 블록 크기", self.v_adapt_block, 3, 99, 2, self.callbacks['update_all'])
            self.add_scale(self.th_param_f, "보정 상수 (C)", self.v_adapt_c, -30, 30, 1, self.callbacks['update_all'])
            
            tk.Label(self.th_param_f, text=" | ", fg="gray").pack(side=tk.LEFT, padx=10)
            tk.Checkbutton(self.th_param_f, text="결과 이미지 반전 (Invert)", variable=self.v_adapt_inv, 
                           command=self.callbacks['update_all'], font=("Arial", 8, "bold"), fg="#d35400").pack(side=tk.LEFT, pady=5)
            
        elif mode == "Otsu (자동-전역)":
            tk.Label(self.th_param_f, text="※ Otsu는 자동으로 임계값을 결정합니다.", font=("Arial", 8), fg="gray").pack(side=tk.LEFT, padx=18)

    def on_thresh_mode_change(self, event=None):
        self.render_thresh_params()  
        self.callbacks['update_all']() 

    def create_img_pane(self, parent, title, col_idx, is_clickable=False):
        f = tk.Frame(parent, relief="sunken", bd=1)
        f.grid(row=0, column=col_idx, sticky="nsew", padx=3, pady=3)
        tk.Label(f, text=title, font=("Arial", 9, "bold")).pack(side=tk.TOP, fill=tk.X)
        
        lbl = tk.Label(f, bg="black")
        lbl.pack(expand=True, fill=tk.BOTH)
        
        lbl_key = f"lbl_{col_idx+1}"
        lbl.bind("<Configure>", lambda event, k=lbl_key: self.on_pane_resize(k))
        lbl.bind("<Button-3>", lambda event, k=lbl_key: self.show_context_menu(event, k))
        
        if is_clickable: 
            lbl.bind("<Button-1>", self.callbacks['on_image_click'])
        return lbl

    def show_context_menu(self, event, key):
        if self.image_cache[key][0] is not None:  
            self.active_context_key = key
            self.context_menu.post(event.x_root, event.y_root)

    # --- [수정] 64비트 가상 주소 충돌(Access Violation)을 완벽 방지한 정밀 복사 엔진 ---
    def copy_image_to_clipboard(self):
        if not self.active_context_key: return
        img, is_gray = self.image_cache[self.active_context_key]
        if img is None: return

        try:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if is_gray else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            
            output = io.BytesIO()
            pil_img.save(output, format="BMP")
            data = output.getvalue()[14:]  # BMP 파일 헤더 제거 (DIB 바이너리 추출)
            output.close()

            # Win32 API 패키지 커스텀 선언 (64비트 포인터 누수 방지 기법 도입)
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            # ctypes 함수들의 인수(argtypes) 및 반환형(restype)을 64비트 크기에 맞게 엄격하게 명시
            kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
            kernel32.GlobalAlloc.restype = ctypes.c_void_p
            
            kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
            kernel32.GlobalLock.restype = ctypes.c_void_p
            
            kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
            kernel32.GlobalUnlock.restype = ctypes.c_int
            
            user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
            user32.SetClipboardData.restype = ctypes.c_void_p

            CF_DIB = 8
            GMEM_MOVEABLE = 0x0002

            if user32.OpenClipboard(None):
                user32.EmptyClipboard()
                
                # 안전하게 전역 메모리 할당 (이제 64비트 환경에서 주소가 0으로 잘리지 않습니다)
                hCd = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
                if not hCd:
                    user32.CloseClipboard()
                    raise MemoryError("클립보드 메모리 할당에 실패했습니다.")

                pCd = kernel32.GlobalLock(hCd)
                if pCd:
                    # 메모리에 안정적으로 쓰기(Writing) 수행
                    ctypes.memmove(pCd, data, len(data))
                    kernel32.GlobalUnlock(hCd)
                    user32.SetClipboardData(CF_DIB, hCd)
                
                user32.CloseClipboard()
                self.res_lbl.config(text="📋 클립보드 이미지 복사 완료 (Ctrl+V 가능)!", fg="blue")
            else:
                raise RuntimeError("Windows 클립보드 제어 권한을 획득하지 못했습니다.")

        except Exception as e:
            messagebox.showerror("오류", f"클립보드 정밀 복사 중 예외가 발생했습니다:\n{e}")

    def on_pane_resize(self, key):
        lbl_widget = getattr(self, key, None)
        if lbl_widget is None: return
        
        img, is_gray = self.image_cache[key]
        if img is None: return

        lbl_w = lbl_widget.winfo_width()
        lbl_h = lbl_widget.winfo_height()
        if lbl_w < 10 or lbl_h < 10: return

        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if is_gray else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = rgb.shape[:2]

        ratio_w = lbl_w / img_w
        ratio_h = lbl_h / img_h
        dynamic_ratio = min(ratio_w, ratio_h)

        new_w = max(1, int(img_w * dynamic_ratio))
        new_h = max(1, int(img_h * dynamic_ratio))
        
        self.display_ratios[key] = dynamic_ratio

        resized = cv2.resize(rgb, (new_w, new_h))
        img_tk = ImageTk.PhotoImage(Image.fromarray(resized))
        lbl_widget.config(image=img_tk)
        lbl_widget.image = img_tk

    def display_img_on_label(self, img, label, is_gray=False):
        if img is None: return
        
        target_key = None
        if label == self.lbl_1: target_key = 'lbl_1'
        elif label == self.lbl_2: target_key = 'lbl_2'
        elif label == self.lbl_3: target_key = 'lbl_3'
        
        if target_key:
            self.image_cache[target_key] = (img.copy(), is_gray)
            self.on_pane_resize(target_key)

    def update_table(self, particle_list):
        for item in self.tree.get_children(): self.tree.delete(item)
        for p in particle_list:
            t_id = self.tree.insert("", tk.END, values=(p['no'], f"{p['x_mm']:.4f}", f"{p['y_mm']:.4f}", p['cnt']))
            p['tid'] = t_id


# ==============================================================================
# 4. Main Controller Module
# ==============================================================================
class MainController:
    def __init__(self, root):
        self.root = root
        self.root.title("TGV PSL Master Station v17.2 - Standard Clipboard Fixed")
        self.root.geometry("1920x980")

        self.current_dir = None
        self.separated_files = {} 
        
        self.img_orig = None
        self.img_binary = None
        self.processed_display_img = None
        self.tgv_center = None
        self.particle_list = []

        self.vision = VisionProcessor()
        
        callbacks = {
            'load_folder': self.load_folder,
            'on_file_select': self.on_file_select,
            'save_recipe': self.save_recipe,
            'load_recipe': self.load_recipe,
            'run_analysis': self.run_analysis,
            'update_all': self.update_all,
            'save_excel': self.save_excel,
            'on_image_click': self.on_image_click,
            'on_tree_select': self.on_tree_select,
            'run_batch_processing': self.run_batch_processing
        }
        self.ui = UIManager(self.root, callbacks)

    def parse_magnification_from_filename(self, filename):
        match = re.search(r'[_]?x(\d+)', filename, re.IGNORECASE)
        if match:
            return f"x{match.group(1)}"
        return "기타"

    def load_folder(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.current_dir = dir_path
            self.separated_files = {}
            
            valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
            try:
                files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(valid_exts)])
                if not files:
                    messagebox.showwarning("알림", "선택한 폴더 내에 분석 가능한 이미지 파일이 없습니다.")
                    return
                
                for f in files:
                    mag_tag = self.parse_magnification_from_filename(f)
                    if mag_tag not in self.separated_files:
                        self.separated_files[mag_tag] = []
                    self.separated_files[mag_tag].append(f)
                
                self.ui.rebuild_tab_notebook(self.separated_files)
                self.ui.res_lbl.config(text=f"📁 폴더 분류 완료 (총 이미지: {len(files)}개)", fg="blue")
                
                if self.ui.tab_listboxes:
                    first_mag_key = list(self.ui.tab_listboxes.keys())[0]
                    first_lbox = self.ui.tab_listboxes[first_mag_key]
                    first_lbox.selection_set(0)
                    self.on_file_select(None, is_initial_load=True)
                
            except Exception as e:
                messagebox.showerror("오류", f"폴더 내 파일을 스캔하고 분류하는 중 예외 발생:\n{e}")

    def on_file_select(self, event, is_initial_load=False):
        if self.current_dir is None: return

        try:
            current_tab_idx = self.ui.notebook.select()
            if not current_tab_idx: return
            active_tab_text = self.ui.notebook.tab(current_tab_idx, "text").split()[0]
            
            active_lbox = self.ui.tab_listboxes.get(active_tab_text)
            if active_lbox is None: return
            
            selection = active_lbox.curselection()
            if not selection: return
            
            filename = active_lbox.get(selection[0])
            full_path = os.path.join(self.current_dir, filename)
            
            img_array = np.fromfile(full_path, np.uint8)
            self.img_orig = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if self.img_orig is None:
                messagebox.showerror("오류", f"[{filename}] 이미지를 읽어올 수 없습니다.")
                return

            res = self.vision.find_tgv_center(self.img_orig)
            if res and res[0] is not None:
                self.tgv_center, inner_r = res
                
                if is_initial_load:
                    params = self.ui.get_recipe_params()
                    params['roi_in'] = max(0, inner_r - 5)
                    params['roi_out'] = inner_r + 300 
                    self.ui.set_recipe_params(params)
            else:
                self.tgv_center = None
                print("TGV 홀을 찾을 수 없습니다.")

            self.ui.display_img_on_label(self.img_orig, self.ui.lbl_1)
            self.update_all()
            
        except Exception as e:
             messagebox.showerror("오류", f"이미지 파일을 읽는 중 예외 발생:\n{e}")

    def update_all(self, *args):
        if self.img_orig is None: return
        try:
            params = self.ui.get_recipe_params()
            self.img_binary = self.vision.process_preview(self.img_orig, self.tgv_center, params)
            self.ui.display_img_on_label(self.img_binary, self.ui.lbl_2, is_gray=True)
            self.run_analysis()
        except tk.TclError:
            pass

    def run_analysis(self, *args):
        if self.img_binary is None: return
        params = self.ui.get_recipe_params()
        
        out_img, self.particle_list, total_ea = self.vision.run_analysis(self.img_orig, self.img_binary, params)
        
        if self.tgv_center:
            cv2.circle(out_img, self.tgv_center, params['roi_out'], (0, 0, 255), 2)
            if params['roi_in'] > 0:
                cv2.circle(out_img, self.tgv_center, params['roi_in'], (255, 0, 255), 2)
        
        self.processed_display_img = out_img.copy()
        
        self.ui.display_img_on_label(out_img, self.ui.lbl_3)
        self.ui.update_table(self.particle_list)
        
        try:
            current_tab_idx = self.ui.notebook.select()
            active_tab_text = self.ui.notebook.tab(current_tab_idx, "text").split()[0]
            active_lbox = self.ui.tab_listboxes.get(active_tab_text)
            selection = active_lbox.curselection()
            filename = active_lbox.get(selection[0]) if selection else ""
            self.ui.res_lbl.config(text=f"📄 {filename} | 결과: {total_ea} ea", fg="black")
        except Exception:
            pass

    def run_batch_processing(self):
        if not self.current_dir or not self.separated_files:
            messagebox.showwarning("경고", "먼저 폴더를 로드해 주세요.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Master Report", "*.xlsx")])
        if not save_path: return

        recipe_params = self.ui.get_recipe_params()
        summary_data = [] 
        detailed_sheets = {} 

        self.ui.res_lbl.config(text="⚡ 일괄 자동 검사 진행 중...", fg="red")
        self.root.update()

        for mag_tag, file_list in self.separated_files.items():
            for filename in file_list:
                full_path = os.path.join(self.current_dir, filename)
                try:
                    img_arr = np.fromfile(full_path, np.uint8)
                    img_batch = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    if img_batch is None: continue

                    center_res, _ = self.vision.find_tgv_center(img_batch)
                    bin_batch = self.vision.process_preview(img_batch, center_res, recipe_params)
                    _, p_list, total_ea = self.vision.run_analysis(img_batch, bin_batch, recipe_params)
                    
                    summary_data.append({
                        "파일명": filename,
                        "현미경 배율": mag_tag,
                        "최종 파티클 검출수 (ea)": total_ea
                    })

                    if p_list:
                        df_det = pd.DataFrame(p_list)[['no', 'x_px', 'y_px', 'x_mm', 'y_mm', 'cnt']]
                        df_det.columns = ['순번', 'X_픽셀좌표', 'Y_픽셀좌표', 'X_위치(mm)', 'Y_위치(mm)', '추정개수']
                        sheet_name = re.sub(r'[\\/*?:\[\]]', '', filename)[:25]
                        detailed_sheets[sheet_name] = df_det
                except Exception:
                    continue

        try:
            with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                df_sum = pd.DataFrame(summary_data)
                df_sum.to_excel(writer, sheet_name="📊 전체 검사 요약 보고서", index=False)
                for sheet_title, df_sheet in detailed_sheets.items():
                    df_sheet.to_excel(writer, sheet_name=sheet_title, index=False)

            self.ui.res_lbl.config(text="📊 폴더 일괄 자동 검사 보고서 출력 완료!", fg="blue")
            messagebox.showinfo("검사 완료", f"총 {len(summary_data)}장의 이미지 자동 검사가 완료되었습니다.\n\n확인을 누르면 파일이 자동으로 즉시 열립니다.")
            
            os.startfile(os.path.abspath(save_path))
            
        except Exception as e:
            messagebox.showerror("엑셀 빌드 에러", f"종합 리포트 파일 생성 중 문제가 발생했습니다:\n{e}")
            self.ui.res_lbl.config(text="상태: 대기 중", fg="black")

    def highlight_particle(self, p):
        temp = self.processed_display_img.copy()
        cv2.circle(temp, (int(p['x_px']), int(p['y_px'])), int(self.vision.PSL_RADIUS_PX)+10, (0, 255, 255), 3)
        self.ui.display_img_on_label(temp, self.ui.lbl_3)

    def on_image_click(self, event):
        if not self.particle_list or self.processed_display_img is None: return
        ratio = self.ui.display_ratios['lbl_3']
        rx, ry = event.x / ratio, event.y / ratio
        near = min(self.particle_list, key=lambda p: np.sqrt((p['x_px']-rx)**2 + (p['y_px']-ry)**2))
        self.ui.tree.selection_set(near['tid'])
        self.ui.tree.see(near['tid'])
        self.highlight_particle(near)

    def on_tree_select(self, event):
        selected = self.ui.tree.selection()
        if not selected or self.processed_display_img is None: return
        p = next((item for item in self.particle_list if item.get('tid') == selected[0]), None)
        if p: self.highlight_particle(p)

    def save_recipe(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if path:
            save_recipe_to_file(path, self.ui.get_recipe_params())
            messagebox.showinfo("완료", "레시피 저장 완료.")

    def load_recipe(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if path:
            try:
                params = load_recipe_from_file(path)
                self.ui.set_recipe_params(params)
                self.update_all()
            except Exception as e:
                messagebox.showerror("오류", f"레시피를 불러오지 못했습니다:\n{e}")

    def save_excel(self):
        if not self.particle_list: return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if path:
            if save_particles_to_excel(path, self.particle_list):
                messagebox.showinfo("완료", "엑셀 저장 완료.\n\n확인을 누르면 파일이 자동으로 즉시 열립니다.")
                os.startfile(os.path.abspath(path))

if __name__ == "__main__":
    root = tk.Tk()
    app = MainController(root)
    root.mainloop()