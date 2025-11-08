import numpy as np
import rasterio
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_laplace
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
from datetime import datetime

# === Configuration ===
output_dir = r"Curvature_Aware_Result_Dir"
os.makedirs(output_dir, exist_ok=True)

def compute_curvature(Z, sigma=1):
    curvature = np.abs(gaussian_laplace(Z, sigma=sigma))
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    return curvature

def quadratic_interpolate_patch(Z_patch, cell_width, cell_height):
    if np.any(np.isnan(Z_patch)):
        Z_patch = np.nan_to_num(Z_patch, nan=np.nanmean(Z_patch))
    m, n = Z_patch.shape
    x = np.linspace(0, cell_width, n)
    y = np.linspace(0, cell_height, m)
    X, Y = np.meshgrid(x, y)
    A = np.vstack([X.ravel()**2, Y.ravel()**2, (X*Y).ravel(), X.ravel(), Y.ravel(), np.ones_like(X.ravel())]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, Z_patch.ravel(), rcond=None)
    def integrand(x, y):
        return coeffs[0]*x**2 + coeffs[1]*y**2 + coeffs[2]*x*y + coeffs[3]*x + coeffs[4]*y + coeffs[5]
    from scipy.integrate import dblquad
    volume, _ = dblquad(integrand, 0, cell_height, 0, cell_width)
    return volume if not np.isnan(volume) else np.mean(Z_patch) * cell_width * cell_height

def spline_integrate_patch(Z_patch, x_coords, y_coords, x0, y0, cell_width, cell_height, local_curv, max_curv):
    if np.all(np.isnan(Z_patch)) or np.std(Z_patch[~np.isnan(Z_patch)]) < 1e-6:
        return quadratic_interpolate_patch(Z_patch, cell_width, cell_height)
    try:
        spline = RectBivariateSpline(y_coords, x_coords, Z_patch, kx=3, ky=3, s=0.1)
        volume = spline.integral(y0, y0 + cell_height, x0, x0 + cell_width)
        return volume
    except:
        return quadratic_interpolate_patch(Z_patch, cell_width, cell_height)

def adaptive_spline_volume(Z, curvature, cell_width, cell_height, curvature_threshold, max_depth=6):
    padded_Z = np.pad(Z, 3, mode='edge')
    padded_curv = np.pad(curvature, 3, mode='edge')
    max_curv = np.max(curvature[~np.isnan(curvature)])
    m, n = Z.shape
    volume, nan_count = 0.0, 0

    def process_cell(i, j, x0, y0, width, height, depth):
        nonlocal volume, nan_count
        if i + 7 > padded_Z.shape[0] or j + 7 > padded_Z.shape[1]:
            return
        Z_patch = padded_Z[i:i+7, j:j+7]
        x_coords = np.linspace(x0 - 3*width, x0 + 3*width, 7)
        y_coords = np.linspace(y0 - 3*height, y0 + 3*height, 7)
        local_curv = padded_curv[i+3, j+3]
        if local_curv < curvature_threshold or depth >= max_depth:
            patch_volume = spline_integrate_patch(Z_patch, x_coords, y_coords, x0, y0, width, height, local_curv, max_curv)
            if np.isnan(patch_volume):
                nan_count += 1
                patch_volume = quadratic_interpolate_patch(Z_patch, width, height)
            volume += patch_volume
        else:
            dx = width / 2
            dy = height / 2
            for di in range(2):
                for dj in range(2):
                    sub_x0 = x0 + dj * dx
                    sub_y0 = y0 + di * dy
                    process_cell(i + di, j + dj, sub_x0, sub_y0, dx, dy, depth + 1)

    for i in range(m):
        for j in range(n):
            y0 = i * cell_height
            x0 = j * cell_width
            process_cell(i, j, x0, y0, cell_width, cell_height, 0)

    if nan_count > 0:
        print(f"Warning: {nan_count} patches returned NaN, fallback used for these patches")
    return volume

def calculate_dtm_difference(dtm1_path, dtm2_path):
    with rasterio.open(dtm1_path) as src1, rasterio.open(dtm2_path) as src2:
        dtm1 = src1.read(1).astype(np.float32)
        dtm2 = src2.read(1).astype(np.float32)
        transform = src1.transform
        cell_width, cell_height = transform.a, -transform.e
        if src1.crs != src2.crs:
            raise ValueError("DTM CRS mismatch.")
    if dtm1.shape != dtm2.shape:
        raise ValueError("DTM shape mismatch.")
    diff = dtm2 - dtm1
    diff[np.isnan(diff)] = 0
    valid_mask = ~np.isnan(dtm1) & ~np.isnan(dtm2)
    pixel_area = cell_width * cell_height
    total_area = np.sum(valid_mask) * pixel_area
    curvature = compute_curvature(diff)
    threshold = np.percentile(curvature[~np.isnan(curvature)], 95)
    volume = adaptive_spline_volume(diff, curvature, cell_width, cell_height, threshold, max_depth=3)
    return volume, total_area

def multi_file_select(title):
    root = tk.Tk()
    root.withdraw()
    return list(filedialog.askopenfilenames(title=title, filetypes=[("GeoTIFF", "*.tif")]))

def save_csv(results):
    df = pd.DataFrame(results, columns=["DTM_BEFORE", "DTM_AFTER", "Area_m2", "Volume_m3"])
    out_path = os.path.join(output_dir, "curvature_aware_volume_results.csv")
    df.to_csv(out_path, index=False)
    print(f"CSV saved to {out_path}")

def save_detailed_log(results):
   
    log_data = []
    for res in results:
        log_data.append({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'DTM_Before': res[0],
            'DTM_After': res[1],
            'Area_m2': res[2],
            'Volume_m3': res[3]
        })
    
    log_df = pd.DataFrame(log_data)
    log_file = os.path.join(output_dir, "volume_detailed_log.csv")
    
    # Append to existing file or create new one
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)
    
    print(f"Detailed log saved to {log_file}")

if __name__ == "__main__":
    print("Select BEFORE DTMs")
    before_dtms = multi_file_select("Select BEFORE DTMs")
    print("Select AFTER DTMs")
    after_dtms = multi_file_select("Select AFTER DTMs")

    if len(before_dtms) != len(after_dtms):
        print("Mismatch in number of files selected.")
    else:
        results = []
        for dtm1, dtm2 in zip(before_dtms, after_dtms):
            try:
                volume, area = calculate_dtm_difference(dtm1, dtm2)
                results.append((os.path.basename(dtm1), os.path.basename(dtm2), area, volume))
                print(f"{os.path.basename(dtm1)} vs {os.path.basename(dtm2)}: Volume: {volume:.2f} m³, Area: {area:.2f} m²")
            except Exception as e:
                print(f"Error with {dtm1} & {dtm2}: {e}")
        save_csv(results)
        save_detailed_log(results)