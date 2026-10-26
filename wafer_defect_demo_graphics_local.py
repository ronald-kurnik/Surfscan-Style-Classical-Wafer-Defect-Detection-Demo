"""
Enhanced Surfscan-style wafer defect detection demo with extensive graphics.
Classical pipeline + rich visualizations of every major step.

Saves figures to the current working directory (or the folder of this script).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import opening, disk
import warnings
warnings.filterwarnings('ignore')

# Style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f8f8',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold'
})

def generate_synthetic_wafer(size=512, n_particles=40, n_scratches=2, noise_level=0.045):
    """Generate synthetic unpatterned wafer with particles, scratches, haze and noise."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Base + mild radial haze
    base = 0.28 + 0.08 * np.exp(-(X**2 + Y**2) / 0.7)
    img = base.copy()
    
    # Particles
    particle_locs = []
    for _ in range(n_particles):
        cx = np.random.randint(30, size-30)
        cy = np.random.randint(30, size-30)
        radius = np.random.uniform(1.8, 5.5)
        intensity = np.random.uniform(0.45, 1.0)
        yy, xx = np.ogrid[:size, :size]
        mask = (xx - cx)**2 + (yy - cy)**2 <= radius**2
        img[mask] += intensity
        particle_locs.append((cy, cx, radius))
    
    # Scratches
    for _ in range(n_scratches):
        start = np.random.randint(60, size-60, 2)
        angle = np.random.uniform(0, np.pi)
        length = np.random.randint(80, 180)
        end = start + (length * np.array([np.cos(angle), np.sin(angle)])).astype(int)
        end = np.clip(end, 30, size-30)
        rr = np.linspace(start[0], end[0], 300).astype(int)
        cc = np.linspace(start[1], end[1], 300).astype(int)
        for r, c in zip(rr, cc):
            if 0 <= r < size and 0 <= c < size:
                img[max(0, r-1):min(size, r+2), max(0, c-1):min(size, c+2)] += 0.7
    
    # Noise
    img += np.random.normal(0, noise_level, img.shape)
    img = np.clip(img, 0, 1.2)
    return img, particle_locs

def full_pipeline(img):
    """Complete classical detection pipeline with intermediate images."""
    # 1. Gaussian blur
    blurred = gaussian(img, sigma=1.2, preserve_range=True)
    
    # 2. Background estimation (large-scale haze)
    background = gaussian(img, sigma=15, preserve_range=True)
    residual = np.clip(img - background * 0.6, 0, None)
    
    # 3. Threshold on residual
    thresh = threshold_otsu(residual)
    binary = residual > thresh * 1.15
    
    # 4. Morphological opening
    opened = opening(binary, disk(1))
    
    # 5. Connected components + properties
    labeled = label(opened)
    props = regionprops(labeled, intensity_image=img)
    
    defects = []
    for p in props:
        if p.area >= 6:
            defects.append({
                'centroid': p.centroid,
                'area': p.area,
                'mean_intensity': p.mean_intensity,
                'max_intensity': p.max_intensity,
                'bbox': p.bbox,
                'eccentricity': p.eccentricity,
                'solidity': p.solidity
            })
    return {
        'blurred': blurred,
        'background': background,
        'residual': residual,
        'binary': binary,
        'opened': opened,
        'labeled': labeled,
        'defects': defects,
        'thresh': thresh
    }

def create_graphics(img, results, particle_locs, out_prefix='wafer_demo'):
    """Save all figures using a local (relative) prefix."""
    defects = results['defects']
    
    # Make sure the output directory exists (current working directory is fine)
    out_dir = os.path.dirname(out_prefix) or '.'
    os.makedirs(out_dir, exist_ok=True)
    
    # ========== Figure 1: Full Pipeline Overview ==========
    fig1, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig1.suptitle('Surfscan-Style Classical Defect Detection Pipeline', fontsize=14, fontweight='bold')
    
    steps = [
        (img, '1. Raw Synthetic Wafer\n(particles + scratches + haze + noise)'),
        (results['blurred'], '2. Gaussian Filtered\n(σ = 1.2)'),
        (results['background'], '3. Background / Haze Estimate\n(large σ)'),
        (results['residual'], '4. Residual Image\n(signal - haze)'),
        (results['binary'], '5. Thresholded Binary'),
        (results['opened'], '6. Morphological Opening'),
        (results['labeled'], '7. Labeled Components'),
        (img, '8. Final Detection Overlay')
    ]
    
    for ax, (im, title) in zip(axes.flat, steps):
        if title.startswith('7'):
            ax.imshow(im, cmap='nipy_spectral')
        elif title.startswith('8'):
            ax.imshow(im, cmap='gray')
            for d in defects:
                cy, cx = d['centroid']
                ax.plot(cx, cy, 'r+', markersize=10, markeredgewidth=1.5)
            ax.set_title(f'{title}\n({len(defects)} defects)', fontweight='bold')
            continue
        else:
            ax.imshow(im, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    fig1.savefig(f'{out_prefix}_pipeline.png', dpi=140, bbox_inches='tight')
    plt.close(fig1)
    
    # ========== Figure 2: Detection Details + Annotations ==========
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('Defect Detection Results', fontsize=14, fontweight='bold')
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(results['residual'], cmap='hot')
    axes[1].set_title('Residual (Defect Signal)')
    axes[1].axis('off')
    
    axes[2].imshow(img, cmap='gray')
    for d in defects:
        cy, cx = d['centroid']
        # Color by size
        color = 'red' if d['area'] < 20 else 'yellow' if d['area'] < 50 else 'lime'
        axes[2].plot(cx, cy, '+', color=color, markersize=12, markeredgewidth=2)
        # Small bounding box
        minr, minc, maxr, maxc = d['bbox']
        rect = Rectangle((minc, minr), maxc-minc, maxr-minr,
                         fill=False, edgecolor=color, linewidth=1.2, alpha=0.8)
        axes[2].add_patch(rect)
    axes[2].set_title(f'Overlay: {len(defects)} defects\n(red=small, yellow=med, green=large)')
    axes[2].axis('off')
    
    plt.tight_layout()
    fig2.savefig(f'{out_prefix}_overlay.png', dpi=140, bbox_inches='tight')
    plt.close(fig2)
    
    # ========== Figure 3: Statistics & Distributions ==========
    fig3, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig3.suptitle('Defect Characterization Statistics', fontsize=14, fontweight='bold')
    
    areas = [d['area'] for d in defects]
    intensities = [d['mean_intensity'] for d in defects]
    eccentricities = [d['eccentricity'] for d in defects]
    solidities = [d['solidity'] for d in defects]
    
    # Histogram of areas
    axes[0, 0].hist(areas, bins=15, color='steelblue', edgecolor='black', alpha=0.8)
    axes[0, 0].set_title('Defect Area Distribution')
    axes[0, 0].set_xlabel('Area (pixels)')
    axes[0, 0].set_ylabel('Count')
    
    # Intensity vs Area
    sc = axes[0, 1].scatter(areas, intensities, c=eccentricities, cmap='viridis', s=40, alpha=0.8)
    axes[0, 1].set_title('Intensity vs Area\n(color = eccentricity)')
    axes[0, 1].set_xlabel('Area')
    axes[0, 1].set_ylabel('Mean Intensity')
    plt.colorbar(sc, ax=axes[0, 1], label='Eccentricity')
    
    # Eccentricity (shape)
    axes[0, 2].hist(eccentricities, bins=12, color='coral', edgecolor='black', alpha=0.8)
    axes[0, 2].set_title('Eccentricity (shape)\n0=circle, 1=line')
    axes[0, 2].set_xlabel('Eccentricity')
    
    # Intensity histogram of whole image
    axes[1, 0].hist(img.ravel(), bins=50, color='gray', alpha=0.7, label='Raw')
    axes[1, 0].hist(results['residual'].ravel(), bins=50, color='orange', alpha=0.6, label='Residual')
    axes[1, 0].axvline(results['thresh'], color='red', linestyle='--', label='Threshold')
    axes[1, 0].set_title('Intensity Distributions')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Intensity')
    
    # Solidity
    axes[1, 1].hist(solidities, bins=12, color='seagreen', edgecolor='black', alpha=0.8)
    axes[1, 1].set_title('Solidity (compactness)')
    axes[1, 1].set_xlabel('Solidity')
    
    # Spatial map of defect density (simple)
    density = np.zeros_like(img)
    for d in defects:
        cy, cx = map(int, d['centroid'])
        density[max(0,cy-8):cy+8, max(0,cx-8):cx+8] += 1
    axes[1, 2].imshow(density, cmap='Reds')
    axes[1, 2].set_title('Defect Spatial Density')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    fig3.savefig(f'{out_prefix}_stats.png', dpi=140, bbox_inches='tight')
    plt.close(fig3)
    
    # ========== Figure 4: Zoomed examples of individual defects ==========
    n_show = min(6, len(defects))
    if n_show == 0:
        print("No defects found to zoom.")
        return
    
    fig4, axes = plt.subplots(2, n_show, figsize=(3*n_show, 6))
    fig4.suptitle('Zoomed Defect Examples (with bounding box & properties)', fontsize=13, fontweight='bold')
    
    # Ensure axes is 2D even if n_show == 1
    if n_show == 1:
        axes = axes.reshape(2, 1)
    
    # Sort by area descending
    sorted_def = sorted(defects, key=lambda x: x['area'], reverse=True)[:n_show]
    
    for i, d in enumerate(sorted_def):
        minr, minc, maxr, maxc = d['bbox']
        # Add padding
        pad = 8
        r0, r1 = max(0, minr-pad), min(img.shape[0], maxr+pad)
        c0, c1 = max(0, minc-pad), min(img.shape[1], maxc+pad)
        crop = img[r0:r1, c0:c1]
        
        axes[0, i].imshow(crop, cmap='gray')
        axes[0, i].set_title(f'Area={d["area"]:.0f}\nInt={d["mean_intensity"]:.2f}')
        axes[0, i].axis('off')
        
        # Residual crop
        crop_res = results['residual'][r0:r1, c0:c1]
        axes[1, i].imshow(crop_res, cmap='hot')
        axes[1, i].set_title(f'Ecc={d["eccentricity"]:.2f}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    fig4.savefig(f'{out_prefix}_zooms.png', dpi=140, bbox_inches='tight')
    plt.close(fig4)
    
    print(f"Saved 4 high-resolution figures in the current directory:")
    print(f"  {out_prefix}_pipeline.png")
    print(f"  {out_prefix}_overlay.png")
    print(f"  {out_prefix}_stats.png")
    print(f"  {out_prefix}_zooms.png")
    print(f"\nDetected {len(defects)} defects (threshold = {results['thresh']:.3f})")

# ==================== Run ====================
if __name__ == '__main__':
    np.random.seed(42)
    img, particle_locs = generate_synthetic_wafer(size=480, n_particles=35, n_scratches=2)
    results = full_pipeline(img)
    
    # Save next to the script / in the current working directory
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
    out_prefix = os.path.join(script_dir, 'wafer_demo')
    create_graphics(img, results, particle_locs, out_prefix=out_prefix)
    
    print("\nTop 5 defects by area:")
    for d in sorted(results['defects'], key=lambda x: -x['area'])[:5]:
        print(f"  area={d['area']:5.0f}  intensity={d['mean_intensity']:.2f}  "
              f"ecc={d['eccentricity']:.2f}  centroid=({d['centroid'][0]:.0f},{d['centroid'][1]:.0f})")
