import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Data: k value, T * K^2, IoU±variance, Dice±variance
data = [
    (2, 32, "0.5712974096604045,0.0091865912", "0.6642553071708625,0.010915174"),
    (2, 512, "0.592026063515026,0.007873285", "0.6869177658838289,0.008971774"),
    (4, 128, "0.596928427666561,0.008340874", "0.6891852826927265,0.009557461"),
    (4, 512, "0.6232896041172539,0.007149501", "0.7146852388446996,0.008285605"),
    (6, 288, "0.618845405061446,0.006759322", "0.71271455193154,0.007777963"),
    (6, 512, "0.6310019119527762,0.006211777", "0.7249789287522597,0.006861101"),
    (8, 512, "0.6292980829847351,0.005984289", "0.7242277990776765,0.00680326")
]

def parse_data(raw_data):
    k_values, params, iou_means, iou_stds, dice_means, dice_stds = [], [], [], [], [], []
    for k, param, iou_str, dice_str in raw_data:
        k_values.append(k)
        params.append(param)
        iou_mean, iou_var = map(float, iou_str.split(','))
        dice_mean, dice_var = map(float, dice_str.split(','))
        iou_means.append(iou_mean)
        iou_stds.append(np.sqrt(iou_var))
        dice_means.append(dice_mean)
        dice_stds.append(np.sqrt(dice_var))
    return (np.array(k_values), np.array(params), np.array(iou_means), 
            np.array(iou_stds), np.array(dice_means), np.array(dice_stds))

# --- Data for T=8 ---
data_t8 = [d for d in data if d[1] / (d[0]**2) == 8]
k_t8, _, iou_t8, iou_std_t8, dice_t8, dice_std_t8 = parse_data(data_t8)

# --- Data for T*K^2=512 ---
data_tk512 = [d for d in data if d[1] == 512]
k_tk512, _, iou_tk512, iou_std_tk512, dice_tk512, dice_std_tk512 = parse_data(data_tk512)

def create_smooth_curve(x, y):
    if len(x) > 3:
        x_smooth = np.linspace(x.min(), x.max(), 300)
        spline = make_interp_spline(x, y, k=3)
        y_smooth = spline(x_smooth)
        return x_smooth, y_smooth
    return x, y

# Create smooth curves
k_smooth_t8, iou_smooth_t8 = create_smooth_curve(k_t8, iou_t8)
_, dice_smooth_t8 = create_smooth_curve(k_t8, dice_t8)

k_smooth_tk512, iou_smooth_tk512 = create_smooth_curve(k_tk512, iou_tk512)
_, dice_smooth_tk512 = create_smooth_curve(k_tk512, dice_tk512)


# --- Plotting ---
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(28, 14))

# Subplot 1: IoU for T=8
ax1.plot(k_smooth_t8, iou_smooth_t8, color='blue', label='IoU')
ax1.errorbar(k_t8, iou_t8, yerr=iou_std_t8, fmt='o', color='blue', capsize=5)
ax1.set_title('IoU vs. k (T=8) on ISIC2016 with SAM', fontsize=24, fontweight='bold')
ax1.set_xlabel('k Value', fontsize=24)
ax1.set_ylabel('IoU Score', fontsize=24)
ax1.grid(True, alpha=0.5)
ax1.legend()
ax1.tick_params(axis='both', which='major', labelsize=20)

# Subplot 2: Dice for T=8
ax2.plot(k_smooth_t8, dice_smooth_t8, color='red', label='Dice')
ax2.errorbar(k_t8, dice_t8, yerr=dice_std_t8, fmt='s', color='red', capsize=5)
ax2.set_title('Dice vs. k (T=8) on ISIC2016 with SAM', fontsize=24, fontweight='bold')
ax2.set_xlabel('k Value', fontsize=24)
ax2.set_ylabel('Dice Score', fontsize=24)
ax2.grid(True, alpha=0.5)
ax2.legend()
ax2.tick_params(axis='both', which='major', labelsize=20)

# Subplot 3: IoU for T*k^2=512
ax3.plot(k_smooth_tk512, iou_smooth_tk512, color='blue', label='IoU')
ax3.errorbar(k_tk512, iou_tk512, yerr=iou_std_tk512, fmt='o', color='blue', capsize=5)
ax3.set_title('IoU vs. k (T*k^2=512) on ISIC2016 with SAM', fontsize=24, fontweight='bold')
ax3.set_xlabel('k Value', fontsize=24)
ax3.set_ylabel('IoU Score', fontsize=24)
ax3.grid(True, alpha=0.5)
ax3.legend()
ax3.tick_params(axis='both', which='major', labelsize=20)

# Subplot 4: Dice for T*k^2=512
ax4.plot(k_smooth_tk512, dice_smooth_tk512, color='red', label='Dice')
ax4.errorbar(k_tk512, dice_tk512, yerr=dice_std_tk512, fmt='s', color='red', capsize=5)
ax4.set_title('Dice vs. k (T*k^2=512) on ISIC2016 with SAM', fontsize=24, fontweight='bold')
ax4.set_xlabel('k Value', fontsize=24)
ax4.set_ylabel('Dice Score', fontsize=24)
ax4.grid(True, alpha=0.5)
ax4.legend()
ax4.tick_params(axis='both', which='major', labelsize=16)

plt.tight_layout()
plt.savefig('iou_dice_curves_by_conditions.png', dpi=300)
print("Chart saved as iou_dice_curves_by_conditions.png")
