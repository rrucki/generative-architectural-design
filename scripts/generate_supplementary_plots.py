import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import struct
from tensorboard.compat.proto import event_pb2

PROJECT_ROOT = Path("/Users/rafalrucki/Documents/IT/generative-architectural-design")
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "manuscript" / "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set clean scientific plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'grid.color': '#eeeeee',
    'grid.linewidth': 0.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

def moving_average(a, n=50):
    ret = []
    for i in range(len(a)):
        start = max(0, i - n // 2)
        end = min(len(a), i + n // 2 + 1)
        ret.append(np.mean(a[start:end]))
    return np.array(ret)

def moving_std(a, n=50):
    ret = []
    for i in range(len(a)):
        start = max(0, i - n // 2)
        end = min(len(a), i + n // 2 + 1)
        ret.append(np.std(a[start:end]))
    return np.array(ret)

def read_tfevents_fast(filepath: Path, tags):
    # Initialize lists for each tag
    data_dict = {tag: ([], []) for tag in tags}
    
    with open(filepath, 'rb') as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            length, = struct.unpack('Q', header)
            f.read(4)  # Skip length CRC
            data = f.read(length)
            f.read(4)  # Skip data CRC
            
            event = event_pb2.Event()
            event.ParseFromString(data)
            
            if event.HasField('summary'):
                for value in event.summary.value:
                    if value.tag in data_dict:
                        steps, vals = data_dict[value.tag]
                        steps.append(event.step)
                        vals.append(value.simple_value)
                        
    ret = {}
    for tag in tags:
        steps, vals = data_dict[tag]
        ret[tag] = (np.array(steps), np.array(vals))
    return ret

def extract_run_data(run_name: str, tags):
    run_dir = LOG_DIR / run_name
    tb_dirs = [d for d in run_dir.parent.iterdir() if d.is_dir() and d.name.startswith(run_name + "_")]
    if not tb_dirs:
        return {tag: (np.array([]), np.array([])) for tag in tags}
    
    tb_dir = tb_dirs[0]
    event_files = list(tb_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return {tag: (np.array([]), np.array([])) for tag in tags}
    
    event_file = event_files[0]
    return read_tfevents_fast(event_file, tags)

def generate_supplementary_plots():
    runs = [
        "stage_1_1754723612",
        "stage_2_1754945765",
        "stage_3_1756213572",
        "stage_4_tunning_1757784606"
    ]
    
    tags = [
        "train/loss",
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/explained_variance"
    ]
    
    # 1. Extract values in a single pass per run
    data = {tag: [] for tag in tags}
    for run in runs:
        print(f"Loading logs for run: {run}...")
        run_data = extract_run_data(run, tags)
        for tag in tags:
            steps, vals = run_data[tag]
            data[tag].append((steps, vals))

    # Plot Concatenated Curriculum Dynamics
    # To keep stages equal width on X-axis (0-1, 1-2, 2-3, 3-4), we normalize the steps within each stage
    # mapping them to [0, 1], [1, 2], [2, 3], [3, 4] respectively.
    
    def get_concatenated_curve(tag):
        x_all = []
        y_all = []
        for i, (steps, vals) in enumerate(data[tag]):
            if len(steps) == 0:
                continue
            # Downsample Stage 3 and Stage 4 by a factor of 16 to keep step density uniform
            if i in [2, 3]:
                steps = steps[::16]
                vals = vals[::16]
            x_norm = np.linspace(i, i + 1, len(steps))
            x_all.append(x_norm)
            y_all.append(vals)
        if not x_all:
            return np.array([]), np.array([])
        return np.concatenate(x_all), np.concatenate(y_all)

    # 1. Plot Total & Value Losses
    print("Plotting total & value losses...")
    x_loss, y_loss = get_concatenated_curve("train/loss")
    _, y_v_loss = get_concatenated_curve("train/value_loss")
    
    if len(x_loss) > 0:
        fig, ax = plt.subplots(figsize=(11, 5.8))
        
        # Smooth
        w = 150
        y_loss_sm = moving_average(y_loss, w)
        y_v_loss_sm = moving_average(y_v_loss, w)
        
        # Raw background cloud (restored)
        ax.plot(x_loss, y_loss, color="#1E3A8A", alpha=0.15, linewidth=0.6)
        ax.plot(x_loss, y_v_loss, color="#EF4444", alpha=0.15, linewidth=0.6)
        
        # Smoothed line
        ax.plot(x_loss, y_loss_sm, label="Total Loss", color="#1E3A8A", linewidth=2.0)
        ax.plot(x_loss, y_v_loss_sm, label="Value Loss", color="#EF4444", linewidth=1.5)
        ax.set_yscale('symlog', linthresh=0.1)
        ax.set_ylim(bottom=100)
        
        # Draw vertical lines for stages (matching Figure 1)
        for stage in [1.0, 2.0, 3.0]:
            ax.axvline(x=stage, color='#7f7f7f', linestyle='--', alpha=0.7)
            
        ax.set_xlim(0, 4)
        ax.set_title("Total and Critic Value Losses (SymLog)", fontsize=12, fontweight="bold", pad=15)
        ax.set_xlabel("Curriculum Learning Stage & Training Progress", fontsize=10, fontweight='bold')
        ax.set_ylabel("Loss Value (SymLog)", fontsize=10, fontweight='bold')
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(['0', '64M', '128M', '1.15B', '2.18B'], fontsize=9)
        
        # Add Stage annotations (using transAxes to place them uniformly at the top of the plot area)
        ax.text(0.125, 0.92, 'Stage 1\n(Containment)', transform=ax.transAxes, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.3'))
        ax.text(0.375, 0.92, 'Stage 2\n(Area)', transform=ax.transAxes, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.3'))
        ax.text(0.625, 0.92, 'Stage 3\n(Tract Width)', transform=ax.transAxes, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.3'))
        ax.text(0.875, 0.92, 'Stage 4\n(Orthogonality)', transform=ax.transAxes, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.3'))
        
        ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0")
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "supplementary_losses_total_value.png", dpi=300)
        plt.close()

    # 1b. Plot Policy Gradient Loss
    print("Plotting policy gradient loss...")
    _, y_pg_loss = get_concatenated_curve("train/policy_gradient_loss")
    
    if len(x_loss) > 0:
        fig, ax = plt.subplots(figsize=(11, 5.8))
        
        # Smooth
        w = 150
        y_pg_loss_sm = moving_average(y_pg_loss, w)
        
        # Raw background cloud (restored)
        ax.plot(x_loss, y_pg_loss, color="#0D9488", alpha=0.15, linewidth=0.6)
        
        # Smoothed line
        ax.plot(x_loss, y_pg_loss_sm, label="Policy Gradient Loss", color="#0D9488", linewidth=2.0)
        
        # Draw vertical lines for stages
        for stage in [1.0, 2.0, 3.0]:
            ax.axvline(x=stage, color='#7f7f7f', linestyle='--', alpha=0.7)
            
        ax.set_xlim(0, 4)
        ax.set_ylim(-0.0125, 0.001)
        ax.set_title("Policy Gradient Loss (Linear)", fontsize=12, fontweight="bold", pad=15)
        ax.set_xlabel("Curriculum Learning Stage & Training Progress", fontsize=10, fontweight='bold')
        ax.set_ylabel("Loss Value", fontsize=10, fontweight='bold')
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(['0', '64M', '128M', '1.15B', '2.18B'], fontsize=9)
        
        # Add Stage annotations (using transAxes to place them uniformly at the top of the plot area)
        ax.text(0.125, 0.92, 'Stage 1\n(Containment)', transform=ax.transAxes, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.3'))
        ax.text(0.375, 0.92, 'Stage 2\n(Area)', transform=ax.transAxes, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.3'))
        ax.text(0.625, 0.92, 'Stage 3\n(Tract Width)', transform=ax.transAxes, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.3'))
        ax.text(0.875, 0.92, 'Stage 4\n(Orthogonality)', transform=ax.transAxes, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.3'))
        
        ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0")
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "supplementary_losses_policy.png", dpi=300)
        plt.close()

    # 2. Plot Policy Entropy
    print("Plotting entropy...")
    x_ent, y_ent = get_concatenated_curve("train/entropy_loss")
    if len(x_ent) > 0:
        fig, ax = plt.subplots(figsize=(11, 5.8))
        # Note: entropy_loss in stable-baselines3 is negative of entropy, so we plot -y_ent
        y_entropy = -y_ent
        w = 80
        y_ent_sm = moving_average(y_entropy, w)
        
        # Raw background cloud (restored)
        ax.plot(x_ent, y_entropy, color="#4F46E5", alpha=0.15, linewidth=0.6)
        
        # Smoothed line
        ax.plot(x_ent, y_ent_sm, color="#4F46E5", linewidth=2.0, label="Policy Entropy")
        
        # Draw vertical lines for stages
        for stage in [1.0, 2.0, 3.0]:
            ax.axvline(x=stage, color='#7f7f7f', linestyle='--', alpha=0.7)
            
        ax.set_xlim(0, 4)
        ax.set_title("Policy Entropy Decay (Exploration vs. Exploitation)", fontsize=12, fontweight="bold", pad=15)
        ax.set_xlabel("Curriculum Learning Stage & Training Progress", fontsize=10, fontweight='bold')
        ax.set_ylabel("Policy Entropy", fontsize=10, fontweight='bold')
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(['0', '64M', '128M', '1.15B', '2.18B'], fontsize=9)
        
        # Add Stage annotations (using transAxes to place them uniformly at the top of the plot area)
        ax.text(0.125, 0.92, 'Stage 1\n(Containment)', transform=ax.transAxes, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.3'))
        ax.text(0.375, 0.92, 'Stage 2\n(Area)', transform=ax.transAxes, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.3'))
        ax.text(0.625, 0.92, 'Stage 3\n(Tract Width)', transform=ax.transAxes, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.3'))
        ax.text(0.875, 0.92, 'Stage 4\n(Orthogonality)', transform=ax.transAxes, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.3'))
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "supplementary_entropy.png", dpi=300)
        plt.close()

    # 3. Plot Explained Variance
    print("Plotting explained variance...")
    x_ev, y_ev = get_concatenated_curve("train/explained_variance")
    if len(x_ev) > 0:
        fig, ax = plt.subplots(figsize=(11, 5.8))
        w = 150
        y_ev_sm = moving_average(y_ev, w)
        
        # Raw background cloud (restored)
        ax.plot(x_ev, y_ev, color="#0891B2", alpha=0.15, linewidth=0.6)
        
        # Smoothed line
        ax.plot(x_ev, y_ev_sm, color="#0891B2", linewidth=2.0, label="Explained Variance")
        
        # Draw vertical lines for stages
        for stage in [1.0, 2.0, 3.0]:
            ax.axvline(x=stage, color='#7f7f7f', linestyle='--', alpha=0.7)
            
        ax.set_xlim(0, 4)
        ax.set_ylim(0.6, 1.02)
        ax.set_title("Value Function Explained Variance", fontsize=12, fontweight="bold", pad=15)
        ax.set_xlabel("Curriculum Learning Stage & Training Progress", fontsize=10, fontweight='bold')
        ax.set_ylabel("Explained Variance", fontsize=10, fontweight='bold')
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(['0', '64M', '128M', '1.15B', '2.18B'], fontsize=9)
        
        # Add Stage annotations (using transAxes to place them uniformly at the top of the plot area)
        ax.text(0.125, 0.92, 'Stage 1\n(Containment)', transform=ax.transAxes, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.3'))
        ax.text(0.375, 0.92, 'Stage 2\n(Area)', transform=ax.transAxes, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.3'))
        ax.text(0.625, 0.92, 'Stage 3\n(Tract Width)', transform=ax.transAxes, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.3'))
        ax.text(0.875, 0.92, 'Stage 4\n(Orthogonality)', transform=ax.transAxes, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.3'))
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "supplementary_explained_variance.png", dpi=300)
        plt.close()

    print("Supplementary plots generated successfully!")

if __name__ == "__main__":
    generate_supplementary_plots()
