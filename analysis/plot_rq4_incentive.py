#!/usr/bin/env python3
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Ensure we can import sibling modules when invoked from repo root
try:
    from plot_style import apply_style, COLORS
except ImportError:  # pragma: no cover
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from plot_style import apply_style, COLORS
apply_style()

# Data from Table RQ4 in the paper
schemes = ["No Incentive", "Fixed Reward", "Dynamic Reward"]
participation = [71.25, 86.25, 87.50]
attack_rate = [28.75, 13.75, 12.50]
accuracy = [98.28, 97.86, 98.01]

colors = {
    'participation': COLORS['participation'],
    'attack': COLORS['attack'],
    'accuracy': COLORS['accuracy'],
}

x = range(len(schemes))
width = 0.25

# Increased figsize for better aesthetics: 6.0 -> 8.0 inches width
fig, ax1 = plt.subplots(figsize=(7.50, 4.50))

# Reduced edgecolor linewidth for less density
ax1.bar([i - width for i in x], participation, width=width, color=colors['participation'], label='Participation (%)', edgecolor='#444444', linewidth=0.3)
ax1.bar(x, attack_rate, width=width, color=colors['attack'], label='Attack Rate (%)', edgecolor='#444444', linewidth=0.3)
ax1.bar([i + width for i in x], accuracy, width=width, color=colors['accuracy'], label='Accuracy (%)', edgecolor='#444444', linewidth=0.3)

ax1.set_xticks(list(x))
ax1.set_xticklabels(schemes, rotation=0)
ax1.set_ylim(0, 100)
ax1.set_ylabel('Percentage (%)')
ax1.grid(axis='y', linestyle='--', alpha=0.3)
ax1.legend(ncol=3, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.20), frameon=False)

# Optimized layout with padding for better spacing
plt.tight_layout(pad=0.5)

# Save into repository-level author-kit figures dir (three levels up)
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'author-kit-CVPR2026-v1-latex-', 'figures'))
os.makedirs(root, exist_ok=True)
out_path = os.path.join(root, 'rq4_incentive.pdf')
# High quality output with minimal padding
plt.savefig(out_path, format='pdf', bbox_inches='tight', pad_inches=0.02, dpi=300)
print(f"Saved {out_path}")

