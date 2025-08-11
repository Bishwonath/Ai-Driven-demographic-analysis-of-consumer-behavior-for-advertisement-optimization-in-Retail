import matplotlib.pyplot as plt
import numpy as np

# Existing data for Starbucks (example numbers for before AI)
starbucks_before = [2, 5, 10]  # hypothetical pre-AI values
starbucks_after = [5, 18, 22]  # your existing values

# Existing data for Levi's (example numbers)
levis_before = [5, 7]  # hypothetical pre-AI values
levis_after = [15, 21]

# Existing data for Coca-Cola (example numbers)
cocacola_before = [3, 30]
cocacola_after = [6.9, 60]

# Labels
starbucks_labels = [
    "Mobile order mix (percentage points)",
    "Year-over-year mobile order growth (%)",
    "Average ticket price increase (%)"
]

levis_labels = [
    "Loose-fit jeans sales increase (%) (Conservative)",
    "Loose-fit jeans sales increase (%) (Optimistic)"
]

cocacola_labels = [
    "Business customers reached (millions)",
    "Marketing budget on AI-driven digital ads (%)"
]

def plot_before_after(labels, before, after, title, xlabel):
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12,6))
    bars1 = ax.barh(x - width/2, before, width, label='Before AI', color='gray')
    bars2 = ax.barh(x + width/2, after, width, label='After AI', color='forestgreen')

    ax.set(yticks=x, yticklabels=labels)
    ax.invert_yaxis()  # Highest on top
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=16, weight='bold')
    ax.legend()
    
    # Add data labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            width_bar = bar.get_width()
            ax.text(width_bar + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{width_bar}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()


# Plot Starbucks before vs after AI
plot_before_after(
    starbucks_labels, 
    starbucks_before, 
    starbucks_after,
    "Starbucks: Growth Before vs After AI Implementation",
    "Percentage (%) or Percentage Points"
)

# Plot Levi's before vs after AI
plot_before_after(
    levis_labels,
    levis_before,
    levis_after,
    "Levi's: Sales Growth Before vs After AI Trend Analysis",
    "Sales Increase (%)"
)

# Plot Coca-Cola before vs after AI
plot_before_after(
    cocacola_labels,
    cocacola_before,
    cocacola_after,
    "Coca-Cola: Marketing Reach & Spend Before vs After AI",
    "Millions or Percentage (%)"
)
