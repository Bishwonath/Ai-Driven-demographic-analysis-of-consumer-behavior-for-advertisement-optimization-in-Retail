import matplotlib.pyplot as plt

# Starbucks AI Impact metrics with descriptions
starbucks_metrics = [
    ("Increase in mobile order mix (percentage points)", 5),
    ("Year-over-year growth in mobile orders (%)", 18),
    ("Increase in average ticket price due to AI personalization (%)", 22)
]
starbucks_labels, starbucks_values = zip(*starbucks_metrics)

# Levi's sales uplift with more context
levis_metrics = [
    ("Conservative estimate of loose-fit jeans sales increase (%)", 15),
    ("Optimistic estimate of loose-fit jeans sales increase (%)", 21)
]
levis_labels, levis_values = zip(*levis_metrics)

# Coca-Cola marketing impact metrics with description
cocacola_metrics = [
    ("Number of business customers reached via AI-driven push messages (millions)", 6.9),
    ("Percentage of marketing budget allocated to AI-driven digital advertising (%)", 60)
]
cocacola_labels, cocacola_values = zip(*cocacola_metrics)

# Plotting with descriptive titles and axis labels
plt.figure(figsize=(16, 9))

plt.subplot(1, 3, 1)
plt.barh(starbucks_labels, starbucks_values, color='forestgreen')
plt.title("Starbucks: AI-Driven Personalization and Operational Efficiency Impact")
plt.xlabel("Improvement / Growth (%) or Percentage Points")
plt.tight_layout()

plt.subplot(1, 3, 2)
plt.barh(levis_labels, levis_values, color='royalblue')
plt.title("Levi's: AI-Based Trend Identification Leading to Sales Growth")
plt.xlabel("Estimated Increase in Loose-Fit Jeans Sales (%)")

plt.subplot(1, 3, 3)
plt.barh(cocacola_labels, cocacola_values, color='crimson')
plt.title("Coca-Cola: AI-Enhanced Retail Messaging and Marketing Spend Shift")
plt.xlabel("Scale (Millions or %)")

plt.tight_layout()
plt.show()
