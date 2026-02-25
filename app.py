# ------------------------------
# LIME Visual Explanation (LIKE your screenshot)
# ------------------------------
st.subheader("🧠 LIME Visual Explanation")

# Create LIME figure
fig = exp.as_pyplot_figure()

# Make it bigger (better readability)
fig.set_size_inches(12, 5)

# Force white background (same as your screenshot)
fig.patch.set_facecolor("white")
for ax in fig.axes:
    ax.set_facecolor("white")

# Show in Streamlit
st.pyplot(fig, use_container_width=True)
