# ------------------------------
# LIME Visual Explanation (NO HTML / NO JS)
# This will NOT be blank on Streamlit Cloud
# ------------------------------
import matplotlib.pyplot as plt  # required for plotting

st.subheader("🧠 LIME Visual Explanation")

# Create the LIME plot as a Matplotlib figure (works everywhere)
fig = exp.as_pyplot_figure()

# Improve readability on dark background
fig.set_facecolor("none")  # transparent figure background

# Show in Streamlit
st.pyplot(fig, use_container_width=True)

# ------------------------------
# Bonus: Show highlighted text (simple version)
# ------------------------------
st.subheader("📝 Highlighted Words (simple)")

# Get top important words from LIME
top_words = [w for w, _ in exp.as_list()]  # words only

# Make a simple highlighted version of the text
highlighted = user_text
for w in top_words:
    highlighted = highlighted.replace(w, f"**{w}**")  # bold important words

st.write(highlighted)
