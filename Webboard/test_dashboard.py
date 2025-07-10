import streamlit as st

# Set page config
st.set_page_config(
    page_title="Test Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.markdown("### Real Time Processing of Data")
    selected_option = st.radio(
        "",
        ["Stickman", "StepLength", "PowerMetrics", "Frequency"],
        label_visibility="collapsed"
    )

st.markdown("### Test Content")
st.write("If this works, we'll add the full content back") 