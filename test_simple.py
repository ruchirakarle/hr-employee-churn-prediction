import streamlit as st

st.title("?? SIMPLE TEST")
st.write("# If you see this, Streamlit works!")
st.success("SUCCESS! Your Streamlit is working!")
st.error("This is an error message (just testing^)")
st.warning("This is a warning message (just testing^)")
st.info("This is an info message (just testing^)")

st.write("---")
st.write("If you see all the colored boxes above, your frontend is working!")

if st.button("Click Me!"):
    st.balloons()
    st.write("Button works!")
