"""InsightTrace - Step 1: hello app.

This page only checks that your machine (and later the live server) is set up
correctly. It will be replaced by the real Home page in Step 9.
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="InsightTrace", page_icon="🔎", layout="wide")

st.title("🔎 InsightTrace")
st.caption("AI Business Intelligence: detect KPI changes, find the cause, act on it.")

st.subheader("Setup check")

# 1. Python version
py = sys.version.split()[0]
st.write(f"**Python version:** {py}")
if sys.version_info < (3, 11):
    st.warning("Python is older than 3.11. Some later packages may fail.")

# 2. Key libraries import correctly
for lib in ["pandas", "duckdb", "plotly", "openpyxl"]:
    try:
        module = __import__(lib)
        st.write(f"✅ **{lib}** {getattr(module, '__version__', '')}")
    except ImportError:
        st.error(f"❌ {lib} is not installed")

# 3. Demo dataset present?
csv_path = ROOT / "data" / "Sample_-_Superstore.csv"
if csv_path.exists():
    st.write(f"✅ **Demo dataset found** ({csv_path.stat().st_size // 1024} KB)")
else:
    st.error("❌ Demo dataset missing. Put Sample_-_Superstore.csv in the data folder.")

# 4. Gemini key present? (we only report yes/no, never show the key)
key = os.getenv("GEMINI_API_KEY")
if not key:
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None
if key and key != "paste_your_key_here":
    st.write("✅ **Gemini key loaded** (hidden)")
else:
    st.info("Gemini key not set yet. That is fine for Step 1.")

st.success("If you can read this page, Step 1 works.")
