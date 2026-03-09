#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import os
import requests
import base64
from spacy import displacy

from pathlib import Path

# IMPORTANT: Path updated for internal package reorganization
image_dir = Path(__file__).parent / "static" / "images"

# API configuration
api_base_url = os.getenv("API_BASE_URL")
if not api_base_url:
    hostname = os.getenv("API_HOSTNAME", "localhost")
    port = os.getenv("API_PORT", "8000")
    api_base_url = f"http://{hostname}:{port}"

page_title = "Atrium Archaeological NER Demo"
st.set_page_config(page_title=page_title, page_icon=f"{image_dir}/atrium.png", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for layout and entities
st.markdown("""
    <style>
        
        /* Fix displaCy entity wrapping and line height */
        .entities {
            line-height: 2.8 !important;
        }
        mark.entity {
            display: inline-block !important;
            margin: 0.15em 0.1em !important;
            padding: 0.25em 0.4em !important;
            border-radius: 0.3em !important;
        }
    </style>
""", unsafe_allow_html=True)

# Header with Logos
col1, col2 = st.columns(2)
with col1:
    if os.path.exists(f"{image_dir}/atrium.png"):
        img_b64 = base64.b64encode(open(f"{image_dir}/atrium.png", "rb").read()).decode()
        st.markdown(f'<a href="https://atrium-research.eu/"><img src="data:image/png;base64,{img_b64}" width="100"></a>', unsafe_allow_html=True)
with col2:
    if os.path.exists(f"{image_dir}/athena_ilsp_logo_horizontal.webp"):
        img_b64 = base64.b64encode(open(f"{image_dir}/athena_ilsp_logo_horizontal.webp", "rb").read()).decode()
        st.markdown(f'<div style="text-align: right;"><a href="https://www.athenarc.gr/ilsp/" target="_blank"><img src="data:image/webp;base64,{img_b64}" width="140"></a></div>', unsafe_allow_html=True)

st.title(page_title)

UI_EXAMPLES = [
    "The cut c702 corresponds to a circular post-hole identified in the north-western quadrant of Trench B. Cut into the natural clay substrate (c701), it is filled by a single, homogenous deposit of dark brown silty loam (c703). In plan, the cut is remarkably symmetrical, with a diameter of 0.35 m and a depth of 0.42 m. The profile reveals near-vertical sides and a flat, compacted base, suggesting it served as a primary structural support for Phase II. Unlike the nearby pit F4, no packing stones were identified within the fill; however, several fragments of burnt daub and a single flint flake (c703 #4) were recovered near the interface with the base.",
    "Cut in context 2055, which represents a linear foundation trench running parallel to the eastern limit of the excavation. It has a width of 0.60 m and was excavated to a maximum depth of 0.55 m into the underlying limestone bedrock. The fill (c2056) is characterized by a very high density of limestone rubble and mortar inclusions, likely representing construction debris. A diagnostic rim sherd of a transport amphora (Find A55) was recovered from the very bottom of the trench. The relationship between this cut and the adjacent drainage channel (c2060) remains unclear, though the channel appears to truncate the upper fill of the trench.",
    "The context c814 corresponds to the poorly preserved remains of a neonatal skeleton belonging to the infant burial F9. Found within a small, shallow scoop (c812) measuring 0.45 m in length, the remains are highly fragmented due to the acidic nature of the surrounding soil (c813). Oriented north-south, the individual was positioned in a contracted posture. No clear evidence of a coffin or shroud was detected, though a concentration of dark, carbonized material near the pelvis (c813 #2) might represent decomposed organic grave goods. The cut is barely distinguishable from the surrounding occupation layer, identified primarily by the slight change in soil compaction and the presence of the skeletal material itself.",
    "Context c921 identifies an oval-shaped cut (0.80 m x 0.65 m) characterized by intense in-situ burning. The sides of the cut exhibit distinct rubefaction of the natural soil, indicating exposure to high temperatures. The fill (c922) is a stratified sequence of fine white ash layers and charcoal-rich silt. Within the ash, several fire-cracked stones were recovered, arranged in a semi-circle at the western edge, potentially acting as pot-supports. A small copper alloy pin (Find C10) was found at the interface between the ash and the base of the cut; however, its presence may be intrusive due to significant root action (bioturbation) visible throughout the northern section.",
    "The context 1105 represents a large, sub-rectangular pit located centrally within the habitation area. Measuring 2.10 m by 1.85 m, it truncates the earlier floor surface (c1090) and is sealed by the later alluvial wash (c1085). The primary fill (c1106) consists of a dense accumulation of organic-rich 'dark earth' containing a high frequency of charred botanical remains and animal bone fragments. Notably, a concentrated lens of ash (c1107) was observed in the southern half of the cut, which yielded a significant assemblage of late-period pottery sherds (Finds B12–B18). The irregular, scalloped edges of the cut suggest multiple episodes of re-cutting before its final abandonment."
]

col_input, col_controls = st.columns([3, 1])

with col_input:
    selected_example = st.selectbox("Choose an example or write your own:", ["Custom Text"] + UI_EXAMPLES)
    default_text = "" if selected_example == "Custom Text" else selected_example
    text = st.text_area("Text to process:", height=100, value=default_text)
    extract_button = st.button("Extract Entities")

with col_controls:
    st.markdown("### Settings")
    confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
    
    # Calculate external API link, falling back to localhost proxy if root path is not defined
    api_root_path = os.getenv("API_ROOT_PATH", "")
    swagger_url = f"{api_root_path}/docs" if api_root_path else "/docs"
    st.markdown(f"<p style='font-size: 0.9em;'><a href='{swagger_url}' target='_blank'>📚 Open API Documentation</a></p>", unsafe_allow_html=True)

if "raw_results" not in st.session_state:
    st.session_state.raw_results = None
if "last_text" not in st.session_state:
    st.session_state.last_text = None

if extract_button:
    if text:
        try:
            with st.spinner("Analyzing text with GLiNER2 (fetching all entities)..."):
                # Fetch with threshold 0.0 to cache everything for local slider filtering
                response = requests.post(f"{api_base_url}/ner", json={"text": text, "threshold": 0.0}, timeout=60)
                response.raise_for_status()
                st.session_state.raw_results = response.json()["sentences"]
                st.session_state.last_text = text
        except Exception as e:
            st.error(f"API Error: {e}. Ensure main.py is running.")
            st.session_state.raw_results = None
    else:
        st.warning("Please input some text.")
        st.session_state.raw_results = None

# If we have successfully extracted results for the current text, render them
if st.session_state.raw_results and text == st.session_state.last_text:
    import copy
    # Deep copy the cached results so we can safely filter them
    doc = copy.deepcopy(st.session_state.raw_results)
    
    # Filter entities locally based on the slider threshold
    for sentence in doc:
        if "ents" in sentence:
            sentence["ents"] = [e for e in sentence["ents"] if e.get("score", 1.0) >= confidence_threshold]
            
    # Fetch dynamic palette from API
    try:
        color_palette = requests.get(f"{api_base_url}/palette").json()
    except:
        color_palette = {}
        
    import re
    
    scores = []
    for sentence in doc:
        for e in sentence.get("ents", []):
            scores.append(e.get("score"))
    
    st.subheader("Results:")
    ent_html = displacy.render(doc, manual=True, style="ent", jupyter=False, options={'colors': color_palette})
    
    # Dynamically inject the score as a hover title tooltip into the HTML
    def inject_title(match):
        if not hasattr(inject_title, "counter"):
            inject_title.counter = 0
        score = scores[inject_title.counter] if inject_title.counter < len(scores) else None
        inject_title.counter += 1
        
        title = f'title="Confidence: {score:.2f}" ' if score is not None else ''
        return f'<mark {title}' + match.group(1) + '>'
    
    ent_html = re.sub(r'<mark\s+(.*?)>', inject_title, ent_html)
    st.markdown(ent_html, unsafe_allow_html=True)

