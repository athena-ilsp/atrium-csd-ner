#!/usr/bin/env python
# coding: utf-8
from multiprocessing import context
from pathlib import Path
import streamlit as st
import os
import torch
import re
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
# or simply:
torch.classes.__path__ = []
from pathlib import Path
import requests
import logging
from logging import getLogger
from spacy import displacy
from huggingface_hub import hf_hub_download
from spacy import displacy
from flair.splitter import SegtokSentenceSplitter
from flair.data import Sentence
from flair.models import SequenceTagger
import seaborn as sns
import base64
import matplotlib
import matplotlib.pyplot as plt

image_dir="images"

logger = getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

NL="\n"
TAB="\t"

def call_fastapi_ner(text):
    response = requests.post(
        "http://10.1.1.76:8000/ner",
        json={"text": text},
        timeout=60
    )
    response.raise_for_status()
    return response.json()["sentences"]

page_title = "Atrium Speech-based NER demo"
st.set_page_config(
    page_title=page_title,
#    page_icon="https://www.athenarc.gr/sites/default/files/ilsp_logo.gif",
    page_icon=f"{image_dir}/atrium.png",
)


col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """<a href="https://atrium-research.eu/">
        <img src="data:image/png;base64,{}" width="150">
        </a>""".format(
            base64.b64encode(open(f"{image_dir}/atrium.png", "rb").read()).decode()
        ),
        unsafe_allow_html=True,
    )

with col2:  
    st.markdown(
        """<div style="text-align: right;">
        <a href="https://www.athenarc.gr/ilsp/">
        <img src="data:image/png;base64,{}" width="150">
        </a></div>""".format(
            base64.b64encode(open(f"{image_dir}/ilsp_athena.png", "rb").read()).decode()
        ),
        unsafe_allow_html=True,
    )

st.title(page_title)
text = st.text_area("Text to process:", height=400, value = """Lower fill of posthole transitioning into fill 1036 gradually.
There is a linear delimitation / demarcation from the block behind the back of the head to the left foot.
It is contemporaneous with contexts 626 and 614, which initially corresponded to the inner elements of the cover.
It cover the original northern face of the structure , which appear to be vertical.
The context 609 corresponds to a fine layer of soil that partially covers the stones of the grave's cover , located in the eastern part of the burial.
C913 is the floor of the suspected c903 oven, it is a thin, irregular whitish layer of clay.
In this season we excavate c214 to reach the same level.
For further details, refer to the description provided for context 614.
An urn found in context 526.
Probably Late Mesolithic.
Animal bones in context 631.""")


if st.button("Process text with NER"):
    if text:
        try:
            with st.spinner("Processing text ..."):
                doc = call_fastapi_ner(text)
                color_palette = requests.get("http://10.1.1.76:8000/palette").json()
                st.subheader("Results:")
                # Use displacy for visualization
                ent_html = displacy.render(doc, manual=True, style="ent", jupyter=False, options={'colors': color_palette})
                st.markdown(ent_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please input some text.")

# "with" notation
with st.sidebar:
    st.markdown(
        """<div style="text-align: right;">
        <a href="http://10.1.1.76:8000/docs/">
        <img src="data:image/png;base64,{}" width="80">
        </a></div>""".format(
            base64.b64encode(open(f"{image_dir}/fastapi.png", "rb").read()).decode()
        ),
        unsafe_allow_html=True,
    )


#displacy.render(dic_ents, manual=True, style="ent")