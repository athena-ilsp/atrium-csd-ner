# Atrium: Context Sheet Description Named Entity Recognition (NER) 

## Description: 

This repository contains code to extract structured information from unstructured text. Its primary function is to process transcribed free-text descriptions of archaeological findings and identify relevant named entities.

## Process Overview

The workflow receives transcribed text from archaeological context sheets as its input and executes the following steps:

-    Text Segmentation: The input text is parsed and segmented into sentences and tokens to prepare it for analysis.
-    Entity Extraction: State-of-the-art Named Entity Recognition (NER) models, including both generic and domain-specific (e.g., ArchaeoBERT-NER), are applied to the tokenized text. These models identify and classify key terms within the descriptions.


