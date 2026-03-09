import re
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

def find_entity_offsets(text: str, entity_text: str) -> List[Tuple[int, int]]:
    """Find all (start, end) offsets of entity_text in text."""
    matches = []
    pattern = re.escape(entity_text)
    for match in re.finditer(pattern, text):
        matches.append((match.start(), match.end()))
    return matches

def build_entity_spans(text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each entity, find all its spans in the text.
    Returns a list of dicts: {'text', 'label', 'start', 'end'}
    """
    spans = []
    for ent in entities:
        ent_text = ent["text"]
        label = ent["label"]
        for start, end in find_entity_offsets(text, ent_text):
            spans.append({'text': ent_text, 'label': label, 'start': start, 'end': end})
    
    # Remove overlapping spans, keep longest first
    spans.sort(key=lambda x: (x['start'], -x['end']))
    non_overlapping = []
    last_end = -1
    for span in spans:
        if span['start'] >= last_end:
            non_overlapping.append(span)
            last_end = span['end']
    return non_overlapping

def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Tokenizes text and returns a list of (token, start_offset, end_offset)."""
    pattern = r'\w+|[^\w\s]'
    tokens = []
    for match in re.finditer(pattern, text):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens

def assign_iob(tokens: List[Tuple[str, int, int]], entity_spans: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Assign IOB tags to tokens based on entity spans."""
    tags = []
    for token, start, end in tokens:
        tag = "O"
        for ent in entity_spans:
            if start >= ent['start'] and end <= ent['end']:
                if start == ent['start']:
                    tag = f"B-{ent['label']}"
                else:
                    tag = f"I-{ent['label']}"
                break
        tags.append((token, tag))
    return tags

def get_argilla_client():
    """Initialize Argilla client from env vars."""
    import os
    import argilla as rg
    
    url = os.getenv("ARGILLA_API_URL")
    api_key = os.getenv("ARGILLA_API_KEY")
    
    if not url or not api_key:
        logger.warning("ARGILLA_API_URL or ARGILLA_API_KEY not set.")
        return None
        
    return rg.Argilla(api_url=url, api_key=api_key)

def fetch_argilla_data(dataset_name: str = None, annotator: str = None) -> List[Dict[str, Any]]:
    """
    Fetch all records from an Argilla dataset.
    If annotator is provided (username or user_id), only use responses from that user.
    """
    import os
    from argilla import Query, Filter
    
    if dataset_name is None:
        dataset_name = os.getenv("ARGILLA_DATASET_NAME", "atrium_context_sheet_descriptions")
    
    if annotator is None:
        annotator = os.getenv("ARGILLA_ANNOTATOR")
    
    client = get_argilla_client()
    if not client:
        return []
    
    workspace = os.getenv("ARGILLA_WORKSPACE")
    try:
        dataset = client.datasets(dataset_name, workspace=workspace)
    except Exception as e:
        logger.error(f"Error fetching dataset {dataset_name}: {e}")
        return []

    # Identify the target user ID for filtering
    target_user_id = None
    if annotator:
        try:
            user = client.users(annotator)
            target_user_id = user.id
            logger.info(f"Filtering for response from annotator: {annotator} (ID: {target_user_id})")
        except:
            logger.warning(f"Could not find user {annotator} by name, will try ID match if applicable.")
            target_user_id = annotator

    # Query to only get submitted/completed records
    query = Query(filter=Filter(("response.status", "==", "submitted")))
    
    records = list(dataset.records(query=query))
    logger.info(f"Retrieved {len(records)} submitted records from {dataset_name}")
    
    processed = []
    for record in records:
        # Try different text fields, prioritize 'sentence_field' from inspection
        text = record.fields.get("sentence_field") or record.fields.get("Context Description") or record.fields.get("text")
        if not text:
            logger.warning(f"No text found in record {record.id}. Fields: {list(record.fields.keys())}")
            continue
            
        entities_source = None
        # In Argilla 2.x, record.responses is a collection of Response objects
        if record.responses:
            for resp in record.responses:
                # Filter by annotator if requested
                if target_user_id and str(resp.user_id) != str(target_user_id):
                    continue
                
                # We specifically want the 'entities' question
                if hasattr(resp, 'question_name') and resp.question_name == "entities":
                    entities_source = resp.value
                    break
                elif not hasattr(resp, 'question_name'):
                    # Fallback for unexpected SDK behavior: check if it's the entities list
                    if isinstance(resp.value, list) and len(resp.value) > 0 and 'label' in resp.value[0]:
                        entities_source = resp.value
                        break
                
        # Only fallback to suggestions if NO annotator filter is active and no human response found
        if not entities_source and not annotator:
            if record.suggestions and "entities" in record.suggestions:
                entities_source = record.suggestions["entities"].value
            
        if entities_source:
            entities = []
            for ent in entities_source:
                start = ent.get("start")
                end = ent.get("end")
                label = ent.get("label")
                # Extract text snippet from the record text using offsets
                snippet = text[start:end] if start is not None and end is not None else None
                
                entities.append({
                    "start": start,
                    "end": end,
                    "label": label,
                    "text": ent.get("text") or snippet
                })
            
            processed.append({
                "text": text,
                "entities": entities,
                "record_id": str(record.id)
            })
            
    logger.info(f"Successfully processed {len(processed)} records with valid entity annotations.")
    return processed
