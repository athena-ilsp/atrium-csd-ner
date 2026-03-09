import re
from pathlib import Path
from typing import Dict

def parse_guidelines(filepath: str) -> Dict[str, str]:
    """
    Parses the guidelines markdown file to extract entity descriptions.
    Expected format: - LABEL -> Description
    """
    path = Path(filepath)
    if not path.exists():
        return {}

    content = path.read_text()
    # Regex to find "- LABEL -> Description"
    pattern = r"-\s+([A-Z]+)\s+->\s+([^-\n]+)"
    matches = re.findall(pattern, content)
    
    # Clean up descriptions (remove extra spaces/newlines)
    guidelines = {label.strip(): desc.strip() for label, desc in matches}
    
    return guidelines

def create_gliner_prompt(label: str, guidelines: Dict[str, str]) -> str:
    """Returns the description for a label or a default if not found."""
    return guidelines.get(label, f"Identify {label.lower()} entities in the text.")
