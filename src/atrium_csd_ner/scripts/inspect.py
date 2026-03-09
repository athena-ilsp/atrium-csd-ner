import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path to use project utilities
sys.path.append(str(Path.cwd() / "src"))

from atrium_csd_ner.data_utils import get_argilla_client

# Suppress verbose logs to focus on the output
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("argilla").setLevel(logging.WARNING)

load_dotenv()

def inspect():
    client = get_argilla_client()
    if not client:
        print("Failed to initialize Argilla client. Check your .env file.")
        return

    dataset_name = os.getenv("ARGILLA_DATASET_NAME", "atrium_context_sheet_descriptions")
    workspace = os.getenv("ARGILLA_WORKSPACE")

    print(f"Connecting to dataset: '{dataset_name}' in workspace: '{workspace}'...")
    
    try:
        dataset = client.datasets(dataset_name, workspace=workspace)
        
        print("\n" + "="*50)
        print("DATASET SCHEMA")
        print("="*50)
        print(f"Dataset Name: {dataset.name}")
        print(f"Workspace: {dataset.workspace.name}")
        
        print("\n[FIELDS]")
        for field in dataset.fields:
            print(f" - {field.name}")
            
        print("\n[QUESTIONS]")
        for question in dataset.questions:
            print(f" - {question.name}")

        # Fetch records using the generator
        records_gen = dataset.records(limit=1)
        record = next(iter(records_gen))

        print("\n" + "="*50)
        print("FIRST RECORD PREVIEW")
        print("="*50)
        print(f"Record ID: {record.id}")
        
        print("\n[FIELD VALUES]")
        for field_name, value in record.fields.items():
            print(f" - {field_name}: {str(value)[:150]}...")

        print("\n[RESPONSES]")
        if record.responses:
            for i, resp in enumerate(record.responses):
                print(f" - Response {i} (User: {resp.user_id}):")
                # Let's inspect all available attributes to find where the values are stored
                attrs = [a for a in dir(resp) if not a.startswith('_')]
                print(f"   Available Attributes: {attrs}")
                
                # Check for common data keys
                for key in ['values', 'value', 'responses']:
                    if hasattr(resp, key):
                        val = getattr(resp, key)
                        print(f"   * Found '{key}':")
                        try:
                            print(json.dumps(val, indent=2, ensure_ascii=False))
                        except:
                            print(val)
        else:
            print(" - No responses yet.")
            
        print("="*50)

    except Exception as e:
        print(f"\nError during inspection: {e}")

if __name__ == "__main__":
    inspect()
