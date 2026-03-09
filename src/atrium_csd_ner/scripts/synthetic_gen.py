import argparse
import json
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from atrium_csd_ner.synthetic import SyntheticGenerator

# Load environment variables from .env
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic archaeological NER data from ASR transcripts.")
    parser.add_argument("--count", type=int, default=10, help="Number of examples to generate.")
    parser.add_argument("--output", type=str, default="data/train_data/synthetic_asr.json", help="Path to save the output.")
    parser.add_argument("--use-seeds", action="store_true", help="Use real Argilla annotations as few-shot seeds.")
    args = parser.parse_args()

    # Ensure output dir exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    generator = SyntheticGenerator()
    
    seeds = None
    if args.use_seeds:
        logger.info("Fetching real annotations from Argilla to use as seeds...")
        seeds = load_from_argilla()
        logger.info(f"Loaded {len(seeds)} real examples.")

    results = []
    logger.info(f"Generating {args.count} synthetic examples...")
    
    for i in tqdm(range(args.count)):
        transcript = generator.generate_single(seed_examples=seeds)
        if transcript:
            results.append(transcript)
        else:
            logger.warning(f"Failed to generate example {i+1}")

    if results:
        logger.info(f"Generated {len(results)} successful transcripts. Saving to {args.output}...")
        generator.save_to_gliner(results, args.output)
        logger.info("Done!")
    else:
        logger.error("No results generated.")

if __name__ == "__main__":
    main()
