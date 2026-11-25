"""Export the direct-action Actor, its limits and observation contract for deployment."""
import argparse
from viserdex.supermimic.inference import export_actor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
export_actor(args.checkpoint, args.output)
print(args.output)
