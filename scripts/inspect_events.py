"""Export finite-value checks and scalar samples from local TensorBoard event files."""
import argparse
import json
import math
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser()
parser.add_argument("directory")
args = parser.parse_args()
events = EventAccumulator(args.directory, size_guidance={"scalars": 0})
events.Reload()
scalars = {tag: [{"step": event.step, "value": event.value} for event in events.Scalars(tag)]
           for tag in events.Tags()["scalars"]}
bad = {tag: [event for event in values if not math.isfinite(event["value"])]
       for tag, values in scalars.items() if any(not math.isfinite(x["value"]) for x in values)}
print(json.dumps({"directory": args.directory, "scalar_tags": len(scalars),
                  "all_logged_scalars_finite": (not bad) if scalars else None,
                  "passed": bool(scalars) and not bad, "nonfinite": bad, "scalars": scalars}, indent=2))
if not scalars or bad:
    raise SystemExit(1)
