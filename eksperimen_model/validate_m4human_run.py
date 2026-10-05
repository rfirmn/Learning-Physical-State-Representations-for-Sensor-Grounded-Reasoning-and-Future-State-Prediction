"""Validate a saved run and independently recompute its metric witness."""
import argparse
import json

from eksperimen_model.utils.m4human_runtime import validate_run_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    args = parser.parse_args()
    try:
        issues = validate_run_artifacts(args.run_directory)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        issues = [f"invalid_artifact:{type(error).__name__}:{error}"]
    print(json.dumps({"run": args.run_directory, "analysis_ready": not issues,
                      "scientific_validity": "requires_real_data_audits", "issues": issues}, indent=2))
    raise SystemExit(bool(issues))


if __name__ == "__main__":
    main()
