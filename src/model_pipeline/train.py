"""
Placeholder for model training entrypoint.

This module will be implemented in later stories (Epic 3).
For now, it serves as the default CMD target for the Docker container
and validates that the model_pipeline package is importable.
"""

import sys


def main():
    """Model training entrypoint placeholder."""
    print("=" * 60)
    print("RewardSense Model Pipeline")
    print("=" * 60)
    print("Model training module loaded successfully.")
    print("Training logic will be implemented in Epic 3 (Stories 3.1-3.5).")
    print()
    print("To run the smoke test instead:")
    print("  python scripts/smoke_test_model_env.py")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
