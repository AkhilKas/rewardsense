"""Create validation checkpoints for automated data quality checks."""

import great_expectations as gx

context = gx.get_context()

print("=" * 70)
print("Creating Validation Checkpoints")
print("=" * 70)

# =====================================================================
# Checkpoint 1: Validate All Data
# =====================================================================

print("\n[1/2] Creating 'validate_all_data' checkpoint...")

checkpoint_config = {
    "name": "validate_all_data",
    "config_version": 1.0,
    "class_name": "Checkpoint",
    "run_name_template": "%Y%m%d-%H%M%S-validate-all",
}

try:
    checkpoint = context.add_or_update_checkpoint(**checkpoint_config)
    print("✅ Created validate_all_data checkpoint")
except Exception as e:
    print(f"⚠️  Error: {e}")

# =====================================================================
# Checkpoint 2: Validate Pipeline Output
# =====================================================================

print("\n[2/2] Creating 'validate_pipeline_output' checkpoint...")

checkpoint_config_pipeline = {
    "name": "validate_pipeline_output",
    "config_version": 1.0,
    "class_name": "Checkpoint",
    "run_name_template": "%Y%m%d-%H%M%S-pipeline-validation",
}

try:
    checkpoint = context.add_or_update_checkpoint(**checkpoint_config_pipeline)
    print("✅ Created validate_pipeline_output checkpoint")
except Exception as e:
    print(f"⚠️  Error: {e}")

# =====================================================================
# Summary
# =====================================================================

print("\n" + "=" * 70)
print("Checkpoints Created!")
print("=" * 70)

checkpoints = context.list_checkpoints()
print(f"\n✅ Total checkpoints: {len(checkpoints)}")
for cp in checkpoints:
    print(f"   - {cp}")  # Fixed: cp is already a string

print("\n" + "=" * 70)
print("Usage:")
print("=" * 70)
print("  # List checkpoints")
print("  great_expectations checkpoint list")
print()
print("  # Run validation")
print("  great_expectations checkpoint run validate_all_data")
