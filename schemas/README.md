# RewardSense Data Schemas

## Overview

This directory contains **Pydantic v2** schema definitions for all data artifacts in the RewardSense pipeline. Schemas serve as contracts between pipeline stages and enable automatic validation, type checking, and documentation.

## Schema Files

| File | Purpose | Schemas Defined |
|------|---------|----------------|
| `credit_card.py` | Credit card schemas | `CreditCardRaw`, `CreditCardCleaned`, `CreditCardFeatures` |
| `transaction.py` | Transaction schemas | `TransactionRaw`, `TransactionCleaned`, `TransactionFeatures` |
| `user_profile.py` | User profile schemas | `UserProfileRaw`, `UserCardMapping`, `UserProfileFeatures` |
| `features.py` | Feature metadata | `FeatureMetadata`, `FeatureRegistry` |
| `validators.py` | Shared validators | Common validation functions |
| `__init__.py` | Central exports | All schemas |

## Pipeline Stages
```
┌─────────────────────────────────────────────────────────┐
│  Raw Data (Input)                                       │
│  - CreditCardRaw (from API/scrapers)                    │
│  - TransactionRaw (from generators)                     │
│  - UserProfileRaw (from generators)                     │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│  Cleaned Data (Story 3.1)                               │
│  - CreditCardCleaned (deduplicated, standardized)       │
│  - TransactionCleaned (validated, suspicious flagged)   │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│  Features (Story 3.2)                                   │
│  - CreditCardFeatures (engineered metrics)              │
│  - TransactionFeatures (aggregated by user)             │
│  - UserProfileFeatures (encoded/transformed)            │
└─────────────────────────────────────────────────────────┘
```

## Usage Examples

### Validate Raw Data
```python
from schemas import TransactionRaw

# Validate a single transaction
txn_data = {
    "transaction_id": "txn_0000123",
    "user_id": "user_0001",
    "date": "2025-08-01",
    "category": "dining",
    "merchant": "Starbucks",
    "mcc_code": 5812,
    "amount": 16.09,
    "card_used": "Chase Sapphire Reserve"
}

txn = TransactionRaw(**txn_data)  # Raises ValidationError if invalid
print(txn.model_dump())
```

### Validate DataFrame
```python
from schemas import TransactionRaw
import pandas as pd

df = pd.read_csv('data/generated/transactions.csv')

# Validate all rows
validated = []
for row in df.to_dict('records'):
    try:
        validated.append(TransactionRaw(**row))
    except ValidationError as e:
        print(f"Invalid row: {e}")

print(f"Validated {len(validated)}/{len(df)} transactions")
```

### Generate JSON Schema
```python
from schemas import TransactionRaw

# Generate JSON Schema for documentation
json_schema = TransactionRaw.model_json_schema()
print(json_schema)
```

## Schema Versioning

**Current Version:** `1.0.0`

### Versioning Strategy

We follow **Semantic Versioning** (MAJOR.MINOR.PATCH):

- **MAJOR** (1.x.x → 2.x.x): Breaking changes
  - Removing required fields
  - Changing field types incompatibly
  - Renaming fields
  - Example: `card_used` → `card_id`
  
- **MINOR** (x.1.x → x.2.x): Backward-compatible additions
  - Adding optional fields
  - Adding new validators
  - Expanding enum values
  - Example: Adding `card_tier` optional field
  
- **PATCH** (x.x.1 → x.x.2): Non-breaking fixes
  - Documentation updates
  - Validator improvements
  - Bug fixes in validation logic

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-17 | Initial schema definitions (Story 6.1) |

## Schema Evolution Policy

### 1. Backward Compatibility

- **Maintain compatibility** for at least **2 major versions** (6 months minimum)
- **Deprecation warnings** added before removal
- **Old schemas** kept in `schemas/deprecated/` during transition

### 2. Adding New Fields

**For backward-compatible additions:**
```python
# Good: Adding optional field
class TransactionRaw(BaseModel):
    # ... existing fields ...
    tip_amount: Optional[float] = None  # ✅ Optional, no breaking change
```

**Update checklist:**
- [ ] Add field with `Optional` or default value
- [ ] Update feature engineering to handle field
- [ ] Update tests
- [ ] Bump MINOR version
- [ ] Document in CHANGELOG

### 3. Breaking Changes

**For incompatible changes:**

1. **Create new schema version:**
```python
   # schemas/transaction_v2.py
   class TransactionRawV2(BaseModel):
       card_id: str  # Changed from card_used
```

2. **Add migration function:**
```python
   # schemas/migrations.py
   def migrate_transaction_v1_to_v2(v1_data: dict) -> dict:
       v2_data = v1_data.copy()
       v2_data['card_id'] = v1_data['card_used']
       del v2_data['card_used']
       return v2_data
```

3. **Support both versions** in pipeline for transition period

4. **Deprecate old version** after 2 releases

5. **Remove old version** after migration complete

### 4. Change Documentation

All changes must be documented in `CHANGELOG.md`:
```markdown
## [1.1.0] - 2026-03-15

### Added
- `TransactionRaw.tip_amount` - Optional tip amount field
- `UserProfileFeatures.avg_spending_growth` - Month-over-month growth rate

### Deprecated
- None

### Changed
- None
```

## Validation Best Practices

### 1. Always Validate at Boundaries
```python
# ✅ Good: Validate when data enters pipeline
raw_data = load_from_csv('transactions.csv')
validated = [TransactionRaw(**row) for row in raw_data]

# ❌ Bad: Assume data is valid
raw_data = load_from_csv('transactions.csv')
# Use directly without validation
```

### 2. Use Appropriate Schema for Pipeline Stage
```python
# ✅ Good: Use correct schema for each stage
raw_txns = [TransactionRaw(**r) for r in raw_data]
cleaned_txns, _ = clean_transaction_data(raw_df)
validated_cleaned = [TransactionCleaned(**r) for r in cleaned_txns.to_dict('records')]

# ❌ Bad: Use raw schema for cleaned data
cleaned_txns, _ = clean_transaction_data(raw_df)
validated = [TransactionRaw(**r) for r in cleaned_txns.to_dict('records')]
# This won't validate 'suspicious' field!
```

### 3. Handle Validation Errors Gracefully
```python
from pydantic import ValidationError

validated = []
errors = []

for row in data:
    try:
        validated.append(TransactionRaw(**row))
    except ValidationError as e:
        errors.append({'row': row, 'error': str(e)})

if errors:
    logger.warning(f"Found {len(errors)} invalid rows")
    # Handle errors appropriately
```

## Testing Schemas

All schemas have comprehensive tests in `tests/schemas/`:
```bash
# Run schema tests
pytest tests/schemas/ -v

# Test specific schema
pytest tests/schemas/test_transaction.py -v
```

## Auto-Generated Documentation

### Generate JSON Schema
```python
from schemas import TransactionRaw

# Export as JSON Schema
json_schema = TransactionRaw.model_json_schema()

# Save to file
with open('docs/schemas/transaction_raw.json', 'w') as f:
    json.dump(json_schema, f, indent=2)
```

### Generate Markdown Documentation
```python
# Using pydantic-markdown or custom script
from schemas import TransactionRaw

# Generate markdown table
for field_name, field_info in TransactionRaw.model_fields.items():
    print(f"| {field_name} | {field_info.annotation} | {field_info.description} |")
```

## Contributing

When adding new schemas:

1. **Follow naming conventions**
   - `{DataType}Raw` - Raw input data
   - `{DataType}Cleaned` - After cleaning
   - `{DataType}Features` - After feature engineering

2. **Add comprehensive docstrings**
   - Class-level: Purpose and guarantees
   - Field-level: Description and constraints

3. **Include example** in `Config.json_schema_extra`

4. **Add validators** for business logic

5. **Update `__init__.py`** exports

6. **Create tests** in `tests/schemas/`

7. **Update this README** with new schema

## Support

For questions about schemas:
- See `docs/data_card.md` for data specifications
- See `src/data_pipeline/preprocessing/` for transformation logic
- Open GitHub issue with `schema` label

---

**Last Updated:** 2026-02-17  
**Schema Version:** 1.0.0  
**Maintained by:** RewardSense Team