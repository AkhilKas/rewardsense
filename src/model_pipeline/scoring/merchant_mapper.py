"""
Merchant Category Code (MCC) to spending category mapper for RewardSense.

Maps 4-digit MCC codes to standardized reward categories used by
the scoring engine. Categories align with common credit card bonus structures.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# MCC range mappings based on ISO 18245 and common issuer category definitions
# Format: (start, end) -> category
_MCC_RANGE_MAP = [
    # Dining
    ((5811, 5814), 'dining'),       # Restaurants, fast food, bars, caterers
    ((5462, 5462), 'dining'),       # Bakeries

    # Travel - Airlines
    ((3000, 3350), 'travel'),       # Airlines
    ((4511, 4511), 'travel'),       # Air carriers
    ((4722, 4722), 'travel'),       # Travel agencies

    # Travel - Hotels/Lodging
    ((3501, 3838), 'travel'),       # Hotels and motels
    ((7011, 7012), 'travel'),       # Hotels, timeshares

    # Travel - Transit
    ((4011, 4011), 'travel'),       # Railroads
    ((4111, 4112), 'travel'),       # Local transit, commuter
    ((4121, 4121), 'travel'),       # Taxis/rideshare
    ((4131, 4131), 'travel'),       # Bus lines
    ((4411, 4411), 'travel'),       # Cruise lines
    ((4789, 4789), 'travel'),       # Transportation services
    ((7512, 7512), 'travel'),       # Car rentals

    # Gas / Fuel
    ((5541, 5542), 'gas'),          # Gas stations, automated fuel

    # Groceries
    ((5411, 5411), 'groceries'),    # Grocery stores
    ((5422, 5422), 'groceries'),    # Meat provisioners
    ((5441, 5441), 'groceries'),    # Candy/confection stores
    ((5451, 5451), 'groceries'),    # Dairy stores
    ((5499, 5499), 'groceries'),    # Misc food stores

    # Streaming / Digital
    ((4899, 4899), 'streaming'),    # Cable/streaming services
    ((5815, 5818), 'streaming'),    # Digital goods, games, software

    # Online Shopping
    ((5942, 5942), 'online_shopping'),  # Bookstores
    ((5943, 5943), 'online_shopping'),  # Stationery
    ((5944, 5947), 'online_shopping'),  # Jewelry, watches, electronics
    ((5964, 5964), 'online_shopping'),  # Direct marketing - catalog
    ((5965, 5966), 'online_shopping'),  # Direct marketing - combo/inbound

    # Drugstores / Pharmacies
    ((5912, 5912), 'drugstores'),   # Drug stores, pharmacies

    # Entertainment
    ((7832, 7833), 'entertainment'),  # Movie theaters
    ((7911, 7911), 'entertainment'),  # Dance halls/studios
    ((7922, 7922), 'entertainment'),  # Theatrical producers
    ((7929, 7929), 'entertainment'),  # Bands/orchestras
    ((7932, 7933), 'entertainment'),  # Billiard/bowling
    ((7941, 7941), 'entertainment'),  # Sports clubs/fields
    ((7991, 7999), 'entertainment'),  # Recreation services, amusement parks

    # Utilities
    ((4812, 4816), 'utilities'),    # Telecom, phone services
    ((4900, 4900), 'utilities'),    # Utilities - electric, gas, water
]

# Exact MCC overrides (takes priority over ranges)
_MCC_EXACT_MAP: Dict[int, str] = {
    # Common specific MCCs that might not fit neatly into ranges
    5812: 'dining',
    5813: 'dining',
    5814: 'dining',
    5541: 'gas',
    5542: 'gas',
    5411: 'groceries',
    5912: 'drugstores',
}

DEFAULT_CATEGORY = 'general'


class MerchantCategoryMapper:
    """
    Maps Merchant Category Codes (MCC) to standardized spending categories.
    
    Categories align with common credit card reward structures:
    dining, travel, groceries, gas, streaming, online_shopping,
    drugstores, entertainment, utilities, general (fallback).
    """

    def __init__(self, custom_mappings: Optional[Dict[int, str]] = None):
        """
        Initialize mapper with optional custom MCC overrides.
        
        Args:
            custom_mappings: Optional dict of {mcc_code: category} to override defaults
        """
        self.exact_map = dict(_MCC_EXACT_MAP)
        if custom_mappings:
            self.exact_map.update(custom_mappings)

        self.range_map = list(_MCC_RANGE_MAP)
        logger.info("Initialized MerchantCategoryMapper")

    def map_mcc_to_category(self, mcc_code: int) -> str:
        """
        Map an MCC code to a spending category.
        
        Priority: exact match -> range match -> default ('general').
        
        Args:
            mcc_code: 4-digit Merchant Category Code
        
        Returns:
            Spending category string
        """
        # 1. Exact match
        if mcc_code in self.exact_map:
            return self.exact_map[mcc_code]

        # 2. Range match
        for (start, end), category in self.range_map:
            if start <= mcc_code <= end:
                return category

        # 3. Fallback
        logger.debug(f"MCC {mcc_code} not mapped, returning '{DEFAULT_CATEGORY}'")
        return DEFAULT_CATEGORY

    def get_all_categories(self) -> list:
        """Return list of all known spending categories."""
        categories = set(self.exact_map.values())
        for _, cat in self.range_map:
            categories.add(cat)
        categories.add(DEFAULT_CATEGORY)
        return sorted(categories)