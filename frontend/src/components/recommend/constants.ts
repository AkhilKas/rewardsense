export const SPENDING_CATEGORIES = [
  { key: "groceries" as const, label: "Groceries", max: 3000 },
  { key: "dining" as const, label: "Dining", max: 3000 },
  { key: "travel" as const, label: "Travel", max: 3000 },
  { key: "gas" as const, label: "Gas", max: 1000 },
  { key: "online_shopping" as const, label: "Online Shopping", max: 3000 },
  { key: "entertainment" as const, label: "Entertainment", max: 1000 },
  { key: "utilities" as const, label: "Utilities", max: 1000 },
  { key: "other" as const, label: "Other", max: 2000 },
] as const;

export type CategoryKey = (typeof SPENDING_CATEGORIES)[number]["key"];

export const REWARD_TYPES = [
  { value: "cashback", label: "Cashback" },
  { value: "travel_points", label: "Travel Points" },
  { value: "hotel_points", label: "Hotel Points" },
  { value: "airline_miles", label: "Airline Miles" },
];

export const INCOME_RANGES = [
  { value: "under_30k", label: "Under $30,000" },
  { value: "30k_50k", label: "$30,000 – $50,000" },
  { value: "50k_75k", label: "$50,000 – $75,000" },
  { value: "75k_100k", label: "$75,000 – $100,000" },
  { value: "over_100k", label: "Over $100,000" },
];

export const POPULAR_CARDS = [
  { value: "chase_sapphire_preferred", label: "Chase Sapphire Preferred" },
  { value: "amex_gold", label: "Amex Gold Card" },
  { value: "citi_double_cash", label: "Citi Double Cash" },
  { value: "capital_one_venture_x", label: "Capital One Venture X" },
  { value: "discover_it_cash_back", label: "Discover it Cash Back" },
  { value: "chase_freedom_unlimited", label: "Chase Freedom Unlimited" },
  { value: "capital_one_savor", label: "Capital One Savor" },
  { value: "wells_fargo_autograph", label: "Wells Fargo Autograph" },
  { value: "blue_cash_preferred", label: "Blue Cash Preferred" },
];

export const INITIAL_SPENDING: Record<CategoryKey, number> = {
  groceries: 0,
  dining: 0,
  travel: 0,
  gas: 0,
  online_shopping: 0,
  entertainment: 0,
  utilities: 0,
  other: 0,
};
