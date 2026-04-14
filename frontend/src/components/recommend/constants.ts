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

export type FrontendArchetype =
  | "traveler"
  | "foodie-dining"
  | "grocery-family"
  | "cashback-generalist"
  | "student"
  | "big-spender";

export const ARCHETYPE_META: Record<
  FrontendArchetype,
  { label: string; emoji: string; description: string; backendPersona: string }
> = {
  traveler: {
    label: "Traveler",
    emoji: "🧳",
    description:
      "You prioritize travel rewards, transfer partners, and premium travel perks.",
    backendPersona: "traveler",
  },
  "foodie-dining": {
    label: "Foodie/Dining",
    emoji: "🍽️",
    description:
      "You spend heavily on restaurants and food delivery and want strong dining returns.",
    backendPersona: "cashback-focused",
  },
  "grocery-family": {
    label: "Grocery/Family",
    emoji: "🛒",
    description:
      "Your budget leans toward grocery, gas, and household categories.",
    backendPersona: "family",
  },
  "cashback-generalist": {
    label: "Cashback Generalist",
    emoji: "💳",
    description:
      "You want reliable flat-value rewards across many categories.",
    backendPersona: "cashback-focused",
  },
  student: {
    label: "Student",
    emoji: "🎓",
    description:
      "You prefer low-fee cards with simple rewards and flexible approval profiles.",
    backendPersona: "student",
  },
  "big-spender": {
    label: "Big Spender",
    emoji: "💼",
    description:
      "You can benefit from premium cards with higher multipliers and richer perks.",
    backendPersona: "traveler",
  },
};

const LOW_SPEND_THRESHOLD = 1000;
const BIG_SPENDER_THRESHOLD = 3000;
const STUDENT_SPEND_THRESHOLD = 500;
const STUDENT_CATEGORY_CAP = 150;
const TRAVEL_THRESHOLD = 200;
const DINING_THRESHOLD = 200;
const GROCERIES_THRESHOLD = 300;

export function detectArchetype(
  spending: Record<CategoryKey, number>,
): FrontendArchetype {
  const entries = Object.entries(spending) as Array<[CategoryKey, number]>;
  const total = entries.reduce((sum, [, amount]) => sum + amount, 0);
  if (total === 0) return "cashback-generalist";
  const sorted = [...entries].sort((a, b) => b[1] - a[1]);
  const [topCategory] = sorted[0] ?? ["other", 0];
  const travelAmount = spending.travel ?? 0;
  const diningAmount = spending.dining ?? 0;
  const groceriesAmount = spending.groceries ?? 0;
  const hasCategoryAboveStudentCap = entries.some(
    ([, amount]) => amount > STUDENT_CATEGORY_CAP,
  );

  if (total > BIG_SPENDER_THRESHOLD) return "big-spender";
  if (total < STUDENT_SPEND_THRESHOLD && !hasCategoryAboveStudentCap) {
    return "student";
  }
  if (topCategory === "groceries" && groceriesAmount > GROCERIES_THRESHOLD) {
    return "grocery-family";
  }
  if (topCategory === "dining" && diningAmount > DINING_THRESHOLD) {
    return "foodie-dining";
  }
  if (topCategory === "travel" && travelAmount > TRAVEL_THRESHOLD) {
    return "traveler";
  }
  if (total < LOW_SPEND_THRESHOLD) {
    return "cashback-generalist";
  }
  return "cashback-generalist";
}
