const CARD_IMAGE_BY_ID: Record<string, string> = {
  chase_sapphire_preferred: "/cards/chase-gradient.svg",
  chase_freedom_unlimited: "/cards/chase-gradient.svg",
  amex_gold: "/cards/amex-gradient.svg",
  blue_cash_preferred: "/cards/amex-gradient.svg",
  capital_one_venture_x: "/cards/capitalone-gradient.svg",
  capital_one_savor: "/cards/capitalone-gradient.svg",
  citi_double_cash: "/cards/citi-gradient.svg",
  discover_it_cash_back: "/cards/discover-gradient.svg",
  wells_fargo_autograph: "/cards/default-gradient.svg",
};

const CARD_IMAGE_BY_ISSUER: Record<string, string> = {
  chase: "/cards/chase-gradient.svg",
  "american express": "/cards/amex-gradient.svg",
  amex: "/cards/amex-gradient.svg",
  "capital one": "/cards/capitalone-gradient.svg",
  citi: "/cards/citi-gradient.svg",
  discover: "/cards/discover-gradient.svg",
};

export function getCardImage(cardId?: string, issuer?: string): string {
  if (cardId && CARD_IMAGE_BY_ID[cardId]) return CARD_IMAGE_BY_ID[cardId];
  if (issuer) {
    const key = issuer.trim().toLowerCase();
    if (CARD_IMAGE_BY_ISSUER[key]) return CARD_IMAGE_BY_ISSUER[key];
  }
  return "/cards/default-gradient.svg";
}
