import { getCardImage } from "../api/cardImages";

interface CardImageProps {
  cardId?: string;
  issuer?: string;
  alt: string;
  className?: string;
}

export default function CardImage({
  cardId,
  issuer,
  alt,
  className = "",
}: CardImageProps) {
  return (
    <img
      src={getCardImage(cardId, issuer)}
      alt={alt}
      className={`rounded-lg border border-border object-cover ${className}`}
      loading="lazy"
    />
  );
}
