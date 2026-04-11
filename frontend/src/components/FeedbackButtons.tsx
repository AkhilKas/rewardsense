import { useState } from "react";
import type { FeedbackReasonTag } from "../types";

const REASON_TAGS: { value: FeedbackReasonTag; label: string }[] = [
  { value: "too_expensive", label: "Too Expensive" },
  { value: "not_relevant", label: "Not Relevant" },
  { value: "already_have", label: "Already Have" },
  { value: "explanation_unclear", label: "Unclear Explanation" },
];

interface FeedbackButtonsProps {
  cardId: string;
  recommendationEventId?: number;
  target: "card" | "explanation";
  onSubmit: (reaction: "like" | "dislike", reasonTag?: FeedbackReasonTag) => void;
}

export default function FeedbackButtons({
  cardId,
  recommendationEventId,
  target,
  onSubmit,
}: FeedbackButtonsProps) {
  const [submitted, setSubmitted] = useState(false);
  const [reaction, setReaction] = useState<"like" | "dislike" | null>(null);
  const [showReasons, setShowReasons] = useState(false);

  if (submitted) {
    return (
      <p className="text-xs text-green-600 dark:text-green-400 py-1">
        Thanks for your feedback!
      </p>
    );
  }

  function handleReaction(r: "like" | "dislike") {
    setReaction(r);
    if (r === "like") {
      onSubmit(r);
      setSubmitted(true);
    } else {
      setShowReasons(true);
    }
  }

  function handleReasonSelect(tag?: FeedbackReasonTag) {
    if (reaction) {
      onSubmit(reaction, tag);
      setSubmitted(true);
    }
  }

  return (
    <div className="mt-2">
      {!showReasons ? (
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500 dark:text-slate-400">Helpful?</span>
          <button
            onClick={() => handleReaction("like")}
            className="p-1 rounded hover:bg-green-50 dark:hover:bg-green-900/30 transition-colors cursor-pointer"
            aria-label="Like this recommendation"
          >
            <svg className="w-4 h-4 text-slate-400 hover:text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 9V5a3 3 0 00-3-3l-4 9v11h11.28a2 2 0 002-1.7l1.38-9a2 2 0 00-2-2.3H14z" />
            </svg>
          </button>
          <button
            onClick={() => handleReaction("dislike")}
            className="p-1 rounded hover:bg-red-50 dark:hover:bg-red-900/30 transition-colors cursor-pointer"
            aria-label="Dislike this recommendation"
          >
            <svg className="w-4 h-4 text-slate-400 hover:text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 15V19a3 3 0 003 3l4-9V2H5.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3H10z" />
            </svg>
          </button>
        </div>
      ) : (
        <div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-1.5">
            What could be better? <span className="text-slate-400">(optional)</span>
          </p>
          <div className="flex flex-wrap gap-1.5 mb-1.5">
            {REASON_TAGS.map((tag) => (
              <button
                key={tag.value}
                onClick={() => handleReasonSelect(tag.value)}
                className="px-2 py-0.5 text-xs rounded-full border border-border text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors cursor-pointer"
              >
                {tag.label}
              </button>
            ))}
          </div>
          <button
            onClick={() => handleReasonSelect()}
            className="text-xs text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 cursor-pointer"
          >
            Skip
          </button>
        </div>
      )}
    </div>
  );
}
