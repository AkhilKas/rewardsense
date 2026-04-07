import { Link } from "react-router-dom";
import Button from "../components/Button";
import Card from "../components/Card";
import { useAuth } from "../context/AuthContext";

const features = [
  {
    title: "Personalized Scoring",
    description:
      "Analyzes your spending patterns to rank cards specifically for you.",
    icon: "\u{1F3AF}",
  },
  {
    title: "Real-Time Recommendations",
    description:
      "Get card rankings in seconds based on your unique spending profile.",
    icon: "\u26A1",
  },
  {
    title: "Clear Explanations",
    description:
      "Understand why each card is recommended with clear, plain-language explanations.",
    icon: "\u{1F4A1}",
  },
  {
    title: "Always Up to Date",
    description:
      "Recommendations stay accurate over time as your spending habits evolve.",
    icon: "\u{1F4CA}",
  },
];

export default function HomePage() {
  const { isAuthenticated } = useAuth();
  const ctaPath = isAuthenticated ? "/recommend" : "/signup";

  return (
    <div className="space-y-16">
      <section className="text-center pt-12 pb-4">
        <h1 className="text-4xl sm:text-5xl font-bold text-secondary tracking-tight">
          Find Your Perfect{" "}
          <span className="text-primary">Credit Card</span>
        </h1>
        <p className="mt-4 text-lg text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
          RewardSense analyzes your spending habits and recommends the credit
          cards that maximize your rewards — personalized just for you.
        </p>
        <div className="mt-8 flex items-center justify-center gap-4">
          <Link to={ctaPath}>
            <Button size="lg">Get Recommendations</Button>
          </Link>
          <Link to="/dashboard">
            <Button variant="secondary" size="lg">
              System Status
            </Button>
          </Link>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-secondary text-center mb-8">
          How It Works
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((f) => (
            <Card key={f.title} className="text-center">
              <div className="text-3xl mb-3">{f.icon}</div>
              <h3 className="font-semibold text-secondary mb-2">{f.title}</h3>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                {f.description}
              </p>
            </Card>
          ))}
        </div>
      </section>

      <section className="text-center">
        <Card
          padding="lg"
          className="bg-linear-to-br from-primary/5 to-accent/5 border-primary/20"
        >
          <h2 className="text-2xl font-bold text-secondary mb-3">
            Ready to Find Your Best Card?
          </h2>
          <p className="text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
            Enter your spending profile and get personalized recommendations in
            seconds.
          </p>
          <div className="mt-6">
            <Link to={ctaPath}>
              <Button size="lg">Get Started</Button>
            </Link>
          </div>
        </Card>
      </section>
    </div>
  );
}
