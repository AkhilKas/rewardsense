import { Link } from "react-router-dom";
import Button from "../components/Button";
import Card from "../components/Card";
import Badge from "../components/Badge";

const features = [
  {
    title: "Personalized Scoring",
    description:
      "ML model learns your spending patterns to rank cards specifically for you.",
    icon: "\u{1F3AF}",
  },
  {
    title: "Real-Time Recommendations",
    description:
      "Get card rankings in seconds, powered by a deployed inference API on Cloud Run.",
    icon: "\u26A1",
  },
  {
    title: "AI Explanations",
    description:
      "Understand why each card is recommended with LLM-generated plain-language explanations.",
    icon: "\u{1F4A1}",
  },
  {
    title: "Continuous Monitoring",
    description:
      "Automated drift detection and retraining keeps recommendations accurate over time.",
    icon: "\u{1F4CA}",
  },
];

const techStack = [
  "Airflow",
  "MLflow",
  "Docker",
  "Cloud Run",
  "Evidently AI",
  "SHAP",
  "FastAPI",
  "React",
];

export default function HomePage() {
  return (
    <div className="space-y-16">
      <section className="text-center pt-12 pb-4">
        <Badge variant="info" className="mb-4">
          Powered by MLOps
        </Badge>
        <h1 className="text-4xl sm:text-5xl font-bold text-secondary tracking-tight">
          Find Your Perfect{" "}
          <span className="text-primary">Credit Card</span>
        </h1>
        <p className="mt-4 text-lg text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
          RewardSense uses machine learning to analyze your spending habits and
          recommend the credit cards that maximize your rewards — personalized
          just for you.
        </p>
        <div className="mt-8 flex items-center justify-center gap-4">
          <Link to="/recommend">
            <Button size="lg">Get Recommendations</Button>
          </Link>
          <Link to="/dashboard">
            <Button variant="secondary" size="lg">
              View Dashboard
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
            End-to-End MLOps Pipeline
          </h2>
          <p className="text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
            From data ingestion and preprocessing to model training, deployment,
            monitoring, and automatic retraining — RewardSense demonstrates a
            production-grade ML system.
          </p>
          <div className="mt-6 flex flex-wrap justify-center gap-2">
            {techStack.map((tech) => (
              <Badge key={tech} variant="info">
                {tech}
              </Badge>
            ))}
          </div>
        </Card>
      </section>
    </div>
  );
}
