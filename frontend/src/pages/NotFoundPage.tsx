import { Link } from "react-router-dom";
import Card from "../components/Card";
import Button from "../components/Button";

export default function NotFoundPage() {
  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <Card className="text-center max-w-md">
        <div className="text-5xl mb-4">404</div>
        <h1 className="text-2xl font-bold text-secondary mb-2">
          Page Not Found
        </h1>
        <p className="text-slate-600 dark:text-slate-400 mb-6">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <Link to="/">
          <Button>Back to Home</Button>
        </Link>
      </Card>
    </div>
  );
}
