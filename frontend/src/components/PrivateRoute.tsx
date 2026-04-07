import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

/**
 * Wraps protected routes. Unauthenticated users are redirected to /login.
 * The original path is saved in location state so LoginPage can redirect
 * back after a successful login.
 */
export default function PrivateRoute() {
  const { isAuthenticated } = useAuth();
  const location = useLocation();

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <Outlet />;
}
