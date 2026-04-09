import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import Layout from "./components/Layout";
import PrivateRoute from "./components/PrivateRoute";
import HomePage from "./pages/HomePage";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import RecommendPage from "./pages/RecommendPage";
import ResultsPage from "./pages/ResultsPage";
import DashboardPage from "./pages/DashboardPage";
import ProfilePage from "./pages/ProfilePage";
import WalletPage from "./pages/WalletPage";
import QuickRecommendPage from "./pages/QuickRecommendPage";
import TransactionHistoryPage from "./pages/TransactionHistoryPage";
import ExpenseSummaryPage from "./pages/ExpenseSummaryPage";
import NotFoundPage from "./pages/NotFoundPage";

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          {/* Public auth pages — no Layout chrome */}
          <Route path="login" element={<LoginPage />} />
          <Route path="signup" element={<SignupPage />} />

          {/* All other pages share the Layout shell */}
          <Route element={<Layout />}>
            <Route index element={<HomePage />} />

            {/* Protected: redirect to /login if not authenticated */}
            <Route element={<PrivateRoute />}>
              <Route path="recommend" element={<RecommendPage />} />
              <Route path="results" element={<ResultsPage />} />
              <Route path="dashboard" element={<DashboardPage />} />
              <Route path="profile" element={<ProfilePage />} />
              <Route path="wallet" element={<WalletPage />} />
              <Route path="quick-recommend" element={<QuickRecommendPage />} />
              <Route path="transactions" element={<TransactionHistoryPage />} />
              <Route path="summary" element={<ExpenseSummaryPage />} />
            </Route>

            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
