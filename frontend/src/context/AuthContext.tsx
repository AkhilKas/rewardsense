import {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";

const API_BASE_URL = import.meta.env.VITE_API_URL || "";

interface AuthUser {
  user_id: number;
  display_name: string;
  email: string;
}

interface AuthContextValue {
  user: AuthUser | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (
    email: string,
    password: string,
    displayName: string,
  ) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

const TOKEN_KEY = "rs_token";
const USER_KEY = "rs_user";

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(
    () => localStorage.getItem(TOKEN_KEY),
  );
  const [user, setUser] = useState<AuthUser | null>(() => {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? (JSON.parse(raw) as AuthUser) : null;
  });

  // Keep localStorage in sync whenever state changes
  useEffect(() => {
    if (token) {
      localStorage.setItem(TOKEN_KEY, token);
    } else {
      localStorage.removeItem(TOKEN_KEY);
    }
  }, [token]);

  useEffect(() => {
    if (user) {
      localStorage.setItem(USER_KEY, JSON.stringify(user));
    } else {
      localStorage.removeItem(USER_KEY);
    }
  }, [user]);

  async function login(email: string, password: string): Promise<void> {
    const res = await fetch(`${API_BASE_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error((err as { detail?: string }).detail ?? "Login failed");
    }
    const data = await res.json() as {
      access_token: string;
      user_id: number;
      display_name: string;
    };
    setToken(data.access_token);
    setUser({ user_id: data.user_id, display_name: data.display_name, email });
  }

  async function signup(
    email: string,
    password: string,
    displayName: string,
  ): Promise<void> {
    const res = await fetch(`${API_BASE_URL}/auth/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, display_name: displayName }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error((err as { detail?: string }).detail ?? "Signup failed");
    }
    const data = await res.json() as {
      access_token: string;
      user_id: number;
      display_name: string;
    };
    setToken(data.access_token);
    setUser({ user_id: data.user_id, display_name: data.display_name, email });
  }

  async function logout(): Promise<void> {
    try {
      if (token) {
        await fetch(`${API_BASE_URL}/auth/logout`, {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
        });
      }
    } finally {
      setToken(null);
      setUser(null);
    }
  }

  return (
    <AuthContext.Provider
      value={{ user, token, isAuthenticated: !!token, login, signup, logout }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used inside <AuthProvider>");
  }
  return ctx;
}
