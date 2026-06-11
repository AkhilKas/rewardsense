/* eslint-disable react-refresh/only-export-components */
import {
  createContext,
  useContext,
  useState,
  useEffect,
} from "react";
import type { ReactNode } from "react";
import {
  getMe,
  login as loginRequest,
  logout as logoutRequest,
  signup as signupRequest,
  verifyEmail as verifyEmailRequest,
  resendOtp as resendOtpRequest,
} from "../api/client";
import type { UserProfile } from "../types";
import { applyThemePreference } from "../hooks/useTheme";
import { TOKEN_KEY, USER_KEY } from "./authStorage";

interface AuthUser {
  user_id: number;
  display_name: string;
  email: string;
  dark_mode: boolean;
  reward_preference: string;
  saved_card_ids: string[];
  is_verified: boolean;
}

interface AuthContextValue {
  user: AuthUser | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoadingAuth: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (
    email: string,
    password: string,
    displayName: string,
  ) => Promise<{ is_verified: boolean }>;
  logout: () => Promise<void>;
  setUserDarkMode: (darkMode: boolean) => void;
  verifyEmail: (otp: string) => Promise<void>;
  resendOtp: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(
    () => localStorage.getItem(TOKEN_KEY),
  );
  const [user, setUser] = useState<AuthUser | null>(() => {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? (JSON.parse(raw) as AuthUser) : null;
  });
  const [isLoadingAuth, setIsLoadingAuth] = useState<boolean>(!!token);

  function toAuthUser(profile: UserProfile): AuthUser {
    return {
      user_id: profile.user_id,
      display_name: profile.display_name,
      email: profile.email,
      dark_mode: profile.dark_mode,
      reward_preference: profile.reward_preference,
      saved_card_ids: profile.saved_card_ids ?? [],
      is_verified: (profile as UserProfile & { is_verified?: boolean }).is_verified ?? false,
    };
  }

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

  // On app boot, if token exists, immediately hydrate full profile from GET /me.
  useEffect(() => {
    if (!token) {
      setIsLoadingAuth(false);
      return;
    }

    let cancelled = false;

    async function bootstrapAuth() {
      setIsLoadingAuth(true);
      try {
        const profile = await getMe();
        if (!cancelled) {
          setUser(toAuthUser(profile));
          applyThemePreference(profile.dark_mode ? "dark" : "light");
        }
      } catch {
        if (!cancelled) {
          setToken(null);
          setUser(null);
          localStorage.removeItem(TOKEN_KEY);
          localStorage.removeItem(USER_KEY);
        }
      } finally {
        if (!cancelled) {
          setIsLoadingAuth(false);
        }
      }
    }

    void bootstrapAuth();

    return () => {
      cancelled = true;
    };
  }, [token]);

  async function login(email: string, password: string): Promise<void> {
    setIsLoadingAuth(true);
    try {
      const auth = await loginRequest({ email, password });
      // Persist token before GET /me since client auth headers read localStorage.
      localStorage.setItem(TOKEN_KEY, auth.access_token);
      setToken(auth.access_token);
      const profile = await getMe();
      setUser(toAuthUser(profile));
      applyThemePreference(profile.dark_mode ? "dark" : "light");
    } finally {
      setIsLoadingAuth(false);
    }
  }

  async function signup(
    email: string,
    password: string,
    displayName: string,
  ): Promise<{ is_verified: boolean }> {
    setIsLoadingAuth(true);
    try {
      const auth = await signupRequest({
        email,
        password,
        display_name: displayName,
      });
      localStorage.setItem(TOKEN_KEY, auth.access_token);
      setToken(auth.access_token);
      const profile = await getMe();
      setUser(toAuthUser(profile));
      applyThemePreference(profile.dark_mode ? "dark" : "light");
      return { is_verified: auth.is_verified };
    } finally {
      setIsLoadingAuth(false);
    }
  }

  async function verifyEmail(otp: string): Promise<void> {
    const auth = await verifyEmailRequest(otp);
    setUser((prev) => prev ? { ...prev, is_verified: auth.is_verified } : prev);
  }

  async function resendOtp(): Promise<void> {
    await resendOtpRequest();
  }

  async function logout(): Promise<void> {
    try {
      await logoutRequest();
    } finally {
      setToken(null);
      setUser(null);
    }
  }

  function setUserDarkMode(darkMode: boolean): void {
    setUser((prev) => (prev ? { ...prev, dark_mode: darkMode } : prev));
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!token && !!user,
        isLoadingAuth,
        login,
        signup,
        logout,
        setUserDarkMode,
        verifyEmail,
        resendOtp,
      }}
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
