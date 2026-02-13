"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { apiClient, getToken, setToken, removeToken } from "@/lib/api-client";
import type { User } from "@/types";

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  canAccessBrand: (brandId: string) => boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Local mode: default super admin user (no login required)
const LOCAL_USER: User = {
  id: "superadmin",
  email: "admin@localhost",
  name: "Local Admin",
  role: "super_admin",
  brand_ids: [],
  is_active: true,
};

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(LOCAL_USER);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Try to fetch actual user data from API (for brand_ids etc.)
    const syncUser = async () => {
      try {
        const userData = await apiClient.getCurrentUser();
        setUser(userData);
      } catch {
        // API not reachable or auth failed — keep local user
      }
    };

    syncUser();
  }, []);

  const login = async (email: string, password: string) => {
    const response = await apiClient.login({ email, password });
    setToken(response.access_token);
    setUser(response.user);
  };

  const logout = () => {
    removeToken();
    setUser(LOCAL_USER);
  };

  const canAccessBrand = (): boolean => {
    return true; // Local mode: full access
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: true, // Always authenticated in local mode
        login,
        logout,
        canAccessBrand,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
