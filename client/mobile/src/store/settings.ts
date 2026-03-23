/**
 * Settings store — persists API URL via AsyncStorage.
 *
 * The mobile app connects to the **provider-facing server** (FastAPI on port 8000).
 * On first launch the default URL is derived from the Expo dev server's LAN IP
 * so the app works immediately during development without manual configuration.
 * In production, users set the URL via the Settings screen.
 */
import { create } from "zustand";
import AsyncStorage from "@react-native-async-storage/async-storage";
import Constants from "expo-constants";

const STORAGE_KEY = "ai_scribe_settings";
const PROVIDER_SERVER_PORT = "8000";

/**
 * Derive a sensible default API URL.
 *
 * During Expo development `Constants.expoConfig?.hostUri` is something like
 * "192.168.1.42:8081". We strip the Expo port and replace it with the
 * provider-facing server port (8000).
 *
 * Falls back to http://localhost:8000 when the host URI is unavailable
 * (e.g. production builds).
 */
function getDefaultApiUrl(): string {
  const hostUri = Constants.expoConfig?.hostUri; // e.g. "192.168.1.42:8081"
  if (hostUri) {
    const host = hostUri.split(":")[0]; // strip Expo port
    return `http://${host}:${PROVIDER_SERVER_PORT}`;
  }
  return `http://localhost:${PROVIDER_SERVER_PORT}`;
}

export const DEFAULT_API_URL = getDefaultApiUrl();

interface SettingsState {
  apiUrl: string;
  loaded: boolean;
  /** Whether the user has explicitly saved a URL (vs. using auto-detected default) */
  configured: boolean;
  setApiUrl: (url: string) => void;
  load: () => Promise<void>;
}

export const useSettings = create<SettingsState>((set, get) => ({
  apiUrl: DEFAULT_API_URL,
  loaded: false,
  configured: false,
  setApiUrl: (url: string) => {
    set({ apiUrl: url, configured: true });
    AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ apiUrl: url }));
  },
  load: async () => {
    try {
      const raw = await AsyncStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed.apiUrl) set({ apiUrl: parsed.apiUrl, configured: true });
      }
    } catch {
      // ignore
    }
    set({ loaded: true });
  },
}));

/** Synchronous getter for API URL (used by api.ts outside of React) */
export function getApiUrl(): string {
  return useSettings.getState().apiUrl;
}
