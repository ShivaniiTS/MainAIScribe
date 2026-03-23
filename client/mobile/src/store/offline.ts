/**
 * Offline queue — caches encounter + audio uploads when offline,
 * retries when connectivity is restored.
 */
import { create } from "zustand";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Network from "expo-network";
import { createEncounter, uploadEncounterAudio } from "../lib/api";

const QUEUE_KEY = "ai_scribe_offline_queue";

export interface QueuedEncounter {
  id: string;
  provider_id: string;
  patient_id: string;
  visit_type: string;
  mode: string;
  audioUri: string;
  filename: string;
  createdAt: string;
  status: "queued" | "uploading" | "failed";
  error?: string;
}

interface OfflineState {
  queue: QueuedEncounter[];
  isOnline: boolean;
  enqueue: (item: Omit<QueuedEncounter, "id" | "createdAt" | "status">) => Promise<void>;
  remove: (id: string) => void;
  processQueue: () => Promise<void>;
  checkConnectivity: () => Promise<boolean>;
  load: () => Promise<void>;
}

function makeId(): string {
  return `offline-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export const useOfflineStore = create<OfflineState>((set, get) => ({
  queue: [],
  isOnline: true,

  enqueue: async (item) => {
    const entry: QueuedEncounter = {
      ...item,
      id: makeId(),
      createdAt: new Date().toISOString(),
      status: "queued",
    };
    const newQueue = [...get().queue, entry];
    set({ queue: newQueue });
    await AsyncStorage.setItem(QUEUE_KEY, JSON.stringify(newQueue));
  },

  remove: (id) => {
    const newQueue = get().queue.filter((q) => q.id !== id);
    set({ queue: newQueue });
    AsyncStorage.setItem(QUEUE_KEY, JSON.stringify(newQueue));
  },

  processQueue: async () => {
    const { queue, isOnline } = get();
    if (!isOnline || queue.length === 0) return;

    for (const item of queue) {
      if (item.status === "uploading") continue;

      // Mark uploading
      const updated = get().queue.map((q) =>
        q.id === item.id ? { ...q, status: "uploading" as const } : q,
      );
      set({ queue: updated });

      try {
        const enc = await createEncounter({
          provider_id: item.provider_id,
          patient_id: item.patient_id,
          visit_type: item.visit_type,
          mode: item.mode,
        });
        await uploadEncounterAudio(enc.encounter_id, item.audioUri, item.filename);
        get().remove(item.id);
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : "Upload failed";
        const failedQueue = get().queue.map((q) =>
          q.id === item.id ? { ...q, status: "failed" as const, error: errMsg } : q,
        );
        set({ queue: failedQueue });
        await AsyncStorage.setItem(QUEUE_KEY, JSON.stringify(failedQueue));
      }
    }
  },

  checkConnectivity: async () => {
    try {
      const state = await Network.getNetworkStateAsync();
      const online = state.isConnected === true && state.isInternetReachable !== false;
      set({ isOnline: online });
      return online;
    } catch {
      set({ isOnline: false });
      return false;
    }
  },

  load: async () => {
    try {
      const raw = await AsyncStorage.getItem(QUEUE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as QueuedEncounter[];
        // Reset any "uploading" items back to "queued"
        const reset = parsed.map((q) =>
          q.status === "uploading" ? { ...q, status: "queued" as const } : q,
        );
        set({ queue: reset });
      }
    } catch {
      // ignore
    }
    const state = await Network.getNetworkStateAsync();
    set({ isOnline: state.isConnected === true });
  },
}));
