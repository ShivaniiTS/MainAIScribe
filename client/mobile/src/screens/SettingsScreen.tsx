/**
 * Settings screen — API URL configuration + offline queue management.
 */
import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  ScrollView,
  Alert,
  useWindowDimensions,
  ActivityIndicator,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";

import Card from "../components/Card";
import Badge from "../components/Badge";
import { colors, fontSize, spacing, radius } from "../lib/theme";
import { useSettings, DEFAULT_API_URL } from "../store/settings";
import { useOfflineStore } from "../store/offline";

export default function SettingsScreen() {
  const { width } = useWindowDimensions();
  const isTablet = width >= 768;
  const { apiUrl, setApiUrl, configured } = useSettings();
  const { queue, remove, processQueue, isOnline, checkConnectivity } = useOfflineStore();
  const [urlDraft, setUrlDraft] = useState(apiUrl);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; msg: string } | null>(null);

  const saveUrl = () => {
    const trimmed = urlDraft.trim().replace(/\/+$/, "");
    if (!trimmed) {
      Alert.alert("Invalid URL", "API URL cannot be empty.");
      return;
    }
    setApiUrl(trimmed);
    setTestResult(null);
    Alert.alert("Saved", `API URL set to: ${trimmed}`);
  };

  const testConnection = async () => {
    const trimmed = urlDraft.trim().replace(/\/+$/, "");
    if (!trimmed) {
      setTestResult({ ok: false, msg: "URL cannot be empty" });
      return;
    }
    setTesting(true);
    setTestResult(null);
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);
      const res = await fetch(`${trimmed}/providers`, { signal: controller.signal });
      clearTimeout(timeout);
      if (res.ok) {
        setTestResult({ ok: true, msg: "Connected to provider-facing server" });
      } else {
        setTestResult({ ok: false, msg: `Server responded with ${res.status}` });
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Connection failed";
      setTestResult({ ok: false, msg: msg.includes("abort") ? "Connection timed out (5s)" : msg });
    }
    setTesting(false);
  };

  const resetToDefault = () => {
    setUrlDraft(DEFAULT_API_URL);
    setTestResult(null);
  };

  const retryQueue = async () => {
    await checkConnectivity();
    await processQueue();
  };

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={[styles.content, isTablet && styles.tabletContent]}
    >
      <Text style={styles.title}>Settings</Text>

      {/* First-launch banner */}
      {!configured && (
        <Card style={{ backgroundColor: "#DBEAFE", borderColor: "#3B82F6" }}>
          <View style={styles.row}>
            <Ionicons name="information-circle" size={18} color="#1D4ED8" />
            <Text style={{ color: "#1D4ED8", fontSize: fontSize.sm, marginLeft: spacing.sm, flex: 1 }}>
              Configure the provider-facing server URL below, then tap "Test Connection" to verify.
            </Text>
          </View>
        </Card>
      )}

      {/* API URL */}
      <Card>
        <Text style={styles.label}>Provider Server URL</Text>
        <Text style={styles.hint}>
          The provider-facing FastAPI server address (port 8000). Must be reachable from this device — use a LAN IP, not localhost.
        </Text>
        <TextInput
          value={urlDraft}
          onChangeText={(t) => { setUrlDraft(t); setTestResult(null); }}
          style={styles.input}
          placeholder="http://192.168.1.100:8000"
          placeholderTextColor={colors.textTertiary}
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
        />

        {/* Auto-detected default hint */}
        {!configured && DEFAULT_API_URL !== "http://localhost:8000" && (
          <Text style={[styles.hint, { color: colors.brand }]}>
            Auto-detected: {DEFAULT_API_URL}
          </Text>
        )}

        {/* Test result */}
        {testResult && (
          <View style={[styles.row, { marginTop: spacing.sm }]}>
            <Ionicons
              name={testResult.ok ? "checkmark-circle" : "close-circle"}
              size={16}
              color={testResult.ok ? colors.success : colors.error}
            />
            <Text style={{
              fontSize: fontSize.xs,
              color: testResult.ok ? colors.success : colors.error,
              marginLeft: spacing.xs,
              flex: 1,
            }}>
              {testResult.msg}
            </Text>
          </View>
        )}

        <View style={[styles.row, { marginTop: spacing.md, gap: spacing.sm }]}>
          <TouchableOpacity style={styles.testBtn} onPress={testConnection} disabled={testing}>
            {testing ? (
              <ActivityIndicator size="small" color={colors.indigo} />
            ) : (
              <Ionicons name="pulse" size={14} color={colors.indigo} />
            )}
            <Text style={styles.testBtnText}>Test Connection</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.saveBtn} onPress={saveUrl}>
            <Text style={styles.saveBtnText}>Save</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={resetToDefault} style={{ marginLeft: "auto" }}>
            <Text style={{ fontSize: fontSize.xs, color: colors.textTertiary }}>Reset to Default</Text>
          </TouchableOpacity>
        </View>
      </Card>

      {/* Connection status */}
      <Card>
        <View style={styles.row}>
          <Ionicons
            name={isOnline ? "cloud-done" : "cloud-offline"}
            size={20}
            color={isOnline ? colors.brand : colors.warning}
          />
          <Text style={[styles.statusText, { color: isOnline ? colors.brand : colors.warning }]}>
            {isOnline ? "Connected" : "Offline"}
          </Text>
          <TouchableOpacity onPress={checkConnectivity} style={{ marginLeft: "auto" }}>
            <Ionicons name="refresh" size={18} color={colors.textSecondary} />
          </TouchableOpacity>
        </View>
      </Card>

      {/* Offline queue */}
      <Card>
        <View style={styles.row}>
          <Text style={styles.label}>Offline Queue</Text>
          <Badge label={`${queue.length}`} variant={queue.length > 0 ? "warning" : "neutral"} />
        </View>

        {queue.length === 0 ? (
          <Text style={styles.hint}>No queued recordings.</Text>
        ) : (
          <>
            {queue.map((item) => (
              <View key={item.id} style={styles.queueItem}>
                <View style={{ flex: 1 }}>
                  <Text style={styles.queueTitle}>{item.provider_id}</Text>
                  <Text style={styles.queueMeta}>
                    {item.mode} · {item.visit_type} · {new Date(item.createdAt).toLocaleString()}
                  </Text>
                  {item.error && <Text style={styles.queueError}>{item.error}</Text>}
                </View>
                <Badge
                  label={item.status}
                  variant={item.status === "failed" ? "error" : item.status === "uploading" ? "info" : "neutral"}
                />
                <TouchableOpacity onPress={() => remove(item.id)} style={{ marginLeft: spacing.sm }}>
                  <Ionicons name="trash-outline" size={18} color={colors.error} />
                </TouchableOpacity>
              </View>
            ))}
            <TouchableOpacity style={styles.retryBtn} onPress={retryQueue}>
              <Ionicons name="cloud-upload" size={16} color={colors.textInverse} />
              <Text style={styles.retryBtnText}>Retry All</Text>
            </TouchableOpacity>
          </>
        )}
      </Card>

      {/* About */}
      <Card>
        <Text style={styles.label}>About</Text>
        <Text style={styles.hint}>AI Scribe v1.0.0</Text>
        <Text style={styles.hint}>Talisman Solutions</Text>
        <Text style={[styles.hint, { marginTop: spacing.sm }]}>
          HIPAA-compliant medical documentation. All audio is processed on your own servers — zero PHI egress.
        </Text>
      </Card>

      <View style={{ height: 40 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  content: { padding: spacing.lg, gap: spacing.md },
  tabletContent: { maxWidth: 640, alignSelf: "center", width: "100%" },
  title: { fontSize: fontSize.xxl, fontWeight: "700", color: colors.text },
  label: { fontSize: fontSize.sm, fontWeight: "600", color: colors.text },
  hint: { fontSize: fontSize.xs, color: colors.textSecondary, marginTop: spacing.xs },
  row: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  input: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    fontSize: fontSize.sm,
    color: colors.text,
    marginTop: spacing.md,
  },
  saveBtn: {
    backgroundColor: colors.brand,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    borderRadius: radius.md,
  },
  saveBtnText: { color: colors.textInverse, fontWeight: "600", fontSize: fontSize.sm },
  testBtn: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.indigo,
  },
  testBtnText: { color: colors.indigo, fontWeight: "600", fontSize: fontSize.sm },
  statusText: { fontSize: fontSize.sm, fontWeight: "600", marginLeft: spacing.sm },
  queueItem: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.borderLight,
  },
  queueTitle: { fontSize: fontSize.sm, fontWeight: "600", color: colors.text },
  queueMeta: { fontSize: fontSize.xs, color: colors.textSecondary },
  queueError: { fontSize: fontSize.xs, color: colors.error, marginTop: 2 },
  retryBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.brand,
    paddingVertical: spacing.sm,
    borderRadius: radius.md,
    marginTop: spacing.md,
    gap: spacing.sm,
  },
  retryBtnText: { color: colors.textInverse, fontWeight: "600", fontSize: fontSize.sm },
});
