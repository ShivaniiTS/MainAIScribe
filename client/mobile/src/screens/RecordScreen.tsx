/**
 * Record screen — audio capture with provider/patient/visit-type selection.
 * Mirrors the web Capture page. Supports offline queueing.
 */
import React, { useEffect, useState, useRef, useCallback } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Alert,
  useWindowDimensions,
  FlatList,
  TextInput,
  ActivityIndicator,
} from "react-native";
import { Audio } from "expo-av";
import { Ionicons } from "@expo/vector-icons";
import { useNavigation } from "@react-navigation/native";

import Card from "../components/Card";
import ProgressBar from "../components/ProgressBar";
import { colors, fontSize, spacing, radius } from "../lib/theme";
import {
  fetchProviders,
  searchPatients,
  createEncounter,
  uploadEncounterAudio,
  getWsUrl,
  type ProviderSummary,
  type PatientSearchResult,
} from "../lib/api";
import { useSettings } from "../store/settings";
import { useOfflineStore } from "../store/offline";

const VISIT_TYPES = [
  { value: "initial_evaluation", label: "Initial Evaluation" },
  { value: "follow_up", label: "Follow-up" },
  { value: "assume_care", label: "Assume Care" },
  { value: "discharge", label: "Discharge" },
];

const MODES = [
  { value: "dictation", label: "Dictation" },
  { value: "ambient", label: "Ambient" },
];

type PipelineStage = "idle" | "recording" | "creating" | "uploading" | "processing" | "complete" | "error";

export default function RecordScreen() {
  const nav = useNavigation<any>();
  const { width } = useWindowDimensions();
  const isTablet = width >= 768;

  // Data
  const [providers, setProviders] = useState<ProviderSummary[]>([]);
  const [patients, setPatients] = useState<PatientSearchResult[]>([]);
  const [patientQuery, setPatientQuery] = useState("");

  // Selections
  const [providerId, setProviderId] = useState("");
  const [selectedPatient, setSelectedPatient] = useState<PatientSearchResult | null>(null);
  const [visitType, setVisitType] = useState("follow_up");
  const [mode, setMode] = useState("dictation");

  // Recording
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [recordingUri, setRecordingUri] = useState<string | null>(null);
  const [duration, setDuration] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Pipeline
  const [stage, setStage] = useState<PipelineStage>("idle");
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState("");
  const [resultSampleId, setResultSampleId] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Offline
  const { isOnline, enqueue, checkConnectivity } = useOfflineStore();

  // Re-fetch when the API URL changes (e.g. user saves cloudflare tunnel URL)
  const apiUrl = useSettings((s) => s.apiUrl);

  // Load data
  useEffect(() => {
    fetchProviders()
      .then((ps) => {
        setProviders(ps);
        if (ps.length > 0 && !providerId) setProviderId(ps[0].id);
      })
      .catch(() => {});
    searchPatients("").then(setPatients).catch(() => {});
  }, [apiUrl]);

  // Patient search
  useEffect(() => {
    if (patientQuery.length === 0) {
      searchPatients("").then(setPatients).catch(() => {});
      return;
    }
    const timer = setTimeout(() => {
      searchPatients(patientQuery).then(setPatients).catch(() => {});
    }, 300);
    return () => clearTimeout(timer);
  }, [patientQuery]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      wsRef.current?.close();
    };
  }, []);

  // ---- Recording ----
  const startRecording = async () => {
    try {
      const perm = await Audio.requestPermissionsAsync();
      if (!perm.granted) {
        Alert.alert("Permission required", "Microphone access is needed to record audio.");
        return;
      }
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });
      const { recording: rec } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY,
      );
      setRecording(rec);
      setRecordingUri(null);
      setDuration(0);
      setStage("recording");
      timerRef.current = setInterval(() => setDuration((d) => d + 1), 1000);
    } catch (err) {
      Alert.alert("Error", "Could not start recording.");
    }
  };

  const stopRecording = async () => {
    if (!recording) return;
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    await recording.stopAndUnloadAsync();
    await Audio.setAudioModeAsync({ allowsRecordingIOS: false });
    const uri = recording.getURI();
    setRecordingUri(uri);
    setRecording(null);
    setStage("idle");
  };

  const discardRecording = () => {
    setRecordingUri(null);
    setDuration(0);
  };

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  // ---- Submit ----
  const canSubmit = providerId && selectedPatient && recordingUri && stage === "idle";

  const handleSubmit = useCallback(async () => {
    if (!selectedPatient || !recordingUri) return;

    // Check connectivity
    const online = await checkConnectivity();
    if (!online) {
      await enqueue({
        provider_id: providerId,
        patient_id: selectedPatient.id,
        visit_type: visitType,
        mode,
        audioUri: recordingUri,
        filename: "recording.m4a",
      });
      Alert.alert(
        "Saved Offline",
        "No network connection. Your recording has been saved and will be uploaded automatically when connectivity is restored.",
      );
      discardRecording();
      return;
    }

    try {
      setStage("creating");
      setStatusMsg("Creating encounter...");
      setProgress(0);

      const enc = await createEncounter({
        provider_id: providerId,
        patient_id: selectedPatient.id,
        visit_type: visitType,
        mode,
      });

      // Connect WebSocket for progress
      const ws = new WebSocket(`${getWsUrl()}/ws/encounters/${enc.encounter_id}`);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "progress") {
          setProgress(data.pct ?? 0);
          setStatusMsg(data.message ?? `Stage: ${data.stage}`);
        } else if (data.type === "complete") {
          setStage("complete");
          setProgress(100);
          setStatusMsg("Pipeline complete — note generated");
          setResultSampleId(data.sample_id ?? null);
          ws.close();
        } else if (data.type === "error") {
          setStage("error");
          setStatusMsg(data.error ?? "Pipeline error");
          ws.close();
        }
      };

      ws.onerror = () => {
        if (stage !== "complete" && stage !== "error") {
          setStatusMsg("Running pipeline (live updates unavailable)");
        }
      };

      setStage("uploading");
      setStatusMsg("Uploading audio...");
      setProgress(5);

      const result = await uploadEncounterAudio(enc.encounter_id, recordingUri, "recording.m4a");

      setStage("processing");
      setStatusMsg(result.message ?? "Pipeline running...");
      setResultSampleId(result.sample_id);
      setProgress(10);
    } catch (err) {
      setStage("error");
      setStatusMsg(err instanceof Error ? err.message : "Unknown error");
    }
  }, [providerId, selectedPatient, recordingUri, visitType, mode, stage, checkConnectivity, enqueue]);

  const resetForm = () => {
    setStage("idle");
    setStatusMsg("");
    setProgress(0);
    setResultSampleId(null);
    discardRecording();
    wsRef.current?.close();
  };

  // ---- Render ----
  const showPatientList = patientQuery.length >= 0 && !selectedPatient;

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={[styles.content, isTablet && styles.tabletContent]}
    >
      <Text style={styles.title}>Record Encounter</Text>
      <Text style={styles.subtitle}>Capture audio to generate a clinical note</Text>

      {!isOnline && (
        <Card style={{ backgroundColor: "#FEF3C7", borderColor: "#F59E0B" }}>
          <View style={styles.row}>
            <Ionicons name="cloud-offline" size={18} color="#92400E" />
            <Text style={{ color: "#92400E", fontSize: fontSize.sm, marginLeft: spacing.sm }}>
              Offline — recordings will be queued for upload
            </Text>
          </View>
        </Card>
      )}

      {/* Provider */}
      <Card>
        <Text style={styles.label}>Provider</Text>
        {providers.length === 0 ? (
          <Text style={styles.sublabel}>Loading providers...</Text>
        ) : (
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={{ marginTop: spacing.sm }}>
            {providers.map((p) => (
              <TouchableOpacity
                key={p.id}
                onPress={() => setProviderId(p.id)}
                style={[
                  styles.chip,
                  providerId === p.id && styles.chipActive,
                ]}
              >
                <Text style={[styles.chipText, providerId === p.id && styles.chipTextActive]}>
                  {p.name ?? p.id}
                </Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        )}
      </Card>

      {/* Patient */}
      <Card>
        <Text style={styles.label}>Patient</Text>
        {selectedPatient ? (
          <View style={styles.selectedPatient}>
            <View style={{ flex: 1 }}>
              <Text style={styles.patientName}>
                {selectedPatient.first_name} {selectedPatient.last_name}
              </Text>
              <Text style={styles.patientMeta}>
                MRN: {selectedPatient.mrn} · DOB: {selectedPatient.date_of_birth}
              </Text>
            </View>
            <TouchableOpacity onPress={() => { setSelectedPatient(null); setPatientQuery(""); }}>
              <Ionicons name="close-circle" size={22} color={colors.textTertiary} />
            </TouchableOpacity>
          </View>
        ) : (
          <>
            <View style={styles.searchBox}>
              <Ionicons name="search" size={16} color={colors.textTertiary} />
              <TextInput
                value={patientQuery}
                onChangeText={setPatientQuery}
                placeholder="Search by name or MRN..."
                style={styles.searchInput}
                placeholderTextColor={colors.textTertiary}
              />
            </View>
            <FlatList
              data={patients.slice(0, 6)}
              keyExtractor={(p) => p.id}
              scrollEnabled={false}
              renderItem={({ item }) => (
                <TouchableOpacity style={styles.patientRow} onPress={() => setSelectedPatient(item)}>
                  <Text style={styles.patientName}>
                    {item.first_name} {item.last_name}
                  </Text>
                  <Text style={styles.patientMeta}>
                    MRN: {item.mrn} · {item.sex} · {item.date_of_birth}
                  </Text>
                </TouchableOpacity>
              )}
            />
          </>
        )}
      </Card>

      {/* Visit type + mode */}
      <Card>
        <Text style={styles.label}>Visit Details</Text>
        <View style={[styles.row, { marginTop: spacing.sm }]}>
          <View style={{ flex: 1, marginRight: spacing.sm }}>
            <Text style={styles.sublabel}>Type</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              {VISIT_TYPES.map((v) => (
                <TouchableOpacity
                  key={v.value}
                  onPress={() => setVisitType(v.value)}
                  style={[styles.chip, visitType === v.value && styles.chipActive]}
                >
                  <Text style={[styles.chipText, visitType === v.value && styles.chipTextActive]}>
                    {v.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
        </View>
        <View style={{ marginTop: spacing.md }}>
          <Text style={styles.sublabel}>Mode</Text>
          <View style={styles.row}>
            {MODES.map((m) => (
              <TouchableOpacity
                key={m.value}
                onPress={() => setMode(m.value)}
                style={[styles.chip, mode === m.value && styles.chipActive]}
              >
                <Text style={[styles.chipText, mode === m.value && styles.chipTextActive]}>
                  {m.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      </Card>

      {/* Audio recording */}
      <Card>
        <Text style={styles.label}>Audio</Text>

        {!recordingUri ? (
          <View style={styles.recordSection}>
            <TouchableOpacity
              style={[styles.recordBtn, stage === "recording" && styles.recordBtnActive]}
              onPress={stage === "recording" ? stopRecording : startRecording}
              disabled={stage !== "idle" && stage !== "recording"}
            >
              <Ionicons
                name={stage === "recording" ? "stop" : "mic"}
                size={32}
                color={colors.textInverse}
              />
            </TouchableOpacity>
            <View style={{ marginLeft: spacing.lg }}>
              {stage === "recording" ? (
                <>
                  <View style={styles.row}>
                    <View style={styles.pulseDot} />
                    <Text style={[styles.recordingLabel, { color: colors.error }]}>Recording...</Text>
                  </View>
                  <Text style={styles.timer}>{formatTime(duration)}</Text>
                </>
              ) : (
                <Text style={styles.tapToRecord}>Tap to start recording</Text>
              )}
            </View>
          </View>
        ) : (
          <View style={styles.recordedRow}>
            <Ionicons name="document-text" size={22} color={colors.brand} />
            <View style={{ flex: 1, marginLeft: spacing.sm }}>
              <Text style={{ fontSize: fontSize.sm, fontWeight: "600", color: colors.text }}>
                Recording ({formatTime(duration)})
              </Text>
            </View>
            <TouchableOpacity onPress={discardRecording}>
              <Ionicons name="close-circle" size={22} color={colors.textTertiary} />
            </TouchableOpacity>
          </View>
        )}
      </Card>

      {/* Submit / Progress */}
      {stage === "idle" && (
        <TouchableOpacity
          style={[styles.submitBtn, !canSubmit && styles.submitBtnDisabled]}
          onPress={handleSubmit}
          disabled={!canSubmit}
        >
          <Text style={styles.submitBtnText}>Run Pipeline</Text>
        </TouchableOpacity>
      )}

      {stage !== "idle" && stage !== "recording" && (
        <Card>
          <ProgressBar
            progress={progress}
            color={stage === "error" ? colors.error : stage === "complete" ? colors.brand : colors.indigo}
          />
          <View style={[styles.row, { marginTop: spacing.md }]}>
            {(stage === "creating" || stage === "uploading" || stage === "processing") && (
              <ActivityIndicator size="small" color={colors.indigo} style={{ marginRight: spacing.sm }} />
            )}
            {stage === "complete" && (
              <Ionicons name="checkmark-circle" size={18} color={colors.brand} style={{ marginRight: spacing.sm }} />
            )}
            {stage === "error" && (
              <Ionicons name="close-circle" size={18} color={colors.error} style={{ marginRight: spacing.sm }} />
            )}
            <Text style={{ fontSize: fontSize.sm, color: stage === "error" ? colors.error : colors.text, flex: 1 }}>
              {statusMsg}
            </Text>
          </View>

          {stage === "complete" && resultSampleId && (
            <View style={[styles.row, { marginTop: spacing.lg }]}>
              <TouchableOpacity
                style={styles.viewNoteBtn}
                onPress={() => {
                  // Switch to Encounters tab with both EncountersList and EncounterDetail
                  // in the stack so the native back button works correctly.
                  nav.navigate("Encounters", {
                    screen: "EncounterDetail",
                    params: { sampleId: resultSampleId },
                    initial: false,
                  });
                }}
              >
                <Text style={{ color: colors.textInverse, fontWeight: "600", fontSize: fontSize.sm }}>View Note</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.newEncounterBtn} onPress={resetForm}>
                <Text style={{ color: colors.textSecondary, fontWeight: "600", fontSize: fontSize.sm }}>New Encounter</Text>
              </TouchableOpacity>
            </View>
          )}

          {stage === "error" && (
            <TouchableOpacity style={[styles.newEncounterBtn, { marginTop: spacing.md }]} onPress={resetForm}>
              <Text style={{ color: colors.textSecondary, fontWeight: "600", fontSize: fontSize.sm }}>Try Again</Text>
            </TouchableOpacity>
          )}
        </Card>
      )}

      <View style={{ height: 40 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  content: { padding: spacing.lg, gap: spacing.md },
  tabletContent: { maxWidth: 640, alignSelf: "center", width: "100%" },
  title: { fontSize: fontSize.xxl, fontWeight: "700", color: colors.text },
  subtitle: { fontSize: fontSize.sm, color: colors.textSecondary, marginBottom: spacing.sm },
  label: { fontSize: fontSize.sm, fontWeight: "600", color: colors.text },
  sublabel: { fontSize: fontSize.xs, fontWeight: "500", color: colors.textSecondary, marginBottom: spacing.xs },
  row: { flexDirection: "row", alignItems: "center" },
  chip: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radius.full,
    borderWidth: 1,
    borderColor: colors.border,
    marginRight: spacing.sm,
    backgroundColor: colors.card,
  },
  chipActive: { backgroundColor: colors.brand, borderColor: colors.brand },
  chipText: { fontSize: fontSize.xs, color: colors.textSecondary, fontWeight: "500" },
  chipTextActive: { color: colors.textInverse },
  searchBox: {
    flexDirection: "row",
    alignItems: "center",
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    paddingHorizontal: spacing.md,
    marginTop: spacing.sm,
    marginBottom: spacing.sm,
  },
  searchInput: { flex: 1, paddingVertical: spacing.sm, marginLeft: spacing.sm, fontSize: fontSize.sm, color: colors.text },
  patientRow: { paddingVertical: spacing.sm, borderBottomWidth: 1, borderBottomColor: colors.borderLight },
  patientName: { fontSize: fontSize.sm, fontWeight: "600", color: colors.text },
  patientMeta: { fontSize: fontSize.xs, color: colors.textSecondary, marginTop: 2 },
  selectedPatient: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#D1FAE5",
    borderRadius: radius.md,
    padding: spacing.md,
    marginTop: spacing.sm,
  },
  recordSection: { flexDirection: "row", alignItems: "center", marginTop: spacing.md },
  recordBtn: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: colors.brand,
    alignItems: "center",
    justifyContent: "center",
  },
  recordBtnActive: { backgroundColor: colors.error },
  pulseDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: colors.error, marginRight: spacing.sm },
  recordingLabel: { fontSize: fontSize.sm, fontWeight: "600" },
  timer: { fontSize: fontSize.xxl, fontWeight: "700", fontVariant: ["tabular-nums"], color: colors.text, marginTop: 2 },
  tapToRecord: { fontSize: fontSize.sm, color: colors.textSecondary },
  recordedRow: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#D1FAE5",
    borderRadius: radius.md,
    padding: spacing.md,
    marginTop: spacing.md,
  },
  submitBtn: {
    backgroundColor: colors.brand,
    paddingVertical: spacing.lg,
    borderRadius: radius.lg,
    alignItems: "center",
  },
  submitBtnDisabled: { opacity: 0.5 },
  submitBtnText: { color: colors.textInverse, fontWeight: "700", fontSize: fontSize.md },
  viewNoteBtn: {
    backgroundColor: colors.brand,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: radius.md,
    marginRight: spacing.sm,
  },
  newEncounterBtn: {
    backgroundColor: "#F3F4F6",
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: radius.md,
  },
});
