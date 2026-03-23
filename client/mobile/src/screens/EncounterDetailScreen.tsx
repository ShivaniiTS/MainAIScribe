/**
 * Encounter detail — transcript + clinical note viewer with tabs.
 */
import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  useWindowDimensions,
} from "react-native";
import Markdown from "react-native-markdown-display";
import { Ionicons } from "@expo/vector-icons";

import { colors, fontSize, spacing, radius } from "../lib/theme";
import Badge from "../components/Badge";
import {
  fetchSample,
  fetchNote,
  fetchTranscript,
  type SampleDetail,
} from "../lib/api";

type Tab = "note" | "transcript";

export default function EncounterDetailScreen({ route }: any) {
  const { sampleId } = route.params as { sampleId: string };
  const { width } = useWindowDimensions();
  const isTablet = width >= 768;

  const [sample, setSample] = useState<SampleDetail | null>(null);
  const [noteContent, setNoteContent] = useState<string | null>(null);
  const [transcriptContent, setTranscriptContent] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("note");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, [sampleId]);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const s = await fetchSample(sampleId);
      setSample(s);

      const version = s.latest_version ?? undefined;

      const [noteRes, transcriptRes] = await Promise.allSettled([
        fetchNote(sampleId, version),
        fetchTranscript(sampleId, version),
      ]);

      if (noteRes.status === "fulfilled") setNoteContent(noteRes.value.content);
      if (transcriptRes.status === "fulfilled") setTranscriptContent(transcriptRes.value.content);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load encounter");
    }
    setLoading(false);
  };

  if (loading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color={colors.brand} />
      </View>
    );
  }

  if (error || !sample) {
    return (
      <View style={styles.centered}>
        <Ionicons name="alert-circle-outline" size={40} color={colors.error} />
        <Text style={styles.errorText}>{error ?? "Not found"}</Text>
      </View>
    );
  }

  const scoreVariant = (score: number | null | undefined) => {
    if (score == null) return "neutral" as const;
    if (score >= 4.5) return "success" as const;
    if (score >= 4.0) return "info" as const;
    if (score >= 3.5) return "warning" as const;
    return "error" as const;
  };

  const score = sample.quality?.overall;

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={[styles.header, isTablet && styles.tabletHeader]}>
        <View style={{ flex: 1 }}>
          <Text style={styles.title} numberOfLines={2}>{sampleId}</Text>
          <View style={styles.metaRow}>
            <Badge label={sample.mode} variant={sample.mode === "dictation" ? "info" : "success"} />
            <Text style={styles.metaText}>{sample.physician}</Text>
            {sample.latest_version && (
              <Text style={styles.metaText}>{sample.latest_version}</Text>
            )}
          </View>
        </View>
        {score != null && (
          <View style={styles.scoreBox}>
            <Text style={styles.scoreLabel}>Quality</Text>
            <Text style={[styles.scoreValue, { color: score >= 4.0 ? colors.brand : colors.warning }]}>
              {score.toFixed(2)}
            </Text>
          </View>
        )}
      </View>

      {/* Patient context */}
      {sample.patient_context?.patient?.name && (
        <View style={[styles.patientBar, isTablet && styles.tabletPatientBar]}>
          <Ionicons name="person" size={14} color={colors.textSecondary} />
          <Text style={styles.patientText}>
            {sample.patient_context.patient.name}
            {sample.patient_context.patient.date_of_birth
              ? ` · DOB: ${sample.patient_context.patient.date_of_birth}`
              : ""}
            {sample.patient_context.patient.sex ? ` · ${sample.patient_context.patient.sex}` : ""}
          </Text>
        </View>
      )}

      {/* Tabs */}
      <View style={[styles.tabRow, isTablet && styles.tabletTabRow]}>
        <TouchableOpacity
          style={[styles.tab, activeTab === "note" && styles.tabActive]}
          onPress={() => setActiveTab("note")}
        >
          <Ionicons name="document-text" size={16} color={activeTab === "note" ? colors.brand : colors.textTertiary} />
          <Text style={[styles.tabText, activeTab === "note" && styles.tabTextActive]}>Clinical Note</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === "transcript" && styles.tabActive]}
          onPress={() => setActiveTab("transcript")}
        >
          <Ionicons name="chatbubbles" size={16} color={activeTab === "transcript" ? colors.brand : colors.textTertiary} />
          <Text style={[styles.tabText, activeTab === "transcript" && styles.tabTextActive]}>Transcript</Text>
        </TouchableOpacity>
      </View>

      {/* Content — iPad uses split view */}
      {isTablet ? (
        <View style={styles.splitView}>
          <ScrollView style={styles.splitPane} contentContainerStyle={styles.markdownContainer}>
            <Text style={styles.paneTitle}>Clinical Note</Text>
            {noteContent ? (
              <Markdown style={markdownStyles}>{noteContent}</Markdown>
            ) : (
              <Text style={styles.emptyContent}>No note available</Text>
            )}
          </ScrollView>
          <View style={styles.splitDivider} />
          <ScrollView style={styles.splitPane} contentContainerStyle={styles.markdownContainer}>
            <Text style={styles.paneTitle}>Transcript</Text>
            {transcriptContent ? (
              <Text style={styles.transcriptText}>{transcriptContent}</Text>
            ) : (
              <Text style={styles.emptyContent}>No transcript available</Text>
            )}
          </ScrollView>
        </View>
      ) : (
        <ScrollView contentContainerStyle={styles.markdownContainer}>
          {activeTab === "note" ? (
            noteContent ? (
              <Markdown style={markdownStyles}>{noteContent}</Markdown>
            ) : (
              <Text style={styles.emptyContent}>No note available</Text>
            )
          ) : transcriptContent ? (
            <Text style={styles.transcriptText}>{transcriptContent}</Text>
          ) : (
            <Text style={styles.emptyContent}>No transcript available</Text>
          )}
        </ScrollView>
      )}
    </View>
  );
}

const markdownStyles = StyleSheet.create({
  heading1: { fontSize: fontSize.xl, fontWeight: "700", color: colors.text, marginBottom: spacing.sm, marginTop: spacing.lg },
  heading2: { fontSize: fontSize.lg, fontWeight: "700", color: colors.text, marginBottom: spacing.sm, marginTop: spacing.lg },
  heading3: { fontSize: fontSize.md, fontWeight: "600", color: colors.text, marginBottom: spacing.xs, marginTop: spacing.md },
  paragraph: { fontSize: fontSize.sm, color: colors.text, lineHeight: 22 },
  strong: { fontWeight: "700" },
  listItem: { fontSize: fontSize.sm, color: colors.text },
  hr: { borderColor: colors.border, marginVertical: spacing.lg },
  body: {},
  bullet_list: {},
  ordered_list: {},
  list_item: {},
  link: {},
  blockquote: {},
  code_inline: {},
  code_block: {},
  fence: {},
  table: {},
  thead: {},
  tbody: {},
  th: {},
  tr: {},
  td: {},
  image: {},
  text: {},
  textgroup: {},
  hardbreak: {},
  softbreak: {},
  pre: {},
  inline: {},
  span: {},
  s: {},
  em: {},
});

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  centered: { flex: 1, justifyContent: "center", alignItems: "center" },
  errorText: { fontSize: fontSize.sm, color: colors.error, marginTop: spacing.md },
  header: { paddingHorizontal: spacing.lg, paddingTop: spacing.lg, paddingBottom: spacing.sm, flexDirection: "row" },
  tabletHeader: { maxWidth: 900, alignSelf: "center", width: "100%" },
  title: { fontSize: fontSize.lg, fontWeight: "700", color: colors.text },
  metaRow: { flexDirection: "row", alignItems: "center", gap: spacing.sm, marginTop: spacing.xs },
  metaText: { fontSize: fontSize.xs, color: colors.textSecondary },
  scoreBox: { alignItems: "center", marginLeft: spacing.lg },
  scoreLabel: { fontSize: fontSize.xs, color: colors.textSecondary },
  scoreValue: { fontSize: fontSize.xxl, fontWeight: "700" },
  patientBar: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    backgroundColor: "#F0FDF4",
    marginHorizontal: spacing.lg,
    borderRadius: radius.md,
    gap: spacing.sm,
  },
  tabletPatientBar: { maxWidth: 900, alignSelf: "center", width: "100%" },
  patientText: { fontSize: fontSize.xs, color: colors.textSecondary },
  tabRow: {
    flexDirection: "row",
    paddingHorizontal: spacing.lg,
    marginTop: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  tabletTabRow: { maxWidth: 900, alignSelf: "center", width: "100%" },
  tab: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    marginRight: spacing.md,
    borderBottomWidth: 2,
    borderBottomColor: "transparent",
    gap: spacing.xs,
  },
  tabActive: { borderBottomColor: colors.brand },
  tabText: { fontSize: fontSize.sm, color: colors.textTertiary, fontWeight: "500" },
  tabTextActive: { color: colors.brand },
  markdownContainer: { padding: spacing.lg },
  emptyContent: { fontSize: fontSize.sm, color: colors.textTertiary, textAlign: "center", marginTop: 40 },
  transcriptText: { fontSize: fontSize.sm, color: colors.text, lineHeight: 22 },
  // iPad split view
  splitView: { flex: 1, flexDirection: "row" },
  splitPane: { flex: 1 },
  splitDivider: { width: 1, backgroundColor: colors.border },
  paneTitle: { fontSize: fontSize.md, fontWeight: "600", color: colors.text, marginBottom: spacing.md },
});
