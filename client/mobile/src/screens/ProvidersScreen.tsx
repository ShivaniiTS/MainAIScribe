/**
 * Providers list screen.
 */
import React, { useEffect, useState, useCallback } from "react";
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  RefreshControl,
  useWindowDimensions,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";

import Badge from "../components/Badge";
import { colors, fontSize, spacing, radius } from "../lib/theme";
import { fetchProviders, type ProviderSummary } from "../lib/api";
import { useSettings } from "../store/settings";

export default function ProvidersScreen() {
  const { width } = useWindowDimensions();
  const isTablet = width >= 768;
  const numColumns = isTablet ? 2 : 1;
  const apiUrl = useSettings((s) => s.apiUrl);

  const [providers, setProviders] = useState<ProviderSummary[]>([]);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    const ps = await fetchProviders();
    setProviders(ps);
  }, []);

  useEffect(() => {
    load();
  }, [load, apiUrl]);

  const onRefresh = async () => {
    setRefreshing(true);
    await load().catch(() => {});
    setRefreshing(false);
  };

  const renderItem = ({ item }: { item: ProviderSummary }) => {
    const score = item.latest_score;
    const scoreVariant = () => {
      if (score == null) return "neutral" as const;
      if (score >= 4.5) return "success" as const;
      if (score >= 4.0) return "info" as const;
      return "warning" as const;
    };

    return (
      <View style={[styles.card, isTablet && styles.tabletCard]}>
        <View style={styles.cardTop}>
          <View style={styles.avatar}>
            <Ionicons name="person" size={22} color={colors.brand} />
          </View>
          <View style={{ flex: 1, marginLeft: spacing.md }}>
            <Text style={styles.name}>{item.name ?? item.id}</Text>
            {item.credentials && (
              <Text style={styles.credentials}>{item.credentials}</Text>
            )}
          </View>
        </View>

        <View style={styles.cardMeta}>
          {item.specialty && <Badge label={item.specialty} variant="info" />}
          {score != null && (
            <Badge label={`Quality: ${score.toFixed(2)}`} variant={scoreVariant()} />
          )}
        </View>

        {/* Version scores */}
        {Object.keys(item.quality_scores).length > 0 && (
          <View style={styles.scoresRow}>
            {Object.entries(item.quality_scores)
              .sort(([a], [b]) => a.localeCompare(b))
              .slice(-4)
              .map(([ver, s]) => (
                <View key={ver} style={styles.scorePill}>
                  <Text style={styles.scoreVersion}>{ver}</Text>
                  <Text style={styles.scoreNum}>{s.toFixed(2)}</Text>
                </View>
              ))}
          </View>
        )}
      </View>
    );
  };

  return (
    <FlatList
      style={styles.container}
      data={providers}
      keyExtractor={(p) => p.id}
      renderItem={renderItem}
      numColumns={numColumns}
      key={numColumns}
      contentContainerStyle={[styles.listContent, isTablet && styles.tabletListContent]}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.brand} />}
      ListEmptyComponent={
        <View style={styles.empty}>
          <Ionicons name="people-outline" size={40} color={colors.textTertiary} />
          <Text style={styles.emptyText}>No providers configured</Text>
        </View>
      }
    />
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  listContent: { padding: spacing.lg, gap: spacing.md },
  tabletListContent: { maxWidth: 900, alignSelf: "center", width: "100%" },
  card: {
    flex: 1,
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
  },
  tabletCard: { marginHorizontal: spacing.xs },
  cardTop: { flexDirection: "row", alignItems: "center" },
  avatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: "#D1FAE5",
    alignItems: "center",
    justifyContent: "center",
  },
  name: { fontSize: fontSize.md, fontWeight: "600", color: colors.text },
  credentials: { fontSize: fontSize.xs, color: colors.textSecondary },
  cardMeta: { flexDirection: "row", gap: spacing.sm, marginTop: spacing.md },
  scoresRow: { flexDirection: "row", gap: spacing.sm, marginTop: spacing.md, flexWrap: "wrap" },
  scorePill: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: colors.borderLight,
    borderRadius: radius.full,
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    gap: spacing.xs,
  },
  scoreVersion: { fontSize: fontSize.xs, color: colors.textTertiary, fontWeight: "500" },
  scoreNum: { fontSize: fontSize.xs, color: colors.text, fontWeight: "600" },
  empty: { alignItems: "center", marginTop: 80 },
  emptyText: { fontSize: fontSize.sm, color: colors.textTertiary, marginTop: spacing.md },
});
