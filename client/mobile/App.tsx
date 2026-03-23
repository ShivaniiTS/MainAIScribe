/**
 * AI Scribe — Mobile App
 *
 * Cross-platform (iPhone, iPad, Android) medical documentation app.
 * Connects to the same FastAPI backend as the web app.
 */
import React, { useEffect } from "react";
import { View, ActivityIndicator, Text, StyleSheet } from "react-native";
import { StatusBar } from "expo-status-bar";
import { NavigationContainer } from "@react-navigation/native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { Ionicons } from "@expo/vector-icons";

import RecordScreen from "./src/screens/RecordScreen";
import EncountersScreen from "./src/screens/EncountersScreen";
import EncounterDetailScreen from "./src/screens/EncounterDetailScreen";
import ProvidersScreen from "./src/screens/ProvidersScreen";
import SettingsScreen from "./src/screens/SettingsScreen";
import { colors, fontSize, spacing } from "./src/lib/theme";
import { useSettings } from "./src/store/settings";
import { useOfflineStore } from "./src/store/offline";

// Stack for Encounters tab (list → detail)
const EncounterStack = createNativeStackNavigator();

function EncountersStackScreen() {
  return (
    <EncounterStack.Navigator
      screenOptions={{
        headerTintColor: colors.brand,
        headerStyle: { backgroundColor: colors.card },
      }}
    >
      <EncounterStack.Screen
        name="EncountersList"
        component={EncountersScreen}
        options={{ title: "Encounters" }}
      />
      <EncounterStack.Screen
        name="EncounterDetail"
        component={EncounterDetailScreen}
        options={{ title: "Encounter" }}
      />
    </EncounterStack.Navigator>
  );
}

const Tab = createBottomTabNavigator();

export default function App() {
  const loadSettings = useSettings((s) => s.load);
  const loaded = useSettings((s) => s.loaded);
  const loadOffline = useOfflineStore((s) => s.load);
  const processQueue = useOfflineStore((s) => s.processQueue);

  useEffect(() => {
    loadSettings();
    loadOffline().then(() => processQueue());
  }, []);

  // Block rendering until settings are loaded from AsyncStorage so screens
  // use the saved API URL (e.g. cloudflare tunnel) instead of the default.
  if (!loaded) {
    return (
      <View style={splashStyles.container}>
        <ActivityIndicator size="large" color={colors.brand} />
        <Text style={splashStyles.text}>Loading settings...</Text>
      </View>
    );
  }

  return (
    <NavigationContainer>
      <StatusBar style="dark" />
      <Tab.Navigator
        screenOptions={({ route }) => ({
          headerShown: route.name !== "Encounters",
          headerStyle: { backgroundColor: colors.card },
          headerTintColor: colors.text,
          tabBarActiveTintColor: colors.brand,
          tabBarInactiveTintColor: colors.textTertiary,
          tabBarStyle: {
            backgroundColor: colors.card,
            borderTopColor: colors.border,
          },
          tabBarIcon: ({ color, size }) => {
            const icons: Record<string, keyof typeof Ionicons.glyphMap> = {
              Record: "mic",
              Encounters: "list",
              Providers: "people",
              Settings: "settings",
            };
            return <Ionicons name={icons[route.name] ?? "ellipse"} size={size} color={color} />;
          },
        })}
      >
        <Tab.Screen name="Record" component={RecordScreen} />
        <Tab.Screen
          name="Encounters"
          component={EncountersStackScreen}
          options={{ headerShown: false }}
        />
        <Tab.Screen name="Providers" component={ProvidersScreen} />
        <Tab.Screen name="Settings" component={SettingsScreen} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}

const splashStyles = StyleSheet.create({
  container: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: colors.bg },
  text: { marginTop: spacing.md, fontSize: fontSize.sm, color: colors.textSecondary },
});
