import { Link } from 'expo-router';
import { type ReactNode } from 'react';
import { ActivityIndicator, Platform, Pressable, StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { Spacing } from '@/constants/theme';
import type { PlayerRow } from '@/lib/api';
import { age, eur, UNCERTAINTY } from '@/lib/format';

export const Accent = { pos: '#1fa971', neg: '#e5484d', warn: '#d39b1f', brand: '#1fa971' } as const;

export function Card({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <ThemedView type="backgroundElement" style={styles.card}>
      {title ? <ThemedText type="smallBold" style={styles.cardTitle}>{title}</ThemedText> : null}
      {children}
    </ThemedView>
  );
}

export function Kpi({ label, value, note, accent }: { label: string; value: string; note?: string; accent?: boolean }) {
  return (
    <ThemedView type="backgroundElement" style={styles.kpi}>
      <ThemedText type="small" themeColor="textSecondary">{label}</ThemedText>
      <ThemedText style={[styles.kpiValue, accent && { color: Accent.brand }]}>{value}</ThemedText>
      {note ? <ThemedText type="small" themeColor="textSecondary">{note}</ThemedText> : null}
    </ThemedView>
  );
}

export function UncertaintyTag({ level }: { level: string }) {
  const color = level === 'low' ? Accent.pos : level === 'medium' ? Accent.warn : Accent.neg;
  return (
    <View style={[styles.tag, { borderColor: color }]}>
      <ThemedText type="small" style={{ color, fontSize: 12, lineHeight: 16 }}>{UNCERTAINTY[level] ?? level}</ThemedText>
    </View>
  );
}

export function PlayerItem({ p, right }: { p: PlayerRow; right?: ReactNode }) {
  return (
    <Link href={{ pathname: '/players/[id]', params: { id: String(p.id) } }} asChild>
      <Pressable style={({ pressed }) => [styles.row, pressed && { opacity: 0.6 }]}>
        <View style={{ flex: 1, minWidth: 0 }}>
          <ThemedText type="smallBold" numberOfLines={1}>{p.name}</ThemedText>
          <ThemedText type="small" themeColor="textSecondary" numberOfLines={1}>
            {age(p.age)} · {p.position} · {p.club ?? '—'}
          </ThemedText>
        </View>
        <View style={{ alignItems: 'flex-end' }}>
          <ThemedText type="smallBold">{eur(p.market_value_eur)}</ThemedText>
          {right ?? <UncertaintyTag level={p.uncertainty} />}
        </View>
      </Pressable>
    </Link>
  );
}

export function Loading() {
  return <ActivityIndicator style={{ marginTop: Spacing.five }} />;
}

export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <Card>
      <ThemedText type="smallBold" style={{ color: Accent.neg }}>Nie udało się pobrać danych</ThemedText>
      <ThemedText type="small" themeColor="textSecondary">{message}</ThemedText>
      {onRetry ? (
        <Pressable onPress={onRetry} style={styles.button}><ThemedText type="smallBold" style={{ color: '#fff' }}>Spróbuj ponownie</ThemedText></Pressable>
      ) : null}
    </Card>
  );
}

export function Chips({ options, value, onChange }: { options: [string, string][]; value: string; onChange: (v: string) => void }) {
  return (
    <View style={styles.chips}>
      {options.map(([v, label]) => (
        <Pressable key={v} onPress={() => onChange(v)}>
          <ThemedView type={v === value ? 'backgroundSelected' : 'backgroundElement'} style={styles.chip}>
            <ThemedText type="small" themeColor={v === value ? 'text' : 'textSecondary'}>{label}</ThemedText>
          </ThemedView>
        </Pressable>
      ))}
    </View>
  );
}

export const styles = StyleSheet.create({
  card: { borderRadius: Spacing.three, padding: Spacing.three, gap: Spacing.two },
  cardTitle: { marginBottom: Spacing.one },
  kpi: { flexGrow: 1, flexBasis: '45%', borderRadius: Spacing.three, padding: Spacing.three, gap: Spacing.half },
  kpiValue: { fontSize: 24, lineHeight: 30, fontWeight: 700 },
  tag: { borderWidth: 1, borderRadius: 99, paddingHorizontal: Spacing.two, marginTop: Spacing.half },
  row: { flexDirection: 'row', alignItems: 'center', gap: Spacing.two, paddingVertical: Spacing.two },
  button: { backgroundColor: Accent.brand, borderRadius: Spacing.two, paddingVertical: Spacing.two, paddingHorizontal: Spacing.three, alignSelf: 'flex-start' },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: Spacing.two },
  chip: { borderRadius: 99, paddingHorizontal: Spacing.three, paddingVertical: Spacing.one },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: Spacing.two },
  // on web the tab bar floats over the top of the screen (app-tabs.web.tsx)
  screen: { paddingHorizontal: Spacing.three, paddingBottom: Spacing.three, paddingTop: Platform.OS === 'web' ? 88 : Spacing.three, gap: Spacing.three,
            width: '100%', maxWidth: 800, alignSelf: 'center' },
});
