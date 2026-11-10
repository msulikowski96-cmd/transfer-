import { Stack, useLocalSearchParams } from 'expo-router';
import { useMemo, useState } from 'react';
import { Pressable, ScrollView, TextInput, View } from 'react-native';

import { Accent, Card, ErrorBox, Kpi, Loading, PlayerItem, styles, UncertaintyTag } from '@/components/fmv';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { api, type Club, type PlayerProfile, type Simulation } from '@/lib/api';
import { age, eur, POSITION, signedPct } from '@/lib/format';
import { useApi } from '@/lib/use-api';

function Change({ from, to }: { from: number; to: number }) {
  const c = (to / from - 1) * 100;
  return <ThemedText type="small" style={{ color: c > 2 ? Accent.pos : c < -2 ? Accent.neg : undefined }}>{signedPct(c)}</ThemedText>;
}

function History({ p }: { p: PlayerProfile }) {
  const max = Math.max(...p.history.map((h) => h.market_value_eur));
  return (
    <View style={{ gap: 4 }}>
      {p.history.slice(-8).map((h) => (
        <View key={h.as_of_date} style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
          <ThemedText type="small" themeColor="textSecondary" style={{ width: 44 }}>{h.as_of_date.slice(0, 4)}</ThemedText>
          <View style={{ flex: 1, height: 10 }}>
            <View style={{ width: `${Math.max(2, (h.market_value_eur / max) * 100)}%`, height: 10, borderRadius: 5, backgroundColor: Accent.brand }} />
          </View>
          <ThemedText type="small" style={{ width: 92, textAlign: 'right' }}>{eur(h.market_value_eur)}</ThemedText>
        </View>
      ))}
    </View>
  );
}

function Simulator({ p }: { p: PlayerProfile }) {
  const theme = useTheme();
  const clubs = useApi('clubs', () => api<Club[]>('/clubs'));
  const [filter, setFilter] = useState('');
  const [result, setResult] = useState<Simulation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<number | null>(null);
  const list = useMemo(() => {
    const f = filter.trim().toLowerCase();
    return (clubs.data ?? []).filter((c) => c.id !== p.club_id && (!f || c.name.toLowerCase().includes(f))).slice(0, 6);
  }, [clubs.data, filter, p.club_id]);

  async function simulate(club: Club) {
    setBusy(club.id); setError(null);
    try {
      setResult(await api<Simulation>('/simulate', undefined, {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ player_id: p.id, buyer_club_id: club.id, window: 'summer' }),
      }));
    } catch (e) {
      setResult(null); setError(e instanceof Error ? e.message : String(e));
    } finally { setBusy(null); }
  }

  return (
    <Card title="Symulator transferu">
      <ThemedText type="small" themeColor="textSecondary">Ile zapłaciłby wybrany klub w letnim oknie.</ThemedText>
      <TextInput value={filter} onChangeText={setFilter} placeholder="Wpisz klub kupujący…" placeholderTextColor={theme.textSecondary}
                 accessibilityLabel="Klub kupujący"
                 style={{ color: theme.text, backgroundColor: theme.background, borderRadius: Spacing.two, paddingHorizontal: Spacing.three, paddingVertical: Spacing.two }} />
      {clubs.loading ? <Loading /> : list.map((c) => (
        <Pressable key={c.id} onPress={() => simulate(c)} style={({ pressed }) => [{ paddingVertical: 6, flexDirection: 'row', justifyContent: 'space-between' }, pressed && { opacity: 0.6 }]}>
          <ThemedText type="small">{c.name} <ThemedText type="small" themeColor="textSecondary">· {c.league}</ThemedText></ThemedText>
          <ThemedText type="smallBold" style={{ color: Accent.brand }}>{busy === c.id ? '…' : 'Symuluj'}</ThemedText>
        </Pressable>
      ))}
      {error ? <ThemedText type="small" style={{ color: Accent.neg }}>{error}</ThemedText> : null}
      {result ? (
        <ThemedText>
          {result.buyer.name}: <ThemedText type="smallBold">{eur(result.transfer_value_eur)}</ThemedText>
          <ThemedText type="small" themeColor="textSecondary"> ({eur(result.transfer_value_low_eur)} – {eur(result.transfer_value_high_eur)}), </ThemedText>
          <ThemedText type="small" style={{ color: result.premium_pct >= 0 ? Accent.pos : Accent.neg }}>{signedPct(result.premium_pct)}</ThemedText>
          <ThemedText type="small"> względem Market Value</ThemedText>
        </ThemedText>
      ) : null}
    </Card>
  );
}

export default function PlayerScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const res = useApi(`player:${id}`, () => api<PlayerProfile>(`/players/${encodeURIComponent(id)}`));
  const p = res.data;
  return (
    <ThemedView style={{ flex: 1 }}>
      <Stack.Screen options={{ title: p?.name ?? 'Zawodnik' }} />
      <ScrollView contentContainerStyle={[styles.screen, { paddingTop: 16 }]}>
        {res.error ? <ErrorBox message={res.error} onRetry={res.reload} /> : null}
        {!p && !res.error ? <Loading /> : null}
        {p ? (
          <>
            <ThemedText type="small" themeColor="textSecondary">
              {age(p.age)} lat · {POSITION[p.position] ?? p.position} · {p.club ?? '—'} ({p.league ?? '—'}) · {p.nationality}
            </ThemedText>
            <View style={styles.grid}>
              <Kpi label="Market Value" value={eur(p.market_value_eur)} accent note={`80%: ${eur(p.market_value_low_eur)} – ${eur(p.market_value_high_eur)}`} />
              <Kpi label="Transfer Value" value={eur(p.transfer_value_eur)} note={`${eur(p.transfer_value_low_eur)} – ${eur(p.transfer_value_high_eur)}`} />
            </View>
            <Card title="Prognoza">
              {([['6 mies.', p.forecast_6m_eur], ['12 mies.', p.forecast_12m_eur], ['24 mies.', p.forecast_24m_eur]] as const).map(([l, v]) => (
                <View key={l} style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                  <ThemedText type="small">{l}</ThemedText>
                  <ThemedText type="small"><ThemedText type="smallBold">{eur(v)}</ThemedText>  {v ? <Change from={p.market_value_eur} to={v} /> : null}</ThemedText>
                </View>
              ))}
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 4 }}>
                <ThemedText type="small" themeColor="textSecondary">Niepewność (względem innych wycen):</ThemedText>
                <UncertaintyTag level={p.uncertainty} />
              </View>
              <ThemedText type="small" themeColor="textSecondary">Wycena referencyjna: {eur(p.reference_value_eur)}</ThemedText>
            </Card>
            <Card title="Dlaczego taka wycena?">
              <ThemedText type="small">{p.details.summary}</ThemedText>
              {p.details.factors.slice(0, 6).map((f) => (
                <View key={f.factor} style={{ flexDirection: 'row', justifyContent: 'space-between', gap: 8 }}>
                  <ThemedText type="small" style={{ flex: 1 }}>{f.factor}</ThemedText>
                  <ThemedText type="smallBold" style={{ color: f.direction === 'positive' ? Accent.pos : f.direction === 'negative' ? Accent.neg : undefined }}>
                    {signedPct(f.effect_pct)}
                  </ThemedText>
                </View>
              ))}
            </Card>
            <Card title="Historia wyceny"><History p={p} /></Card>
            <Simulator p={p} />
            <Card title="Ostatnie sezony">
              {p.seasons.slice(0, 5).map((s, i) => (
                <View key={i} style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                  <ThemedText type="small" style={{ flex: 1 }} numberOfLines={1}>{s.season} · {s.club}</ThemedText>
                  <ThemedText type="small" themeColor="textSecondary">{s.minutes}′ · {s.goals} G · {s.assists} A</ThemedText>
                </View>
              ))}
            </Card>
            <Card title="Podobni zawodnicy">{p.similar.map((s) => <PlayerItem key={s.id} p={s} />)}</Card>
          </>
        ) : null}
      </ScrollView>
    </ThemedView>
  );
}
