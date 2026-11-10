import { useState } from 'react';
import { RefreshControl, ScrollView, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Accent, Card, Chips, ErrorBox, Kpi, Loading, PlayerItem, styles } from '@/components/fmv';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { BottomTabInset } from '@/constants/theme';
import { api, type PlayerRow, type Summary } from '@/lib/api';
import { eur, GROUPS, int, pct } from '@/lib/format';
import { useApi } from '@/lib/use-api';

export default function MarketScreen() {
  const [group, setGroup] = useState('');
  const summary = useApi('summary', () => api<Summary>('/summary'));
  const under = useApi(`under:${group}`, () => api<PlayerRow[]>('/rankings/undervalued', { limit: 15, position_group: group }));
  const top = useApi('top', () => api<{ items: PlayerRow[] }>('/players', { limit: 10 }));
  const s = summary.data;
  const refresh = () => { summary.reload(); under.reload(); top.reload(); };

  return (
    <ThemedView style={{ flex: 1 }}>
      <SafeAreaView style={{ flex: 1 }} edges={['top']}>
        <ScrollView contentContainerStyle={[styles.screen, { paddingBottom: BottomTabInset + 16 }]}
                    refreshControl={<RefreshControl refreshing={summary.loading && !!s} onRefresh={refresh} />}>
          <View>
            <ThemedText type="subtitle">FMV Intelligence</ThemedText>
            <ThemedText type="small" themeColor="textSecondary">
              {s ? `Wyceny na dzień ${s.as_of} · model ${s.model_version}` : 'Wycena piłkarzy na europejskim rynku'}
            </ThemedText>
          </View>
          {summary.error ? <ErrorBox message={summary.error} onRetry={refresh} /> : null}
          {!s && !summary.error ? <Loading /> : null}
          {s ? (
            <>
              {s.synthetic_data ? (
                <ThemedText type="small" style={{ color: Accent.warn }}>Dane syntetyczne: zawodnicy i kwoty są fikcyjne.</ThemedText>
              ) : (
                <ThemedText type="small" themeColor="textSecondary">Źródło: {s.data_sources.join(', ')} (dane badawcze).</ThemedText>
              )}
              <View style={styles.grid}>
                <Kpi label="Zawodnicy z wyceną" value={int(s.valued_players)} note={`${int(s.players)} w bazie`} />
                <Kpi label="Typowy błąd Transfer Value" value={pct(s.test_metrics.transfer_value.median_ape)}
                     note={`${int(s.test_metrics.transfer_value.n)} transferów testowych`} />
                <Kpi label="Błąd wyceny referencyjnej" value={pct(s.test_metrics.reference_valuation.median_ape)} note="ten sam zbiór testowy" />
                <Kpi label="Pokrycie przedziału 80%" value={pct(s.test_metrics.market_value.interval_coverage)} note="Market Value" />
              </View>
            </>
          ) : null}

          <Card title="Niedowartościowani">
            <ThemedText type="small" themeColor="textSecondary">
              Market Value modelu najwyżej ponad wyceną referencyjną, tylko pewne wyceny.
            </ThemedText>
            <Chips options={GROUPS} value={group} onChange={setGroup} />
            {under.loading ? <Loading /> : under.error ? <ErrorBox message={under.error} /> : (
              under.data?.length ? under.data.map((p) => (
                <PlayerItem key={p.id} p={p} right={
                  <ThemedText type="small" style={{ color: Accent.pos }}>×{(p.value_ratio ?? 0).toFixed(2)} vs {eur(p.reference_value_eur)}</ThemedText>
                } />
              )) : <ThemedText type="small" themeColor="textSecondary">Brak zawodników spełniających kryteria.</ThemedText>
            )}
          </Card>

          <Card title="Najwyżej wyceniani">
            {top.loading ? <Loading /> : top.error ? <ErrorBox message={top.error} /> : top.data?.items.map((p) => <PlayerItem key={p.id} p={p} />)}
          </Card>
        </ScrollView>
      </SafeAreaView>
    </ThemedView>
  );
}
