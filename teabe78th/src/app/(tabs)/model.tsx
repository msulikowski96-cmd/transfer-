import { ScrollView, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Accent, Card, ErrorBox, Loading, styles } from '@/components/fmv';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { BottomTabInset } from '@/constants/theme';
import { api, type Metrics, type ModelInfo } from '@/lib/api';
import { eur, int, pct } from '@/lib/format';
import { useApi } from '@/lib/use-api';

function Row({ label, m }: { label: string; m: Metrics }) {
  return (
    <View style={{ flexDirection: 'row', justifyContent: 'space-between', gap: 8, paddingVertical: 4 }}>
      <ThemedText type="small" style={{ flex: 1 }}>{label}</ThemedText>
      <ThemedText type="smallBold">{pct(m.median_ape)}</ThemedText>
      <ThemedText type="small" themeColor="textSecondary" style={{ width: 92, textAlign: 'right' }}>MAE {eur(m.mae_eur)}</ThemedText>
    </View>
  );
}

export default function ModelScreen() {
  const res = useApi('model', () => api<ModelInfo>('/model'));
  const m = res.data?.metrics;
  return (
    <ThemedView style={{ flex: 1 }}>
      <SafeAreaView style={{ flex: 1 }} edges={['top']}>
        <ScrollView contentContainerStyle={[styles.screen, { paddingBottom: BottomTabInset + 16 }]}>
          <ThemedText type="subtitle">Jakość modelu</ThemedText>
          {res.error ? <ErrorBox message={res.error} onRetry={res.reload} /> : null}
          {!m && !res.error ? <Loading /> : null}
          {m ? (
            <>
              <ThemedText type="small" themeColor="textSecondary">
                Wersja {res.data?.name}. Test na {int(Number(m.split.n_test))} transferach z {m.split.test_from} – {m.split.test_to},
                których model nie widział. Wartość: mediana błędu względnego.
              </ThemedText>
              <Card title="Błąd na transferach testowych">
                <Row label="Transfer Value" m={m.transfer_value.overall} />
                <Row label="Market Value" m={m.market_value.overall} />
                <Row label="Wycena referencyjna" m={m.baselines.reference_valuation} />
                <Row label="Mediana dla pozycji" m={m.baselines.position_median} />
              </Card>
              <Card title="Market Value według lig">
                {Object.entries(m.market_value.by_league).map(([l, v]) => <Row key={l} label={l} m={v} />)}
              </Card>
              <Card title="Wyuczona trudność lig">
                {Object.entries(m.league_difficulty).sort((a, b) => b[1] - a[1]).map(([l, v]) => (
                  <View key={l} style={{ flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 2 }}>
                    <ThemedText type="small">{l}</ThemedText>
                    <ThemedText type="smallBold" style={{ color: v >= 0 ? Accent.pos : Accent.neg }}>{v > 0 ? '+' : ''}{v.toFixed(2)}</ThemedText>
                  </View>
                ))}
              </Card>
            </>
          ) : null}
        </ScrollView>
      </SafeAreaView>
    </ThemedView>
  );
}
