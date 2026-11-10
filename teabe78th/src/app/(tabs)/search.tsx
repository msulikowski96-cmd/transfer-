import { useEffect, useState } from 'react';
import { FlatList, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Chips, ErrorBox, Loading, PlayerItem, styles } from '@/components/fmv';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { BottomTabInset, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { api, type PlayerRow } from '@/lib/api';
import { GROUPS, int } from '@/lib/format';
import { useApi } from '@/lib/use-api';

export default function SearchScreen() {
  const theme = useTheme();
  const [text, setText] = useState('');
  const [query, setQuery] = useState('');
  const [group, setGroup] = useState('');
  useEffect(() => {
    const t = setTimeout(() => setQuery(text.trim()), 350);
    return () => clearTimeout(t);
  }, [text]);
  const res = useApi(`search:${query}:${group}`, () => api<{ total: number; items: PlayerRow[] }>('/players', { q: query, position_group: group, limit: 50 }));

  return (
    <ThemedView style={{ flex: 1 }}>
      <SafeAreaView style={{ flex: 1 }} edges={['top']}>
        <View style={[styles.screen, { paddingBottom: 0 }]}>
          <ThemedText type="subtitle">Szukaj</ThemedText>
          <TextInput value={text} onChangeText={setText} placeholder="Zawodnik lub klub…" placeholderTextColor={theme.textSecondary}
                     autoCorrect={false} clearButtonMode="while-editing" accessibilityLabel="Szukaj zawodnika lub klubu"
                     style={{ color: theme.text, backgroundColor: theme.backgroundElement, borderRadius: Spacing.two,
                              paddingHorizontal: Spacing.three, paddingVertical: Spacing.two, fontSize: 16 }} />
          <Chips options={GROUPS} value={group} onChange={setGroup} />
          {res.data ? <ThemedText type="small" themeColor="textSecondary">{int(res.data.total)} zawodników</ThemedText> : null}
        </View>
        {res.error ? <View style={styles.screen}><ErrorBox message={res.error} onRetry={res.reload} /></View> : null}
        {res.loading && !res.data ? <Loading /> : (
          <FlatList data={res.data?.items ?? []} keyExtractor={(p) => String(p.id)} keyboardShouldPersistTaps="handled"
                    contentContainerStyle={[styles.screen, { paddingTop: 0, paddingBottom: BottomTabInset + 16, gap: 0 }]}
                    renderItem={({ item }) => <PlayerItem p={item} />} />
        )}
      </SafeAreaView>
    </ThemedView>
  );
}
