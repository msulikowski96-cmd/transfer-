import { useCallback, useEffect, useState } from 'react';

type State<T> = { data: T | null; error: string | null; key: string | null; nonce: number };

/**
 * Minimal data hook: data / error / loading, with reload.
 * ``key`` identifies the request: when it changes the data is fetched again. The previous data stays while reloading.
 */
export function useApi<T>(key: string, load: () => Promise<T>) {
  const [nonce, setNonce] = useState(0);
  const [state, setState] = useState<State<T>>({ data: null, error: null, key: null, nonce: -1 });

  useEffect(() => {
    let alive = true;
    load().then(
      (data) => alive && setState({ data, error: null, key, nonce }),
      (e: unknown) => alive && setState((s) => ({ data: s.data, error: e instanceof Error ? e.message : String(e), key, nonce })),
    );
    return () => { alive = false; };
    // the request is identified by key; load is a new closure on every render
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  const loading = state.key !== key || state.nonce !== nonce;
  return { data: state.data, error: loading ? null : state.error, loading, reload };
}
