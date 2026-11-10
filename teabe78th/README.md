# FMV Intelligence – aplikacja mobilna (Expo)

Klient mobilny API wycen z `../backend`. Zakładki:

- **Rynek**: liczby z bazy, jakość modelu, niedowartościowani zawodnicy, najwyżej wyceniani.
- **Szukaj**: wyszukiwarka zawodników z filtrem pozycji.
- **Model**: błędy na transferach testowych, podział na ligi, wyuczona trudność lig.
- **Profil zawodnika**: Market Value i Transfer Value z przedziałami, prognoza 6/12/24 mies., czynniki wyceny,
  historia, symulator transferu, ostatnie sezony, podobni zawodnicy.

Projekt EAS: `@teabe78th/teabe78th` (ID `43353551-7365-4641-8143-d710adbf8bbd`, ustawione przez `eas init`).
Expo SDK 58, Expo Router, trasy w `src/app/`.

## Uruchomienie

```bash
npm install
EXPO_PUBLIC_API_URL=http://localhost:8000 npx expo start   # web: klawisz w, telefon: Expo Go i kod QR
```

`EXPO_PUBLIC_API_URL` to adres API wycen (`uvicorn fmv.api:app` w `../backend`). `localhost` działa tylko w wersji
web i w emulatorze; na telefonie potrzebny jest adres widoczny z sieci telefonu (IP komputera w tej samej sieci
albo API wdrożone publicznie). Wartość jest wkompilowywana w aplikację przy starcie bundlera lub buildzie.

## Buildy (EAS)

```bash
npx eas-cli@latest build --profile preview --platform android   # instalowalny APK do testów
npx eas-cli@latest build --profile production --platform all
```

Przed buildem ustaw adres API dla buildu, np. `npx eas-cli@latest env:create --name EXPO_PUBLIC_API_URL --value https://... --environment preview`.

## Sprawdzanie

```bash
npx tsc --noEmit
npx expo lint
```
