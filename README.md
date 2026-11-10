# Football Market Value AI

Niezależny system wyceny piłkarzy na europejskim rynku transferowym. Dla każdego zawodnika liczy:

- **Market Value** – wartość w standardowych warunkach rynkowych (bez wiedzy o konkretnej transakcji),
- **Transfer Value** – kwotę oczekiwaną w konkretnej transakcji (kupujący, okno, kontrakt),
- **prognozę** wartości za 6, 12 i 24 miesiące,
- przedział 80%, poziom niepewności, jakość danych i wyjaśnienie wyceny.

Modele uczą się na **rzeczywistych kwotach transferów**. Wyceny publikowane przez portale są wyłącznie
punktem odniesienia: nie są ani etykietą, ani cechą modelu.

> **Stan danych.** Są dwa źródła:
> 1. **Prawdziwe dane**: importer otwartego zbioru [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)
>    (`fmv import-tm`), dane do lipca 2026. Zbiór ma licencję CC0, ale treść pochodzi z Transfermarkt, więc
>    traktujemy go jako dane badawcze, nie licencjonowane źródło dla produktu komercyjnego.
> 2. **Dane syntetyczne** (`fmv seed`): fikcyjny świat do testów i rozwoju bez dostępu do sieci.
>
> Docelowe źródło komercyjne (np. Sportmonks) wymaga osobnego importera i licencji.

## Szybki start

```bash
# całość w Dockerze: PostgreSQL + API + panel (pierwszy start ładuje dane, trenuje i wycenia, ok. 1-2 min)
docker compose up --build
# panel: http://localhost:3000   API: http://localhost:8000/docs
```

Lokalnie, bez Dockera (SQLite):

```bash
cd backend
python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"

# prawdziwe dane (pobiera ok. 70 MB, import ok. 1 min, trening i wyceny ok. 2 min)
python -m fmv.cli import-tm
# opcjonalnie (ok. 45 min): kontrakty, kontuzje i Ekstraklasa z API Transfermarktu, potem ponowny import
python -m fmv.cli fetch-tm-live
python -m fmv.cli import-tm --no-download
FMV_TRAIN_END=2022-06-20 FMV_VAL_END=2023-06-20 python -m fmv.cli train
python -m fmv.cli value
# albo: dane syntetyczne
python -m fmv.cli all            # seed + train + value
uvicorn fmv.api:app --port 8000
pytest                           # testy

cd ../frontend
npm install && npm run dev       # http://localhost:3000 (FMV_API_URL domyślnie http://localhost:8000)
```

| Komenda | Działanie |
| --- | --- |
| `fmv import-tm` | tworzy schemat i ładuje otwarty zbiór transfermarkt-datasets (kasuje poprzednie dane); używa danych z `fetch-tm-live`, jeśli są |
| `fmv fetch-tm-live` | pobiera z API Transfermarktu kontrakty, kontuzje i statystyki Ekstraklasy do pamięci podręcznej (można przerwać i wznowić) |
| `fmv seed` | tworzy schemat i ładuje dane syntetyczne (kasuje poprzednie) |
| `fmv train` | trenuje wszystkie silniki, zapisuje nową wersję modelu i raport jakości |
| `fmv value` | wycenia wszystkich zawodników (historia roczna + pełna wycena bieżąca) |
| `fmv report` | wypisuje raport jakości wybranej wersji |

Konfiguracja przez zmienne środowiskowe: `FMV_DATABASE_URL`, `FMV_ARTIFACTS_DIR`, `FMV_TRAIN_END`,
`FMV_VAL_END`, `FMV_WARMUP_YEARS`, `FMV_MODEL_VERSION`, `FMV_API_URL` (frontend).
Docker przy pierwszym starcie ładuje dane syntetyczne; prawdziwe dane ładuje się komendą `import-tm` w kontenerze `api`.

## Architektura

```
dane (PostgreSQL, 12 tabel)
  └─ features.py           cechy point-in-time dla pary (zawodnik, data)
       ├─ league_adjustment.py   Model D: trudność lig z zawodników zmieniających ligę
       └─ engines.py
            ├─ MarketValueEngine   Model A: CatBoost + LightGBM, wagi NNLS, przedział konforemny
            ├─ TransferFeeEngine   Model B: XGBoost kwantylowy, premia względem Market Value
            ├─ PotentialEngine     Model C: LightGBM kwantylowy, zmiana wartości za 12 i 24 mies.
            └─ DriftCorrector      bieżąca korekta dryfu rynku
  └─ pipeline.py           trening, ewaluacja, wyceny wsadowe, wyjaśnienia (Model E), podobni zawodnicy
  └─ api.py                FastAPI
frontend/                  Next.js: dashboard, profil, symulator, porównywarka, jakość modelu
teabe78th/                 Expo (iOS/Android/web): rynek, wyszukiwarka, profil z symulatorem, jakość modelu
```

### Jak powstaje wycena

1. **Cechy point-in-time.** Wycena na dany dzień korzysta tylko z tego, co było wtedy znane: sezonów
   zakończonych przed tą datą, biegnącego kontraktu z pierwotną datą końca, kontuzji rozpoczętych
   wcześniej, transferów z poprzednich 5 lat. Test `test_features_do_not_use_the_future` usuwa z danych
   wszystko po dacie wyceny i sprawdza, że cechy się nie zmieniają.
2. **Etykiety.** Tylko transfery definitywne z potwierdzoną, dodatnią kwotą stałą. Wypożyczenia, powroty,
   wolne transfery i kwoty nieujawnione są pomijane; bonusy warunkowe są trzymane osobno i nie wchodzą do etykiety.
3. **Indeks rynku.** Modele przewidują `log(kwota) − indeks`, gdzie indeks to średni log kwot z poprzednich
   24 miesięcy. Drzewa nie muszą dzięki temu ekstrapolować inflacji transferowej.
4. **Korekta lig (Model D).** Indeks gry zawodnika jest standaryzowany w grupie pozycji, a trudność ligi
   wynika z regresji na tych samych zawodnikach w kolejnych sezonach: zostający wyznaczają krzywą wieku,
   zmieniający ligę – różnicę między ligami.
5. **Market Value (Model A)** nie widzi żadnych informacji o kupującym ani transakcji.
   **Transfer Value (Model B)** modeluje premię lub dyskonto względem Market Value w zależności od
   kupującego (liga, wyniki, wydatki z 3 lat), sprzedającego, okna i dni do jego zamknięcia.
   Gdy kupujący nie jest podany, Transfer Value to mediana po 10 klubach, które typowo kupują w tej półce cenowej.
6. **Prognoza (Model C)** uczy się na rocznych migawkach tego samego zawodnika, jak zmienia się jego
   wycena po 12 i 24 miesiącach. Prognoza 6-miesięczna to połowa zmiany 12-miesięcznej (dane są sezonowe).
   Prognoza jest w cenach dzisiejszych: nie zakłada dalszej inflacji kwot.
7. **Przedziały** są kalibrowane konforemnie na danych walidacyjnych, więc deklarowane 80% jest sprawdzane
   na teście, a nie zakładane.
8. **Wyjaśnienie (Model E)** to wartości SHAP komponentu CatBoost zsumowane w 10 czytelnych grup oraz tekst
   złożony z nich szablonem. Cena nigdy nie pochodzi z modelu językowego.

### Podział chronologiczny

| Zbiór | Okres (domyślnie) | Rola |
| --- | --- | --- |
| trening | do 2021-06-20 | dopasowanie modeli i korekty lig |
| walidacja | do 2022-06-20 | early stopping, wagi ensemble, kalibracja przedziałów |
| test | później | pomiar jakości; modele nie widziały tych transferów |

## Jakość na prawdziwych danych (transfermarkt-datasets + API Transfermarktu)

Dane: otwarty zbiór transfermarkt-datasets (`fmv import-tm`) uzupełniony przez `fmv fetch-tm-live`
(biblioteka [transfermarkt-api](https://github.com/felipeall/transfermarkt-api)) o:
- kontrakt ze sprzedającym w dniu każdego transferu,
- historię kontuzji,
- statystyki Ekstraklasy.

Trening na transferach do 06.2022, walidacja do 06.2023, test na 2212 transferach z 07.2023–07.2026,
których model nie widział. Etykieta: stała kwota transferu definitywnego, sprzedający z jednej z 8 lig.

| | MAE | WAPE | Median APE | Pokrycie przedziału 80% |
| --- | --- | --- | --- | --- |
| Transfer Value | €4,34 mln | 42,7% | 43,7% | 81,8% |
| Market Value | €4,83 mln | 47,5% | 49,3% | 81,8% |
| wycena Transfermarkt sprzed transferu | – | 34,5% | 40,0% | – |
| mediana dla pozycji | – | 84,4% | 80,2% | – |

Bez danych z API (sam zbiór, 7 lig, 2110 transferów) Transfer Value miał medianę błędu 45,0%,
a Market Value 51,2%. Najważniejszą cechą modelu jest teraz długość kontraktu. Najwięcej zyskały
Bundesliga i Eredivisie oraz transfery powyżej €20 mln (mediana błędu z 38% do 33%).

**Wniosek: model nadal jest gorszy od wycen Transfermarkt**, choć różnica się zmniejszyła. Wyceny
Transfermarkt bywają aktualizowane tuż przed transferem, gdy kwota jest już znana z mediów, a modelowi
wciąż brakuje xG i wiedzy skautów. Najgorzej wypadają kwoty poniżej €1 mln (mediana błędu powyżej 100%).

Inne obserwacje:
- Wyuczona trudność lig jest sensowna: Premier League > La Liga ≈ Bundesliga ≈ Serie A > Ligue 1 > Eredivisie ≈ Liga Portugal.
- Ranking aktualnych wycen zgadza się z Transfermarkt w dużej mierze (korelacja rang 0,88, mediana stosunku 1,04).
- Ekstraklasa jest wyceniana, ale z dużym błędem (mediana 54% na 92 transferach testowych).
- Przedziały 80% są skalibrowane, ale bardzo szerokie.
- Prognoza zmiany wartości jest tylko trochę lepsza od założenia „bez zmian” (błąd log 0,48 wobec 0,52 dla 12 mies.).
- Starsi zawodnicy, zwłaszcza bramkarze po 35. roku życia, są wyraźnie przeszacowani.

Ograniczenia: brak xG i statystyk defensywnych. Kontrakt jest znany tylko w dniu transferu i dziś
(przedłużenia między transferami nie są widoczne). Dla starszych sezonów Ekstraklasy brakuje wyników
meczów, więc nie ma kontekstu drużyny. Wypożyczenia są rozpoznawane heurystycznie (powrót do tego samego
klubu w ciągu 400 dni).

**Kwestia prawna:** oba źródła to dane Transfermarktu. `fetch-tm-live` pobiera je z ich wewnętrznego API,
co regulamin Transfermarktu wyklucza. Tylko do prywatnych badań, nie do produktu komercyjnego.

## Jakość na danych syntetycznych (1603 transfery testowe)

| | MAE | WAPE | Median APE | Pokrycie przedziału 80% |
| --- | --- | --- | --- | --- |
| Transfer Value | €3,61 mln | 28,6% | 25,8% | 80,5% |
| Market Value | €4,03 mln | 32,0% | 28,4% | 82,7% |
| wycena referencyjna „portalu” | – | 34,2% | 31,9% | – |
| mediana dla pozycji | – | 77,0% | 70,3% | – |

Prognoza zmiany wartości: błąd (log) 0,26 dla 12 mies. i 0,33 dla 24 mies., wobec 0,46 i 0,77 przy założeniu
„bez zmian”; pokrycie pasm 80% i 81%. Raport zawiera też podział na pozycje, ligi i przedziały wartości
(`fmv report`, strona „Jakość modelu”, endpoint `/model`).

Market Value porównujemy z kwotą transferu, bo to jedyna obserwowalna cena. Część różnicy to warunki
konkretnej transakcji, a nie błąd wyceny.

## API

| Endpoint | Opis |
| --- | --- |
| `GET /summary` | liczniki, data wycen, wersja modelu, główne metryki |
| `GET /players?q=&league_id=&position_group=&sort=&limit=&offset=` | wyszukiwarka z aktualną wyceną |
| `GET /players/{id}` | profil: wycena, czynniki, prognoza, historia, podobni, sezony, transfery |
| `POST /simulate` | Transfer Value dla pary zawodnik–kupujący, okna i długości kontraktu |
| `GET /rankings/undervalued` | Market Value istotnie powyżej wyceny referencyjnej |
| `GET /compare?ids=1,2,3` | porównanie 2–4 zawodników |
| `GET /model` | parametry i pełny raport jakości wersji modelu |
| `GET /leagues`, `GET /clubs` | słowniki |

## Baza danych

`players`, `clubs`, `leagues`, `seasons`, `player_statistics`, `transfers`, `contracts`, `injuries`,
`market_valuations` (zewnętrzne wyceny referencyjne), `model_versions`, `model_predictions`
(wyceny systemu: data, zawodnik, wersja modelu, Market Value i Transfer Value z przedziałami, prognozy,
jakość danych, niepewność), `player_comparisons`. Definicje: `backend/fmv/db.py`.

## Czego jeszcze nie ma

- **Licencjonowanego źródła danych** z xG, kontraktami i Ekstraklasą (np. Sportmonks) oraz dopasowania
  tych samych zawodników między źródłami.
- **LSTM, MLflow, Redis, Celery, Evidently, wyjaśnień pisanych przez LLM** – świadomie odłożone do czasu,
  aż będzie na czym je zmierzyć. Wersje modeli i ich metryki są na razie w tabeli `model_versions`.
- **Automatycznych aktualizacji** (etap 6): `fmv train` i `fmv value` uruchamia się ręcznie lub z crona.
- **Finansów klubów**: siłę nabywczą przybliżają wydatki transferowe z 3 lat.

## Znane ograniczenia metody

- Etykiety pochodzą tylko od zawodników, którzy zostali sprzedani; wycena pozostałych jest ekstrapolacją.
- Modele produkcyjne są uczone na danych do końca okresu treningowego (walidacja służy kalibracji);
  nowsze transfery wpływają na wyceny tylko przez korektę dryfu.
- Statystyki są sezonowe, więc forma z trwającego sezonu nie wpływa na wycenę do jego zakończenia.
- Wyceny historyczne sprzed końca okresu treningowego pochodzą z modelu, który te sezony widział
  (na wykresie oznaczone pustym punktem).
- Prognoza uczy się na zawodnikach, którzy po roku lub dwóch nadal grali w obsługiwanych ligach.
- Symulator dla kupującego znacznie bogatszego lub biedniejszego niż typowy dla danej półki cenowej
  wychodzi poza zakres danych treningowych.
