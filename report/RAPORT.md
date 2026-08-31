# Od ε-zachłannej do Boltzmanna: wyżarzanie eksploracji w DQN

**Projekt zaliczeniowy z uczenia ze wzmocnieniem — pierwsza iteracja, środowiska OpenAI Gym.**

> Polska wersja `report/REPORT.md`. Numeracja rozdziałów, tabele, liczby i ścieżki
> do rysunków są identyczne, więc obie wersje można czytać równolegle.
> Nazwy plików, opcji konfiguracyjnych i wariantów pozostawiono w oryginale,
> bo tak nazywają się w kodzie.

---

## 1. Problem

Strategia ε-zachłanna (*ε-greedy*) to domyślny sposób eksploracji dla agentów
opartych na funkcji wartości z dyskretnym zbiorem akcji. Jest **niepoinformowana**:
z prawdopodobieństwem ε losuje jednostajnie z całego zbioru akcji, wydając większość
budżetu eksploracji na akcje, które funkcja Q już teraz stawia na ostatnim miejscu.
Eksploracja Boltzmanna (softmax) jest **poinformowana**: próbkuje proporcjonalnie do
`exp(Q/τ)`, więc eksploracja koncentruje się na akcjach, które wyglądają dobrze.

Haczyk polega na tym, że Boltzmann dziedziczy dowolne przekonanie zakodowane
w funkcji Q, a na początku uczenia to przekonanie jest szumem. Softmax po losowo
zainicjalizowanej sieci **nie jest** polityką jednostajną — jest polityką
*obciążoną szumem inicjalizacji*, co jest ściśle gorsze niż uczciwe losowanie
jednostajne, które wykonuje ε-zachłanna. Rozsądnie jest więc zacząć od ε-zachłannej
i przechodzić w stronę Boltzmanna w miarę, jak Q staje się wiarygodne.

Ten projekt pyta, **jak** przeprowadzić to przejście, i mierzy, czy to pomaga.

Zmierzona odpowiedź (§7.1) jest taka, że powyższa przesłanka jest **warunkowa,
a jej znak odwraca się między środowiskami**. Całkowite usunięcie progu ε jest
zdecydowanie gorsze niż ε-zachłanna na CartPole (`P = 0.04 [0.00, 0.24]`)
i zdecydowanie *lepsze* na LunarLanderze (`P = 0.92 [0.68, 1.00]`) — oba przedziały
leżą po przeciwnych stronach 0.5 i żaden jej nie zawiera. Próg ε to ubezpieczenie
od nieinformatywnej funkcji Q i — jak każde ubezpieczenie — warto je kupować tylko
wtedy, gdy składka, czyli koszt jednostajnie losowej akcji, jest niska.
Na CartPole losowa akcja jest niemal darmowa; na LunarLanderze odpala losowo
silniki i rozbija lądownik.

### Podsumowanie wyników

| | wynik |
|---|---|
| **Rozstrzygnięte** (przedział nie zawiera 0.5) | Wartość progu ε zmienia znak między CartPole a LunarLanderem (§7.1). Ścieżka nośnika `anneal_k` bije ε-zachłanną na zwrocie końcowym w Acrobocie, `P = 0.96 [0.76, 1.00]` (§7.2). |
| **Rozstrzygnięte** (powtórzone na dwóch maszynach) | Normalizacja skali Q decyduje o tym, gdzie ustala się entropia polityki zachowania, a tryb `none` nie ustala się nigdy (§8.1). Warianty bramkowane niepewnością załamują się, `P = 0.00`, i winna jest bramka, a nie architektura zespołowa (§8.2). |
| **Zdiagnozowane** | Bramka niepewności nigdy nie zadziałała, bo pewność normalizowano względem bieżącego kwantyla *własnego sygnału*, co śledzi każdy trend i go kasuje. Zamrożenie referencji sprawia, że przekazanie działa zgodnie z projektem — pewność 0.46 → 0.71, ε 1.00 → 0.30 — i rozdziela sygnał epistemiczny (niezgoda zespołu, spada 2×) od nieodpowiedniego (błąd TD, rośnie 3×). Nadal przegrywa z kontrolą (§8.2). |
| **Nierozstrzygnięte** | Jakikolwiek ranking wariantów z progiem ε na CartPole czy LunarLanderze. To, czy skalowanie `per_state` kosztuje wydajność. §6.2 pokazuje, że ranking z 3 ziaren nie przeżywa zmiany biblioteki BLAS. |

## 2. Metoda: jedna rodzina polityk, kilka ścieżek przez nią

Decyzja projektowa, która porządkuje tutaj wszystko, to odmowa traktowania
ε-zachłannej i Boltzmanna jako dwóch osobnych strategii mieszanych doraźnie.
Zamiast tego każdy wariant jest punktem w jednej sparametryzowanej rodzinie:

```
π(a|s) = ε_t · Uniform(A)  +  (1 − ε_t) · Softmax_{a ∈ TopK_t(Q)}( Q̃(s,·) / τ_t )
```

z trzema sterowanymi harmonogramem parametrami — `ε_t` (próg jednostajny),
`τ_t` (temperatura), `k_t` (rozmiar nośnika) — oraz odwzorowaniem skalującym `Q̃`
omówionym w §3.

Obie klasyczne strategie są **dokładnymi** punktami tej rodziny:

| konfiguracja | to dokładnie |
|---|---|
| `k = 1`, dowolne `τ` | ε-zachłanna (nośnikiem softmaksu jest zbiór argmaksów) |
| `τ → 0`, `k = \|A\|` | ε-zachłanna (niemaksymalne wykładniki zanikają do 0) |
| `ε = 0`, `k = \|A\|` | czysty Boltzmann |

To nie jest wygodne przybliżenie: `tests/test_policies.py` sprawdza te równości
jako **dokładne** równości tablic względem niezależnie napisanych implementacji
referencyjnych, na siatce wartości ε i τ, wraz z zachowaniem przy remisach
(oba przejścia graniczne muszą rozłożyć masę jednostajnie po remisujących
maksimach — a remisy są normą przy inicjalizacji, gdy każda akcja ma niemal
identyczne Q).

Dokładne punkty krańcowe są ważne, bo dzięki nim „wyżarzanie między nimi" staje się
dobrze określoną operacją: *ścieżką w przestrzeni (ε, τ, k)*. Dwa pomysły z tematu
projektu to wtedy dwie różne ścieżki.

**Ścieżka temperatury** (`eps_boltzmann`, pomysł 1 z tematu). Utrzymaj `k = |A|`
i zacznij od `τ = 10⁻⁴`, co czyni softmax dokładnym argmaksem, więc polityka
*jest* ε-zachłanna. Następnie podnoś `τ` geometrycznie do 0.3, podczas gdy `ε`
spada z 1.0 do 0.01. Polityka odkształca się w sposób ciągły w czystego Boltzmanna,
ani razu nie opuszczając rodziny.

**Ścieżka nośnika** (`anneal_k`, pomysły 1+2). Utrzymaj stałe `τ` i zwiększaj `k`
od 1 (znów dokładnie ε-zachłanna) do `|A|` (pełny Boltzmann). Przekazanie następuje
przez *poszerzanie nośnika*, a nie przez podnoszenie temperatury.

**Ścieżka mieszaniny** (`mixture_anneal`, pomysł 1a). Interpoluj wprost dwa
*rozkłady* akcji: `π = (1−β)·π_ε-zachłanna + β·π_Boltzmann`. To jest naprawdę coś
innego niż ścieżka temperatury, a nie jej przeparametryzowanie: przy β = 0.5
polityka zachowuje realną szansę na akcję jednostajnie losową, odziedziczoną po
składniku ε-zachłannym, nawet w stanach, w których Boltzmann jest pewny — podczas
gdy ścieżka temperatury przechodzi przez rozkłady, które nie są żadnym z krańców.
Która zachowuje się lepiej, to pytanie empiryczne i jedna z rzeczy tu mierzonych.

**Stały mały nośnik** (`topk_boltzmann`, `topk3_boltzmann`, pomysł 2). Boltzmann
po 2 lub 3 najlepszych akcjach z opadającym progiem ε. Uzasadnienie z tematu:
zaszumiona funkcja Q często stawia faktycznie najlepszą akcję na drugim miejscu,
więc rozłożenie prawdopodobieństwa na kilka czołowych akcji eksploruje dokładnie
te wiarygodne alternatywy. Próg ε jest zachowany (opada do 0.02), żeby akcja
błędnie wykluczona z top-k nie stała się nieosiągalna na zawsze.

**Nośnik adaptujący się do stanu** (`topp_boltzmann`). Próbkowanie jądrowe
(*nucleus sampling*): nośnikiem jest najmniejszy zbiór akcji o najwyższym Q,
którego masa softmaksowa osiąga `p = 0.9`. Tam, gdzie Q jest ostro zarysowane,
nośnik zapada się do zachłannego; tam, gdzie Q jest płaskie — czyli dokładnie tam,
gdzie agent nie ma podstaw do preferencji — poszerza się. To wydaje eksplorację
tam, gdzie funkcja Q jest niezdecydowana, a nie jednostajnie w czasie.

## 3. Szczegół, który decyduje o tym, czy cokolwiek z tego działa: skala wartości Q

Temperatura ma sens wyłącznie względem rozrzutu wartości, które dzieli,
a ten rozrzut nie jest stały:

- między środowiskami różni się o rzędy wielkości (zwroty w CartPole ~10–500,
  LunarLander ~−400–300, przycięte Atari ~0–50);
- **w obrębie jednego przebiegu rośnie systematycznie**, gdy funkcja wartości
  „pompuje się" od inicjalizacji bliskiej zeru do prawdziwej skali zwrotów.

Zatem stałe τ, które daje zdrową eksplorację przy 20 tys. kroków, przy 300 tys.
kroków jest efektywnie zachłanne — bez żadnej zmiany w harmonogramie. Każde
porównanie „ε-zachłanna kontra Boltzmann", które to ignoruje, w istocie mierzy
niekontrolowany, zależny od środowiska harmonogram wyżarzania, którego
eksperymentator nie wybrał.

Zaimplementowano i zbadano ablacyjnie trzy tryby:

- **`none`** — `Q̃ = Q`. Temperatura niesie surowe jednostki Q. Włączony po to,
  żeby *pokazać* tryb awarii, a nie tylko go zadeklarować.
- **`per_state`** — `Q̃ = (Q − mean_a Q)/(std_a Q + δ)`, liczone niezależnie
  w każdym stanie. Bezwymiarowe, ale *niszczy informację*: stan, w którym każda
  akcja jest naprawdę równie dobra, ma swój znikomy rozrzut Q rozdmuchany do
  wariancji jednostkowej, więc polityka staje się pewnie zaostrzona na czymś,
  co jest w rzeczywistości szumem numerycznym. Dokładnie odwrotnie, niż powinna
  działać eksploracja.
- **`running`** (domyślny) — `Q̃ = (Q − mean_a Q)/σ̂`, gdzie `σ̂` to EMA rozrzutu
  Q między akcjami po ostatnio odwiedzonych stanach. Bezwymiarowe, *a przy tym*
  normalizator jest wspólny dla stanów, więc płaski wiersz Q pozostaje płaski
  i daje politykę bliską jednostajnej, a wiersz zaostrzony pozostaje zaostrzony.

`tests/test_scaling.py` weryfikuje, że przy `running` dwa wiersze Q o identycznym
kształcie, lecz rzędach wielkości różniących się 1000×, dają *ten sam* rozkład
akcji z dokładnością do błędu zmiennoprzecinkowego, a przy `none` — nie dają
(jeden jest sensownie stochastyczny, drugi zapadł się do zachłannego).

## 4. Rozszerzenie: przekazanie bramkowane niepewnością

Harmonogram oparty na liczbie kroków to zgadywanie, kiedy Q staje się wiarygodne.
Rozszerzenie zastępuje je pomiarem.

Mając pewność `c ∈ [0,1]`, parametry są interpolowane między krańcem niepewnym
(`ε = 1`, `k = 1` — czyli ε-zachłanna) a krańcem pewnym (`ε = 0.01`, `k = |A|`,
`τ = 0.3` — czyli Boltzmann). Temperatura jest interpolowana geometrycznie,
ponieważ obejmuje rzędy wielkości.

Dwa sygnały:

- **`ensemble`** — 5 głowic Q typu bootstrap na wspólnym korpusie (maski
  Bernoulli(0.8) zapisywane per przejście), `u(s) = mean_a std_k Q_k(s,a)`.
  Ten sygnał jest **per stan**: agent może być Boltzmannem w dobrze odwiedzonym
  regionie, pozostając ε-zachłannym w nieznanym. Żaden harmonogram oparty na czasie
  tego nie wyrazi. Mierzy niepewność epistemiczną — to, czego dane nie przygwoździły
  — a nie wewnętrzną losowość zwrotu.
- **`td_error`** — EMA z |błędu TD|, który priorytetyzowany bufor powtórek i tak
  już liczy, więc jest darmowy. To pojedynczy skalar **globalny**, może więc
  wytworzyć tylko harmonogram zmienny w czasie — choć sterowany zmierzonym
  postępem uczenia, a nie licznikiem kroków.

Oba normalizują surowy sygnał względem bieżącego wysokiego kwantyla własnej
historii, więc pewność jest bezskalowa. Rozumowanie było takie, że niepewność
spada o rzędy wielkości w trakcie uczenia, więc jakikolwiek stały próg zadziałałby
raz i już nigdy się nie ruszył, sprowadzając adaptacyjny schemat do funkcji
schodkowej.

> **To właśnie ta decyzja projektowa zepsuła rozszerzenie, a §8.2 wyjaśnia dlaczego.**
> Normalizowanie sygnału względem bieżącego oszacowania wyliczonego z *tego samego
> sygnału* usuwa dokładnie ten trend, który bramka ma wykrywać: referencja podąża
> za sygnałem, iloraz zbiega do stałej, a pewność zostaje na niej przyszpilona na
> cały przebieg. Przesłanka też była błędna — zmierzone na Acrobocie, żaden
> z surowych sygnałów nie spada w trakcie uczenia; oba rosną, bo oba są wyrażone
> w wartościach Q, które pompują się w miarę uczenia funkcji wartości.
> `reference_freeze_step` (domyślnie wyłączony) to poprawka, a §8.2 pokazuje
> przekazanie, które po jej włączeniu wreszcie działa.

Dwa zabezpieczenia okazały się w praktyce konieczne. **Rozgrzewka**: zanim kwantyl
referencyjny zbierze dane, pewność jest przyszpilona do 0 — w przeciwnym razie
pierwsze stany raportują fałszywie wysoką pewność i przebieg startuje w niemal
zachłannym Boltzmannie na losowej sieci, czyli w najgorszej możliwej kombinacji.
**Wygładzanie**: `confidence_smoothing` interpoluje między surowym sygnałem
per stan (1.0) a globalną EMA (0.0), co przy okazji stanowi ablację oddzielającą
„rozdzielczość per stan" od „adaptacyjnego momentu przełączenia".

### Dlaczego wariant kontrolny jest sednem

Bootstrapowe głowice zespołu zmieniają *uczenie wartości*, a nie tylko pomiar
niepewności. Porównanie `uncertainty_gated`-z-zespołem z jednogłowicową
ε-zachłanną mieszałoby efekt bramkowania z efektem architektury. Seria
eksperymentów dotycząca niepewności zawiera więc `eps_greedy_ensemble` — zwykłą
ε-zachłanną działającą na **tej samej** 5-głowicowej architekturze — jako wariant,
który faktycznie odpowiada na pytanie badawcze, plus wariant ε-zachłanny
jednogłowicowy, aby zmierzyć, co robi sama zmiana architektury.

## 5. Agent bazowy

**Double + Dueling + n-krokowy(3) + priorytetyzowany bufor powtórek** — Rainbow
minus NoisyNets minus rozkładowy (*distributional*).

**NoisyNets celowo pominięto.** Sam jest mechanizmem eksploracji — wyuczonym
szumem parametrów — i wiadomo, że czyni ε-zachłanną zbędną. Włączenie go dałoby
każdemu wariantowi drugą, niekontrolowaną strategię eksploracji pod spodem tej
mierzonej, tłumiąc dokładnie te różnice, które to badanie ma wykryć. Kiedy
przedmiotem badania *jest* strategia eksploracji, agent bazowy nie może mieć
własnej.

**C51 pominięto** ze względu na koszt (około 2× czasu CPU na krok) oraz dlatego,
że dostarczany przez niego rozrzut jest aleatoryczny, podczas gdy rozszerzenie
potrzebuje niepewności epistemicznej — którą głowice zespołu dostarczają wprost.

Każdy inny komponent jest identyczny we wszystkich wariantach. `configs/base.yaml`
jest wspólny dla wszystkich wariantów, a loader konfiguracji **odrzuca nieznane
klucze**, zamiast je ignorować, więc literówka w konfiguracji eksperymentu kończy
się głośnym błędem, a nie po cichu przebiegiem, który testował coś innego.

## 6. Protokół

- **Ewaluacja jest zawsze zachłanna**, w osobnym środowisku z własnym strumieniem
  ziaren losowych, 10 epizodów na punkt. Tylko tak porównanie ma sens: polityka
  zachowania typu Boltzmann punktuje inaczej niż ε-zachłanna z powodów niezwiązanych
  z tym, jak dobrze się nauczyła, więc mierzenie zwrotu samej polityki zachowania
  mieszałoby koszt eksploracji z jakością uczenia. Remisy argmaksu są rozstrzygane
  jednostajnie losowo, a nie po indeksie — rozstrzyganie po indeksie obciąża
  ewaluację w stronę akcji o niskich numerach dla niedouczonej sieci.
- **Agregacja** używa **IQM** (średniej międzykwartylowej) z **warstwowymi
  bootstrapowymi** 95% przedziałami ufności po ziarnach, za Agarwal i in. (2021).
  Średnia po 3–5 ziarnach to zła statystyka dla DQN, który generuje ziarna
  z katastrofalną porażką wystarczająco często, by istotnie ją przesunąć;
  IQM jest znacznie odporniejsza i dużo mniej zaszumiona niż mediana. Raportowane
  jest też `probability_of_improvement`: zadaje pytanie, które praktyk faktycznie
  ma („jeśli uruchomię to raz, czy pobije punkt odniesienia?") i jest odporne na
  pojedyncze odstające ziarno.
- **Dwie główne metryki.** *Zwrot końcowy* uśrednia 3 ostatnie punkty ewaluacji
  (pojedynczy punkt na zaszumionej krzywej DQN jest niemal bez znaczenia).
  *AUC* to średni zwrot po wszystkich punktach ewaluacji — przybliżenie
  efektywności próbkowej. W badaniu eksploracji droga liczy się tak samo jak cel:
  dwa warianty mogą skończyć na równi przy bardzo różnych kosztach po drodze.
- **Diagnostyka eksploracji.** ε, τ i k *nie są* porównywalne między strategiami —
  ε równe 0.05 i τ równe 0.3 nie są „tą samą ilością" eksploracji, a to, ile
  eksploracji kupuje dane τ, dryfuje wraz ze wzrostem skali Q. Porównywalne miary
  to więc **entropia polityki zachowania** oraz **P(akcja ≠ argmax Q)**, obie
  uśredniane w każdym oknie logowania. To one pozwalają raportowi odróżnić
  „harmonogram się przesunął" od „zachowanie się zmieniło".

### 6.1 Budżet obliczeniowy i na co poszedł

Każda liczba w tym raporcie powstała na **Apple M1 Pro (10 rdzeni, 32 GB,
tylko CPU)**, `--workers 8`. Przepustowość identycznego agenta jest zbliżona we
wszystkich trzech środowiskach:

| środowisko | akcje | kroków/s na workera (8 workerów) |
|---|---|---|
| CartPole-v1 | 2 | 1607-1858 (mediana 1735) |
| Acrobot-v1 | 3 | 1423-1650 (mediana 1590) |
| LunarLander-v3 | 4 | 1391-1671 (mediana 1534) |

LunarLander **nie jest** istotnie wolniejszy niż klasyczne sterowanie — w granicach
12% CartPole na krok. Wcześniejsza wersja raportu zanotowała ~50 kroków/s
i przeskalowała wokół tej liczby całe badanie; był to artefakt osieroconych
procesów workerów po zabitej serii eksperymentów, konkurujących o CPU w momencie
pomiaru. Lekcja warta zapamiętania: pomiar przepustowości wykonany, gdy na maszynie
jest niepowiązane obciążenie, nie jest pomiarem twojego programu.

| badanie | środowiska | kroki | warianty | ziarna | przebiegi |
|---|---|---|---|---|---|
| **badanie pełne** (`full_gym`) | CartPole / Acrobot / LunarLander | 100k / 150k / 400k | 8 | **5** | 120 |
| zredukowane (`reduced_gym`) | CartPole | 60k | 7 | 3 | 21 |
| zredukowane (`acrobot_gym`) | Acrobot | 100k | 8 | 3 | 24 |
| zredukowane (`main_gym`) | LunarLander | 400k | 8 | 3 | 24 |
| ablacja skalowania Q | Acrobot | 100k | 3 | 3 | 9 |
| rozszerzenie z niepewnością | Acrobot | 100k | 9 | 3 | 27 |

Badanie pełne to 26.0 mln kroków środowiska: 4.6 h zsumowanego czasu workerów,
35 min czasu zegarowego. **§7 raportuje badanie pełne na 5 ziarnach.** Serie
3-ziarnowe są zachowane, bo ablacja i rozszerzenie z §8 działają w tym rozmiarze —
a także dlatego, że porównanie obu poziomów jest samo w sobie najużyteczniejszym
metodologicznym wynikiem tego raportu.

### 6.2 Trzy ziarna nie rozstrzygają rankingu — pokazane, nie zadeklarowane

Serie 3-ziarnowe uruchomiono dwukrotnie na różnych maszynach: najpierw na
4-rdzeniowej maszynie z Linuksem, potem ponownie tutaj. Ten sam kod, te same
ziarna, te same konfiguracje. Różni się wyłącznie build torch/BLAS, co zaburza
arytmetykę zmiennoprzecinkową na tyle, by trajektorie rozeszły się chaotycznie.

| wariant, CartPole | 4-rdzeniowy Linux | M1 Pro |
|---|---|---|
| ε-zachłanna | 344.1 [303.0, 378.6] | 275.1 [218.8, 317.0] |
| Boltzmann (bez progu ε) | 157.3 [108.2, 221.8] | 211.9 [113.9, 393.5] |

| wariant, Acrobot | 4-rdzeniowy Linux | M1 Pro |
|---|---|---|
| top-3 Boltzmann | −110.4 (**1.** z 8) | −173.7 (**7.** z 8) |
| mieszanina ε-zachłanna⊕Boltzmann | −140.3 (5. z 8) | −89.7 (**1.** z 8) |

Uporządkowanie wariantów z progiem ε nie jest stabilne wobec zmiany biblioteki
algebry liniowej, a co dopiero wobec zmiany ziarna. **Cokolwiek odczytane
z rankingu na 3 ziarnach w tej rodzinie jest szumem.** Dwie rzeczy przetrwały
zmianę maszyny bez zmian — mechanizm entropii przy skalowaniu Q (§8.1) oraz
awaria bramkowania niepewnością (§8.2) — i różnica między nimi a rankingami to
dokładnie ta linia, którą ten raport rysuje między wynikiem a estymatorem punktowym.

## 7. Wyniki: porównanie główne (5 ziaren, pełne budżety)

IQM z warstwowymi bootstrapowymi 95% przedziałami ufności, `n = 5` ziaren.
Rysunki: `report/figures/full_gym_*`.

### CartPole-v1 (100 tys. kroków, |A| = 2)

| Wariant | Zwrot końcowy | AUC | P(bije ε-zachłanną) |
|---|---|---|---|
| **ε-zachłanna (punkt odniesienia)** | 319.8 [266.0, 370.2] | 295.9 [251.6, 329.0] | — |
| mieszanina ε-zachłanna⊕Boltzmann | 309.1 [264.9, 364.8] | 276.1 [253.1, 327.7] | 0.44 [0.08, 0.84] |
| ε→Boltzmann (ścieżka τ) | 287.5 [267.3, 299.4] | 274.7 [254.8, 313.3] | 0.24 [0.00, 0.64] |
| ε→Boltzmann (ścieżka k) | 283.0 [254.4, 324.0] | 274.2 [250.9, 309.1] | 0.28 [0.00, 0.64] |
| top-2 Boltzmann | 272.7 [240.0, 330.5] | 259.6 [237.7, 299.2] | 0.24 [0.00, 0.60] |
| top-3 Boltzmann | 272.7 [240.0, 330.5] | 259.6 [237.7, 299.2] | 0.24 [0.00, 0.60] |
| top-p Boltzmann | 268.3 [231.4, 333.1] | 257.0 [225.4, 300.4] | 0.24 [0.00, 0.60] |
| Boltzmann (bez progu ε) | 209.1 [121.4, 426.3] | 134.4 [107.7, 251.3] | 0.24 [0.00, 0.60] |

*(`top-2` i `top-3` są **identyczne co do bitu na wszystkich pięciu ziarnach** —
w środowisku o 2 akcjach oba przycinają się do pełnego nośnika, więc to ta sama
polityka. To niezaplanowany test end-to-end, że przycięcie top-k jest dokładne,
a potok jest deterministyczny per ziarno.)*

### Acrobot-v1 (150 tys. kroków, |A| = 3)

| Wariant | Zwrot końcowy | AUC | P(bije ε-zachłanną) |
|---|---|---|---|
| ε→Boltzmann (ścieżka k) | **−79.6 [−81.4, −78.1]** | −256.8 [−285.3, −231.6] | **0.96 [0.76, 1.00]** |
| top-2 Boltzmann | −81.9 [−86.2, −79.0] | −281.6 [−309.1, −215.7] | 0.80 [0.44, 1.00] |
| top-p Boltzmann | −82.5 [−92.9, −80.0] | −233.7 [−281.3, −213.7] | 0.60 [0.20, 1.00] |
| mieszanina ε-zachłanna⊕Boltzmann | −82.6 [−87.8, −78.3] | −235.9 [−275.5, −165.3] | 0.64 [0.24, 1.00] |
| Boltzmann (bez progu ε) | −83.2 [−214.1, −77.8] | −240.3 [−321.8, −228.4] | 0.56 [0.16, 1.00] |
| **ε-zachłanna (punkt odniesienia)** | −84.9 [−86.8, −81.4] | −250.3 [−294.8, −199.0] | — |
| top-3 Boltzmann | −90.7 [−101.0, −78.4] | −231.3 [−258.3, −174.6] | 0.32 [0.00, 0.72] |
| ε→Boltzmann (ścieżka τ) | −98.3 [−128.2, −80.9] | −236.7 [−330.9, −184.1] | 0.32 [0.00, 0.72] |

### LunarLander-v3 (400 tys. kroków, |A| = 4)

| Wariant | Zwrot końcowy | AUC | P(bije ε-zachłanną) |
|---|---|---|---|
| **ε-zachłanna (punkt odniesienia)** | 165.4 [76.2, 217.1] | 44.8 [−11.0, 79.8] | — |
| ε→Boltzmann (ścieżka τ) | 162.6 [122.8, 207.0] | 36.1 [8.0, 77.2] | 0.48 [0.12, 0.88] |
| top-2 Boltzmann | 152.8 [138.6, 178.8] | 34.0 [5.8, 78.1] | 0.48 [0.08, 0.88] |
| Boltzmann (bez progu ε) | 148.3 [119.2, 174.7] | **114.5 [70.1, 122.2]** | 0.40 [0.04, 0.80] |
| top-3 Boltzmann | 122.8 [46.1, 168.5] | 32.8 [12.0, 56.2] | 0.28 [0.00, 0.64] |
| mieszanina ε-zachłanna⊕Boltzmann | 117.2 [7.8, 165.4] | 62.2 [24.1, 73.1] | 0.28 [0.00, 0.64] |
| top-p Boltzmann | 104.7 [51.1, 150.8] | 12.8 [−22.0, 59.8] | 0.24 [0.00, 0.64] |
| ε→Boltzmann (ścieżka k) | 100.5 [95.3, 123.4] | 26.4 [12.1, 37.0] | 0.20 [0.00, 0.60] |

### 7.1 Jedyny jednoznacznie rozstrzygnięty wynik: wartość progu ε zmienia znak

Przesłanką projektu jest to, że softmax po nienauczonym Q jest *gorszy* niż
eksploracja jednostajna. Mierzone jako efektywność próbkowa (AUC) względem
punktu odniesienia ε-zachłannego, wariant Boltzmanna bez progu daje:

| środowisko | akcje | P(AUC Boltzmanna bije AUC ε-zachłannej) |
|---|---|---|
| CartPole-v1 | 2 | **0.04 [0.00, 0.24]** — zdecydowanie gorzej |
| Acrobot-v1 | 3 | 0.48 [0.12, 0.80] — nie do odróżnienia |
| LunarLander-v3 | 4 | **0.92 [0.68, 1.00]** — zdecydowanie lepiej |

Oba skrajne przypadki wykluczają 0.5 ze swoich przedziałów. To jedyny efekt
w całym porównaniu głównym na tyle duży, by rozstrzygnąć się przy 5 ziarnach,
i jest to **odwrócenie znaku**, a nie różnica wielkości: usunięcie progu ε jest
najgorszą rzeczą, jaką można zrobić na CartPole, i najlepszą na LunarLanderze.

Ślady entropii polityki zachowania (średnia po 5 ziarnach) pokazują mechanizm:

| | krok 0 | 2k | 10k | późno |
|---|---|---|---|---|
| **CartPole**, maks. entropia ln 2 = 0.69 | | | | *(100k)* |
| ε-zachłanna | 0.693 | 0.692 | 0.647 | 0.117 |
| Boltzmann | **0.569** | 0.559 | 0.458 | 0.163 |
| **LunarLander**, maks. entropia ln 4 = 1.39 | | | | *(400k)* |
| ε-zachłanna | 1.386 | 1.386 | 1.379 | 0.201 |
| Boltzmann | 1.261 | 1.261 | 1.147 | **0.882** |

W obu środowiskach wariant Boltzmanna startuje *poniżej* maksymalnej entropii —
to obciążenie wynikające z softmaksowania nienauczonego Q, widoczne już
w pierwszej ewaluowanej polityce, dokładnie tak, jak przewiduje §1. Różni się to,
ile to obciążenie kosztuje:

- Na **CartPole** obie akcje są symetryczne, a jednostajnie losowa akcja jest
  niemal darmowa, więc wczesne obciążenie to czysta strata i wariant nigdy nie
  odrabia swojej straty w efektywności próbkowej.
- Na **LunarLanderze** jednostajnie losowa akcja jest *droga* — losowe odpalanie
  silników rozbija lądownik — więc jednostajny próg ε-zachłannej wydaje budżet
  eksploracji na akcje już rozpoznane jako katastrofalne. Boltzmann wydaje go
  zamiast tego proporcjonalnie i utrzymuje naprawdę informatywne 0.88 nata
  eksploracji przy 400 tys. kroków, gdy ε-zachłanna zapadła się do 0.20.

**Właściwe sformułowanie przesłanki projektu jest więc warunkowe**: próg ε to
ubezpieczenie od funkcji Q, która nie jest jeszcze informatywna, i jak każde
ubezpieczenie warto je kupować tylko wtedy, gdy składka — koszt jednostajnie
losowej akcji — jest niska względem ryzyka. To ostrzejsze stwierdzenie niż
„wyżarzaj od jednego do drugiego" i przewiduje, *gdzie* pomysł z przekazaniem
powinien się opłacić: w środowiskach z wieloma akcjami o mocno różnym koszcie.

### 7.2 Jedyne rozstrzygnięte porównanie wśród wariantów z progiem ε

Na Acrobocie **ścieżka nośnika** (`anneal_k`: k rośnie 1 → |A| przy stałym τ)
bije ε-zachłanną na zwrocie końcowym z `P = 0.96 [0.76, 1.00]` — przedział
wyklucza 0.5, więc ten wynik jest prawdziwy. Jej AUC *nie* jest lepsze
(`P = 0.48`): dochodzi do lepszej polityki końcowej, nie ucząc się szybciej.

To pomysł 2 z tematu działający zgodnie z zamierzeniem, a Acrobot jest miejscem,
gdzie powinno to stać się widoczne najwcześniej — |A| = 3 to najmniejszy zbiór
akcji, w którym poszerzanie nośnika ma więcej niż jeden krok do wykonania.
Warto zauważyć, że ścieżka temperatury jest jednocześnie *najgorszym* wariantem
w tym samym środowisku (−98.3, `P = 0.32`), więc rzecz dotyczy konkretnie ścieżki
nośnika, a nie wyżarzania w ogóle.

### 7.3 Co nie jest rozstrzygnięte

Cała reszta. Na CartPole i LunarLanderze przedział `P(bije ε-zachłanną)` każdego
wariantu z progiem ε zawiera 0.5, a każdy punktowy estymator zwrotu końcowego
leży wewnątrz przedziałów ufności sąsiadów. ε-zachłanna ma najwyższy estymator
punktowy w obu — i to **również** nie jest wynikiem: jej przedziały ufności
nakładają się na całą stawkę równie mocno.

Przy 5 ziarnach to badanie potrafi rozstrzygnąć odwrócenie znaku w efektywności
próbkowej i jedną wygraną ścieżki nośnika w jednym środowisku. Nie potrafi
uszeregować ośmiu harmonogramów eksploracji, a §6.2 pokazuje, co się dzieje,
gdy próbuje się tego przy trzech.

## 8. Wyniki: ablacja skalowania Q i rozszerzenie z niepewnością

### 8.1 Ablacja skalowania Q (Acrobot-v1, harmonogram `eps_boltzmann`, 100 tys. kroków, 3 ziarna)

| tryb `q_scaling` | Zwrot końcowy | P(bije `running`) |
|---|---|---|
| `per_state` | −151.4 [−237.9, −75.5] | 0.56 [0.00, 1.00] |
| `running` (domyślny) | −154.5 [−207.1, −86.1] | — (punkt odniesienia) |
| `none` | −307.7 [−500.0, −106.9] | 0.22 [0.00, 0.67] |

`per_state` i `running` są nie do odróżnienia (estymatory punktowe oddalone o 3
jednostki zwrotu, `P = 0.56` z przedziałem obejmującym [0, 1]). Oddziela się tylko
`none`, a i to nie jest rozstrzygnięte przy 3 ziarnach (`P = 0.22 [0.00, 0.67]`).

Kolumna *wydajności* jest więc słaba. Kolumna **entropii** nie jest — i to dla
niej ta ablacja znalazła się w raporcie. Uruchomienie identycznych konfiguracji
na dwóch różnych maszynach (§6.2) odtwarza ślad entropii niemal dokładnie,
jednocześnie mieszając ranking wydajności:

| tryb | entropia @100k (4-rdzeniowy Linux → M1 Pro) | P(niezachłanna) @100k |
|---|---|---|
| `none` | 0.93 → **0.90** | 0.46 → **0.45** |
| `running` | 0.38 → **0.34** | 0.15 → **0.13** |
| `per_state` | 0.13 → **0.14** | 0.05 → **0.04** |

To dokładnie to, co przewiduje §3, i jest niezależne od maszyny:

- **`none` nigdy się nie ustala.** Entropia utrzymuje się blisko 0.9 nata przez
  cały przebieg: gdy Q pompuje się od inicjalizacji bliskiej zeru, stałe τ wciąż
  znaczy coś innego, więc polityka nigdy nie zbiega do stabilnego poziomu
  eksploracji. Jedyna liczba, która ma sterować eksploracją Boltzmanna, nie jest
  w rzeczywistości pod kontrolą eksperymentatora.
- **`per_state` jest ~2.5× pewniej zaostrzony niż `running`** (0.14 wobec 0.34
  nata; 4% wobec 13% akcji niezachłannych), bo wymuszenie jednostkowego rozrzutu
  na każdym wierszu Q fabrykuje pewną preferencję w wierszach, w których akcje są
  naprawdę równoważne.

**Uczciwy podział: mechanizm jest potwierdzony, jego konsekwencja wydajnościowa
nie.** Na wejściu oczekiwanie z §3 było takie, że fałszywa pewność `per_state`
powinna *szkodzić*. Na Acrobocie nie szkodzi — nagroda jest gęsta (−1 za każdy
krok do celu), a wczesna eksploatacja nie jest oczywiście zła, więc ten sam
mechanizm, który fabrykuje fałszywą pewność, ogranicza też marnowaną eksplorację.
To, czy kosztuje wydajność, powinno zależeć od tego, ile stanów ma naprawdę
płaski wiersz Q — prawdopodobnie więcej na LunarLanderze, gdzie kilka akcji może
być podobnie rozsądnych w locie, niż w bardziej rozstrzygniętej dynamice Acrobota.
Ten zbiór danych tego nie testuje; `running` pozostaje domyślny, bo to tryb,
którego zachowanie odpowiada deklarowanemu zamiarowi, a nie dlatego, że wygrał
w sposób mierzalny.

### 8.2 Rozszerzenie bramkowane niepewnością (Acrobot-v1, 100 tys. kroków, 3 ziarna)

Rozszerzenie przeszło **trzy iteracje**. Pierwsza zawiodła, druga sfalsyfikowała
oczywiste wyjaśnienie dlaczego, a trzecia zidentyfikowała i naprawiła rzeczywistą
usterkę. Wszystkie dziewięć wariantów jest w jednej serii, więc dzielą ziarna
i architekturę.

| Wariant | Zwrot końcowy | P(bije ε-zachłanną, zespół) |
|---|---|---|
| ε→Boltzmann, architektura zespołowa | −84.1 [−92.1, −77.2] | 0.56 [0.00, 1.00] |
| **ε-zachłanna, zespół (kontrola)** | −93.2 [−117.7, −80.3] | — |
| ε-zachłanna, jedna głowica | −105.4 [−138.3, −81.1] | 0.33 [0.00, 0.89] |
| bramkowana, zespół, **zamrożona ref.** (v3) | −296.1 [−500.0, −93.4] | 0.11 [0.00, 0.44] |
| bramkowana, błąd TD, skalowana (v2) | −312.4 [−461.2, −97.9] | 0.11 [0.00, 0.44] |
| bramkowana, błąd TD, **zamrożona ref.** (v3) | −357.1 [−452.9, −241.0] | 0.00 [0.00, 0.00] |
| bramkowana, zespół, skalowana (v2) | −375.1 [−500.0, −226.8] | 0.00 [0.00, 0.00] |
| bramkowana, zespół (v1) | −385.8 [−500.0, −290.1] | 0.00 [0.00, 0.00] |
| bramkowana, błąd TD (v1) | −398.5 [−500.0, −335.9] | 0.00 [0.00, 0.00] |

**Każdy wariant bramkowany przegrywa z kontrolą.** Rozszerzenie nie działa przy
tym budżecie i uczciwy nagłówek brzmi: zawiodło. To, co następuje, to wyjaśnienie
dlaczego — bo awaria okazała się diagnozowalna i częściowo naprawialna, a diagnoza
jest ciekawsza niż ranking.

Kontrola z dopasowaną architekturą (§4) spełnia swoje zadanie przez cały czas:
ε-zachłanna z zespołem (−93.2) bije ε-zachłanną jednogłowicową (−105.4), więc
dodatkowe głowice są co najwyżej *pomocne*, a załamanie należy przypisać
bramkowaniu, nie architekturze. To także powtórzyło się na obu maszynach (§6.2) —
warianty v1 uzyskały −330/−347 na maszynie z Linuksem i −386/−399 tutaj,
`P = 0.00` w obu przypadkach.

#### v1: bramka nigdy się nie otwiera

Pewność jest przyszpilona do 0 w czasie rozgrzewki (10 tys. kroków), potem
skacze i płaszczy się:

| krok | 5k | 10k | 12k | 20k | 50k | 100k |
|---|---|---|---|---|---|---|
| pewność (zespół) | 0.00 | 0.00 | **0.33** | 0.34 | 0.31 | 0.26 |
| ε (zespół) | 1.00 | 1.00 | 0.67 | 0.67 | 0.69 | 0.74 |
| pewność (błąd TD) | 0.00 | 0.00 | **0.83** | 0.28 | 0.04 | 0.04 |
| ε (błąd TD) | 1.00 | 1.00 | 0.18 | 0.73 | **0.96** | 0.96 |

Oba warianty tkwią między `ε ≈ 0.67` a `ε ≈ 0.96` przez praktycznie cały przebieg,
podczas gdy każdy wariant o stałym harmonogramie osiąga `ε ≤ 0.05` do 30 tys.
kroków. Przy `eps_uncertain = 1.0, eps_confident = 0.01` pewność 0.3 interpoluje
się do `ε = 0.70`, dokładnie jak zaobserwowano: bramka robi dokładnie to, co jej
kazano, a to, co jej kazano, jest błędne. **Przekazanie, dla którego to rozszerzenie w ogóle
istnieje, nigdy nie zachodzi**, co samo w sobie wystarcza za wyjaśnienie
załamania.

#### v2: oczywiste wyjaśnienie, przetestowane i sfalsyfikowane

Oba sygnały są mierzone w **surowych jednostkach Q** — niezgoda zespołu to
odchylenie standardowe wartości Q, a błąd TD to ich różnica — a rzędy wielkości Q
pompują się w trakcie uczenia. Mierzone przez 100 tys. kroków *surowe* sygnały nie
maleją, lecz rosną: niezgoda zespołu 0.076 → 0.136, błąd TD 0.27 → 1.45. Wyglądało
to na tryb awarii z §3 pojawiający się tam, gdzie §3 nigdy nie zastosowano, więc
`normalise_uncertainty` dzieli sygnał przez tę samą bieżącą skalę Q, której
polityka używa dla swojej temperatury (`policy/uncertainty_gated_scaled`).

Uczyniło to sygnał bezwymiarowym i **nic nie zmieniło**:

| `u / referencja` | 11k | 40k | 100k |
|---|---|---|---|
| zespół, v1 (jednostki surowe) | 0.699 | 0.671 | 0.741 |
| zespół, v2 (normalizacja skalą) | 0.691 | 0.712 | 0.831 |
| błąd TD, v1 | 0.142 | 0.954 | 0.964 |
| błąd TD, v2 | 0.327 | 0.958 | 0.963 |

Zwroty końcowe przesunęły się z −385.8 na −375.1 (zespół) i z −398.5 na −312.4
(TD), oba mocno w granicach szumu. **Jednostki były prawdziwą usterką, ale nie tą
wiążącą.**

#### v3: rzeczywista usterka jest strukturalna

Iloraz się nie rusza z powodu samej definicji pewności:

```
pewność = 1 − u / kwantyl(własnej niedawnej historii u)
```

Referencja jest szacowana **z tego samego strumienia, który normalizuje**.
Cokolwiek robi `u` — rośnie, maleje czy stoi — bieżący kwantyl podąża za nim,
`u/ref` zbiega do stałej, a pewność zostaje na niej przyszpilona. To nie jest błąd
kalibracji, który naprawiłby lepszy współczynnik uczenia; **żaden wybór jednostek,
kwantyla ani długości kroku nie sprawi, że samonormalizujący się sygnał wykryje
własny trend.** Zaprojektowano go jako bezskalowy, a przy okazji wyszedł
bez-trendowy.

Wersja v3 zamraża referencję na końcu rozgrzewki (`reference_freeze_step`),
czyniąc z niej stałą kalibracyjną zmierzoną we wczesnym uczeniu. Pewność rośnie
wtedy i tylko wtedy, gdy niepewność spada poniżej swojego wczesnego poziomu
— czyli dokładnie to, co bramka miała zawsze znaczyć. Ślad wariantu zespołowego:

| `gated_ensemble_frozen` | 11k | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|---|
| niepewność (zamrożona ref. = 5.25) | 3.24 | 2.81 | 1.84 | 1.95 | 1.66 | **1.53** |
| pewność | 0.00 | 0.46 | 0.64 | 0.62 | 0.68 | **0.71** |
| ε | 1.00 | 0.55 | 0.37 | 0.39 | 0.33 | **0.30** |
| k | 1.0 | 1.9 | 2.3 | 2.2 | 2.4 | **2.3** |

**Przekazanie działa.** Niepewność faktycznie spada, pewność rośnie monotonicznie,
ε opada, a nośnik Boltzmanna się poszerza — mechanizm opisany w §4, działający,
po raz pierwszy w trzech iteracjach. Zwrot końcowy poprawia się z −385.8 do
−296.1, a najciekawszy jest rozrzut per ziarno: ziarno 1 kończy na **−93.4**,
na równi z kontrolą, podczas gdy ziarno 0 kończy na −500. Mechanizm potrafi
zadziałać; nie jest jeszcze niezawodny.

Dwie rzeczy wciąż nie pozwalają mu wygrać i obie są teraz konkretne, a nie
tajemnicze:

1. **Jest o wiele za wolny.** ε osiąga zaledwie 0.30 do 100 tys. kroków, podczas
   gdy stałe harmonogramy osiągają 0.05 do 30 tys. Wariant spędza cały budżet
   bardziej eksploracyjnie niż każdy konkurent, więc wciąż mierzy słabiej
   wytrenowaną sieć. To odwzorowanie pewność→ε, a nie sygnał, wymaga teraz
   przestrojenia.
2. **Istnieje pułapka sprzężenia zwrotnego.** Wysokie ε daje gorszą funkcję
   wartości, która daje wyższą niezgodę, która utrzymuje wysokie ε. Bramka
   sterowana pomiarem zamyka pętlę, której harmonogram czasowy nie zamyka,
   a projekt nic z tym nie robi.

#### Sygnał błędu TD jest nieodpowiedni, i teraz da się to wykazać

Przy zamrożonej referencji niepewność wariantu z błędem TD **rośnie** — 8.9 →
24.9 — a pewność zapada się do 0, utrzymując `ε ≈ 0.95`. To także nie jest problem
kalibracji: `|błąd TD|` odzwierciedla wielkość nagrody, szum bootstrapu
i nieaktualność sieci docelowej, a nic z tego nie maleje tylko dlatego, że agent
się nauczył. To sygnał *o charakterze aleatorycznym* proszony o wykonanie pracy
epistemicznej. Niezgoda zespołu, która jest naprawdę epistemiczna, spada w tym
samym oknie dwukrotnie. **To porównanie było celem włączenia obu sygnałów** (§4),
a zamrożenie referencji jest tym, co wreszcie uczyniło je widocznym: dopiero gdy
referencja przestaje gonić sygnał, widać, który sygnał ma trend wart gonienia.

#### Status

Rozszerzenie jest **zaimplementowane, zdiagnozowane w trzech iteracjach i wciąż
przegrywa z kontrolą**. Usterka, która powodowała przegraną, jest zrozumiana
i naprawiona; pozostaje źle dostrojone odwzorowanie pewność→ε oraz pułapka
sprzężenia zwrotnego — oba to zwykłe problemy strojenia, a nie wady projektu.
Wobec ustalenia z §6.2, że 3 ziarna nie rozstrzygają rankingu w tej rodzinie,
następnym krokiem nie jest dalsze strojenie pod te liczby, lecz przebieg na
5 ziarnach wariantu zespołowego z zamrożoną referencją przeciwko kontroli,
z szybszym odwzorowaniem pewność→ε. `normalise_uncertainty`
i `reference_freeze_step` są domyślnie **wyłączone**, więc powyższe wyniki v1
pozostają dokładnie odtwarzalne.

## 9. Co daje sam projekt, niezależnie od liczb

Trzy rzeczy w tym projekcie warto zachować niezależnie od tego, jak uszeregowały
się warianty, bo to one *umożliwiają* porównanie:

**Zunifikowana rodzina czyni krańce dokładnymi.** Ponieważ `k = 1` oraz `τ → 0`
redukują się do ε-zachłannej *z dokładnością maszynową*, a `ε = 0` z pełnym
nośnikiem jest dokładnie softmaksem, harmonogram wyżarzania naprawdę zaczyna się
w jednej klasycznej strategii i kończy w drugiej. Ręcznie sklecona mieszanka
zaczynałaby i kończyła *blisko* nich, a każde twierdzenie o tym, „jak przekazanie
wpływa na uczenie", byłoby skażone tą różnicą. `tests/test_policies.py` przypina
to jako dokładną równość tablic względem niezależnie napisanych implementacji
referencyjnych.

**Normalizator skali Q czyni temperaturę przenośną.** Bez niego „τ = 0.3" znaczy
co innego w każdym środowisku i w każdym momencie uczenia, więc jedyna liczba
definiująca eksplorację Boltzmanna nie jest w rzeczywistości pod kontrolą
eksperymentatora. §8.1 pokazuje, co się dzieje bez niego: ślad entropii, który
nigdy się nie ustala, kontra taki, który się ustala.

**Jawne rozkłady akcji czynią eksplorację mierzalną.** Liczenie `π(a|s)` zamiast
proceduralnego próbkowania kosztuje mikrosekundy i daje za darmo entropię polityki
zachowania oraz P(akcja ≠ argmax) — jedyne dwie wielkości porównywalne między
strategiami, których parametry porównywalne nie są. Bez nich raport mógłby
powiedzieć, że harmonogram się przesunął, ale nie że zachowanie się zmieniło.

## 10. Ograniczenia

Podane wprost, bo wyznaczają granice tego, co powyższe wyniki uzasadniają.

- **Pięć ziaren wystarcza na odwrócenie znaku i niewiele więcej.** Wynik dotyczący
  progu ε z §7.1 i wygrana ścieżki nośnika z §7.2 to jedyne dwa porównania,
  których przedziały prawdopodobieństwa poprawy wykluczają 0.5. Każdy inny ranking
  w §7 to estymator punktowy, a §6.2 empirycznie pokazuje, co się dzieje, gdy
  zaufa się takiemu rankingowi przy 3 ziarnach: uporządkowanie nie przeżywa zmiany
  biblioteki BLAS. Literatura o ewaluacji w głębokim RL wymaga od dziesięciu do
  trzydziestu ziaren; to badanie ma pięć.
- **Badania z §8 wciąż są na 3 ziarnach.** Ablacji skalowania Q i rozszerzenia
  z niepewnością nie uruchomiono ponownie na 5. Ich kolumny *wydajności* są
  odpowiednio słabe — §8.1 mówi to wprost. Ich nośnymi twierdzeniami są mechanizm
  entropii i załamanie bramkowania, oba powtórzone na dwóch maszynach (§6.2), co
  jest innym i mocniejszym rodzajem dowodu niż ranking z 3 ziaren.
- **Brak wyników na Atari.** Ścieżka kodu ALE jest zaimplementowana, przetestowana
  jednostkowo względem prawdziwych środowisk ALE i zweryfikowana pod kątem
  wykonania pełnego przebiegu uczenia od początku do końca — ale nietrenowana.
  Przy ~10 mln klatek na przebieg 8 wariantów × 5 ziaren to rząd wielkości
  miesiąca GPU. Ma to bezpośrednie znaczenie dla wniosku z §7.1: zidentyfikowany
  tam mechanizm — próg ε jest wart swojej składki tylko wtedy, gdy jednostajnie
  losowa akcja jest tania — przewiduje, że efekt powinien *rosnąć* wraz ze zbiorem
  akcji, a Atari z liczbą akcji do 18 jest miejscem, gdzie tę przepowiednię
  należałoby sprawdzić. Nie sprawdzono jej.
- **Małe zbiory akcji ograniczają pomysł 2.** Na CartPole (|A| = 2) `top-2`
  i `top-3` Boltzmann są dowodliwie tą samą polityką, a tabela z §7 potwierdza to
  co do ostatniego miejsca po przecinku na wszystkich 5 ziarnach. `anneal_k` ma
  jeden krok do wykonania. Acrobot (3) i LunarLander (4) są lepsze, ale wciąż
  wąskie — a jedyna rozstrzygnięta wygrana ścieżki nośnika z §7.2 jest na
  Acrobocie, czyli w środowisku, w którym |A| po raz pierwszy czyni ten wariant
  nietrywialnym. To sugestia, nie rozstrzygnięcie.
- **Trzy środowiska, wszystkie o gęstej nagrodzie i niskiej wymiarowości.**
  Mechanizm z §7.1 dotyczy *kosztu losowej akcji*, który te trzy środowiska
  akurat użytecznie rozpinają (darmowy na CartPole, zabójczy na LunarLanderze).
  Nie rozpinają natomiast w ogóle rzadkiej nagrody, długich horyzontów ani
  obserwacji pikselowych.
- **Rozszerzenie zostaje w trakcie naprawy.** §8.2 naprawia usterkę, która
  blokowała działanie bramki niepewności, ale wariant z zamrożoną referencją wciąż
  jest wolny (ε osiąga tylko 0.30 do 100 tys. kroków) i wciąż przegrywa ze swoją
  kontrolą. Pozostałe problemy są nazwane — źle dostrojone odwzorowanie pewność→ε
  oraz pułapka sprzężenia wartość/niepewność — ale nierozwiązane, a strojenie pod
  liczby z 3 ziaren, które §6.2 pokazuje jako niewiarygodne, byłoby złym ruchem.
- **Jedno ustawienie hiperparametrów na wariant.** Krańce i długości harmonogramu
  każdego wariantu wybrano raz, na podstawie rozumowania, a nie przeszukiwania.
  Wariant, który wygląda gorzej, może być po prostu źle dostrojony. §8.1 pokazuje,
  jak wrażliwa jest ta rodzina na jeden taki wybór (`q_scaling`), a awaria z §8.2
  jest *sama w sobie* błędem kalibracji w jednym konkretnym schemacie normalizacji,
  a nie obaleniem eksploracji bramkowanej niepewnością.
- **Ewaluacja jest wyłącznie zachłanna.** Każda liczba to zwrot polityki
  zachłannej. To właściwy wybór do porównywania jakości uczenia między wariantami
  (§6), ale oznacza, że raport nie mówi nic o zwrocie online, który agent
  faktycznie zebrałby w trakcie eksploracji — a dla systemu wdrożonego to często
  właśnie ta wielkość ma znaczenie.

## 11. Odtworzenie wyników

```bash
make setup                     # swig PRZED gymnasium[box2d] -- zob. README
make test                      # 194 testy

make experiment-full           # BADANIE z §7: 3 środowiska x 8 wariantów x 5 ziaren
make experiment-ablation       # §8.1  Acrobot, skalowanie Q
make experiment-uncertainty    # §8.2  Acrobot, rozszerzenie bramkowane niepewnością

make plots SWEEP=full_gym      # regeneracja wszystkich rysunków i tabel dla serii
```

`make experiment-full WORKERS=8` odtwarza §7 od początku do końca: 120 przebiegów,
26.0 mln kroków środowiska, **35 minut czasu zegarowego na M1 Pro** (4.6 h
zsumowanego czasu workerów). Trzy zredukowane serie 3-ziarnowe
(`experiment-reduced`, `experiment-acrobot`, `experiment-main`) są zachowane na
potrzeby porównania międzymaszynowego z §6.2 i razem wykonują się w niecałe
20 minut.

Każdy rysunek w tym raporcie jest regenerowany z wyników przebiegów przez
`scripts/make_plots.py`; nic tutaj nie zostało narysowane ręcznie. Serie są
wznawialne — komórka, której `result.json` już istnieje, jest pomijana — więc
przerwany przebieg wznawia się tym samym poleceniem.

**O dokładnej odtwarzalności:** pojedyncze przebiegi są deterministyczne per
ziarno *na ustalonej maszynie i ustalonym stosie bibliotek* — tabela CartPole
z §7 pokazuje `top-2` i `top-3` zgodne co do ostatniego miejsca po przecinku na
wszystkich 5 ziarnach. **Nie** są natomiast odtwarzalne między buildami
torch/BLAS; §6.2 kwantyfikuje, jak bardzo przesuwają się zagregowane rankingi.
To normalne w głębokim RL i właśnie dlatego ten raport podaje przedziały
i prawdopodobieństwa, a nie estymatory punktowe.
