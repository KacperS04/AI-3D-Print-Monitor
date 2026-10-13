# Hybrydowy agent AI do monitoringu druku 3D

Prototyp asystenta, który prowadzi użytkownika przez wydruk 3D i analizuje klatki z kamery pod kątem błędów. Projekt powstał na przedmiot *Sztuczna inteligencja w biznesie* (Politechnika Warszawska, 2026). Główne pytanie projektu: **czy da się na tym zarobić?**

## Problem biznesowy
Analiza każdej klatki przez multimodalny model w chmurze jest dokładna, ale przy zdjęciu co minutę koszt API rośnie liniowo z czasem druku. Model kosztów w Excelu pokazał, że w wersji „każda klatka do chmury” marża przy cenie akceptowalnej dla hobbysty jest zbyt mała.

## Rozwiązanie: architektura dwustopniowa
```
klatka z kamery ──> YOLOv8 (lokalnie, za darmo) ──podejrzana?──> Gemini API (płatne) ──> decyzja
                                   │                                   │
                                  nie ──> drukujemy dalej       błąd krytyczny ──> M112 (stop)
```
- **Etap 1 – YOLOv8 (Ultralytics):** lokalny, darmowy filtr wstępny. W prototypie użyty jest model pretrenowany (`yolov8n.pt`), a próg pewności można ustawić osobno dla każdego pakietu.
- **Etap 2 – Gemini API:** analiza tylko klatek przepuszczonych przez filtr, z regułami ograniczającymi fałszywe alarmy (porównanie z geometrią modelu, odbicia światła, cienie).

## Przebieg sesji (prompt systemowy)
1. Wywiad o wydruku i rekomendacja parametrów (temperatury, obudowa).
2. Użytkownik wybiera, które błędy są krytyczne (warping, spaghetti, layer shift, zatkanie dyszy, stringing, nadmiar materiału).
3. Automatyczny monitoring klatek.
4. Ankieta po wydruku i rekomendacje eksploatacyjne.
5. Automatycznie generowany wielorozdziałowy raport techniczny PDF.

## Model kosztów (plik `Interaktywny_Kalkulator_Pakietow.xlsx`)
Pakiet bazowy: 1 drukarka, 50 h monitoringu i 15 wydruków miesięcznie, 1 klatka na minutę. Kurs przyjęty w modelu: 4 zł/USD.

| Wariant | Klatki do Gemini / mies. | Koszt API / drukarkę / mies. |
|---|---|---|
| Każda klatka do chmury | 3 000 | ok. 11,50 zł |
| Z filtrem YOLO (25% klatek) | 750 | ok. 3,10 zł |

Filtr obniża koszt API o ok. 73%. Kalkulator pozwala zmieniać liczbę drukarek, limity, częstotliwość zdjęć, odsetek klatek przepuszczanych przez filtr i marżę; arkusz „Marketing” szacuje koszt pozyskania klientów.

## Ograniczenia i dalszy rozwój
- Pretrenowany YOLOv8 nie rozpoznaje wad druku jako osobnych klas. Kolejny krok to douczenie (fine-tuning) na oznaczonym zbiorze błędów druku 3D i pomiar skuteczności (precision / recall).
- Testy przeprowadzono na sekwencjach klatek z nagrań timelapse, a nie na żywej kamerze.

## Uruchomienie
```bash
pip install ultralytics google-genai markdown-pdf
export GEMINI_API_KEY="twój_klucz"   # klucza nie umieszczaj w kodzie
python agent.py
```
Klatki testowe umieść w folderze `symulacja_druku/`.

## Technologie
Python · Ultralytics YOLOv8 · Google Gen AI SDK (Gemini) · markdown-pdf · MS Excel
