import os
import time
from ultralytics import YOLO
from google import genai
from google.genai import types
from markdown_pdf import MarkdownPdf, Section
import datetime

# ==========================================
# 1. INICJALIZACJA I KONFIGURACJA
# ==========================================
API_KEY = "KLUCZ_API"

client = genai.Client(api_key=API_KEY)

# KLUCZOWA POPRAWKA: Zmiana modelu na ten, który masz w AI Studio
MODEL_NAME = "gemini-3.1-flash-lite"
FOLDER_Z_FOTKAMI = "symulacja_druku"

print("🔄 Ładowanie modelu YOLO...")
model_yolo = YOLO('yolov8n.pt')

SYSTEM_PROMPT = """Jesteś elitarnym ekspertem druku 3D, automatyzacji oraz analizy wizyjnej. Twoim zadaniem jest przeprowadzenie ustrukturyzowanego wywiadu, doradztwo w zakresie parametrów, ustalenie kryteriów bezpieczeństwa, analiza materiałów wizualnych, a na koniec przeprowadzenie ankiety ewaluacyjnej i nauka na błędach.

PRZEBIEG PROCESU (PROSZĘ ŚCIŚLE TRZYMAĆ SIĘ TYCH FAZ):

### FAZA 1: WYWIAD I REKOMENDACJA PARAMETRÓW
1. Gdy użytkownik przywita się ("Cześć"), odpowiedz entuzjastycznie i zadaj pierwsze pytanie: "Cześć! Co chcesz dzisiaj wydrukować?". Użytkownik opisze ci co chce wydrukować
2. W kolejnych krokach zbieraj informacje, zadając TYLKO JEDNO PYTANIE NA WIADOMOŚĆ. Zapytaj o:
   - Oczekiwania wobec wydruku, jego zastosowanie oraz specyfika kształtu (np. czy ma być wytrzymały mechanicznie, ozdobny, elastyczny? ORAZ czy model posiada specyficzną geometrię – np. naturalne zagięcia, zaokrąglenia lub sferyczne kształty jak Gwiazda Śmierci – które system wizyjny mógłby pomylić z warpingiem?).
3. **REKOMENDACJA:** Po zebraniu tych danych, zanim przejdziesz dalej, przedstaw użytkownikowi analizę. Ocen, czy wybrany filament pasuje do zastosowania. Podaj rekomendowane parametry: sugerowaną temperaturę dyszy i stołu roboczego oraz instrukcję techniczną dotyczącą obudowy (np. "otwórz obudowę, aby zapobiec heat creep dla PLA" lub "zamknij komorę, aby uniknąć warping ABS").
4. Po przedstawieniu rekomendacji przejdź do Fazy 2.

### FAZA 2: DEFINIOWANIE BŁĘDÓW KRYTYCZNYCH
1. Wyświetl użytkownikowi listę najczęstszych błędów w druku 3D w formie punktowej z numerkami:
   1. Warping (podwijanie od stołu)
   2. Spaghetti / Layer Detachment (druk w powietrzu)
   3. Layer Shift (przesunięcie warstw w osi X/Y)
   4. Clogging / Under-extrusion (zatykanie dyszy, braki w materiale)
   5. Stringing / Oozing (nitkowanie)
   6. Over-extrusion (nadmiar materiału, bąble)
2. Poproś użytkownika: "Napisz mi numery błędów, które uważasz za KRYTYCZNE (czyli takie, przy których chcesz natychmiast zatrzymać druk). Pozostałe uznam za AKCEPTOWALNE (wydruk leci dalej, ale dostaniesz ostrzeżenie)."
3. Poczekaj na odpowiedź użytkownika i zapamiętaj jego wybór. Następnie poproś o przesłanie zdjęcia lub filmu z wydruku.

### FAZA 3: ANALIZA WIZUALNA I REAKCJA (SKRYPT)
Gdy użytkownik prześle zdjęcie lub film:
1. Zanim dokonasz diagnozy, przeprowadź rygorystyczny proces "Sceptycyzmu Wizualnego", aby wykluczyć fałszywe alarmy (False Positives):
   - KROK A (Analiza kształtu): Porównaj zaobserwowaną anomalię z opisem geometrii modelu z Fazy 1. Jeśli model ma naturalnie zaokrąglone brzegi, sferyczną strukturę lub geometryczne przetłoczenia, NIE klasyfikuj ich jako warping czy deformation.
   - KROK B (Analiza otoczenia): Upewnij się, że rzekomy błąd nie jest jedynie refleksem światła na błyszczącym filamencie, cieniem rzucanym przez głowicę, czy elementem interfejsu (np. migającym kursem na ekranie).
   - Zasada "In dubio pro reo": Dopiero po wizualnym i 100% pewnym wykryciu błędu zacznij korzystać z bazy wiedzy. Jeśli masz jakiekolwiek wątpliwości, czy to błąd, czy celowa geometria – uznaj, że to poprawny element modelu, nie zatrzymuj procesu i przejdź do punktu 3 (Nie wykryto błędu).
2. Jeśli wykryjesz błąd, który użytkownik oznaczył jako KRYTYCZNY (i masz 100% pewności, że to nie jest naturalna geometria obiektu):
   - Opisz szczegółowo anomalię i czas/miejsce jej wystąpienia.
   - WYŚLIJ w bloku kodu uniwersalną komendę ratunkową G-code (M112 - Emergency Stop) oraz krótki skrypt w Pythonie (używający webhooka / API dla OctoPrint lub Klippera/Moonrakera), który użytkownik może wywołać, aby natychmiast przerwać zasilanie/wydruk.
3. Jeśli NIE wykryjesz błędu LUB analiza wykaże, że kształt wynika z naturalnej geometrii modelu:
   - Napisz dokładnie: "Nie wykryto błędu (kształt zgodny z przewidywaną geometrią modelu)".
   - Dodaj komunikat: "Kontynuuję monitorowanie. Czekam na kolejne zdjęcie".
3. Jeśli wykryjesz błąd, który jest AKCEPTOWALNY:
   - Opisz go, ostrzeż użytkownika, ale napisz, że zgodnie z jego instrukcjami wydruk może być kontynuowany. Podaj krótką poradę, jak naprawić to przy kolejnym drukowaniu.

### FAZA 4: ANKIETA PO WYDRUKU I REKOMENDACJE BIEŻĄCE
Uruchom tę fazę, gdy użytkownik zasygnalizuje, że proces się skończył (np. napisze "wydruk zakończono", "druk gotowy", "koniec").
1. Przeprowadź krótką ankietę ewaluacyjną. Zapytaj użytkownika:
   - Czy wydruk ostatecznie przebiegł pomyślnie i czy jesteś zadowolony z jakości?
   - Czy w trakcie (poza moją analizą) wystąpiły jakieś inne błędy? Jeśli tak, to jakie, w którym momencie i co Twoim zdaniem mogło być ich przyczyną?
2. Po otrzymaniu odpowiedzi wyświetl sekcję **"Rekomendacje eksploatacyjne"**:
   - Jak zadbać o drukarkę po tym konkretnym druku (np. czyszczenie stołu izopropanolem (IPA), sprawdzenie napięcia pasków, usunięcie resztek filamentu z dyszy).
   - Jak prawidłowo przechowywać użyty właśnie filament (np. schowanie do szczelnego pojemnika/worka strunowego z żelem krzemionkowym/pochłaniaczem wilgoci, aby nie chłonął wody).
3. **NAUKA MODELU:** Wpisy z tej ankiety traktuj jako cenne źródło wiedzy o specyfice drukarki użytkownika. Wykorzystaj te wnioski przy kolejnych sesjach z tym użytkownikiem, aby lepiej przewidywać anomalie i personalizować porady.

### BAZA WIEDZY (KNOWLEDGE BASE)
Podczas diagnozy i analizy błędów sugeruj się wiedzą, wytycznymi i instrukcjami zawartych w załączonych przeze mnie plikach Word. Traktuj je jako poradę dotyczącą identyfikacji wad w druku 3D
"""

config = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    temperature=0.4
)

chat = client.chats.create(model=MODEL_NAME, config=config)

# ==========================================
# START PROCESU
# ==========================================
print("\n🚀 [SYSTEM] Uruchomiono inteligentnego asystenta druku 3D.\n")

print("📋 --- ROZPOCZYNAMY FAZĘ 1 & 2: ANKIETA WSTĘPNA ---")
print(
    "(Napisz 'Cześć' aby zacząć. Przejdź przez wywiad. Gdy zdefiniujesz błędy krytyczne, napisz 'START', aby odpalić monitoring wizyjny).\n")

while True:
    user_input = input("👤 Ty: ")
    if user_input.upper() == "START":
        print("\n🔒 Zamykam ankietę wstępną. Przechodzę do automatycznego monitorowania wydruku...")
        break

    try:
        response = chat.send_message(user_input)
        print(f"\n🤖 Agent AI:\n{response.text}\n")
    except Exception as e:
        print(f"\n❌ Błąd komunikacji z chmurą: {e}\n")

# ==========================================
# FAZA 3: AUTOMATYCZNY MONITORING WIZYJNY
# ==========================================
print("\n📹 --- ROZPOCZYNAMY FAZĘ 3: AUTOMATYCZNY MONITORING WIZYJNY ---")

lista_zdjec = sorted([f for f in os.listdir(FOLDER_Z_FOTKAMI) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

druk_przerwany = False

for nazwa_pliku in lista_zdjec:
    sciezka_do_foto = os.path.join(FOLDER_Z_FOTKAMI, nazwa_pliku)
    print(f"\n📸 [MONITORING] Pobieram klatkę: {nazwa_pliku}")

    results = model_yolo(sciezka_do_foto, verbose=False)
    podejrzenie_bledu = False

    for result in results:
        if len(result.boxes) > 0 and result.boxes.conf[0] > 0.20:
            podejrzenie_bledu = True
            print(f"⚠️ [EDGE AI] YOLO widzi anomalię (pewność: {result.boxes.conf[0]:.2f}). Wysyłam do Gemini...")

    if podejrzenie_bledu:
        with open(sciezka_do_foto, "rb") as f:
            img_bytes = f.read()

        try:
            mime_type = "image/png" if nazwa_pliku.lower().endswith('.png') else "image/jpeg"

            response = chat.send_message([
                types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                "Oto aktualne zdjęcie z wydruku. Przeanalizuj je zgodnie z FAZĄ 3. Pamiętaj o wytycznych krytycznych."
            ])
            print(f"\n🤖 Agent AI (Analiza obrazu):\n{response.text}\n")

            if "M112" in response.text:
                print("🚨 [PROCES] Wykryto błąd krytyczny! G-code M112 aktywowany.")
                print("🛑 Drukarka została awaryjnie zatrzymana przez AI.")
                print("--------------------------------------------------")
                druk_przerwany = True
                break
        except Exception as e:
            print(f"\n❌ Wystąpił błąd podczas analizy obrazu: {e}")
    else:
        print("✅ [EDGE AI] Wszystko w porządku, brak anomalii kształtu. Drukujemy dalej.")

    time.sleep(3)

# ==========================================
# FAZA 4: ANKIETA PO WYDRUKU
# ==========================================
print("\n📝 --- ROZPOCZYNAMY FAZĘ 4: ANKIETA PO WYDRUKU ---")
if druk_przerwany:
    msg = "Wydruk zakończono przedwcześnie z powodu awarii. Uruchom Fazę 4 (krótka ankieta)."
else:
    msg = "Wydruk zakończono pomyślnie. Uruchom Fazę 4 (krótka ankieta)."

try:
    response = chat.send_message(msg)
    print(f"\n🤖 Agent AI:\n{response.text}\n")
except Exception as e:
    print(f"❌ Błąd wywołania ankiety końcowej: {e}")

# Pętla rozmowy (użytkownik odpowiada na ankietę)
while True:
    user_input = input("👤 Ty (Napisz 'raport', aby zakończyć i wygenerować dokument): ")

    if user_input.lower() == "raport":
        break

    try:
        response = chat.send_message(user_input)
        print(f"\n🤖 Agent AI:\n{response.text}\n")
    except Exception as e:
        print(f"❌ Wystąpił błąd komunikacji: {e}")

# ==========================================
# FAZA 5: SEKWENCYJNE GENEROWANIE RAPORTU PDF
# ==========================================
print("\n⚙️ [SYSTEM] Rozpoczynam generowanie GIGANTYCZNEGO raportu PDF z twardym podziałem na fazy.")
print("⏳ To potrwa około 45-80 sekund, ponieważ Agent pisze bardzo obszerne analizy rozdział po rozdziale...\n")

pelny_raport_md = "# PEŁNA DOKUMENTACJA TECHNICZNA PROCESU DRUKU 3D\n\n"

# SŁOWNIK STRUKTURY: Zwiększony wolumen tekstu + prawdziwe linki inżynieryjne
rozdziały = [
    {
        "naglowek": "# CZĘŚĆ I: FAZA DIAGNOSTYCZNA\n\n## 1. Karta Kwalifikacji i Podsumowanie Menedżerskie\n",
        "prompt": "Wygeneruj treść do Rozdziału 1. Nie pisz własnego tytułu. Napisz to bardzo formalnym, akademickim językiem. Bardzo szeroko i szczegółowo opisz architekturę hybrydowego systemu monitoringu (Edge AI + Cloud AI), cel wydruku oraz dokonaj głębokiej analizy dobranych parametrów z Fazy 1. Użyj minimum dwóch tabel do zestawienia wszystkich danych wejściowych, parametrów temperaturowych oraz konfiguracji sprzętowej."
    },
    {
        "naglowek": "## 2. Analiza Fizyko-Chemiczna Materiału i Termodynamika Procesu\n",
        "prompt": "Wygeneruj treść do Rozdziału 2. Nie pisz tytułu. Napisz niezwykle wyczerpujący, wielostronicowy elaborat akademicki o właściwościach reologicznych i termicznych wybranego filamentu. Szczegółowo omów zjawiska takie jak: przejście szkliste, skurcz przetwórczy, adhezja międzywarstwowa (interlayer bonding), naprężenia szczątkowe (residual stress) oraz wpływ zamkniętej lub otwartej komory na krystalizację polimeru. Tekst musi być długi, bogaty w zaawansowaną terminologię techniczną i podzielony na podsekcje."
    },
    {
        "naglowek": "## 3. Dziennik Inspekcji Wizyjnej i Przebieg Fazy 3\n",
        "prompt": "Wygeneruj treść do Rozdziału 3. Nie pisz tytułu. Stwórz rozbudowaną, wielokolumnową tabelę logów systemowych dokumentującą analizę klatek sekwencyjnych (symulowaną minuta po minucie dla całego procesu). Dla każdej klatki rozpisz szeroki, algorytmiczny komentarz: jak zadziałał lokalny filtr YOLO i jak przebiegła zaawansowana weryfikacja logiczna chmury w ramach protokołu Sceptycyzmu Wizualnego. Opisz dokładnie eskalację wad wizualnych (lub decyzję o ich zignorowaniu ze względu na specyfikę bryły) aż do momentu aktywacji stopu awaryjnego."
    },
    {
        "naglowek": "# CZĘŚĆ II: FAZA REKOMENDACYJNA\n\n## 4. Ewaluacja Użytkownika i Procedury Serwisowe\n",
        "prompt": "Wygeneruj treść do Rozdziału 4, który stanowi początek Fazy Rekomendacyjnej. Nie pisz tytułu. Dokonaj szczegółowej, krytycznej analizy odpowiedzi z ankiety użytkownika (Faza 4) i zestaw je z logami systemowymi. Następnie stwórz gigantyczny, podręcznikowy poradnik serwisowy krok po kroku. Rozpisz na osobne, bardzo długie akapity procedury: czyszczenia i odtłuszczania platformy roboczej za pomocą IPA, inspekcji i naciągu pasków, prewencyjnego czyszczenia dyszy metodą cold pull, smarowania prowadnic liniowych oraz optymalnego przechowywania polimerów w kapsułach higroskopijnych."
    },
    {
        "naglowek": "## 5. Profesjonalne Źródła Wiedzy i Baza Literatury\n",
        "prompt": """Wygeneruj treść do Rozdziału 5. Nie pisz żadnego wstępu, podsumowania ani własnych opisów. 
        Twoim jedynym zadaniem jest wypluć czystą, gęstą listę punktowaną zawierającą dokładnie poniższe linki sformatowane jako klikalny Markdown [Nazwa](URL). 
        Oto lista, którą musisz bezwzględnie i w całości przepisać:

        * [Prusa Research Quality Troubleshooting Guide](https://help.prusa3d.com/category/print-quality-troubleshooting_225)
        * [Simplify3D Print Quality Troubleshooting](https://www.simplify3d.com/support/print-quality-troubleshooting/)
        * [Bambu Lab Official Wiki](https://wiki.bambulab.com/)
        * [All3DP Ultimate 3D Printing Troubleshooting Guide](https://all3dp.com/1/common-3d-printing-problems-troubleshooting-3d-printer-issues/)
        * [Ellis' Print Tuning Guide - Advanced Calibration](https://ellis3dp.com/Print-Tuning-Guide/)
        * [Teaching Tech 3D Printer Calibration Internet Tool](https://teachingtechyt.github.io/calibration.html)
        * [RepRap Print Quality Troubleshooting Pictorial Guide](https://reprap.org/wiki/Print_Quality_Troubleshooting_Pictorial_Guide)
        * [UltiMaker Hardware and Print Quality Support](https://support.ultimaker.com/hc/en-us/categories/360002283080-Troubleshooting)
        * [MatterHackers 3D Printer Troubleshooting Guide](https://www.matterhackers.com/articles/3d-printer-troubleshooting-guide)
        * [Creality Official Wiki and Troubleshooting Base](https://wiki.creality.com/)
        * [Klipper 3D Printer Firmware Documentation](https://www.klipper3d.org/)
        * [Voron Design Official Documentation](https://docs.vorondesign.com/)
        * [OctoPrint Official Documentation and API Reference](https://docs.octoprint.org/)
        * [Marlin Firmware G-code Reference (M112 Emergency Stop)](https://marlinfw.org/docs/gcode/M112.html)
        * [CNC Kitchen - Material Testing and Engineering Data](https://www.cnckitchen.com/)"""
    }
]

# Pętla generująca rozdziały
for i, sekcja in enumerate(rozdziały, 1):
    print(f"✍️ Generowanie Rozdziału {i}/{len(rozdziały)}...")

    # 1. Doklejamy twardy nagłówek bezpośrednio z kodu Pythona (Gwarancja podziału!)
    pelny_raport_md += sekcja["naglowek"]

    # 2. Wysyłamy prośbę do Agenta o wygenerowanie samego "mięsa" i doklejamy je pod nagłówkiem
    try:
        odpowiedz = chat.send_message(sekcja["prompt"])
        pelny_raport_md += odpowiedz.text + "\n\n---\n\n"
    except Exception as e:
        print(f"❌ Błąd przy generowaniu rozdziału {i}: {e}")

# Zapisywanie do PDF
print("\n💾 Składanie tekstu do pliku PDF...")
try:
    from markdown_pdf import MarkdownPdf, Section
    import datetime

    pdf = MarkdownPdf(toc_level=2)  # Spis treści
    pdf.add_section(Section(pelny_raport_md))

    nazwa_pliku = f"Raport_Techniczny_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    pdf.save(nazwa_pliku)
    print(f"✅ SUKCES! Pomyślnie wygenerowano raport z podziałem na fazy: {nazwa_pliku}")
except Exception as e:
    print(f"❌ Błąd zapisu PDF: {e}")
    with open("awaryjny_raport.md", "w", encoding="utf-8") as f:
        f.write(pelny_raport_md)

print("\n🏁 Koniec procesu.")