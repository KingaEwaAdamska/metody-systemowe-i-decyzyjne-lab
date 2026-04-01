# Analiza Danych o Zdrowiu Psychicznym i Wypaleniu Studentów

## Przegląd

Celem tego projektu jest porównanie dwóch rodzajów systemów decyzyjnych:

1. Systemów opartych na słabej sztucznej inteligencji (AI), znanych również jako systemy oparte na regułach, działających na bazie logiki rozmytej (fuzzy logic).
2. Systemów inteligentnych, wykorzystujących techniki uczenia maszynowego (ML) lub głębokiego (DL).

Projekt będzie dotyczył analizy danych z zestawu danych o zdrowiu psychicznym studentów oraz wypaleniu zawodowym, co czyni go odpowiednim do porównania tych dwóch metodologii.

## Opis Zadania

### I. Ogólny Opis Zadania

Zadanie polega na porównaniu dwóch systemów decyzyjnych:

* **Systemy oparte na logice rozmytej (fuzzy logic)** – systemy oparte na regułach.
* **Systemy oparte na uczeniu maszynowym (ML/DL)** – systemy wykorzystujące techniki uczenia maszynowego lub głębokiego.

### II. Szczegółowy Opis

Do porównania tych systemów użyjemy zestawu danych o zdrowiu psychicznym studentów oraz wypaleniu zawodowym. Zestaw ten zawiera informacje o różnych atrybutach, które wpływają na zdrowie psychiczne studentów, co pozwoli na przewidywanie ryzyka wypalenia zawodowego oraz innych problemów związanych ze zdrowiem psychicznym. Celem jest porównanie skuteczności obu systemów pod kątem dokładności, czasu reakcji oraz innych metryk.

#### Metodologia:

* **Systemy oparte na logice rozmytej**: Implementacja logiki rozmytej z użyciem biblioteki `scikit-fuzzy` lub podobnej. Kluczowe aspekty do analizy:

  * Liczba atrybutów branych pod uwagę przy formułowaniu reguł.
  * Liczba funkcji przynależności.
  * Rodzaj funkcji przynależności (np. `trimf`, `gaussmf`).

* **Systemy oparte na uczeniu maszynowym**: Implementacja modeli ML z użyciem biblioteki `scikit-learn` do zadań klasyfikacji:

  * Regresja logistyczna
  * Drzewo decyzyjne
  * Las losowy (Random Forest)

Dla każdego modelu należy przeanalizować, jak różne czynniki, takie jak liczba atrybutów lub typ funkcji przynależności, wpływają na dokładność.

#### Metryki do porównania:

* Dokładność przewidywań.
* Czas odpowiedzi na zapytanie.
* Porównanie wydajności modeli za pomocą standardowych metryk, takich jak precyzja, recall, F1-score.

### III. Kryteria Zaliczenia

Aby zdać przedmiot, wymagane są następujące rezultaty:

* Demonstracja działającego oprogramowania implementującego zarówno systemy oparte na logice rozmytej, jak i uczeniu maszynowym.
* Sporządzenie skondensowanego raportu zawierającego:

  * Opis problemu i szczegóły dotyczące zestawu danych.
  * Metodologię używaną w obu systemach (fuzzy logic i ML/DL).
  * Szczegóły eksperymentalne oraz metryki oceny.
  * Wyniki i porównanie obu systemów.
  * Dyskusję wyników i wnioski.

## Szczegóły Zestawu Danych

Zestaw danych używany w tym projekcie, [Student Mental Health and Burnout Dataset](https://www.kaggle.com/datasets/sehaj1104/student-mental-health-and-burnout-dataset/data), zawiera informacje dotyczące zdrowia psychicznego studentów, wypalenia zawodowego oraz czynników wpływających na te problemy, takich jak:

* Informacje demograficzne
* Wyniki akademickie
* Czynniki społeczne
* Samopoczucie emocjonalne
* Problemy zdrowia psychicznego i wskaźniki wypalenia zawodowego

### Struktura Zestawu Danych:

* Zestaw danych zawiera kolumny takie jak wiek, płeć, godziny nauki, czas spędzany na mediach społecznościowych oraz odpowiedzi na oceny psychologiczne.
* Zmienna docelowa wskazuje, czy student jest zagrożony wypaleniem zawodowym.

## Ewaluacja

Wydajność obu systemów będzie porównywana na podstawie:

* **Dokładności**: Jak dobrze każdy model przewiduje ryzyko wypalenia zawodowego.
* **Czasu reakcji**: Czas, jaki każdy model potrzebuje, aby wygenerować wynik.
* **Inne metryki**: Precyzja, recall oraz F1-score.

## Podsumowanie

Celem projektu jest dostarczenie kompleksowego porównania między systemami opartymi na regułach (logika rozmyta) i systemami wykorzystującymi uczenie maszynowe, co pozwoli lepiej zrozumieć ich mocne strony i słabości w kontekście rzeczywistych aplikacji decyzyjnych. Wyniki oraz wnioski z porównania będą pomocne w doborze odpowiednich metodologii do różnych zastosowań decyzyjnych.
