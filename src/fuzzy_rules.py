import skfuzzy as fuzz
import numpy as np
from skfuzzy import control as ctrl
from collections import Counter

# --- Inicjalizacja Zmiennych ---
social_media_while_eating = ctrl.Antecedent(
    np.linspace(0, 3, 500), "social_media_while_eating"
)
overeating_level = ctrl.Antecedent(np.linspace(0, 12, 500), "overeating_level")
employment_status = ctrl.Antecedent(np.linspace(0, 4, 500), "employment_status")
low_energy = ctrl.Antecedent(np.linspace(0, 2, 500), "low_energy")

depression_type = ctrl.Consequent(np.linspace(0, 11, 1000), "depression_type")

# --- Zbiory Rozmyte ---
social_media_while_eating["never"] = fuzz.trimf(
    social_media_while_eating.universe, [0, 0, 1]
)
social_media_while_eating["rarely"] = fuzz.trimf(
    social_media_while_eating.universe, [0, 1, 2]
)
social_media_while_eating["often"] = fuzz.trimf(
    social_media_while_eating.universe, [1, 2, 3]
)
social_media_while_eating["always"] = fuzz.trimf(
    social_media_while_eating.universe, [2, 3, 3]
)

overeating_level["none"] = fuzz.trimf(overeating_level.universe, [0, 0, 1])
overeating_level["mild"] = fuzz.trimf(overeating_level.universe, [1, 2.5, 5])
overeating_level["moderate"] = fuzz.trimf(overeating_level.universe, [5, 6.5, 8])
overeating_level["severe"] = fuzz.trimf(overeating_level.universe, [8, 10.5, 12])

employment_status["unemployed"] = fuzz.trimf(employment_status.universe, [0, 0, 0.8])
employment_status["student"] = fuzz.trimf(employment_status.universe, [0.5, 1, 1.5])
employment_status["employed"] = fuzz.trimf(employment_status.universe, [1.5, 2, 2.5])
employment_status["self-employed"] = fuzz.trimf(
    employment_status.universe, [2.5, 3, 3.5]
)
employment_status["other"] = fuzz.trimf(employment_status.universe, [3.2, 4, 4])

low_energy["no"] = fuzz.trimf(low_energy.universe, [0, 0, 0.5])
low_energy["yes"] = fuzz.trimf(low_energy.universe, [0.5, 1, 1.5])
low_energy["sometimes"] = fuzz.trimf(low_energy.universe, [1.5, 2, 2])

# Etykiety wyjściowe
depression_type["result_0"] = fuzz.trimf(depression_type.universe, [0, 0, 0.5])
depression_type["result_1_2"] = fuzz.trimf(depression_type.universe, [1, 1.5, 2])
depression_type["result_3_4"] = fuzz.trimf(depression_type.universe, [3, 3.5, 4])
depression_type["result_5"] = fuzz.trimf(depression_type.universe, [4.5, 5, 5.5])
depression_type["result_6_7"] = fuzz.trimf(depression_type.universe, [6, 6.5, 7])
depression_type["result_8_9"] = fuzz.trimf(depression_type.universe, [8, 8.5, 9])
depression_type["result_10_11"] = fuzz.trimf(depression_type.universe, [10, 10.5, 11])

# ============================================================
# INICJALIZACJA LISTY REGUŁ
# ============================================================
rules = []

# ============================================================
# REGUŁY Social + Overeating (bez sprzeczności)
# ============================================================
social_overeating_rules = {
    ("always", "severe"): "result_8_9",  # 165 przypadków
    ("often", "severe"): "result_1_2",  # 155 przypadków
    ("never", "severe"): "result_5",  # 70 przypadków
    ("always", "moderate"): "result_1_2",  # 65 przypadków
    ("often", "moderate"): "result_8_9",  # 61 przypadków
}

for (social, overeating), output in social_overeating_rules.items():
    rules.append(
        ctrl.Rule(
            social_media_while_eating[social] & overeating_level[overeating],
            depression_type[output],
        )
    )

# ============================================================
# REGUŁY Overeating + Energy (bez sprzeczności)
# ============================================================
overeating_energy_rules = {
    ("severe", "yes"): "result_1_2",  # 249 przypadków
    ("severe", "no"): "result_8_9",  # 152 przypadków
    ("none", "yes"): "result_8_9",  # 105 przypadków
    ("mild", "yes"): "result_5",  # 88 przypadków
    ("moderate", "yes"): "result_8_9",  # 84 przypadków
}

for (overeating, energy), output in overeating_energy_rules.items():
    rules.append(
        ctrl.Rule(
            overeating_level[overeating] & low_energy[energy], depression_type[output]
        )
    )

# ============================================================
# REGUŁY Social + Energy (bez sprzeczności)
# ============================================================
social_energy_rules = {
    ("always", "yes"): "result_1_2",  # 177 przypadków
    ("often", "yes"): "result_8_9",  # 172 przypadków
    ("always", "no"): "result_6_7",  # 106 przypadków
    ("often", "no"): "result_5",  # 86 przypadków
    ("rarely", "yes"): "result_8_9",  # 62 przypadków
}

for (social, energy), output in social_energy_rules.items():
    rules.append(
        ctrl.Rule(
            social_media_while_eating[social] & low_energy[energy],
            depression_type[output],
        )
    )

# ============================================================
# REGUŁY Z PEŁNYMI WARUNKAMI (4 zmienne) - najważniejsze
# ============================================================
full_rules_resolved = {
    ("often", "severe", "employed", "yes"): "result_8_9",  # 91
    ("often", "severe", "employed", "no"): "result_5",  # 86
    ("always", "moderate", "employed", "yes"): "result_1_2",  # 65
    ("always", "severe", "employed", "yes"): "result_6_7",  # 57 (kompromis)
    ("rarely", "severe", "other", "yes"): "result_1_2",  # 59
    ("often", "none", "employed", "yes"): "result_1_2",  # 42
    ("always", "moderate", "employed", "no"): "result_6_7",  # 39
}

for (social, overeating, employment, energy), output in full_rules_resolved.items():
    rules.append(
        ctrl.Rule(
            social_media_while_eating[social]
            & overeating_level[overeating]
            & employment_status[employment]
            & low_energy[energy],
            depression_type[output],
        )
    )

# ============================================================
# REGUŁY DOMYŚLNE DLA NIE POKRYTYCH PRZYPADKÓW
# ============================================================
# Na podstawie globalnego rozkładu True_value:
# result_8_9: 648 (32.4%) - najczęstsze
# result_1_2: 482 (24.1%)
# result_5: 386 (19.3%)
# result_6_7: 225 (11.3%)

# Reguły domyślne dla ekstremalnych wartości
rules.append(
    ctrl.Rule(
        overeating_level["severe"] & social_media_while_eating["always"],
        depression_type["result_8_9"],
    )
)

rules.append(
    ctrl.Rule(
        overeating_level["severe"] & low_energy["yes"], depression_type["result_8_9"]
    )
)

rules.append(
    ctrl.Rule(
        social_media_while_eating["always"] & low_energy["yes"],
        depression_type["result_8_9"],
    )
)

# Reguła dla braku objawów
rules.append(
    ctrl.Rule(
        overeating_level["none"]
        & social_media_while_eating["never"]
        & low_energy["no"],
        depression_type["result_0"],
    )
)

# ============================================================
# UTWORZENIE SYSTEMU
# ============================================================
depression_ctrl = ctrl.ControlSystem(rules)
depression_sim = ctrl.ControlSystemSimulation(depression_ctrl)

print(f"System utworzony pomyślnie z {len(rules)} regułami")


# Funkcja testowa
def test_system(X, y):
    """Test systemu na danych"""
    results = []
    matches = 0

    for idx, row in X.iterrows():
        depression_sim.reset()
        depression_sim.input["social_media_while_eating"] = row[
            "SocialMedia_WhileEating"
        ]
        depression_sim.input["overeating_level"] = row["Your overeating level"]
        depression_sim.input["employment_status"] = row["Employment_Status"]
        depression_sim.input["low_energy"] = row["Low_Energy"]

        try:
            depression_sim.compute()
            result = depression_sim.output["depression_type"]
        except Exception as e:
            print(f"Błąd dla indeksu {idx}: {e}")
            result = 5.0

        pred = round(result)
        results.append(pred)

        if pred == y[idx]:
            matches += 1

    accuracy = matches / len(y) * 100
    print(f"\nDokładność: {accuracy:.2f}%")
    print(f"Poprawne: {matches}/{len(y)}")

    return results


# Automatyczny test jeśli dane są dostępne
if __name__ == "__main__":
    try:
        import utils

        DATA_PATH = "data/raw/Mental Health Classification.csv"
        TARGET_COL = "Depression_Type"

        X, y = utils.load_data(DATA_PATH, TARGET_COL)
        test_system(X, y)
    except Exception as e:
        print(f"Nie można załadować danych testowych: {e}")
