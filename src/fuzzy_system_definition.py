import skfuzzy as fuzz
import numpy as np
from skfuzzy import control as ctrl

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

# --- REGUŁY OPARTE NA UPROSZCZONYCH WZORCACH (top 2 zmienne) ---
rules = [
    # ============================================================
    # REGUŁY Social + Overeating (165, 160, 155, 116, 92, 82, 70, 65, 62, 61)
    # ============================================================
    # IF Social=always AND Overeating=severe THEN result_8_9 (165 przypadków)
    ctrl.Rule(
        social_media_while_eating["always"] & overeating_level["severe"],
        depression_type["result_8_9"],
    ),
    # IF Social=often AND Overeating=severe THEN result_8_9 (160 przypadków)
    ctrl.Rule(
        social_media_while_eating["often"] & overeating_level["severe"],
        depression_type["result_8_9"],
    ),
    # IF Social=often AND Overeating=severe THEN result_1_2 (155 przypadków)
    ctrl.Rule(
        social_media_while_eating["often"] & overeating_level["severe"],
        depression_type["result_1_2"],
    ),
    # IF Social=often AND Overeating=severe THEN result_5 (116 przypadków)
    ctrl.Rule(
        social_media_while_eating["often"] & overeating_level["severe"],
        depression_type["result_5"],
    ),
    # IF Social=always AND Overeating=severe THEN result_6_7 (92 przypadków)
    ctrl.Rule(
        social_media_while_eating["always"] & overeating_level["severe"],
        depression_type["result_6_7"],
    ),
    # IF Social=always AND Overeating=severe THEN result_1_2 (82 przypadków)
    ctrl.Rule(
        social_media_while_eating["always"] & overeating_level["severe"],
        depression_type["result_1_2"],
    ),
    # IF Social=never AND Overeating=severe THEN result_5 (70 przypadków)
    ctrl.Rule(
        social_media_while_eating["never"] & overeating_level["severe"],
        depression_type["result_5"],
    ),
    # IF Social=always AND Overeating=moderate THEN result_1_2 (65 przypadków)
    ctrl.Rule(
        social_media_while_eating["always"] & overeating_level["moderate"],
        depression_type["result_1_2"],
    ),
    # IF Social=often AND Overeating=severe THEN result_6_7 (62 przypadków)
    ctrl.Rule(
        social_media_while_eating["often"] & overeating_level["severe"],
        depression_type["result_6_7"],
    ),
    # IF Social=often AND Overeating=moderate THEN result_8_9 (61 przypadków)
    ctrl.Rule(
        social_media_while_eating["often"] & overeating_level["moderate"],
        depression_type["result_8_9"],
    ),
    # ============================================================
    # REGUŁY Overeating + Energy (249, 248, 152, 148, 105, 92, 90, 88, 84, 79)
    # ============================================================
    # IF Overeating=severe AND Energy=yes THEN result_1_2 (249 przypadków)
    ctrl.Rule(
        overeating_level["severe"] & low_energy["yes"],
        depression_type["result_1_2"],
    ),
    # IF Overeating=severe AND Energy=yes THEN result_8_9 (248 przypadków)
    ctrl.Rule(
        overeating_level["severe"] & low_energy["yes"],
        depression_type["result_8_9"],
    ),
    # IF Overeating=severe AND Energy=no THEN result_8_9 (152 przypadków)
    ctrl.Rule(
        overeating_level["severe"] & low_energy["no"],
        depression_type["result_8_9"],
    ),
    # IF Overeating=severe AND Energy=no THEN result_5 (148 przypadków)
    ctrl.Rule(
        overeating_level["severe"] & low_energy["no"],
        depression_type["result_5"],
    ),
    # IF Overeating=none AND Energy=yes THEN result_8_9 (105 przypadków)
    ctrl.Rule(
        overeating_level["none"] & low_energy["yes"],
        depression_type["result_8_9"],
    ),
    # IF Overeating=severe AND Energy=yes THEN result_5 (92 przypadków)
    ctrl.Rule(
        overeating_level["severe"] & low_energy["yes"],
        depression_type["result_5"],
    ),
    # IF Overeating=severe AND Energy=yes THEN result_3_4 (90 przypadków)
    ctrl.Rule(
        overeating_level["severe"] & low_energy["yes"],
        depression_type["result_3_4"],
    ),
    # IF Overeating=mild AND Energy=yes THEN result_5 (88 przypadków)
    ctrl.Rule(
        overeating_level["mild"] & low_energy["yes"],
        depression_type["result_5"],
    ),
    # IF Overeating=moderate AND Energy=yes THEN result_8_9 (84 przypadków)
    ctrl.Rule(
        overeating_level["moderate"] & low_energy["yes"],
        depression_type["result_8_9"],
    ),
    # IF Overeating=severe AND Energy=no THEN result_1_2 (79 przypadków)
    ctrl.Rule(
        overeating_level["severe"] & low_energy["no"],
        depression_type["result_1_2"],
    ),
    # ============================================================
    # REGUŁY Social + Energy (177, 172, 158, 138, 119, 106, 104, 86, 82, 62)
    # ============================================================
    # IF Social=always AND Energy=yes THEN result_1_2 (177 przypadków)
    ctrl.Rule(
        social_media_while_eating["always"] & low_energy["yes"],
        depression_type["result_1_2"],
    ),
    # IF Social=often AND Energy=yes THEN result_8_9 (172 przypadków)
    ctrl.Rule(
        social_media_while_eating["often"] & low_energy["yes"],
        depression_type["result_8_9"],
    ),
    # IF Social=always AND Energy=yes THEN result_8_9 (158 przypadków)
    ctrl.Rule(
        social_media_while_eating["always"] & low_energy["yes"],
        depression_type["result_8_9"],
    ),
    # IF Social=often AND Energy=yes THEN result_1_2 (138 przypadków)
    ctrl.Rule(
        social_media_while_eating["often"] & low_energy["yes"],
        depression_type["result_1_2"],
    ),
    # IF Social=always AND Energy=yes THEN result_5 (119 przypadków)
    ctrl.Rule(
        social_media_while_eating["always"] & low_energy["yes"],
        depression_type["result_5"],
    ),
    # IF Social=always AND Energy=no THEN result_6_7 (106 przypadków)
    ctrl.Rule(
        social_media_while_eating["always"] & low_energy["no"],
        depression_type["result_6_7"],
    ),
    # IF Social=always AND Energy=no THEN result_8_9 (104 przypadków)
    ctrl.Rule(
        social_media_while_eating["always"] & low_energy["no"],
        depression_type["result_8_9"],
    ),
    # IF Social=often AND Energy=no THEN result_5 (86 przypadków)
    ctrl.Rule(
        social_media_while_eating["often"] & low_energy["no"],
        depression_type["result_5"],
    ),
    # IF Social=often AND Energy=no THEN result_8_9 (82 przypadków)
    ctrl.Rule(
        social_media_while_eating["often"] & low_energy["no"],
        depression_type["result_8_9"],
    ),
    # IF Social=rarely AND Energy=yes THEN result_8_9 (62 przypadków)
    ctrl.Rule(
        social_media_while_eating["rarely"] & low_energy["yes"],
        depression_type["result_8_9"],
    ),
]

depression_ctrl = ctrl.ControlSystem(rules)
depression_sim = ctrl.ControlSystemSimulation(depression_ctrl)

