import skfuzzy as fuzz
import numpy as np
from skfuzzy import control as ctrl

# --- Inicjalizacja Zmiennych ---
social_media_while_eating = ctrl.Antecedent(
    np.linspace(0, 3, 500), "social_media_while_eating"
)
overeating_level = ctrl.Antecedent(np.linspace(0, 12, 500), "overeating_level")
employment_status = ctrl.Antecedent(np.linspace(0, 4, 500), "employment_status")

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

employment_status["unemployed"] = fuzz.trimf(employment_status.universe, [0, 0, 0])
employment_status["student"] = fuzz.trimf(employment_status.universe, [1, 1, 1])
employment_status["employed"] = fuzz.trimf(employment_status.universe, [2, 2, 2])
employment_status["self-employed"] = fuzz.trimf(employment_status.universe, [3, 3, 3])
employment_status["other"] = fuzz.trimf(employment_status.universe, [4, 4, 4])

depression_type["no_depression"] = fuzz.trimf(depression_type.universe, [0, 0, 2])
depression_type["moderate"] = fuzz.trimf(depression_type.universe, [2, 3, 4])
depression_type["severe"] = fuzz.trimf(depression_type.universe, [4, 5, 7])
depression_type["seasonal"] = fuzz.trimf(depression_type.universe, [6, 7, 8])
depression_type["bipolar"] = fuzz.trimf(depression_type.universe, [8, 9, 10])
depression_type["psychotic_depression"] = fuzz.trimf(
    depression_type.universe, [10, 11, 11]
)

rules = [
    ctrl.Rule(
        overeating_level["none"]
        & (social_media_while_eating["never"] | social_media_while_eating["rarely"]),
        depression_type["no_depression"],
    ),
    ctrl.Rule(
        employment_status["unemployed"] & overeating_level["moderate"],
        depression_type["severe"],
    ),
    ctrl.Rule(
        employment_status["unemployed"] & overeating_level["mild"],
        depression_type["moderate"],
    ),
    ctrl.Rule(
        employment_status["unemployed"] & overeating_level["severe"],
        depression_type["psychotic_depression"],
    ),
    ctrl.Rule(
        employment_status["student"] & overeating_level["mild"],
        depression_type["moderate"],
    ),
    ctrl.Rule(
        employment_status["student"] & overeating_level["moderate"],
        depression_type["seasonal"],
    ),
    ctrl.Rule(
        employment_status["student"] & overeating_level["severe"],
        depression_type["severe"],
    ),
    ctrl.Rule(
        (employment_status["employed"] | employment_status["self-employed"])
        & overeating_level["severe"],
        depression_type["bipolar"],
    ),
    ctrl.Rule(
        employment_status["employed"]
        & overeating_level["moderate"]
        & social_media_while_eating["often"],
        depression_type["severe"],
    ),
    ctrl.Rule(
        employment_status["self-employed"] & overeating_level["moderate"],
        depression_type["moderate"],
    ),
    ctrl.Rule(
        employment_status["other"] & overeating_level["severe"],
        depression_type["bipolar"],
    ),
    ctrl.Rule(
        employment_status["other"] & overeating_level["moderate"],
        depression_type["moderate"],
    ),
    ctrl.Rule(
        overeating_level["severe"] & social_media_while_eating["always"],
        depression_type["psychotic_depression"],
    ),
    ctrl.Rule(
        overeating_level["moderate"] & social_media_while_eating["often"],
        depression_type["severe"],
    ),
    ctrl.Rule(
        overeating_level["mild"] & social_media_while_eating["often"],
        depression_type["moderate"],
    ),
]

depression_ctrl = ctrl.ControlSystem(rules)
depression_sim = ctrl.ControlSystemSimulation(depression_ctrl)

