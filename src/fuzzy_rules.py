import numpy as np
import utils
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# =========================
# FUZZIFICATION HELPERS
# =========================
def best_membership(value, antecedent):
    best_label = None
    best_score = -1

    for label in antecedent.terms:
        score = fuzz.interp_membership(
            antecedent.universe,
            antecedent[label].mf,
            value,
        )
        if score > best_score:
            best_score = score
            best_label = label

    return best_label, best_score


# =========================
# RULE GENERATION
# =========================
def generate_rules(X, y, antecedents):
    rules = []

    antecedent_items = list(antecedents.items())

    for i, row in X.iterrows():
        premise = []
        weights = []

        for col, antecedent in antecedent_items:
            label, score = best_membership(row[col], antecedent)
            premise.append(label)
            weights.append(score)

        premise = tuple(premise)
        weight = np.prod(weights)

        target = float(y.iloc[i])

        if premise not in rules or weight > rules[premise][1]:
            rules.append((premise, target, weight))

    return rules


# =========================
# BUILD FUZZY SYSTEM
# =========================
def build_fuzzy_system(rules, antecedents, consequent):
    rule_list = []
    antecedent_items = list(antecedents.items())

    for premise, target, weight in rules:
        conditions = []
        for (col, antecedent), label in zip(antecedent_items, premise):
            conditions.append(antecedent[label])

        # map regression target → fuzzy output
        if target <= 4:
            out = consequent["low"]
        elif target <= 8:
            out = consequent["medium"]
        else:
            out = consequent["high"]

        rule = ctrl.Rule(conditions[0] & conditions[1] & conditions[2], out)

        rule_list.append(rule)

    return ctrl.ControlSystem(rule_list)


# =========================
# MAIN
# =========================
def main():
    DATA_PATH = "data/raw/Mental Health Classification.csv"
    TARGET_COL = "Depression_Type"

    X, y = utils.load_data(DATA_PATH, TARGET_COL)

    print(X.head())
    print("DATA SHAPE:", X.shape)

    # =========================
    # INPUT FUZZY VARIABLES
    # =========================
    social_media = ctrl.Antecedent(np.linspace(0, 3, 100), "social_media")
    overeating = ctrl.Antecedent(np.linspace(0, 12, 100), "overeating")
    employment = ctrl.Antecedent(np.linspace(0, 4, 100), "employment")

    antecedents = {
        "SocialMedia_WhileEating": social_media,
        "Your overeating level": overeating,
        "Employment_Status": employment,
    }

    # =========================
    # OUTPUT
    # =========================
    depression = ctrl.Consequent(np.linspace(0, 12, 200), "depression")

    depression["low"] = fuzz.trimf(depression.universe, [0, 0, 4])
    depression["medium"] = fuzz.trimf(depression.universe, [3, 6, 9])
    depression["high"] = fuzz.trimf(depression.universe, [8, 12, 12])

    # =========================
    # INPUT MEMBERSHIP FUNCTIONS
    # =========================
    social_media["never"] = fuzz.trimf(social_media.universe, [0, 0, 1])
    social_media["often"] = fuzz.trimf(social_media.universe, [1, 2, 3])

    overeating["low"] = fuzz.trimf(overeating.universe, [0, 3, 6])
    overeating["high"] = fuzz.trimf(overeating.universe, [6, 9, 12])

    employment["student"] = fuzz.trimf(employment.universe, [0, 1, 2])
    employment["employed"] = fuzz.trimf(employment.universe, [1, 3, 4])

    # =========================
    # RULES
    # =========================
    rules = generate_rules(X, y, antecedents)

    print("Generated rule patterns:", len(rules))

    system = build_fuzzy_system(rules, antecedents, depression)

    sim = ctrl.ControlSystemSimulation(system)

    # =========================
    # TEST
    # =========================
    for i, row in X.iterrows():
        sim.reset()

        sim.input["social_media"] = row["SocialMedia_WhileEating"]
        sim.input["overeating"] = row["Your overeating level"]
        sim.input["employment"] = row["Employment_Status"]

        sim.compute()

        pred = sim.output.get("depression", None)

        print("pred:", pred, "true:", y.iloc[i])

        if i > 10:
            break


if __name__ == "__main__":
    main()
