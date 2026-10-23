"""CardiaNova - Medical Diet & Clinical Remedies Generator

Generates evidence-based, disease-tailored nutrition protocols (DASH, Mediterranean,
Low-Acid, Electrolyte Balance) and safe home remedies based on the patient's extracted
clinical profile.
"""
from typing import Any, Dict


def generate_diet_and_remedies(entities: Dict[str, Any]) -> Dict[str, Any]:
    """Assembles a tailored clinical diet plan and daily home remedy checklist."""
    category = entities.get("primary_symptom_category", "UNKNOWN")
    has_htn = entities.get("has_hypertension", False) or (entities.get("systolic_bp") or 120) >= 130
    has_diabetes = entities.get("has_diabetes", False)
    pain_char = entities.get("pain_character", [])
    is_smoker = entities.get("is_smoker", False)

    # 1. Determine Appropriate Clinical Diet Protocol
    if "BURNING_ACID_REFLUX" in pain_char or category == "GASTROESOPHAGEAL_REFLUX_GERD":
        protocol_name = "LOW-ACID CARDIOPROTECTIVE GI DIET"
        goal = "Soothe esophageal and gastric mucosal lining, prevent nocturnal reflux, and eliminate acid-induced chest spasms."
        foods_to_eat = [
            {"food": "Oatmeal & Porridge", "reason": "Binds stomach acid without stimulating bile secretion."},
            {"food": "Ripe Bananas & Melons", "reason": "Naturally low-acid, alkaline fruits that coat the esophagus."},
            {"food": "Steamed Zucchini & Green Beans", "reason": "Easily digestible vegetables that do not provoke gastric gas."},
            {"food": "Lean Boiled Chicken or Tofu", "reason": "Pure protein without heavy animal fat that delays stomach emptying."},
            {"food": "Almond Milk & Herbal Chamomile Tea", "reason": "Non-dairy soothing beverages that neutralize gastric acidity."},
        ]
        foods_to_avoid = [
            {"food": "Fried, Greasy & Deep-Fried Foods", "reason": "Relaxes the lower esophageal sphincter, allowing acid into the chest."},
            {"food": "Citrus Fruits & Tomatoes", "reason": "High citric/malic acid directly irritates inflamed chest tissue."},
            {"food": "Spicy Curries, Raw Garlic & Onions", "reason": "Strong pungent compounds that trigger esophageal spasms."},
            {"food": "Chocolate, Mint & Caffeine", "reason": "Contains methylxanthines that chemically weaken esophageal closure."},
            {"food": "Carbonated Sodas & Energy Drinks", "reason": "Expands the stomach with gas, forcing stomach contents upward."},
        ]
        sample_menu = {
            "Breakfast": "Warm oatmeal cooked with almond milk, topped with sliced banana and a pinch of cinnamon.",
            "Lunch": "Steamed white rice or soft quinoa with baked chicken breast / steamed lentils and zucchini.",
            "Evening Snack": "A handful of non-salted raw almonds and soothing lukewarm chamomile tea.",
            "Dinner": "Clear vegetable broth with boiled potatoes and steamed fish / tofu. Finished 3 hours before sleep.",
        }

    elif has_htn or category == "HYPERTENSIVE_VASCULAR_CONCERN":
        protocol_name = "CLINICAL DASH PROTOCOL (Dietary Approaches to Stop Hypertension)"
        goal = "Target sodium reduction (<1,500 mg/day) and elevate dietary potassium and magnesium to relax vascular walls and reduce arterial pressure."
        foods_to_eat = [
            {"food": "Dark Leafy Greens (Spinach, Kale)", "reason": "High in natural nitrates and potassium; relaxes vascular endothelial tone."},
            {"food": "Crushed Raw Garlic & Flaxseeds", "reason": "Allicin in garlic has shown 5–8 mm Hg reduction in systolic pressure."},
            {"food": "Beetroot & Beetroot Juice", "reason": "Generates nitric oxide, directly dilating constricted blood vessels."},
            {"food": "Bananas, Sweet Potatoes & Avocados", "reason": "Rich in potassium; encourages kidneys to excrete excess sodium via urine."},
            {"food": "Raw Walnuts & Chia Seeds", "reason": "Plant-based ALA omega-3s reduce vascular inflammation and stiffness."},
        ]
        foods_to_avoid = [
            {"food": "Pickles, Papads & Salt-Cured Snacks", "reason": "Single servings often exceed the entire daily sodium allowance of 1,500 mg."},
            {"food": "Processed Deli Meats & Sausages", "reason": "High in both sodium nitrates and saturated fats that stiffen arteries."},
            {"food": "Packaged Instant Noodles & Canned Soups", "reason": "Monosodium glutamate (MSG) and hidden salt causes immediate fluid retention."},
            {"food": "Table Salt Shaker (Direct Adding)", "reason": "Never add extra raw salt at the table; flavor with lemon, herbs, and cumin instead."},
            {"food": "Licorice & High-Sodium Energy Drinks", "reason": "Licorice contains glycyrrhizin, which directly raises blood pressure."},
        ]
        sample_menu = {
            "Breakfast": "Steamed vegetable oats or whole-wheat toast with avocado slices and a bowl of fresh papaya.",
            "Lunch": "Brown rice or 2 whole-wheat rotis with a thick bowl of spinach dal, steamed beetroot salad, and fresh curd.",
            "Evening Snack": "Roasted unsalted makhana (foxnuts) or roasted pumpkin seeds with unsweetened green tea.",
            "Dinner": "Lentil vegetable soup with grilled paneer / steamed fish and sautéed carrots and peas.",
        }

    elif "PALPITATIONS_ARRHYTHMIA" in pain_char or category == "CARDIAC_ARRHYTHMIA_PALPITATIONS":
        protocol_name = "CARDIAC ELECTROLYTE & ANTI-ARRHYTHMIA DIET"
        goal = "Restore cellular potassium and magnesium balance across myocardial membranes, eliminate cardiac stimulants, and prevent rhythm irritability."
        foods_to_eat = [
            {"food": "Fresh Coconut Water", "reason": "Nature's balanced isotonic source of potassium, magnesium, and hydration."},
            {"food": "Pumpkin Seeds & Almonds", "reason": "Dense in magnesium (essential mineral cofactor for cardiac rhythm stability)."},
            {"food": "Steamed Broccoli & Sweet Potatoes", "reason": "Maintains cellular electrolyte potential in ventricular myocytes."},
            {"food": "Salmon, Mackerel or Walnuts", "reason": "Omega-3 fatty acids exert proven anti-arrhythmic stabilizing effects on heart cells."},
            {"food": "Adequate Water (2.5L / Day)", "reason": "Dehydration is the #1 silent trigger of benign cardiac palpitations."},
        ]
        foods_to_avoid = [
            {"food": "High-Caffeine Coffee & Energy Drinks", "reason": "Potent adenosine-receptor blockers that trigger premature ventricular beats."},
            {"food": "Alcohol & Binge Drinking", "reason": "'Holiday Heart Syndrome' — alcohol directly irritates atrial conduction pathways."},
            {"food": "High-Sugar Refined Desserts", "reason": "Rapid glucose spikes followed by reactive hypoglycemia trigger adrenaline surges."},
            {"food": "Pre-Workout Stimulant Powders", "reason": "High beta-alanine and caffeine concentrations induce acute tachyarrhythmias."},
            {"food": "Excessive Salt & Monosodium Glutamate", "reason": "Shifts fluid balance and triggers nocturnal heart flutters in sensitive individuals."},
        ]
        sample_menu = {
            "Breakfast": "Bowl of Greek yogurt or curd topped with crushed pumpkin seeds, chia seeds, and sliced banana.",
            "Lunch": "Steamed brown rice or quinoa with baked salmon/tofu and a large green salad dressed in olive oil.",
            "Evening Snack": "Fresh tender coconut water with 5–6 soaked almonds and walnuts.",
            "Dinner": "Moong dal khichdi or steamed vegetable soup with a side of steamed broccoli and sweet potato.",
        }

    else:
        # Mediterranean Heart Defense (Ischemic / Angina / General Heart Health)
        protocol_name = "MEDITERRANEAN CARDIOPROTECTIVE ARTERIAL DEFENSE DIET"
        goal = "Reduce circulating LDL cholesterol oxidation, clear early arterial plaque deposits, and maintain clean endothelial blood flow."
        foods_to_eat = [
            {"food": "Extra Virgin Olive Oil (Cold-Pressed)", "reason": "Rich in oleic acid and polyphenols; protects coronary endothelial lining."},
            {"food": "Whole Oats & Barley", "reason": "Beta-glucan soluble fiber binds to bile acids in the gut, forcing liver to draw down circulating blood cholesterol."},
            {"food": "Oily Fish (Sardines, Salmon) or Flaxseeds", "reason": "High in EPA/DHA omega-3s; lowers serum triglycerides by 20–30%."},
            {"food": "Mixed Berries (Blueberries, Strawberries)", "reason": "Anthocyanins reduce arterial stiffness and prevent LDL oxidation."},
            {"food": "Legumes, Chickpeas & Lentils", "reason": "Low glycemic index, rich in plant protein, zero dietary cholesterol."},
        ]
        foods_to_avoid = [
            {"food": "Commercial Fried Snacks & Palm Oil", "reason": "Trans-fatty acids directly penetrate artery walls, creating atherosclerotic plaques."},
            {"food": "Full-Fat Butter & Ghee (Excess)", "reason": "Saturated fats downregulate hepatic LDL receptors, spiking circulating bad cholesterol."},
            {"food": "Refined White Flour (Maida) & White Bread", "reason": "Causes rapid glycemic spikes that trigger systemic vascular inflammation."},
            {"food": "Sugary Soft Drinks & Syrups", "reason": "High-fructose corn syrup drives non-alcoholic fatty liver and dyslipidemia."},
            {"food": "Processed Red Meats (Bacon, Sausages)", "reason": "Contains TMAO precursors and high saturated fats linked to arterial hardening."},
        ]
        sample_menu = {
            "Breakfast": "Steel-cut oatmeal prepared with water or soy milk, sprinkled with flaxseeds and fresh berries.",
            "Lunch": "Mediterranean salad: chickpeas, cucumbers, tomatoes, extra virgin olive oil, and 2 multigrain rotis.",
            "Evening Snack": "Unsweetened green tea with a small bowl of mixed raw walnuts and pumpkin seeds.",
            "Dinner": "Grilled or steamed fish/tofu served over a generous bed of sautéed spinach, mushrooms, and garlic.",
        }

    # 2. Evidence-Based Home Remedies & Daily Dos / Don'ts
    remedies = [
        {
            "practice": "The 4-7-8 Cardiac Vagal Breathing Technique",
            "how_to_do": "Inhale through your nose for 4 seconds, hold your breath gently for 7 seconds, exhale slowly through your mouth for 8 seconds. Repeat 4 times.",
            "clinical_benefit": "Activates the parasympathetic vagus nerve, immediately dropping elevated heart rate and reducing systolic blood pressure by 6–10 mm Hg within minutes.",
        },
        {
            "practice": "Safe Physical Recovery Posture (If Chest Discomfort Appears)",
            "how_to_do": "Immediately stop all walking or exertion. Sit upright in a chair with back supported and legs uncrossed. Loosen shirt collar and belt.",
            "clinical_benefit": "Reduces cardiac preload (venous blood return volume) and myocardial oxygen demand, helping ischemic heart tissue recover without further strain.",
        },
        {
            "practice": "Targeted Hydration Protocol",
            "how_to_do": "Drink 250 mL of room-temperature water every 2 hours throughout the day (target 2.0 to 2.5 Liters total). Avoid ice-cold water during meals.",
            "clinical_benefit": "Prevents hemoconcentration (thick, viscous blood that requires greater cardiac pumping force) and eliminates dehydration-triggered palpitations.",
        },
        {
            "practice": "Early Evening Meal Timing (The 3-Hour Golden Rule)",
            "how_to_do": "Finish your evening meal at least 3 full hours before lying down to sleep. Keep dinner lightweight.",
            "clinical_benefit": "Prevents nocturnal gastro-cardiac syndrome (where a bloated stomach presses against the diaphragm, compressing the inferior heart wall and disturbing sleep rhythm).",
        },
    ]

    dos_and_donts = {
        "DOs": [
            "DO rest immediately if any tightness, breathlessness, or unusual fatigue occurs.",
            "DO take your blood pressure at the same time each morning and evening after sitting quietly for 5 minutes.",
            "DO keep your medications easily accessible and never skip prescribed blood pressure tablets without clinical supervision.",
            "DO replace regular table salt with potassium-enriched salt or natural lemon, ginger, and garlic seasonings.",
        ],
        "DONTs": [
            "DON'T dismiss recurring exertional chest heaviness as 'just gas' or 'acidity' without getting an ECG.",
            "DON'T perform heavy sudden physical lifting, sprinting, or snow-shoveling if you have unverified chest symptoms.",
            "DON'T smoke, vape, or consume energy drinks, which cause acute coronary artery vasospasm.",
            "DON'T take scalding hot baths or sauna immediately after heavy exertion, as sudden peripheral vasodilation can induce syncope (fainting).",
        ]
    }

    return {
        "protocol_name": protocol_name,
        "goal": goal,
        "foods_to_eat": foods_to_eat,
        "foods_to_avoid": foods_to_avoid,
        "sample_menu": sample_menu,
        "remedies": remedies,
        "dos_and_donts": dos_and_donts,
    }
