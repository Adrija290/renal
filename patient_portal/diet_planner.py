"""
AI CKD diet planner.
Generates stage-specific daily meal plans and nutrient targets.
Handles the complex restrictions: potassium, phosphorus, protein, sodium, fluid.
"""
import json
import os
import random
from models.gfr_forecaster import classify_ckd_stage
from config import Config


def _load_guidelines() -> dict:
    path = os.path.join(Config.REFERENCE_DIR, 'ckd_diet_guidelines.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


MEAL_TEMPLATES = {
    'G1_G2': {
        'breakfast': [
            {'name': 'Oatmeal with blueberries and honey', 'protein_g': 5, 'potassium_mg': 220, 'phosphorus_mg': 180, 'sodium_mg': 50},
            {'name': 'Scrambled eggs (2) with white toast', 'protein_g': 14, 'potassium_mg': 130, 'phosphorus_mg': 200, 'sodium_mg': 280},
            {'name': 'Greek yogurt with strawberries', 'protein_g': 18, 'potassium_mg': 300, 'phosphorus_mg': 230, 'sodium_mg': 70},
        ],
        'lunch': [
            {'name': 'Chicken salad sandwich on white bread', 'protein_g': 28, 'potassium_mg': 320, 'phosphorus_mg': 260, 'sodium_mg': 620},
            {'name': 'Tuna wrap with lettuce and cucumber', 'protein_g': 22, 'potassium_mg': 350, 'phosphorus_mg': 220, 'sodium_mg': 540},
            {'name': 'Lentil soup with white roll', 'protein_g': 18, 'potassium_mg': 480, 'phosphorus_mg': 290, 'sodium_mg': 700},
        ],
        'dinner': [
            {'name': 'Grilled salmon with rice and green beans', 'protein_g': 34, 'potassium_mg': 610, 'phosphorus_mg': 380, 'sodium_mg': 320},
            {'name': 'Roast chicken breast with pasta and salad', 'protein_g': 30, 'potassium_mg': 520, 'phosphorus_mg': 340, 'sodium_mg': 380},
            {'name': 'Turkey stir-fry with white rice', 'protein_g': 28, 'potassium_mg': 540, 'phosphorus_mg': 310, 'sodium_mg': 420},
        ],
        'snacks': [
            {'name': 'Apple with rice crackers', 'protein_g': 2, 'potassium_mg': 150, 'phosphorus_mg': 40, 'sodium_mg': 60},
            {'name': 'Grapes (1 cup)', 'protein_g': 1, 'potassium_mg': 175, 'phosphorus_mg': 30, 'sodium_mg': 2},
        ],
    },
    'G3a_G3b': {
        'breakfast': [
            {'name': 'White rice porridge with egg white and chives', 'protein_g': 8, 'potassium_mg': 110, 'phosphorus_mg': 90, 'sodium_mg': 150},
            {'name': 'Cream of wheat with blueberries (no milk)', 'protein_g': 4, 'potassium_mg': 100, 'phosphorus_mg': 60, 'sodium_mg': 120},
            {'name': 'White toast (2 slices) with jam and egg whites', 'protein_g': 10, 'potassium_mg': 80, 'phosphorus_mg': 70, 'sodium_mg': 200},
        ],
        'lunch': [
            {'name': 'Chicken breast (leached veg) over white rice', 'protein_g': 22, 'potassium_mg': 250, 'phosphorus_mg': 190, 'sodium_mg': 280},
            {'name': 'Egg salad on white bread with iceberg lettuce', 'protein_g': 14, 'potassium_mg': 130, 'phosphorus_mg': 140, 'sodium_mg': 380},
            {'name': 'Low-sodium vegetable soup with white crackers', 'protein_g': 6, 'potassium_mg': 280, 'phosphorus_mg': 100, 'sodium_mg': 350},
        ],
        'dinner': [
            {'name': 'Poached fish (cod) with white pasta and cauliflower', 'protein_g': 26, 'potassium_mg': 360, 'phosphorus_mg': 230, 'sodium_mg': 310},
            {'name': 'Baked chicken with white rice and boiled cabbage', 'protein_g': 28, 'potassium_mg': 340, 'phosphorus_mg': 220, 'sodium_mg': 280},
            {'name': 'Turkey patty with white bread bun and coleslaw (rinsed)', 'protein_g': 24, 'potassium_mg': 310, 'phosphorus_mg': 200, 'sodium_mg': 400},
        ],
        'snacks': [
            {'name': 'Cranberry juice (4oz) + rice cakes', 'protein_g': 1, 'potassium_mg': 50, 'phosphorus_mg': 20, 'sodium_mg': 30},
            {'name': 'Blueberries (½ cup) + white crackers', 'protein_g': 1, 'potassium_mg': 65, 'phosphorus_mg': 15, 'sodium_mg': 55},
        ],
    },
    'G4': {
        'breakfast': [
            {'name': 'Cream of wheat with rice milk and berries', 'protein_g': 3, 'potassium_mg': 80, 'phosphorus_mg': 50, 'sodium_mg': 100},
            {'name': 'Egg white omelet (no cheese) with white toast', 'protein_g': 8, 'potassium_mg': 100, 'phosphorus_mg': 40, 'sodium_mg': 180},
        ],
        'lunch': [
            {'name': 'Small chicken breast (2oz) over white rice with green beans (boiled)', 'protein_g': 14, 'potassium_mg': 190, 'phosphorus_mg': 130, 'sodium_mg': 200},
            {'name': 'Low-protein pasta with olive oil and herbs', 'protein_g': 6, 'potassium_mg': 80, 'phosphorus_mg': 70, 'sodium_mg': 150},
        ],
        'dinner': [
            {'name': 'Poached white fish (2oz) with white rice and cabbage', 'protein_g': 14, 'potassium_mg': 200, 'phosphorus_mg': 140, 'sodium_mg': 180},
            {'name': 'Low-protein bread with olive oil, sliced apple, and egg white', 'protein_g': 8, 'potassium_mg': 130, 'phosphorus_mg': 60, 'sodium_mg': 120},
        ],
        'snacks': [
            {'name': 'Apple slices with hard candy', 'protein_g': 0, 'potassium_mg': 80, 'phosphorus_mg': 10, 'sodium_mg': 5},
        ],
    },
    'G5': {
        'breakfast': [
            {'name': 'Egg white scramble with white rice (for HD patients: more protein OK)', 'protein_g': 12, 'potassium_mg': 100, 'phosphorus_mg': 60, 'sodium_mg': 180},
        ],
        'lunch': [
            {'name': 'Grilled chicken breast (3oz) with white rice and boiled cabbage', 'protein_g': 21, 'potassium_mg': 260, 'phosphorus_mg': 170, 'sodium_mg': 250},
        ],
        'dinner': [
            {'name': 'Egg white pasta with chicken and cauliflower', 'protein_g': 20, 'potassium_mg': 220, 'phosphorus_mg': 150, 'sodium_mg': 200},
        ],
        'snacks': [
            {'name': 'White bread with jam', 'protein_g': 2, 'potassium_mg': 40, 'phosphorus_mg': 20, 'sodium_mg': 130},
        ],
    },
}


def generate_meal_plan(egfr: float, weight_kg: float = 70.0,
                        diabetes: bool = False, hyperkalemia: bool = False,
                        seed: int = None) -> dict:
    """Generate a personalized daily meal plan for CKD patient."""
    stage = classify_ckd_stage(egfr)
    guidelines = _load_guidelines()

    stage_key_map = {
        'G1': 'G1_G2', 'G2': 'G1_G2',
        'G3a': 'G3a_G3b', 'G3b': 'G3a_G3b',
        'G4': 'G4', 'G5': 'G5',
    }
    template_key = stage_key_map.get(stage, 'G3a_G3b')
    templates = MEAL_TEMPLATES.get(template_key, MEAL_TEMPLATES['G3a_G3b'])

    rng = random.Random(seed)
    plan = {
        'breakfast': rng.choice(templates['breakfast']),
        'lunch': rng.choice(templates['lunch']),
        'dinner': rng.choice(templates['dinner']),
        'snack': rng.choice(templates['snacks']),
    }

    # Daily totals
    total_protein = sum(m['protein_g'] for m in plan.values())
    total_potassium = sum(m['potassium_mg'] for m in plan.values())
    total_phosphorus = sum(m['phosphorus_mg'] for m in plan.values())
    total_sodium = sum(m['sodium_mg'] for m in plan.values())

    # Targets from guidelines
    stage_guideline_key = 'G3a_G3b' if stage in ('G3a', 'G3b') else (
        'G1_G2' if stage in ('G1', 'G2') else stage
    )
    targets = guidelines.get('stages', {}).get(stage_guideline_key, {})

    protein_target = targets.get('protein', {}).get('target_g_per_kg', 0.8)
    protein_goal = round(protein_target * weight_kg, 0)

    # Warnings
    warnings = []
    if total_potassium > (targets.get('potassium', {}).get('limit_mg_day') or 9999):
        warnings.append(f"Potassium limit exceeded: {total_potassium}mg vs limit {targets['potassium']['limit_mg_day']}mg")
    if total_phosphorus > (targets.get('phosphorus', {}).get('limit_mg_day') or 9999):
        warnings.append(f"Phosphorus limit exceeded: {total_phosphorus}mg vs limit {targets['phosphorus']['limit_mg_day']}mg")
    if total_sodium > (targets.get('sodium', {}).get('limit_mg_day') or 9999):
        warnings.append(f"Sodium limit exceeded: {total_sodium}mg vs limit {targets['sodium']['limit_mg_day']}mg")
    if diabetes and any(m['name'].lower().__contains__('honey') or m['name'].lower().__contains__('jam') for m in plan.values()):
        warnings.append("High-sugar items present — review with dietitian given diabetes diagnosis")

    # Food avoidance / emphasis
    avoid = guidelines.get('stages', {}).get(stage_guideline_key, {}).get('foods_to_avoid', [])
    emphasize = guidelines.get('stages', {}).get(stage_guideline_key, {}).get('foods_to_emphasize', [])

    return {
        'stage': stage,
        'meal_plan': plan,
        'daily_totals': {
            'protein_g': total_protein,
            'potassium_mg': total_potassium,
            'phosphorus_mg': total_phosphorus,
            'sodium_mg': total_sodium,
        },
        'targets': {
            'protein_g': protein_goal,
            'potassium_mg': targets.get('potassium', {}).get('limit_mg_day', 'No restriction'),
            'phosphorus_mg': targets.get('phosphorus', {}).get('limit_mg_day', 1200),
            'sodium_mg': targets.get('sodium', {}).get('limit_mg_day', 2300),
            'fluid_ml': targets.get('fluid', {}).get('limit_ml_day', 'No restriction'),
        },
        'warnings': warnings,
        'foods_to_avoid': avoid,
        'foods_to_emphasize': emphasize,
        'tips': _diet_tips(stage, hyperkalemia, diabetes),
        'leaching_required': stage in ('G3b', 'G4', 'G5'),
        'leaching_guide': guidelines.get('leaching_technique', {}),
    }


def _diet_tips(stage: str, hyperkalemia: bool, diabetes: bool) -> list:
    tips = []
    if stage in ('G4', 'G5'):
        tips.append("Read all food labels — look for 'phosphate' in ingredients (processed foods add inorganic phosphate that is 90% absorbed).")
        tips.append("Avoid 'lite salt' or 'salt substitutes' — these contain potassium chloride which is dangerous in CKD.")
    if hyperkalemia:
        tips.append("Leach all vegetables before cooking: peel, cut small, soak in water 2-4 hours, drain, then cook in fresh water.")
        tips.append("Avoid raw fruits and vegetables high in potassium (bananas, oranges, tomatoes, potatoes).")
    if diabetes:
        tips.append("Space carbohydrates evenly throughout the day to maintain blood glucose control.")
        tips.append("Choose low-glycemic index foods when possible — white rice in small portions is preferred over brown rice in CKD.")
    tips.append("Cook from scratch when possible — restaurant and processed foods are high in sodium and phosphate additives.")
    tips.append("Track your fluid intake: include soups, ice cream, gelatin — all count toward your daily fluid limit.")
    return tips
