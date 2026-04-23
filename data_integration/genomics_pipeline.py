"""
Genomics pipeline for CKD polygenic risk scoring.
Computes PRS from known CKD genetic variants (APOL1, UMOD, SHROOM3, etc.)
and integrates genetic risk into the clinical risk assessment.
"""
import numpy as np

CKD_VARIANTS = {
    'APOL1': {
        'gene': 'APOL1',
        'rsid': 'rs73885319 / rs71785313',
        'chromosome': '22q13.1',
        'risk_alleles': ['G1 (rs73885319)', 'G2 (rs71785313)'],
        'effect_size': 1.89,
        'population': 'African ancestry (West African origin)',
        'mechanism': 'Apolipoprotein L1 G1/G2 variants — lysosomal membrane disruption in podocytes',
        'clinical_significance': (
            'Two APOL1 risk alleles confer ~7-10x increased risk of FSGS and hypertensive CKD '
            'in individuals of African ancestry. Most important genetic risk factor for CKD in this population.'
        ),
    },
    'UMOD': {
        'gene': 'UMOD',
        'rsid': 'rs12917707',
        'chromosome': '16p12.3',
        'risk_alleles': ['T allele'],
        'effect_size': 1.15,
        'population': 'European ancestry',
        'mechanism': 'Uromodulin regulation — affects tubular function and electrolyte handling',
        'clinical_significance': 'Common variant; each C→T allele reduces eGFR ~1.5 mL/min/1.73m²',
    },
    'SHROOM3': {
        'gene': 'SHROOM3',
        'rsid': 'rs17319721',
        'chromosome': '4q21.1',
        'risk_alleles': ['A allele'],
        'effect_size': 1.12,
        'population': 'All ancestries',
        'mechanism': 'Podocyte cytoskeletal organization and filtration barrier integrity',
        'clinical_significance': 'Moderate effect; associated with albuminuria and CKD progression',
    },
    'CUBN': {
        'gene': 'CUBN',
        'rsid': 'rs1801239',
        'chromosome': '10p13',
        'risk_alleles': ['A allele'],
        'effect_size': 1.08,
        'population': 'All ancestries',
        'mechanism': 'Cubilin — albumin reabsorption in proximal tubule',
        'clinical_significance': 'Associated with albuminuria independent of eGFR',
    },
    'PRKAG2': {
        'gene': 'PRKAG2',
        'rsid': 'rs7805747',
        'chromosome': '7q36.1',
        'risk_alleles': ['G allele'],
        'effect_size': 1.10,
        'population': 'All ancestries',
        'mechanism': 'AMP-activated protein kinase — cellular energy sensing in tubular cells',
        'clinical_significance': 'Associated with eGFR decline in diabetic CKD',
    },
    'SLC34A1': {
        'gene': 'SLC34A1',
        'rsid': 'rs6420094',
        'chromosome': '5q35',
        'risk_alleles': ['C allele'],
        'effect_size': 1.09,
        'population': 'All ancestries',
        'mechanism': 'Sodium-phosphate cotransporter — phosphate reabsorption',
        'clinical_significance': 'Variants affect phosphate handling; relevant to CKD-MBD',
    },
}


def compute_polygenic_risk_score(genotype_data: dict) -> dict:
    """
    Compute PRS from patient's genotype data.
    genotype_data: dict mapping rsid/gene to dosage (0, 1, or 2 risk alleles) or None.
    Returns PRS score, risk percentile, and per-variant contributions.
    """
    contributions = []
    total_log_or = 0.0
    variants_with_data = 0
    missing_variants = []

    for gene, variant_info in CKD_VARIANTS.items():
        dosage = genotype_data.get(gene) or genotype_data.get(variant_info['rsid'].split('/')[0].strip())
        if dosage is None:
            missing_variants.append(gene)
            dosage = _impute_from_population(gene, genotype_data.get('ancestry', 'European'))

        effect = np.log(variant_info['effect_size'])
        contribution = float(dosage) * effect
        total_log_or += contribution
        variants_with_data += 1

        contributions.append({
            'gene': gene,
            'rsid': variant_info['rsid'],
            'dosage': dosage,
            'effect_size_or': variant_info['effect_size'],
            'log_contribution': round(contribution, 4),
            'clinical_significance': variant_info['clinical_significance'],
        })

    # Normalize PRS to 0-100 scale (based on approximate population distribution)
    raw_prs = total_log_or
    prs_mean = 0.82
    prs_std = 0.45
    z_score = (raw_prs - prs_mean) / prs_std
    percentile = round(_normal_cdf(z_score) * 100, 1)
    prs_score = round(max(0, min(100, 50 + z_score * 10)), 1)

    # Risk category
    if percentile >= 90:
        risk_category = 'Very High Genetic Risk'
        risk_color = '#dc3545'
        clinical_note = 'Top decile genetic risk — early and aggressive CKD screening recommended'
    elif percentile >= 75:
        risk_category = 'High Genetic Risk'
        risk_color = '#fd7e14'
        clinical_note = 'Above-average genetic predisposition — intensive lifestyle modification and close monitoring'
    elif percentile >= 25:
        risk_category = 'Average Genetic Risk'
        risk_color = '#ffc107'
        clinical_note = 'Genetic risk in population average range — standard screening protocols apply'
    else:
        risk_category = 'Below Average Genetic Risk'
        risk_color = '#28a745'
        clinical_note = 'Below-average genetic predisposition — standard care; other risk factors still important'

    # APOL1 special case
    apol1_high_risk = False
    apol1_dosage = genotype_data.get('APOL1', 0)
    if apol1_dosage == 2:
        apol1_high_risk = True

    recommendations = _genomic_recommendations(percentile, apol1_high_risk,
                                                 genotype_data.get('ancestry', ''))

    return {
        'prs_score': prs_score,
        'prs_percentile': percentile,
        'risk_category': risk_category,
        'risk_color': risk_color,
        'clinical_note': clinical_note,
        'variant_contributions': sorted(contributions, key=lambda x: abs(x['log_contribution']), reverse=True),
        'variants_tested': variants_with_data,
        'missing_variants': missing_variants,
        'apol1_high_risk': apol1_high_risk,
        'recommendations': recommendations,
    }


def _impute_from_population(gene: str, ancestry: str) -> float:
    """Impute missing genotype from population allele frequencies."""
    freq_map = {
        'APOL1': {'African': 0.22, 'European': 0.01, 'Asian': 0.005, 'Hispanic': 0.06},
        'UMOD': {'African': 0.25, 'European': 0.30, 'Asian': 0.22, 'Hispanic': 0.27},
        'SHROOM3': {'African': 0.18, 'European': 0.35, 'Asian': 0.40, 'Hispanic': 0.32},
        'CUBN': {'African': 0.15, 'European': 0.20, 'Asian': 0.18, 'Hispanic': 0.19},
        'PRKAG2': {'African': 0.28, 'European': 0.33, 'Asian': 0.30, 'Hispanic': 0.31},
        'SLC34A1': {'African': 0.22, 'European': 0.25, 'Asian': 0.24, 'Hispanic': 0.23},
    }
    anc_key = ancestry.split()[0] if ancestry else 'European'
    freq = freq_map.get(gene, {}).get(anc_key, 0.25)
    # Expected dosage under HWE
    return 2 * freq


def _normal_cdf(z: float) -> float:
    """Approximate standard normal CDF using error function."""
    import math
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def _genomic_recommendations(percentile: float, apol1_high: bool, ancestry: str) -> list:
    recs = []
    if apol1_high:
        recs.append("APOL1 high-risk genotype (G1/G2) detected — nephrology referral for genetic counseling recommended")
        recs.append("Avoid nephrotoxic medications; consider annual urine ACR screening from age 18")
        recs.append("Discuss implications for family members (first-degree relatives at 25% risk)")
    if percentile >= 90:
        recs.append("Begin CKD screening from age 30 (standard recommendation: age 45)")
        recs.append("Annual eGFR and urine ACR monitoring even without traditional risk factors")
        recs.append("Lifestyle intervention: aggressive BP control, plant-based diet, smoking cessation")
    elif percentile >= 75:
        recs.append("Begin CKD screening from age 40")
        recs.append("Annual urine ACR monitoring")
    recs.append("Genetic risk is fixed — environmental and lifestyle factors remain modifiable")
    return recs


def demo_genotype(ancestry: str = 'European') -> dict:
    """Return a sample genotype dict for demonstration."""
    rng = np.random.RandomState(42)
    freq_ref = {
        'APOL1': 0.01 if ancestry == 'European' else 0.22,
        'UMOD': 0.30,
        'SHROOM3': 0.35,
        'CUBN': 0.20,
        'PRKAG2': 0.33,
        'SLC34A1': 0.25,
    }
    genotype = {'ancestry': ancestry}
    for gene, freq in freq_ref.items():
        genotype[gene] = int(rng.binomial(2, freq))
    return genotype
