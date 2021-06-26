from collections import OrderedDict
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd

DIR_PATH = Path(__file__).parent.absolute()

FLI_THRESHOLD = 60
USFLI_THRESHOLD = 30


def load(path):
    df = pd.read_sas(path)
    df = df.set_index('SEQN')
    return df


def non_alcoholic(df):
    return ((df['sex'] == 'female') & (df['drinks'] <= 1)) | ((df['sex'] == 'male') & (df['drinks'] <= 2))


def fli(df):
    z = (
        0.953 * np.log(df['triglycerides'])
        + 0.139 * df['bmi']
        + 0.718 * np.log(df['ggt'])
        + 0.053 * df['waist_circumference']
        - 15.745
    )
    return np.exp(z) / (1 + np.exp(z)) * 100


def usfli(df):
    z = (
        -0.8073 * df['non_hispanic_black'].apply(lambda x: 1 if x is True else 0)
        + 0.3458 * df['mexican_american'].apply(lambda x: 1 if x is True else 0)
        + 0.0093 * df['age']
        + 0.6151 * np.log(df['ggt'])
        + 0.0249 * df['waist_circumference']
        + 1.1792 * np.log(df['insulin'])
        + 0.8242 * np.log(df['glucose'])
        - 14.7812
    )
    return np.exp(z) / (1 + np.exp(z)) * 100


def baseline(df):

    def mean(df, col):
        return (df[col] * df['weight']).sum() / df['weight'].sum()

    def percent(df, expr):
        return df[expr]['weight'].sum() / df['weight'].sum() * 100

    return OrderedDict([
        ('mean age', mean(df, 'age')),
        ('% female', percent(df, df.sex == 'female')),
        ('% male', percent(df, df.sex == 'male')),
        ('% mexican american', percent(df, df.mexican_american)),
        ('% other hispanic', percent(df, df.other_hispanic)),
        ('% non-hispanic white', percent(df, df.non_hispanic_white)),
        ('% non-hispanic black', percent(df, df.non_hispanic_black)),
        ('% non-hispanic asian', percent(df, df.non_hispanic_asian)),
        ('% other race', percent(df, df.other_race)),
        ('mean drinks/day female', mean(df[df.sex == 'female'], 'drinks')),
        ('mean drinks/day male', mean(df[df.sex == 'male'], 'drinks')),
        ('mean bmi', mean(df, 'bmi')),
        ('mean waist circumference', mean(df, 'waist_circumference')),
        ('mean triglycerides', mean(df, 'triglycerides')),
        ('mean HDL', mean(df, 'HDL')),
        ('mean LDL', mean(df, 'LDL')),
        ('mean TC', mean(df, 'TC')),
        ('mean AST', mean(df, 'AST')),
        ('mean ALT', mean(df, 'ALT')),
        ('mean ALP', mean(df, 'ALP')),
        ('% diabetes', percent(df, df.diabetes)),
        ('% htn', percent(df, df.htn)),
        ('% PIR low', percent(df, df.pir_low)),
        ('% PIR medium', percent(df, df.pir_medium)),
        ('% PIR high', percent(df, df.pir_high)),
        ('% smoker', percent(df, df.smoker)),
    ])


def analyze_2017_2018():

    demo = load('2017-2018/DEMO_J.XPT')

    # Questionnaire data:
    alc = load('2017-2018/ALQ_J.XPT')
    med = load('2017-2018/MCQ_J.XPT')
    hep = load('2017-2018/HEQ_J.XPT')
    pre = load('2017-2018/RXQ_RX_J.XPT')

    # Laboratory data:
    biochem = load('2017-2018/BIOPRO_J.XPT')
    glu = load('2017-2018/GLU_J.XPT')
    ins = load('2017-2018/INS_J.XPT')
    trigly = load('2017-2018/TRIGLY_J.XPT')

    # Examination data:
    body = load('2017-2018/BMX_J.XPT')

    # For statin analysis:
    total = load('2017-2018/TCHOL_J.XPT')
    hdl = load('2017-2018/HDL_J.XPT')

    # For other analysis:
    diab = load('2017-2018/DIQ_J.XPT')
    bp = load('2017-2018/BPQ_J.XPT')
    income = load('2017-2018/INQ_J.XPT')
    smoke = load('2017-2018/SMQ_J.XPT')

    pre['drugs'] = pre['RXDDRUG'].apply(lambda x: x.decode().lower())
    pre = pre[pre.drugs.str.endswith('statin')]  # with statins
    pre = pre.groupby(['SEQN']).transform(lambda x: ','.join(x))  # join rows on SEQN
    pre = pre[~pre.index.duplicated(keep='first')]  # drop duplicates resulting from group_by

    df = pd.DataFrame()

    df['diabetes'] = diab['DIQ010'].apply(lambda x: x == 1)
    df['htn'] = bp['BPQ020'].apply(lambda x: x == 1)
    df['pir_low'] = income['INDFMMPC'].apply(lambda x: x == 1)  # poverty income ratio
    df['pir_medium'] = income['INDFMMPC'].apply(lambda x: x == 2)
    df['pir_high'] = income['INDFMMPC'].apply(lambda x: x == 3)
    df['smoker'] = smoke['SMQ040'].apply(lambda x: x == 1 or x == 2)

    df['drinks'] = alc['ALQ130']  # per day
    df['age'] = demo['RIDAGEYR']

    df['mexican_american'] = demo['RIDRETH3'].apply(lambda x: x == 1)
    df['other_hispanic'] = demo['RIDRETH3'].apply(lambda x: x == 2)
    df['non_hispanic_white'] = demo['RIDRETH3'].apply(lambda x: x == 3)
    df['non_hispanic_black'] = demo['RIDRETH3'].apply(lambda x: x == 4)
    df['non_hispanic_asian'] = demo['RIDRETH3'].apply(lambda x: x == 6)
    df['other_race'] = demo['RIDRETH3'].apply(lambda x: x == 7)

    df['sex'] = demo['RIAGENDR'].apply(lambda x: 'male' if x == 1 else ('female' if x == 2 else None))
    df['hep_b'] = hep['HEQ010'].apply(lambda x: x == 1)
    df['hep_c'] = hep['HEQ030'].apply(lambda x: x == 1)
    df['other_liver_conditions'] = (
        med['MCQ230A'].apply(lambda x: x == 22) |  # cancer
        med['MCQ230B'].apply(lambda x: x == 22) |  # cancer
        med['MCQ230C'].apply(lambda x: x == 22) |  # cancer
        med['MCQ510B'].apply(lambda x: x == 2)  |  # fibrosis
        med['MCQ510C'].apply(lambda x: x == 3)  |  # cirrhosis
        med['MCQ510D'].apply(lambda x: x == 4)  |  # viral hepatitis
        med['MCQ510E'].apply(lambda x: x == 5)  |  # autoimmune hepatitis
        med['MCQ510F'].apply(lambda x: x == 6)     # other
    )
    df['had_liver_condition'] = med['MCQ160L'].apply(lambda x: x == 1)
    df['still_have_liver_condition'] = med['MCQ170L'].apply(lambda x: x == 1)
    df['had_and_still_have_liver_condition'] = df.had_liver_condition & df.still_have_liver_condition
    df['fatty_liver'] = med['MCQ510A'].apply(lambda x: x == 1)
    df['ggt'] = biochem['LBXSGTSI']
    df['bmi'] = body['BMXBMI']
    df['waist_circumference'] = body['BMXWAIST']
    df['glucose'] = glu['LBXGLU']
    df['insulin'] = ins['LBDINSI']
    df['triglycerides'] = trigly['LBXTR']

    df['fli'] = fli(df)
    df['high_fli'] = df['fli'] > FLI_THRESHOLD
    df['usfli'] = usfli(df)
    df['high_usfli'] = df['usfli'] > USFLI_THRESHOLD
    df['non_alcoholic'] = non_alcoholic(df)

    # For statin analysis:
    df['prescribed_statins'] = pre['drugs']
    df['statins'] = ~df.prescribed_statins.isnull()
    df['AST'] = biochem['LBXSASSI']
    df['ALT'] = biochem['LBXSATSI']
    df['ALP'] = biochem['LBXSAPSI']
    df['AST_ALT_ratio'] = df['AST'] / df['ALT']
    df['TC'] = total['LBXTC']
    df['TG'] = trigly['LBXTR']
    df['LDL'] = trigly['LBDLDL']
    df['HDL'] = hdl['LBDHDD']

    # Improved:
    df['fld'] = df.fatty_liver & ~(df.hep_b | df.hep_c | df.other_liver_conditions)
    df['nafld'] = df.non_alcoholic & df.fld
    df['questionnaire_fld'] = df.had_and_still_have_liver_condition & ~(df.hep_b | df.hep_c | df.other_liver_conditions)
    df['questionnaire_nafld'] = df.non_alcoholic & df.questionnaire_fld
    df['fld_fli'] = df.high_fli & ~(df.hep_b | df.hep_c | df.other_liver_conditions)
    df['nafld_fli'] = df.non_alcoholic & df.fld_fli
    df['fld_usfli'] = df.high_usfli & ~(df.hep_b | df.hep_c | df.other_liver_conditions)
    df['nafld_usfli'] = df.non_alcoholic & df.fld_usfli

    df['weight'] = demo['WTMEC2YR']

    df.to_csv('2017-2018/all.csv')  # dump an "Excel spreadsheet"

    df_had = df[df.had_liver_condition]
    print(f'Number of rows had liver condition: {len(df_had)}')

    df_still = df[df.still_have_liver_condition]
    print(f'Number of rows still have liver condition: {len(df_still)}')

    df_had_and_still = df[df.had_liver_condition & df.still_have_liver_condition]
    print(f'Number of rows had and still have liver condition: {len(df_had_and_still)}')

    df_fli = df.dropna(subset=['fli'])
    print(f'Number of rows with FLI: {len(df_fli)}')
    df_fli = df_fli[df_fli.fli >= FLI_THRESHOLD]
    print(f'Number of rows with FLI >= {FLI_THRESHOLD}: {len(df_fli)}')

    df_usfli = df.dropna(subset=['usfli'])
    print(f'Number of rows with USFLI: {len(df_usfli)}')
    df_usfli = df_usfli[df_usfli.usfli >= USFLI_THRESHOLD]
    print(f'Number of rows with USFLI >= {USFLI_THRESHOLD}: {len(df_usfli)}')

    # Start changing things...
    df_fld = df[df.fld]
    print(f'Number of FLD only (fatty liver but not hepB, hepC, or other liver conditions): {len(df_fld)}')

    df_nafld = df[df.nafld]
    print(f'Number of NAFLD (fatty liver and non-alcoholic): {len(df_nafld)}')

    df_questionnaire_fld = df[df.questionnaire_fld]
    print(f'Number of FLD by questionnaire (had and still have liver condition but don\'t have HepB, HepC, liver cancer, etc.): {len(df_questionnaire_fld)}')
    df_questionnaire_nafld = df[df.questionnaire_nafld]
    print(f'Number of NAFLD by questionnaire (had and still have liver condition but don\'t have HepB, HepC, liver cancer, etc., or significant alcoholism): {len(df_questionnaire_nafld)}')

    df_fld_fli = df[df.fld_fli]
    print(f'Number of FLD by (FLI > {FLI_THRESHOLD}): {len(df_fld_fli)}')
    df_nafld_fli = df[df.nafld_fli]
    print(f'Number of NAFLD by (FLI > {FLI_THRESHOLD}): {len(df_nafld_fli)}')

    df_fld_usfli = df[df.fld_usfli]
    print(f'Number of FLD by (USFLI > {USFLI_THRESHOLD}): {len(df_fld_usfli)}')
    df_nafld_usfli = df[df.nafld_usfli]
    print(f'Number of NAFLD by (USFLI > {USFLI_THRESHOLD}): {len(df_nafld_usfli)}')

    overlap = df[df.fld_fli & df.fld_usfli]
    print(f'Number of FLD patients identified by both (USFLI > {USFLI_THRESHOLD}) and (FLI > {FLI_THRESHOLD}): {len(overlap)}')

    overlap = df[df.nafld_fli & df.nafld_usfli]
    print(f'Number of NAFLD patients identified by both (USFLI > {USFLI_THRESHOLD}) and (FLI > {FLI_THRESHOLD}): {len(overlap)}')

    def print_overlap(*attrs):
        first, *others = attrs
        _all = getattr(df, first)
        for attr in others:
            _all &= getattr(df, attr)
        overlap = df[_all]
        print(f'Overlap of {attrs}: {len(overlap)}')

    print('-' * 40)
    print_overlap('nafld_fli', 'nafld_usfli')
    print_overlap('nafld', 'nafld_fli', 'nafld_usfli')
    print_overlap('questionnaire_nafld', 'nafld_fli', 'nafld_usfli')
    print('-' * 40)
    print_overlap('fld_fli', 'fld_usfli')
    print_overlap('fld', 'fld_fli', 'fld_usfli')
    print_overlap('questionnaire_fld', 'fld_fli', 'fld_usfli')
    print('-' * 40)

    df_statin = df[df.nafld & df.statins]
    df_nonstatin = df[df.nafld & ~df.statins]
    print(f'Number using statins: {len(df_statin)}')
    print(f'Number not using statins: {len(df_nonstatin)}')
    df_statin.to_csv('csv/2017-2018/statin.csv')
    df_nonstatin.to_csv('csv/2017-2018/nonstatin.csv')

    df_statin_weight = df_statin['weight'].sum()
    df_nonstatin_weight = df_nonstatin['weight'].sum()
    df_nafld_weight = df[df.nafld]['weight'].sum()
    print('df_statin weight sum:', df_statin_weight)
    print('df_nonstatin weight sum:', df_nonstatin_weight)
    print('df_nafld weight sum:', df_nafld_weight)

    df_nafld_index = df[df.nafld_fli & df.nafld_usfli]

    df_nafld_index_weight = df[df.nafld_fli & df.nafld_usfli]['weight'].sum()
    print('df_nafld_index weight sum:', df_nafld_index_weight)
    df_statin_index = df[df.nafld_fli & df.nafld_usfli & df.statins]
    df_statin_index_weight = df_statin_index['weight'].sum()
    print(f'df_statin_index: length={len(df_statin_index)}, weight sum={df_statin_index_weight}')
    df_nonstatin_index = df[df.nafld_fli & df.nafld_usfli & ~df.statins]
    df_nonstatin_index_weight = df_nonstatin_index['weight'].sum()
    print(f'df_nonstatin_index: length={len(df_nonstatin_index)}, weight sum={df_nonstatin_index_weight}')


    df_index_statin = df[df.nafld_fli & df.nafld_usfli & df.statins]
    df_index_nonstatin = df[df.nafld_fli & df.nafld_usfli & ~df.statins]
    print(f'Number using statins: {len(df_index_statin)}')
    print(f'Number not using statins: {len(df_index_nonstatin)}')
    df_index_statin.to_csv('csv/2017-2018/index_statin.csv')
    df_index_nonstatin.to_csv('csv/2017-2018/index_nonstatin.csv')

    print(f'NAFLD (n = {len(df_nafld)})')
    pprint(baseline(df_nafld))
    print(f'statin (n = {len(df_statin)})')
    pprint(baseline(df_statin))
    print(f'non-statin (n = {len(df_nonstatin)})')
    pprint(baseline(df_nonstatin))
    print(f'NAFLD index (n = {len(df_nafld_index)})')
    pprint(baseline(df_nafld_index))
    print(f'statin index (n = {len(df_statin_index)})')
    pprint(baseline(df_statin_index))
    print(f'non-statin index (n = {len(df_nonstatin_index)})')
    pprint(baseline(df_nonstatin_index))



    # print(df_statin['AST_ALT_ratio'].values)
    # print(df_index_statin['AST_ALT_ratio'].values)


    # from scipy.stats import ttest_ind

    # def run_t_test(df_a, df_b, col, **kwargs):
    #     a = df_a[col].dropna()
    #     b = df_b[col].dropna()
    #     t, p = ttest_ind(a, b, **kwargs)
    #     print(col)
    #     print('t =', t)
    #     print('p =', p)

    # run_t_test(df_statin, df_nonstatin, 'AST_ALT_ratio', equal_var=False)
    # run_t_test(df_statin, df_nonstatin, 'TC', equal_var=False)
    # run_t_test(df_statin, df_nonstatin, 'TG', equal_var=False)


def main():
    analyze_2017_2018()


if __name__ == '__main__':
    main()
