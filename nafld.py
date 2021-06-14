from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

DIR_PATH = Path(__file__).parent.absolute()

FLI_THRESHOLD = 60
USFLI_THRESHOLD = 30


def load_dfs(dirname, suffix):

    def _load(fmt):
        path = DIR_PATH / dirname / fmt.format(suffix=suffix)
        path = str(path)
        df = pd.read_sas(path)
        df = df.set_index('SEQN')
        return df

    d = {
        # Demographics data
        'demo': 'DEMO_{suffix}.XPT',
        # Questionnaire data
        'alc': 'ALQ_{suffix}.XPT',
        'med': 'MCQ_{suffix}.XPT',
        'hep': 'HEQ_{suffix}.XPT',
        'pre': 'RXQ_RX_{suffix}.XPT',
        # Laboratory data
        'biochem': 'BIOPRO_{suffix}.XPT',
        'glu': 'GLU_{suffix}.XPT',
        'ins': 'INS_{suffix}.XPT',
        'trigly': 'TRIGLY_{suffix}.XPT',
        # Examination data
        'body': 'BMX_{suffix}.XPT',

        # For statin analysis:
        'total': 'TCHOL_{suffix}.XPT',
        'hdl': 'HDL_{suffix}.XPT',
    }
    d = {k: _load(v) for k, v in d.items()}
    return SimpleNamespace(**d)


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


def analyze_2017_2018():
    dfs = load_dfs('2017-2018', 'J')
    demo = dfs.demo
    print(f'Number of rows of demographic: {len(demo)}')
    alc = dfs.alc
    med = dfs.med
    hep = dfs.hep
    pre = dfs.pre
    biochem = dfs.biochem
    glu = dfs.glu
    ins = dfs.ins
    trigly = dfs.trigly
    body = dfs.body

    total = dfs.total
    hdl = dfs.hdl

    pre['drugs'] = pre['RXDDRUG'].apply(lambda x: x.decode().lower())
    pre = pre[pre.drugs.str.endswith('statin')]  # with statins
    pre = pre.groupby(['SEQN']).transform(lambda x: ','.join(x))  # join rows on SEQN
    pre = pre[~pre.index.duplicated(keep='first')]  # drop duplicates resulting from group_by

    df = pd.DataFrame()
    df['drinks'] = alc['ALQ130']
    df['age'] = demo['RIDAGEYR']
    df['mexican_american'] = demo['RIDRETH3'].apply(lambda x: x == 1)
    df['non_hispanic_black'] = demo['RIDRETH3'].apply(lambda x: x == 4)
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

    df.to_csv('2017-2018.csv')  # dump an "Excel spreadsheet"

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
    df_statin.to_csv('statin.csv')
    df_nonstatin.to_csv('nonstatin.csv')

    df_index_statin = df[df.nafld_fli & df.nafld_usfli & df.statins]
    df_index_nonstatin = df[df.nafld_fli & df.nafld_usfli & ~df.statins]
    print(f'Number using statins: {len(df_index_statin)}')
    print(f'Number not using statins: {len(df_index_nonstatin)}')
    df_index_statin.to_csv('index_statin.csv')
    df_index_nonstatin.to_csv('index_nonstatin.csv')


def main():
    analyze_2017_2018()


if __name__ == '__main__':
    main()
