from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


ALCOHOL = 'alcohol'
DEMOGRAPHICS = 'demographics'
HEPATITIS = 'hepatitis'
MEDICAL_CONDITIONS = 'medical_conditions'
BIOCHEMISTRY = 'biochemistry',
BODY_MEASURES = 'body_measures',
GLUCOSE = 'glucose',
INSULIN = 'insulin',
TRIGLYCERIDES = 'triglycerides'


BASENAMES = {
    ALCOHOL: 'ALQ_{suffix}.XPT',
    DEMOGRAPHICS: 'DEMO_{suffix}.XPT',
    HEPATITIS: 'HEQ_{suffix}.XPT',
    MEDICAL_CONDITIONS: 'MCQ_{suffix}.XPT',
    BIOCHEMISTRY: 'BIOPRO_{suffix}.XPT',
    BODY_MEASURES: 'BMX_{suffix}.XPT',
    GLUCOSE: 'GLU_{suffix}.XPT',
    INSULIN: 'INS_{suffix}.XPT',
    TRIGLYCERIDES: 'TRIGLY_{suffix}.XPT',
}


CODES = {
    '2015-2016': {
        'drinks': (ALCOHOL, lambda df: df['ALQ120U']),
        'age': (DEMOGRAPHICS, lambda df: df['RIDAGEYR']),
        'mexican_american': (DEMOGRAPHICS, lambda df: df['RIDRETH3'].apply(lambda x: x == 1)),
        'non_hispanic_black': (DEMOGRAPHICS, lambda df: df['RIDRETH3'].apply(lambda x: x == 4)),
        'sex': (DEMOGRAPHICS, lambda df: df['RIAGENDR'].apply(lambda x: 'male' if x == 1 else 'female')),
        'hep_b': (HEPATITIS, lambda df: df['HEQ010'].apply(lambda x: x == 1)),
        'hep_c': (HEPATITIS, lambda df: df['HEQ030'].apply(lambda x: x == 1)),
        'other_liver_conditions': (MEDICAL_CONDITIONS, lambda df: (
            df['MCQ230A'].apply(lambda x: x == 22) |  # cancer
            df['MCQ230B'].apply(lambda x: x == 22) |  # cancer
            df['MCQ230C'].apply(lambda x: x == 22) |  # cancer
            df['MCQ230D'].apply(lambda x: x == 22)    # cancer
        )),
        'ggt': (BIOCHEMISTRY, lambda df: df['LBXSGTSI']),
        'bmi': (BODY_MEASURES, lambda df: df['BMXBMI']),
        'waist_circumference': (BODY_MEASURES, lambda df: df['BMXWAIST']),
        'glucose': (GLUCOSE, lambda df: df['LBXGLU']),
        'insulin': (INSULIN, lambda df: df['LBDINSI']),
        'triglycerides': (TRIGLYCERIDES, lambda df: df['LBXTR']),
    },
    '2017-2018': {
        'drinks': (ALCOHOL, lambda df: df['ALQ130']),
        'age': (DEMOGRAPHICS, lambda df: df['RIDAGEYR']),
        'mexican_american': (DEMOGRAPHICS, lambda df: df['RIDRETH3'].apply(lambda x: x == 1)),
        'non_hispanic_black': (DEMOGRAPHICS, lambda df: df['RIDRETH3'].apply(lambda x: x == 4)),
        'sex': (DEMOGRAPHICS, lambda df: df['RIAGENDR'].apply(lambda x: 'male' if x == 1 else 'female')),
        'hep_b': (HEPATITIS, lambda df: df['HEQ010'].apply(lambda x: x == 1)),
        'hep_c': (HEPATITIS, lambda df: df['HEQ030'].apply(lambda x: x == 1)),
        'other_liver_conditions': (MEDICAL_CONDITIONS, lambda df: (
            df['MCQ230A'].apply(lambda x: x == 22) |  # cancer
            df['MCQ230B'].apply(lambda x: x == 22) |  # cancer
            df['MCQ230C'].apply(lambda x: x == 22) |  # cancer
            df['MCQ510B'].apply(lambda x: x == 2)  |  # fibrosis
            df['MCQ510C'].apply(lambda x: x == 3)  |  # cirrhosis
            df['MCQ510D'].apply(lambda x: x == 4)  |  # viral hepatitis
            df['MCQ510E'].apply(lambda x: x == 5)  |  # autoimmune hepatitis
            df['MCQ510F'].apply(lambda x: x == 6)     # other
        )),
        'fatty_liver': (MEDICAL_CONDITIONS, lambda df: df['MCQ510A'].apply(lambda x: x == 1)),
        'ggt': (BIOCHEMISTRY, lambda df: df['LBXSGTSI']),
        'bmi': (BODY_MEASURES, lambda df: df['BMXBMI']),
        'waist_circumference': (BODY_MEASURES, lambda df: df['BMXWAIST']),
        'glucose': (GLUCOSE, lambda df: df['LBXGLU']),
        'insulin': (INSULIN, lambda df: df['LBDINSI']),
        'triglycerides': (TRIGLYCERIDES, lambda df: df['LBXTR']),
    },
}


def load(directory, suffix, codes):
    directory = Path(directory)
    paths = {name: str(directory / basename.format(suffix=suffix)) for name, basename in BASENAMES.items()}
    dfs = {name: pd.read_sas(path).set_index('SEQN') for name, path in paths.items()}
    df = pd.DataFrame()
    for field, (name, extractor) in codes.items():
        df[field] = extractor(dfs[name])
    return df


def set_alcoholic(df):
    df['alcoholic'] = ((df['sex'] == 'female') & (df['drinks'] > 1)) | ((df['sex'] == 'male') & (df['drinks'] > 2))


def set_liver_disease(df):
    df['liver_disease'] = df['hep_b'] | df['hep_c'] | df['other_liver_conditions']


def set_fli(df):
    df['z'] = (
        0.953 * np.log(df['triglycerides'])
        + 0.139 * df['bmi']
        + 0.718 * np.log(df['ggt'])
        + 0.053 * df['waist_circumference']
        - 15.745
    )
    df['fli'] = np.exp(df['z']) / (1 + np.exp(df['z'])) * 100
    del df['z']


def set_usfli(df):
    df['z'] = (
        -0.8073 * df['non_hispanic_black'].apply(lambda x: 1 if x else 0)
        + 0.3458 * df['mexican_american'].apply(lambda x: 1 if x else 0)
        + 0.0093 * df['age']
        + 0.6151 * np.log(df['ggt'])
        + 0.0249 * df['waist_circumference']
        + 1.1792 * np.log(df['insulin'])
        + 0.8242 * np.log(df['glucose'])
        - 14.7812
    )
    df['usfli'] = np.exp(df['z']) / (1 + np.exp(df['z'])) * 100
    del df['z']


def analyze(df):
    fli_threshold = 60
    usfli_threshold = 30

    set_alcoholic(df)
    set_liver_disease(df)

    set_fli(df)
    set_usfli(df)

    df_fli = df.dropna(subset=['fli'])
    print(f'Number of rows with FLI: {len(df_fli)}')
    df_fli = df_fli[df_fli.fli >= fli_threshold]
    print(f'Number of rows with FLI >= {fli_threshold}: {len(df_fli)}')

    df_usfli = df.dropna(subset=['usfli'])
    print(f'Number of rows with USFLI: {len(df_usfli)}')
    df_usfli = df_usfli[df_usfli.usfli >= usfli_threshold]
    print(f'Number of rows with USFLI >= {usfli_threshold}: {len(df_usfli)}')


def main():
    print('2015-2016')
    df = load('2015-2016', 'I', CODES['2015-2016'])
    analyze(df)
    print('2017-2018')
    df = load('2017-2018', 'J', CODES['2017-2018'])
    analyze(df)


# def usfli(df, threshold=30):
#     columns = [
#         'drinks', 'hep_b', 'hep_c',
#         'liver_cancer', 'fatty_liver', 'other_liver_condition',
#         'age', 'sex',
#         'mexican_american', 'non_hispanic_black',
#         'ggt', 'insulin', 'glucose',
#         'waist_circumference',
#     ]
#     df = df[columns]
#     print('Number of rows:', len(df))

#     df = df.dropna()
#     df['z'] = (
#         -0.8073 * df['non_hispanic_black']
#         + 0.3458 * df['mexican_american']
#         + 0.0093 * df['age']
#         + 0.6151 * np.log(df['ggt'])
#         + 0.0249 * df['waist_circumference']
#         + 1.1792 * np.log(df['insulin'])
#         + 0.8242 * np.log(df['glucose'])
#         - 14.7812
#     )
#     df['usfli'] = np.exp(df['z']) / (1 + np.exp(df['z'])) * 100
#     print('Number of rows with USFLI:', len(df))

#     nafld_usfli = df[
#         (df.usfli >= threshold) & non_alcoholic(df) & no_liver_disease(df)
#     ]
#     print(f'NAFLD (USFLI >= {threshold}):', len(nafld_usfli))

#     nafld_diagnosed = df[
#         df.fatty_liver & non_alcoholic(df) & no_liver_disease(df)
#     ]
#     print('NAFLD (diagnosed with FLD):', len(nafld_diagnosed))


# def fli(df, threshold=60):
#     columns = [
#         'drinks', 'hep_b', 'hep_c',
#         'liver_cancer', 'fatty_liver', 'other_liver_condition',
#         'sex',
#         'ggt', 'triglycerides',
#         'bmi', 'waist_circumference',
#     ]
#     df = df[columns]
#     print('Number of rows:', len(df))

#     df = df.dropna()
#     df['z'] = (
#         0.953 * np.log(df['triglycerides'])
#         + 0.139 * df['bmi']
#         + 0.718 * np.log(df['ggt'])
#         + 0.053 * df['waist_circumference']
#         - 15.745
#     )
#     df['fli'] = np.exp(df['z']) / (1 + np.exp(df['z'])) * 100
#     print('Number of rows with FLI:', len(df))

#     nafld_fli = df[
#         (df.fli >= threshold) & non_alcoholic(df) & no_liver_disease(df)
#     ]
#     print(f'NAFLD (FLI >= {threshold}):', len(nafld_fli))

#     nafld_diagnosed = df[
#         df.fatty_liver & non_alcoholic(df) & no_liver_disease(df)
#     ]
#     print('NAFLD (diagnosed with FLD):', len(nafld_diagnosed))


# def main():
#     df = load('2017-2018', 'J', CODES['2017-2018'])
#     usfli(df)
#     print('-' * 40)
#     fli(df)


if __name__ == '__main__':
    main()

