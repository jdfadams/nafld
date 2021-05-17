from pprint import pprint

import numpy as np
import pandas as pd


non_alcoholic = lambda df: (((df.sex == 'female') & (df.drinks <= 1)) | ((df.sex == 'male') & (df.drinks <= 2)))

no_liver_disease = lambda df: ~(df.hep_b | df.hep_c | df.liver_cancer | df.other_liver_condition)


def load_data():
    df = pd.DataFrame()

    alcohol = pd.read_sas('ALQ_J.XPT')
    df['drinks'] = alcohol['ALQ130']

    demographics = pd.read_sas('DEMO_J.XPT')
    df['age'] = demographics['RIDAGEYR']
    df['mexican_american'] = demographics['RIDRETH3'].apply(lambda x: 1 if x == 1 else 0)
    df['non_hispanic_black'] = demographics['RIDRETH3'].apply(lambda x: 1 if x == 4 else 0)
    df['sex'] = demographics['RIAGENDR'].apply(lambda x: 'male' if x == 1 else 'female')

    hepatitis = pd.read_sas('HEQ_J.XPT')
    df['hep_b'] = hepatitis['HEQ010'].apply(lambda x: x == 1)
    df['hep_c'] = hepatitis['HEQ030'].apply(lambda x: x == 1)

    medical_conditions = pd.read_sas('MCQ_J.XPT')
    df['liver_cancer'] = (
        medical_conditions['MCQ230A'].apply(lambda x: x == 22) |
        medical_conditions['MCQ230B'].apply(lambda x: x == 22) |
        medical_conditions['MCQ230C'].apply(lambda x: x == 22)
    )
    df['fatty_liver'] = medical_conditions['MCQ510A'].apply(lambda x: x == 1)
    df['other_liver_condition'] = (
        medical_conditions['MCQ510B'].apply(lambda x: x == 2) |
        medical_conditions['MCQ510C'].apply(lambda x: x == 3) |
        medical_conditions['MCQ510D'].apply(lambda x: x == 4) |
        medical_conditions['MCQ510E'].apply(lambda x: x == 5) |
        medical_conditions['MCQ510F'].apply(lambda x: x == 6)
    )

    biochemistry = pd.read_sas('BIOPRO_J.XPT')
    df['ggt'] = biochemistry['LBXSGTSI']

    body = pd.read_sas('BMX_J.XPT')
    df['bmi'] = body['BMXBMI']
    df['waist_circumference'] = body['BMXWAIST']

    glucose = pd.read_sas('GLU_J.XPT')
    df['glucose'] = glucose['LBXGLU']

    insulin = pd.read_sas('INS_J.XPT')
    df['insulin'] = insulin['LBDINSI']

    triglycerides = pd.read_sas('TRIGLY_J.XPT')
    df['triglycerides'] = triglycerides['LBXTR']

    return df


def usfli(threshold=30):
    df = load_data()

    columns = [
        'drinks', 'hep_b', 'hep_c',
        'liver_cancer', 'fatty_liver', 'other_liver_condition',
        'age', 'sex',
        'mexican_american', 'non_hispanic_black',
        'ggt', 'insulin', 'glucose',
        'waist_circumference',
    ]
    df = df[columns]
    print('Number of rows:', len(df))

    df = df.dropna()
    df['z'] = (
        -0.8073 * df['non_hispanic_black']
        + 0.3458 * df['mexican_american']
        + 0.0093 * df['age']
        + 0.6151 * np.log(df['ggt'])
        + 0.0249 * df['waist_circumference']
        + 1.1792 * np.log(df['insulin'])
        + 0.8242 * np.log(df['glucose'])
        - 14.7812
    )
    df['usfli'] = np.exp(df['z']) / (1 + np.exp(df['z'])) * 100
    print('Number of rows with USFLI:', len(df))

    nafld_usfli = df[
        (df.usfli >= threshold) & non_alcoholic(df) & no_liver_disease(df)
    ]
    print(f'NAFLD (USFLI >= {threshold}):', len(nafld_usfli))

    nafld_diagnosed = df[
        df.fatty_liver & non_alcoholic(df) & no_liver_disease(df)
    ]
    print('NAFLD (diagnosed with FLD):', len(nafld_diagnosed))


def fli(threshold=60):
    df = load_data()

    columns = [
        'drinks', 'hep_b', 'hep_c',
        'liver_cancer', 'fatty_liver', 'other_liver_condition',
        'sex',
        'ggt', 'triglycerides',
        'bmi', 'waist_circumference',
    ]
    df = df[columns]
    print('Number of rows:', len(df))

    df = df.dropna()
    df['z'] = (
        0.953 * np.log(df['triglycerides'])
        + 0.139 * df['bmi']
        + 0.718 * np.log(df['ggt'])
        + 0.053 * df['waist_circumference']
        - 15.745
    )
    df['fli'] = np.exp(df['z']) / (1 + np.exp(df['z'])) * 100
    print('Number of rows with FLI:', len(df))

    nafld_fli = df[
        (df.fli >= threshold) & non_alcoholic(df) & no_liver_disease(df)
    ]
    print(f'NAFLD (FLI >= {threshold}):', len(nafld_fli))

    nafld_diagnosed = df[
        df.fatty_liver & non_alcoholic(df) & no_liver_disease(df)
    ]
    print('NAFLD (diagnosed with FLD):', len(nafld_diagnosed))


def main():
    usfli()
    print('-' * 40)
    fli()


if __name__ == '__main__':
    main()
