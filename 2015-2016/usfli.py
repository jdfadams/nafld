import numpy as np
import pandas as pd


non_alcoholic = lambda df: (((df.sex == 'female') & (df.drinks <= 1)) | ((df.sex == 'male') & (df.drinks <= 2)))


def load_data():

    def _load_xpt(basename):
        return pd.read_sas(basename).set_index('SEQN')

    df = pd.DataFrame()

    alcohol = _load_xpt('ALQ_I.XPT')
    df['drinks'] = alcohol['ALQ120U']  # different

    demographics = _load_xpt('DEMO_I.XPT')
    df['age'] = demographics['RIDAGEYR']
    df['mexican_american'] = demographics['RIDRETH3'].apply(lambda x: 1 if x == 1 else 0)
    df['non_hispanic_black'] = demographics['RIDRETH3'].apply(lambda x: 1 if x == 4 else 0)
    df['sex'] = demographics['RIAGENDR'].apply(lambda x: 'male' if x == 1 else 'female')

    hepatitis = _load_xpt('HEQ_I.XPT')
    df['hep_b'] = hepatitis['HEQ010'].apply(lambda x: x == 1)
    df['hep_c'] = hepatitis['HEQ030'].apply(lambda x: x == 1)

    medical_conditions = _load_xpt('MCQ_I.XPT')
    df['liver_cancer'] = (
        medical_conditions['MCQ230A'].apply(lambda x: x == 22) |
        medical_conditions['MCQ230B'].apply(lambda x: x == 22) |
        medical_conditions['MCQ230C'].apply(lambda x: x == 22) |
        medical_conditions['MCQ230D'].apply(lambda x: x == 22)  # different
    )
    df['liver_condition'] = medical_conditions['MCQ160L'].apply(lambda x: x == 1)  # different

    biochemistry = _load_xpt('BIOPRO_I.XPT')
    df['ggt'] = biochemistry['LBXSGTSI']

    body = _load_xpt('BMX_I.XPT')
    df['bmi'] = body['BMXBMI']
    df['waist_circumference'] = body['BMXWAIST']

    glucose = _load_xpt('GLU_I.XPT')
    df['glucose'] = glucose['LBXGLU']

    insulin = _load_xpt('INS_I.XPT')
    df['insulin'] = insulin['LBDINSI']

    triglycerides = _load_xpt('TRIGLY_I.XPT')
    df['triglycerides'] = triglycerides['LBXTR']

    return df


def usfli(threshold=30):
    df = load_data()

    columns = [
        'drinks', 'hep_b', 'hep_c',
        'liver_cancer', 'liver_condition',
        'age', 'sex',
        'mexican_american', 'non_hispanic_black',
        'ggt', 'insulin', 'glucose',
        'waist_circumference',
    ]
    df = df[columns]
    print('Number of rows:', len(df))

    df = df[df.liver_condition]
    print('Number of rows with liver condition:', len(df))

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
        (df.usfli >= threshold) &
        non_alcoholic(df) &
        ~(df.hep_b | df.hep_c | df.liver_cancer)
    ]
    print(f'NAFLD (USFLI >= {threshold}):', len(nafld_usfli))


def fli(threshold=60):
    df = load_data()

    columns = [
        'drinks', 'hep_b', 'hep_c',
        'liver_cancer', 'liver_condition',
        'sex',
        'ggt', 'triglycerides',
        'bmi', 'waist_circumference',
    ]
    df = df[columns]
    print('Number of rows:', len(df))
    df.to_csv('all_data.csv')

    df = df[df.liver_condition]
    print('Number of rows with liver condition:', len(df))

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
        (df.fli >= threshold) &
        non_alcoholic(df) &
        ~(df.hep_b | df.hep_c | df.liver_cancer)
    ]
    print(f'NAFLD (FLI >= {threshold}):', len(nafld_fli))


def main():
    print('2015-2016')
    print('-' * 40)
    usfli()
    print('-' * 40)
    fli()


if __name__ == '__main__':
    main()
