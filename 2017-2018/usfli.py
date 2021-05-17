import numpy as np
import pandas as pd


def main():
    alcohol = pd.read_sas('ALQ_J.XPT')
    hepatitis = pd.read_sas('HEQ_J.XPT')
    medical_conditions = pd.read_sas('MCQ_J.XPT')

    biochemistry = pd.read_sas('BIOPRO_J.XPT')
    body = pd.read_sas('BMX_J.XPT')
    demographics = pd.read_sas('DEMO_J.XPT')
    glucose = pd.read_sas('GLU_J.XPT')
    insulin = pd.read_sas('INS_J.XPT')

    lengths = {
        'alcohol': len(alcohol),
        'hepatitis': len(hepatitis),
        'medical_conditions': len(medical_conditions),
        'biochemistry': len(biochemistry),
        'body': len(body),
        'demographics': len(demographics),
        'glucose': len(glucose),
        'insulin': len(insulin),
    }
    print('Numbers of rows:', lengths)

    df = pd.DataFrame()
    df['drinks'] = alcohol['ALQ130']
    df['hep_b'] = hepatitis['HEQ010'].apply(lambda x: x == 1)
    df['hep_c'] = hepatitis['HEQ030'].apply(lambda x: x == 1)
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
    df['sex'] = demographics['RIAGENDR'].apply(lambda x: 'male' if x == 1 else 'female')
    df['age'] = demographics['RIDAGEYR']
    df['mexican_american'] = demographics['RIDRETH3'].apply(lambda x: 1 if x == 1 else 0)
    df['non_hispanic_black'] = demographics['RIDRETH3'].apply(lambda x: 1 if x == 4 else 0)
    df['ggt'] = biochemistry['LBXSGTSI']
    df['insulin'] = insulin['LBDINSI']
    df['glucose'] = glucose['LBXGLU']
    df['waist_circumference'] = body['BMXWAIST']
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
    nafld_fli = df[
        (df.usfli >= 60) &
        (((df.sex == 'female') & (df.drinks <= 1)) | ((df.sex == 'male') & (df.drinks <= 2))) &
        (~(df.hep_b | df.hep_c | df.liver_cancer | df.other_liver_condition))
    ]
    nafld_diagnosed = df[
        df.fatty_liver &
        (((df.sex == 'female') & (df.drinks <= 1)) | ((df.sex == 'male') & (df.drinks <= 2))) &
        (~(df.hep_b | df.hep_c | df.liver_cancer | df.other_liver_condition))
    ]
    print('Number of rows with USFLI:', len(df))
    print('NAFLD (according to USFLI):', len(nafld_fli))
    print('NAFLD (diagnosed with FLD):', len(nafld_diagnosed))


if __name__ == '__main__':
    main()
