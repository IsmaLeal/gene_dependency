import pandas as pd

def get_genes(complex):
    '''
     Returns a list of different gene names within a complexes
     Example input: 'PRORP;TRMT10C;HSD17B10'
     Example output: ['PRORP', 'TRMT10C', 'HSD17B10']
    '''
    return complex.split(';')


df = pd.read_csv('../../datasets/humanComplexes.txt', delimiter='\t')

# Will select rows containing these substrings in their 'Cell line' value
substrings = ['T cell line ED40515',
              'mucosal lymphocytes',
              'CLL cells',
              'monocytes',
              'THP-1 cells',
              'bone marrow-derived',
              'monocytes, LPS-induced',
              'THP1 cells',
              'human blood serum',
              'human blood plasma',
              'plasma',
              'CSF',
              'human leukemic T cell JA3 cells',
              'erythrocytes',
              'peripheral blood mononuclear cells',
              'African Americans',
              'SKW 6.4',
              'BJAB cells',
              'Raji cells',
              'HUT78',
              'J16',
              'H9',
              'U-937',
              'Jurkat T',
              'NB4 cells',
              'U937',
              'early B-lineage',
              'T-cell leukemia',
              'lymphoblasts',
              'whole blood and lymph',
              'human neutrophil-differentiating HL-60 cells',
              'human peripheral blood neutrophils',
              'human neutrophils from fresh heparinized human peripheral blood',
              'human peripheral blood',
              'HCM',
              'liver-hematopoietic',
              'cerebral cortex',
              'human brain',
              'pancreatic islet',
              'human hepatocyte carcinoma HepG2 cells',
              'Neurophils',
              'H295R adrenocortical',
              'frontal cortex',
              'myometrium',
              'vascular smooth muscle cells',
              'Dendritic cells',
              'intestinal epithelial',
              'Primary dermal fibroblasts',
              'HK2 proximal',
              'brain pericytes',
              'HepG2',
              'HEK 293 cells, liver',
              'normal human pancreatic duct epithelial',
              'pancreatic ductal adenocarcinoma',
              'OKH cells',
              'cultured podocytes',
              'renal glomeruli',
              'VSMCs',
              'differentiated HL-60 cells',
              'SH-SY5Y cells',
              'frontal and entorhinal cortex',
              'SHSY-5Y cells',
              'hippocampal HT22 cells',
              'primary neurons',
              'neurons',
              'renal cortex membranes',
              'Kidney epithelial cells',
              'skeletal muscle cells',
              'Skeletal muscle fibers',
              'differentiated 3T3-L1',
              'brain cortex',
              'cortical and hippocampal areas',
              'human H4 neuroglioma',
              'Thalamus',
              'HISM',
              'pancreas',
              'RCC4',
              'C2C12 myotube',
              'XXVI muscle',
              'SH-SY5Y neuroblastoma',
              'HCC1143',
              'Hep-2',
              'PANC-1',
              'HEK293T cells',
              'HEK-293 cells',
              'heart',
              'epithelium',
              'kidney',
              'heart muscle',
              'central nervous system',
              'COS-7 cells',
              'ciliary ganglion',
              'striated muscle',
              'PC12',
              '293FR cells']
pattern = '|'.join(substrings)
# And rows whose 'Cell line' value is exactly one of these
only = ['muscle', '293 cells', 'brain', 'HEK 293 cells']

# Create Boolean mask selecting all the rows described
partial_mask = df['Cell line'].str.contains(pattern, case=False, na=False)
total_mask = df['Cell line'].isin(only)
mask = partial_mask | total_mask

# Obtain filtered dataframe
complexes_full = df[mask]

# Select specific columns
complexes = complexes_full[['ComplexID', 'ComplexName', 'Cell line', 'subunits(Gene name)', 'GO description', 'FunCat description']]

# Sort by 'Cell line'
complexes = complexes.sort_values(by=['Cell line'])

# Exclude 'complexes' including only one subunit/ gene
mask_mono = [len(complexes['subunits(Gene name)'].values[i].split(';')) > 1 for i in range(len(complexes))]
complexes = complexes.loc[mask_mono]

# Save as .csv file
complexes.to_csv('filtered_complexes.csv', index=False)
