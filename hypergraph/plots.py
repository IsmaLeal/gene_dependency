import matplotlib.pyplot as plt
import pandas as pd

def found_fraction_vs_threshold():
    # Plotting settings
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 21})

    ns = [5, 6, 7, 8, 85, 9, 95, 96, 97, 98, 99]
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]
    fractions = []
    fractions_wo = []

    for idx, n in enumerate(ns):
        df = pd.read_csv(f'./../1_graph/results_{str(n)}.csv')
        df_filt = df[df['Significant']]
        fraction = 100 * len(df_filt) / len(df)
        df_wo = df[df['Number of genes'] > 6]
        fraction_wo = 100 * len(df_filt) / len(df_wo)
        fractions.append(fraction)
        fractions_wo.append(fraction_wo)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(thresholds, fractions, label='All complexes')
    ax.plot(thresholds, fractions_wo, label='Complexes with $\geq$ 6 nodes')
    ax.set_title('Fraction of found complexes per threshold')
    ax.set_xlabel('Thresholds')
    ax.set_ylabel('Percentage')
    ax.legend()

    plt.tight_layout()
    plt.show()


found_fraction_vs_threshold()
