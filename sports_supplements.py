"""Analiza e suplementeve sportive sipas kërkesave të Projektit 1."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_preparation import prepare_data
from regression_analysis import TARGET, fit_models

VARIABLES = ["popularity", "citations", "mean_studies"]


def show_plots(data, residuals):
    """Gjeneron tre figurat e dokumentimit me të njëjtat të dhëna."""
    sns.set_theme(style="whitegrid")

    # Figura 1: tre histograme dhe tre boxplot-e, në shkallën origjinale.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for j, col in enumerate(VARIABLES):
        sns.histplot(data=data, x=col, bins=12, ax=axes[0, j])
        sns.boxplot(data=data, y=col, ax=axes[1, j])
    fig.suptitle("Shpërndarjet dhe vlerat ekstreme: një rresht për suplement")
    fig.tight_layout()
    fig.canvas.manager.set_window_title("Figura 1 - Histogramet dhe boxplot-et")

    # Figura 2: lidhjet me secilin parashikues veçmas, pas ln(1+x).
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col in zip(axes, ["log_citations", "log_mean_studies"]):
        sns.regplot(data=data, x=col, y=TARGET, ci=None, ax=ax)
        ax.set_title("Lidhje përshkruese; pa interpretim shkakor")
    fig.tight_layout()
    fig.canvas.manager.set_window_title("Figura 2 - Scatter plot-et")

    # Figura 3: mbetjet e modelit M2 dhe shpërndarja e tyre.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.scatterplot(data=residuals, x="fitted", y="residual", ax=axes[0])
    axes[0].axhline(0, color="black", linestyle="--")
    axes[0].set(title="Mbetjet kundrejt parashikimeve: M2",
                xlabel="Vlera e parashikuar e ln(1+P)", ylabel="Mbetja")
    sns.histplot(data=residuals, x="residual", bins=12, ax=axes[1])
    axes[1].set(title="Shpërndarja e mbetjeve: M2", xlabel="Mbetja",
                ylabel="Numri i suplementeve")
    fig.tight_layout()
    fig.canvas.manager.set_window_title("Figura 3 - Analiza e mbetjeve")
    plt.show()


def build_interpretation(models, audit, flags):
    simple, multiple = models["M1_simple"], models["M2_multiple"]
    a, b = simple.params, multiple.params
    outliers = ", ".join(f"{col}: {int(flags[col + '_IQR_flag'].sum())}" for col in VARIABLES)
    text = f"""PYETJA KËRKIMORE
A lidhet popullariteti me citimet dhe mesataren e studimeve për pretendim?

HIPOTEZAT
H1: suplementet me më shumë citime kanë popullaritet më të lartë.
H2: suplementet me mesatare më të lartë të studimeve kanë popullaritet më të lartë.

PASTRIMI
Skedari: {audit['raw_rows']} rreshta dhe {audit['raw_columns']} kolona.
Rreshta të pavlefshëm të hequr: {audit['invalid_rows_removed']}.
Dublikata identike të hequra: {audit['exact_duplicates_removed']}.
Pretendime të ruajtura: {audit['valid_claim_rows']}; suplemente: {audit['supplements']}.
Kolona bosh: {len(audit['empty_columns'])}.
Vlera ekstreme sipas kufirit 1.5 IQR: {outliers}.
Vlerat ekstreme ruhen; madhësia e tyre nuk provon gabim.

MODELET
Y = ln(1+popullariteti); X1 = ln(1+citimet); X2 = ln(1+mesatarja e studimeve).
M1: Y_hat = {a['const']:.4f} + {a['log_citations']:.4f} X1
R2 = {simple.rsquared:.4f}: shpjegon {100 * simple.rsquared:.2f}% të variacionit të Y.
M2: Y_hat = {b['const']:.4f} + {b['log_citations']:.4f} X1 + {b['log_mean_studies']:.4f} X2
R2 = {multiple.rsquared:.4f}: shpjegon {100 * multiple.rsquared:.2f}% të variacionit të Y.
Rritja e R2: {100 * (multiple.rsquared - simple.rsquared):.2f} pikë përqindjeje.

INTERPRETIMI
Në M1, rritja 1% e citimeve+1 lidhet afërsisht me {a['log_citations']:.4f}% rritje të popullaritetit+1.
Në M2, rritja 1% e citimeve+1 lidhet afërsisht me {b['log_citations']:.4f}% rritje të popullaritetit+1, kur studimet mbahen konstante.
Në M2, rritja 1% e mesatares së studimeve+1 lidhet afërsisht me {b['log_mean_studies']:.4f}% rritje të popullaritetit+1, kur citimet mbahen konstante.
Intercepti është parashikimi i Y kur parashikuesit janë zero; këto raste janë jashtë kampionit.
Mbetja = Y - Y_hat. Pozitivja tregon nënvlerësim; negativja mbivlerësim.
Kontrolloni Figurën 3 për shpërndarjen rreth zeros dhe mbetjet e mëdha.

PËRFUNDIMI PËR KAMPIONIN E DOKUMENTIMIT
Koeficientët pozitivë përputhen me drejtimin e të dy hipotezave.
Shtimi i mesatares së studimeve sjell vetëm një rritje të vogël të R2.
Këto janë lidhje përshkruese, jo vërtetim statistikor i hipotezave ose shkakësi.

BURIMI
Joakim Arvidsson, Sports Supplements, Kaggle:
https://www.kaggle.com/datasets/joebeachcapital/sports-supplements
"""
    return text


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=project / "Sports Supplements.csv")
    args = parser.parse_args()
    claims, data, dictionary, missing, flags, rejected, audit = prepare_data(args.data)
    models, comparison, coefficients, residuals = fit_models(data)

    # Tabela 3 e dokumentimit: vetëm tre variablat e analizuara.
    descriptive = data[VARIABLES].agg(["mean", "median", "std", "min", "max"]).T
    tables = {
        "Tabela 2 - Klasifikimi Stevens": dictionary,
        "Vlerat e munguara": missing.rename_axis("column").reset_index(),
        "Vlerat ekstreme (True = jashtë kufijve IQR)": flags.loc[flags.filter(like="_IQR_flag").any(axis=1)],
        "Tabela 3 - Statistikat përshkruese": descriptive.rename_axis("variable").reset_index(),
        "Tabela 4 - Krahasimi i modeleve": comparison,
        "Tabela 5 - Koeficientët (M1 dhe M2)": coefficients,
        "Mbetjet e modelit M2": residuals,
    }
    structure = pd.DataFrame({"Përshkrimi": ["Rreshta origjinalë", "Kolona origjinale", "Suplemente për regresion"],
                              "Vlera": [audit["raw_rows"], audit["raw_columns"], len(data)]})
    print("\nTabela 1 - Struktura e të dhënave")
    print(structure.to_string(index=False))
    for name, table in tables.items():
        print(f"\n{name}")
        print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    text = build_interpretation(models, audit, flags)
    print(text)
    print("Grafikët hapen në tri dritare. Mbyllini dritaret për të përfunduar ekzekutimin.")
    show_plots(data, residuals)


if __name__ == "__main__":
    main()
