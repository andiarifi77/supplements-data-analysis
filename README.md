# Popullariteti i suplementeve sportive

Nga terminali i PyCharm, brenda dosjes së projektit:

```powershell
python -m pip install -r requirements.txt
python sports_supplements.py
```

Nëse paketat janë instaluar, mjafton komanda e dytë.

Tabelat, klasifikimi Stevens, mungesat, vlerat ekstreme, statistikat, koeficientët, R², mbetjet dhe interpretimi shfaqen në terminal.

Grafikët hapen automatikisht në tri dritare Matplotlib:

1. Tre histograme dhe tre boxplot-e, si Figura 1 e dokumentimit.
2. Dy scatter plot-e me vijat lineare, si Figura 2.
3. Mbetjet kundrejt parashikimeve dhe histogrami i tyre, si Figura 3.

Mbyllini dritaret e grafikëve për të përfunduar ekzekutimin. Në disa versione të PyCharm, grafikët mund të shfaqen në panelin Plots.

Programi nuk ruan tabela apo figura në skedarë dhe nuk ndryshon CSV-në origjinale.

- `data_preparation.py`: klasifikimi, pastrimi, grupimi dhe vlerat ekstreme.
- `regression_analysis.py`: regresioni i thjeshtë M1, i shumëfishtë M2 dhe mbetjet.
- `sports_supplements.py`: ekzekutimi, shfaqja e tabelave dhe figurave.

Përdoren 46 suplemente nga 96 rreshtat origjinalë, me transformimin ln(1+x), si në dokumentim. Tabelat shfaqin katër shifra dhjetore; llogaritjet përdorin saktësinë e plotë.

Burimi: https://www.kaggle.com/datasets/joebeachcapital/sports-supplements
