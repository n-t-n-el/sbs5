# SBS5

Code and data accompanying:

> Spisak N., de Manuel M., Przeworski M. **Collateral mutagenesis funnels multiple sources of DNA damage into a ubiquitous mutational signature.** *bioRxiv* (2025). [doi.org/10.1101/2025.08.28.672844](https://doi.org/10.1101/2025.08.28.672844)

The repository contains the processed data and Jupyter notebooks needed to reproduce the main-text and supplementary figures of the paper.

## Repository layout

```
sbs5/
├── data/                       # Processed mutation data and genomic features
│   ├── Abascal2021/            # 
│   ├── Ganz2024/               # 
│   ├── Ng2021/                 # 
│   ├── Olafsson2020/           # 
│   ├── Olafsson2023/           # 
│   ├── Wang2023/               # 
│   ├── Yoshida2020/            # 
│   └── Clusters/               # Inferred mutation clusters
└── notebooks/
    ├── fig1.ipynb              # Figure 1 (SBS5 across cell types)
    ├── fig3.ipynb              # Figure 3 (SBS5 vs. damage-specific signatures)
    ├── fig4.ipynb              # Figure 4 (co-localization with sites of DNA damage)
    ├── fig5.ipynb              # Figure 5 (repair dependence)
    ├── clusters_inference.ipynb            # Inference of mutation clusters
    ├── cell_lines.ipynb                    # SBS5 mutation rates in cell line experiments
    ├── features2windows.ipynb              # Genomic features along the genome
    ├── signet_attribution_windows.ipynb    # SigNet signature attribution 
    ├── signet_attribution_repair.ipynb     # Attribution in repair rate quantiles
    ├── sigprofiler_attribution.ipynb       # SigProfiler signature attribution
    ├── simulations.ipynb                   # SI simulations to test model solution
    ├── simulations.py                      # Stochastic simulation code
    └── tools.py                            # Shared utilities
```

### Requirements

- [SigNet](https://github.com/weghornlab/SigNet)
- [SigProfilerAssignment](https://github.com/AlexandrovLab/SigProfilerAssignment)

## Reproducing the figures

Each `fig*.ipynb` regenerates the corresponding figure from the files in `data/`. The other notebooks produce the intermediate objects they rely on (signature attributions, cluster calls, genomic features).

## Data sources

Raw sequencing data are not redistributed here. Accessions for all public datasets are listed in Table S3 of the paper.

## Contact

natanael.spisak@gmail.com, marcdemanuelmontero@gmail.com
