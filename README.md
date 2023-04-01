
# BUQEYE Guide to Projection-Based Emulators in Nuclear Physics (Frontiers in Physics Article)

The [BUQEYE collaboration](https://buqeye.github.io/) presents a pedagogical introduction to projection-based, reduced-order emulators for applications in low-energy nuclear physics.
All examples discussed in our Frontiers in Physics Article "BUQEYE Guide to Projection-Based Emulators in Nuclear Physics" ([arXiv:2212.04912](https://arxiv.org/abs/2212.04912)) and more are available here as interactive, open-source Python code so that practitioners can readily adapt projection-based emulators for their own work.


## Documentation

The documentation is written using `Quarto`.
Download the latest version of `Quarto` from [here](https://quarto.org/) and then spin it up in development using

```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:`pwd`
quarto preview docs
```

Alternatively, you can spin it up using the management system `conda`:
```bash
conda env create -f environment.yml
conda activate frontiers-emulator-env
pip3 install .
quarto preview docs
```

The following flags are helpful:

* `--no-browser` if you are developing in a supported IDE (VS Code)
* `--no-watch-inputs` if you don't want the docs updating after every save, which is helpful if it takes a long time to build
* `--port` Choose a default port (e.g., `1234`) so that your browser link will not change. This is not necessary because a default is set in the `_quarto.yml` file.
* Rather than specify `docs` you can instead use a file name to only render that file.

## Cite this work

Please cite this repository as:

```bibtex
@article{Drischler:2022ipa,
    author = "Drischler, C. and Melendez, J. A. and Furnstahl, R. J. and Garcia, A. J. and Zhang, Xilin",
    title = "{BUQEYE Guide to Projection-Based Emulators in Nuclear Physics}",
    eprint = "2212.04912",
    archivePrefix = "arXiv",
    primaryClass = "nucl-th",
    month = "12",
    JOURNAL={Front. Phys.},      
    VOLUME={10},
    pages=92931,
    YEAR={2023},      
    URL={https://www.frontiersin.org/articles/10.3389/fphy.2022.1092931},       
    DOI={10.3389/fphy.2022.1092931},    
    note          = {supplemental, interactive Python code can be found on the companion website~\url{https://github.com/buqeye/frontiers-emulator-review}},
    ISSN={2296-424X}   
}
```

See also our published literature guide [Model reduction methods for nuclear emulators](https://inspirehep.net/literature/2049517).
