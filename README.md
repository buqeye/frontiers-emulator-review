
# BUQEYE Guide to Projection-Based Emulators in Nuclear Physics (Frontiers in Physics Article)

All examples discussed in our Frontiers in Physics Article "BUQEYE Guide to Projection-Based Emulators in Nuclear Physics" and more are available here as interactive, open-source Python code so that practitioners can readily adapt projection-based emulators for their own work.


## Documentation

The documentation is written using `Quarto`.
Download the latest version of `Quarto` from [here](https://quarto.org/) and then spin it up in development using

```bash
conda env create -f environment.yml
conda activate frontiers-emulator-env
export PYTHONPATH=$PYTHONPATH:`pwd`
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
@inproceedings{Drischler:2022ipa,
    author = "Drischler, C. and Melendez, J. A. and Furnstahl, R. J. and Garcia, A. J. and Zhang, Xilin",
    title = "{BUQEYE Guide to Projection-Based Emulators in Nuclear Physics}",
    eprint = "2212.04912",
    archivePrefix = "arXiv",
    primaryClass = "nucl-th",
    month = "12",
    year = "2022"
}
```
