
# BUQEYE Guide to Emulators in Nuclear Physics (Frontiers Review)

Supplemental Material for "BUQEYE Guide to Emulators in Nuclear Physics"

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

The following flags are helpful:

* `--no-browser` if you are developing in a supported IDE (VS Code)
* `--no-watch-inputs` if you don't want the docs updating after every save, which is helpful if it takes a long time to build
* `--port` Choose a default port (e.g., `1234`) so that your browser link will not change. This is not necessary because a default is set in the `_quarto.yml` file.
* Rather than specify `docs` you can instead use a file name to only render that file.
