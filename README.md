# Work with conda local env:
1. Create conda env `conda env create -f environment.yml`
2. Update env file `conda update --all` then  `conda env export --no-builds > environment.yml`
3. Update env file after installing some packages `conda env update`

## Generate a requirements.txt file:
`conda list -e > requirements.txt`

## Troubleshooting
* In case the new kernel didn't show up in Jupyter, you can add it manually `python -m ipykernel install --user --name keras`

# Work with virtualenv

## Create a new env
`virtualenv -p python3 .env`

## Install a Jupyter kernel
`ipython kernel install --user --name=.venv`

## Activate the env
`source .env/bin/activate`

## Deactivate the env
`deactivate`

# Enable jupyter lab git extension:
```
$ jupyter labextension install @jupyterlab/git
$ jupyter serverextension enable --py jupyterlab_git
```

