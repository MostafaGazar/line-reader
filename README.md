Readings:
- Sequence Modeling With CTC https://distill.pub/2017/ctc/

Setup local env:
1. Create conda env `conda env create -f environment.yml –n line-reader`
2. Update env file `conda env export > environment.yml`
3. Update env file after installing some packages `conda env update –f environment.yml`

Enable jupyter lab git extension:
```
$ jupyter labextension install @jupyterlab/git
$ jupyter serverextension enable --py jupyterlab_git
```









