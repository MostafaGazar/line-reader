In this project I finally switched to using `pipenv` instead of `conda`. I also tried `virtualenv` :|


First you need to install linuxbrew or macbrew then use to install `pipenv`

## Install pipenv
```
brew install pipenv
pipenv lock
```

## Add env to kernel
```
pipenv run python -m ipykernel install --user --name=line-reader
```

## Install/update dependencies
```
pipenv install --dev
```

## Install new dependency
```
pipenv install some_dependency
``` 

## Remove the environment
```
pipenv --rm
```

## To avoid switching between tensorflow gpu and cpu versions when moving my code between my local and the cloud I installed relevant version using pip instead
```
pipenv run pip install tensorflow==2.0.0-rc0
# Or
pipenv run pip install tensorflow-gpu==2.0.0-rc0
```