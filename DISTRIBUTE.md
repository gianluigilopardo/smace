## Ref:

* https://python-packaging.readthedocs.io/en/latest/
* https://packaging.python.org/tutorials/packaging-projects/

### files

* README.md        : simple README for the packaging
* LICENCE.md       : license (what else ?)
* setup.*          : setup script + config
* smace            : the library -> distributed in the pip package
* DISTRIBUTE.md    : how to build the package
* conda/smace.yaml : conda env to ease the build and test


### building steps

* install conda and all dependancies (linux/macosx)

```
conda env update -f ./conda/smace.yaml
conda activate smace
```

* test/fix the files against python coding conventions

```
flake8 smace
isort smace
black smace
```

* IF NECESSARY (software update), change the version number in **setup.cfg** and push to github

* compile the package

```
python setup.py bdist_wheel
```

* verify the content of the package

```
unzip -l dist/smace-0.0-py3-none-any.whl
```

* check the install in a empty environment

```
conda deactivate
conda env  -n smace-test python=3.9
conda activate smace-test
python -m pip install dist/smace-0.0-py3-none-any.whl
```

```
unset PYTHONPATH
mkdir ~/tmp/empty_dir
cd ~/tmp/empty_dir
python -c 'from smace.decisions import DM;'
```

* clean the env

```
rm -fr dist build  smace.egg-info
```

### publish the package on pypi.org

#### test the package on test.pypi.org

* logon https://test.pypi.org/account/register/
* create an API token to upload packages on command line (https://test.pypi.org/manage/account/token/)
* upload the package with this token (use **__token__** as username)

```
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/smace-0.0-py3-none-any.whl
```

```
USER   = __token__
PASSWD = pypi-AgENdGVzdC5weXBpLm9yZwIkZTA2MmFiMzItYWE0OC00MGU4LTkxMzctNWQyMDE3ZDc2MTk2AAIleyJwZXJtaXNzaW9ucyI6ICJ1c2VyIiwgInZlcnNpb24iOiAxfQAABiAQ8k2jsDpUQKqGyCNOwPYj55MS9eJvDfsmHD3PpYdXhA
```

* project is now at:  https://test.pypi.org/project/smace/0.0/
* test from distrib:

```
pip install -i https://test.pypi.org/simple/ smace
```

#### install on https://pypi.org

* if everything is OK, you can now install it on pypi.org to make it available to all
* you can save the token into **~/.pypirc** file:

```
[pypi]
  username = __token__
  password = pypi-AgEIcHlwaS5vcmcCJDFjZGIzYjM0LTQ2YTItNDYzZi1hZmI0LTZiMTVkM2E4NWNkMgACJXsicGVybWlzc2lvbnMiOiAidXNlciIsICJ2ZXJzaW9uIjogMX0AAAYgNlahB-dQuiQk12jjlPtNLCWaYE0q_XqGrPsavMM6sg8
```

* upload with
```
python -m twine upload --verbose dist/smace-0.0-py3-none-any.whl
```

* package is available at: https://pypi.org/project/smace/0.0/
* test with:

* pip install should now work for everyone

```
pip install smace
```
