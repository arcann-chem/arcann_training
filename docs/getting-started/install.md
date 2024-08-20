# Installation #

## If your machine has access to the internet ##

To use `ArcaNN`, you first need to clone or download this repository using the green `Code` button in the top right corner of the repository's main page.
We recommend keeping a local copy of this repository on each computer that will be used to prepare, run, or analyze any part of the iterative training process, though it is not mandatory.

After the download is complete, go to the repository's main folder.
Create a Python environment with all the required packages indicated in the `tools/arcann_conda_linux-64_env.txt` file using the following command:

```bash
conda create --name <ENVNAME> --file tools/arcann_conda_linux-64_env.txt
```

Load this environment with `conda activate <ENVNAME>` and run:

```bash
pip install .
```

That's it! `ARCANN` has now been installed as a module of your `<ENVNAME>` Python environment.

To verify the installation, you can run:

```bash
python -m arcann_training --help
```

This command should print the basic usage message of the code.

After installation, the repository folder can be deleted if you wish.

**Note:** You can also install the program with:

```bash
pip install -e .
```

This way, any modifications to the source files will immediately take effect during program execution.
This method is **only** recommended if you plan to modify the source files, and it requires you to **keep** the repository folder on your machine.

## If your machine does not have access to the internet ##

Download the repository onto a machine that has access to the internet and to your offline working computer.
In a `tmp/` folder (outside of the repository), copy the `arcann_training/tools/download_arcann_environment.sh` and `arcann_training/tools/arcann_conda_linux-64_env.txt` files, then run

```bash
chmod +x download_arcann_environment.sh ; ./download_arcann_environment.sh arcann_conda_linux-64_env.txt
```

This script will download all the required Python packages into a `arcann_conda_linux-64_env_offline_files/` folder and create a `arcann_conda_linux-64_env_offline.txt` file.
Then upload this to the working computer along with the `arcann_training` repository:

```bash
rsync -rvu tmp/* USER@WORKMACHINE:/PATH/TO/INSTALLATION/FOLDER/.
rsync -rvu arcann_training USER@WORKMACHINE:/PATH/TO/INSTALLATION/FOLDER/.
```

Now you can simply create the required Python environment:

```bash
conda create --name <ENVNAME> --file arcann_conda_linux-64_env_offline.txt
```

and then install `ArcaNN` as a Python module as detailed above (with pip).

<div id="machine"></div>
