# ArcaNN Installation Guide #

## Installation on Machines with Internet Access ##

To install `ArcaNN`, follow these steps:

- **Clone or Download the Repository:**

Use the green `Code` button on the repository's main page to either clone or download the repository.
While it's recommended to keep a local copy of this repository on any computer that will be used for preparing, running, or analyzing the iterative training process, this is not mandatory.

- **Navigate to the Repository Folder:**

After downloading or cloning, navigate to the main folder of the repository.

- **Create a Python Environment:**

Create a new Python environment with all the required packages listed in the `tools/arcann_conda_linux-64_env.txt` file using the following command:

```bash
conda create --name <ENVNAME> --file tools/arcann_conda_linux-64_env.txt
```

- **Activate the Environment and Install the Package:**
Activate the environment using:

```bash
conda activate <ENVNAME>
```

Then, install `ArcaNN` as a Python module:

```bash
pip install .
```

- **Verify the Installation:**
To ensure that `ArcaNN` has been installed correctly, run the following command:

```bash
python -m arcann_training --help
```

This command should display the basic usage message of the code.

- **Optional:**
If you wish, you can delete the repository folder after installation is complete.

**Note:** Alternatively, you can install the program in "editable" mode with:

```bash
pip install -e .
```

This method allows any modifications to the source files to take effect immediately during program execution. It is only recommended if you plan to modify the source files and requires you to keep the repository folder on your machine.

## Installation on Machines without Internet Access ##

If your machine does not have access to the internet, follow these steps:

- **Download the Repository and Required Files:**

On a machine with internet access, download the `ArcaNN` repository. Then, copy the following files into a `tmp/` folder (outside of the repository):

```bash
arcann_training/tools/download_arcann_environment.sh
arcann_training/tools/arcann_conda_linux-64_env.txt
```

- **Download the Required Python Packages:**

In the `tmp/` folder, make the script executable and run it to download all the required Python packages:

```bash
chmod +x download_arcann_environment.sh
./download_arcann_environment.sh arcann_conda_linux-64_env.txt
```

This script will download all the necessary Python packages into a `arcann_conda_linux-64_env_offline_files/` folder and create a `arcann_conda_linux-64_env_offline.txt` file.

- **Transfer Files to the Offline Machine:**

Use rsync to transfer the downloaded packages and the ArcaNN repository to your offline machine:

```bash
rsync -rvu tmp/* USER@WORKMACHINE:/PATH/TO/INSTALLATION/FOLDER/.
rsync -rvu arcann_training USER@WORKMACHINE:/PATH/TO/INSTALLATION/FOLDER/.
```

- **Create the Python Environment on the Offline Machine:**

On your offline machine, create the required Python environment using the downloaded packages:

``bash
conda create --name <ENVNAME> --file arcann_conda_linux-64_env_offline.txt
``

- **Install ArcaNN:**

Finally, install ArcaNN as a Python module by following the installation steps provided earlier:

```bash
pip install .
```
