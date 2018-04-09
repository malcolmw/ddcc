# ddcc
Double-difference cross-correlation code using pyasdf

# Installing

## 1 - Install Anaconda or Miniconda
If you haven't already, install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://conda.io/miniconda.html).

## 2 - Create and activate a new conda environment with Python2.7

```bash
@maple>> conda create --name py27 python=2.7
@maple>> source activate py27
(py27) @maple>>
```

## 3 - Install some *pyasdf* dependencies
```bash
(py27) @maple>> conda install -c conda-forge obspy colorama pytest pip flake8 dill prov
```

## 4 - Install *mpi4py*
First make sure that your *MPI* environment is properly initialized - skipping this step *should* (I haven't tested this) be fine if *MPI* is not already provided for you.
```bash
(py27) @maple>> source /path/to/openmpi/setup.sh
```
Use *pip* to install *mpi4py*; it will build against your environment's *MPI*, but *conda* won't:
```bash
(py27) @maple>> pip install mpi4py
```

## 5 - Install *h5py* with parallel features enabled
Two scenarios are possible here: i) the machine you are using provides a build of *HDF5* with parallel features enabled - follow step 5a - or ii) the machine you are using does not provide a build of *HDF5-parallel* (*HDF5* with parallel features enabled) - follow step 5b.

### 5a - Link *h5py* to prebuilt *HDF5-parallel*
Make sure that your parallel build of *HDF5* is initialized in your environment:
```bash
(py27) @maple>> source /path/to/paralle/hdf5/setup.sh
```
Build and install *h5py* using *pip*. Make sure that *pip* uses the *mpicc* compiler (set the **CC** environment variable appropriately) and set the **HDF5_MPI** environment variable to **"ON"**
```bash
(py27) @maple>> export CC=mpicc
(py27) @maple>> export HDF5_MPI="ON"
(py27) @maple>> pip install --no-binary=h5py --no-deps h5py
```

### 5b - Build *HDF5-parallel* features enabled, then link *h5py*
First download the [source code](https://www.hdfgroup.org/downloads/) for the latest version of *HDF5*. Then enter the top-level directory of the download and configure the installation to include parallel features and a shared library. By default *HDF5* will install in the source code directory; use the *--prefix* option to specify another location.
```bash
(py27) @maple>> cd hdf5-1.10.1
(py27) @maple>> ./configure --enable-parallel --enable-shared [--prefix=/path/to/desired/install/dir]
(py27) @maple>> make
(py27) @maple>> make install
```

After installing *HDF5-parallel*, [download](https://pypi.python.org/pypi/h5py), build with link to *HDF5-parallel*, and install *h5py*.
```bash
(py27) @maple>> export CC=mpicc
(py27) @maple>> python setup.py configure --mpi --hdf5=/path/to/parallel/hdf5
(py27) @maple>> pip install -e .
```

## 6 - Install *pandas*
*pytables* is a dependency of *pandas* that depends on *HDF5*. We want to make sure that *pytables* links against the *HDF5-parallel* just built.
```bash
(py27) @maple>> export HDF5_DIR=/path/to/hdf5/parallel
(py27) @maple>> pip install --no-binary=tables --no-deps tables
```
*numexpr* is another of dependency of *pandas*, so install it before install *pandas*.
```bash
(py27) @maple>> pip install --no-binary=numexpr --no-deps numexpr
(py27) @maple>> pip install --no-binary=pandas --no-deps pandas
```

## 7 - Install *pyasdf*
Last, install *pyasdf*:
```bash
(py27) @maple>> pip install pyasdf
```
