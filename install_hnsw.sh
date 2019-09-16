apt-get update && apt-get install -y --no-install-recommends \
    python-setuptools python-pip
pip3 install --upgrade pip3
pip3 install pybind11 numpy setuptools

git clone --depth 1 https://github.com/nmslib/hnswlib.git
cd hnswlib
cd python_bindings
python3 setup.py install
cd ../..
rm -r hnswlib
