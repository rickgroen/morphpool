# We first compile the CUDA operations for the pooling and unpooling.
cd extensions || exit
python setup.py install
# Then test whether the operations are correctly compiled and yield expected results.
cd ..
pytest pooling_tests.py
pytest unpooling_tests.py