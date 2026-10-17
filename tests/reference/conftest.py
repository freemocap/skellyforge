"""Explicit selection for local reference-data tests; no implicit downloads."""
def pytest_addoption(parser):
    parser.addoption('--dataset', choices=('test_data', 'sample_data'), default='test_data')
    parser.addoption('--parquet', help='Use this prepared recording instead of default discovery')
    parser.addoption('--sensor-group', help='Select a recording sensor group')
