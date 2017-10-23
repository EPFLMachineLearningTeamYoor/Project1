from scripts import implementation
from scripts import feature_processing

if feature_processing.test_all():
    print("Feature processing OK")

if implementation.test_all():
    print("Implementation OK")
