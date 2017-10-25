from scripts import implementation
from scripts import feature_processing
from scripts import model_logistic

if feature_processing.test_all():
    print("Feature processing OK")

if implementation.test_all():
    print("Implementation OK")

if model_logistic.test_all():
    print("Logistic regression OK")
