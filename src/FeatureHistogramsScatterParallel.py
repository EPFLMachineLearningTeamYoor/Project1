# run: j=0;i=0;while (( $i < 30 )); do echo $i; python a.py $i & let j=j+1;if (( $j == 4 )); then wait; echo "joined"; j=0;fi;let i=i+1;done

from sys import argv
from scripts.plot_histograms import *
from scripts import proj1_helpers
from scripts.helpers import get_header, impute_with_mean

n = str(argv[1])
print("Running iteration %s" % n)

train_path = '../data/train.csv'

print("Loading data")
header = get_header(train_path)
y, X_tr, ids_tr = proj1_helpers.load_csv_data(train_path)
X_tr_imp = impute_with_mean(X_tr)

for i, (a, b) in enumerate(get_ranges(X_tr_imp.shape[1], 6)):
    if str(i) == n:
        print("Running hist")
        do_hist_scatter(X_tr_imp, y, header, idx_x = a, idx_y = b, bins = 15)
        print("Saving files...")
        savefig('hist-scatter-tr-imp-%d' % i)
    else:
        pass
