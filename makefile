MAIN_FOLDER := $(shell git rev-parse --show-toplevel)

install:
	python3 -m pip install -r requirements.txt

# REPRODUCING RESULTS FOR REPORT
# In all cases, the parameters can be adjusted in the file refered in the makefile if necessary.

## Synthetic data:

make best_lr_nce_tg:
	cd $(MAIN_FOLDER) && python3 src/results/NCE_lr.py

make best_lr_nce_tgmm:
	cd $(MAIN_FOLDER) && python3 src/results/NCE_lr_tgmm.py

# make sm_vs_tg_visualization

make sm_vs_tg_boxplot:
	cd $(MAIN_FOLDER) && python3 src/results/sm_vs_tg_boxplot.py

make nce_loss:
	cd $(MAIN_FOLDER) && python3 src/results/see_nce_loss.py

# make nce_loss_tgmm:
# 	cd $(MAIN_FOLDER)

## Real data

