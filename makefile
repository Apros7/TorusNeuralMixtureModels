MAIN_FOLDER := $(shell git rev-parse --show-toplevel)

install:
	python3 -m pip install -r requirements.txt

# REPRODUCING RESULTS FOR REPORT
# In all cases, the parameters can be adjusted in the file referred in the makefile if necessary.

## Synthetic data:

best_lr_nce_tg:
	cd $(MAIN_FOLDER) && python3 src/results/NCE_lr.py

best_lr_nce_tgmm:
	cd $(MAIN_FOLDER) && python3 src/results/NCE_lr_tgmm.py

sm_vs_nce_viz:
	cd $(MAIN_FOLDER) && python3 src/results/sm_vs_nce_viz.py

sm_vs_nce_boxplot:
	cd $(MAIN_FOLDER) && python3 src/results/sm_vs_nce_boxplot.py

nce_loss:
	cd $(MAIN_FOLDER) && python3 src/results/see_nce_loss.py

nce_loss_tgmm:
	cd $(MAIN_FOLDER) && python3 src/results/loss_over_time.py

see_preds_dist:
	cd $(MAIN_FOLDER) && python3 src/results/preds_dist.py

## Real data

real_best_lr_nce_tgmm:
	cd $(MAIN_FOLDER) && python3 src/results/real_nce_lr_tgmm.py

real_tgmm_boxplot:
	cd $(MAIN_FOLDER) && python3 src/results/real_tgmm_boxplot.py

real_tgmm_viz:
	cd $(MAIN_FOLDER) && python3 src/results/real_phi_nce_plot.py

real_nce_loss:
	cd $(MAIN_FOLDER) && python3 src/results/real_nce_loss_tgmm.py

real_nce_loss_fitting_tgmm:
	cd $(MAIN_FOLDER)

real_see_preds_dist:
	cd $(MAIN_FOLDER) && python3 src/results/real_preds_dist.py

