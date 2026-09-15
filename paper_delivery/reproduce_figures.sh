#!/usr/bin/env bash
# Regenerate every model figure of Sec. VI from the precomputed data in data/ (about 30 s in total).
set -e; cd "$(dirname "$0")"
python3 plots/plot_joint_dd.py        # Fig. 7   joint_dd_v8
python3 plots/plot_power_path_k.py    # Fig. 8   fig_power_path_0915   + Fig. 14 fig13_k_0915
python3 plots/plot_plos.py            # Fig. 9   fig_plos_state_0915   (closed form, no data needed)
python3 plots/plot_cir.py             # Fig. 10  cir_v2
python3 plots/plot_pdp.py             # Fig. 12  fig11_pdp_state_0915
python3 plots/plot_ds_doppler.py      # Fig. 13  fig12_ds_state_0914   + Fig. 15 fig14_doppler_state_0914
python3 plots/plot_tsta.py            # Fig. 16  fig_tsta_0915
ls -1 figures/*.pdf
