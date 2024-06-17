
import sys

sys.path.insert(0,'.')

import os
import numpy as np
import pickle

def load_sample_data():
    directory = "src/data/phases_7"
    phases = np.empty((0, 7))
    stages = np.array([])
    for filename in sorted(os.listdir(directory)):
        if filename.startswith('sub-') and filename.endswith('_phases.txt'):
            sub_id = filename.split('-')[1].split('_')[0]
            phase_file = os.path.join(directory, filename)
            stage_filename = f'sub-{sub_id}_mainlysleep_sleep_stage.txt' if "mainlysleep" in filename else f'sub-{sub_id}_mainlywake_sleep_stage.txt'
            stage_file = os.path.join(directory, stage_filename)
            phase_data = np.loadtxt(phase_file)
            stage_data = np.loadtxt(stage_file)
            phases = np.concatenate((phases, phase_data))
            stages = np.concatenate((stages, stage_data))
    indices_to_keep = stages != 3
    filtered_stages = stages[indices_to_keep]
    filtered_phases = phases[indices_to_keep]
    return filtered_phases, filtered_stages



if __name__ == "__main__":
    phases, stages = load_sample_data()

    with open("src/data/real_data_phases.pickle", "wb") as file:
        pickle.dump(phases, file)
    with open("src/data/real_data_stages.pickle", "wb") as file:
        pickle.dump(stages, file)




