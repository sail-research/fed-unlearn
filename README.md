# Structure of the repository
- result_sample: This folder contains sample results and the usage file.
- results: This folder will stores all information when running an experiment.
- unlearn: This folder contains the implementation of the unlearning methods.
- utils: This folder contains the utility files for Federated Learning.

# How to reproduce the experiment results:
- Step 1: Go to config.py file to config the experimen. Important factors include: dataset, num_rounds, num_unlearn_rounds, num_post_training_rounds, num_onboarding_rounds and poisoned_percent. Note that we currently fix the num_clients to 5.
- Step 2: Create a folder name "models" in folder results.
- Step 3: In config.py, set is_onboarding to False and run case0.py. Then, run case1.py, case2.py, case3.py, case4.py and case5.py.
- Step 4: In config.py, set is_onboarding to True and run case1.py, case2.py, case3.py, case4.py and case5.py again.
- Step 5: copy the generated pkl files in the results folder into the folder result_sample/with_onboarding.
- Step 6: Adjust the configuration and run cells in the usage.ipynb in the result_sample folder.