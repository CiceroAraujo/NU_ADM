import os
import yaml
parent_dir=os.getcwd()
with open(parent_dir+'/inputs/finescale.yml', 'r') as f:
    finescale_inputs = yaml.safe_load(f)
