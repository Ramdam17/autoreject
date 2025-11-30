import os
import yaml
import subprocess
import sys

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs_initial')
SINGLE_SCRIPT = os.path.join(os.path.dirname(__file__), 'run_initial_single.py')

os.makedirs(LOGS_DIR, exist_ok=True)

def main():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    subset_configs = config.get('subset_configs', [])
    print(f"Total configs: {len(subset_configs)}")
    for i, cfg in enumerate(subset_configs, 1):
        name = cfg.get('name', f'config_{i}')
        log_path = os.path.join(LOGS_DIR, f'{name}.log')
        print(f"[{i}/{len(subset_configs)}] Running {name}...")
        cmd = [sys.executable, SINGLE_SCRIPT, '--config', name]
        with open(log_path, 'w') as log_file:
            subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        print(f"Log saved to: {log_path}")

if __name__ == '__main__':
    main()
