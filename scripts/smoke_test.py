"""
Quick smoke test that:
1) runs the hello_torch sanity script
2) runs XGBoost baseline on a tiny synthetic classification task
3) logs a row to results/run_log.csv
"""
import subprocess, sys

def run(cmd):
    print(f"\n$ {' '.join(cmd)}\n")
    subprocess.check_call(cmd)

def main():
    run([sys.executable, "-m", "scripts.hello_torch"])
    run([sys.executable, "-m", "scripts.run_xgb_on_synth", "--task_type", "cls", "--rule", "blobs",
         "--n_support", "64", "--n_query", "64", "--n_features", "8", "--classes", "3"])
    print("\nAll good! Check results/run_log.csv for the logged row.\n")

if __name__ == "__main__":
    main()