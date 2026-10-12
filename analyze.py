from pathlib import Path
import json
import numpy as np
import pandas as pd
p=Path(__file__).resolve().parent
rng=np.random.default_rng(42)
results={}
for model in ["CNN","LSTM"]:
    a=pd.read_csv(p/f"{model}_output_only_predictions.csv")
    b=pd.read_csv(p/f"{model}_input_output_predictions.csv")
    assert a[["prompt_sha256","true_model"]].equals(b[["prompt_sha256","true_model"]])
    a["difference"]=(b.true_model==b.predicted_model).astype(int)-(a.true_model==a.predicted_model).astype(int)
    # Preserve original prompt ordering for reproduction of the stored bootstrap.
    # The saved hash mapping has original prompt order; sort=False preserves it.
    delta=a.groupby("prompt_sha256",sort=False)["difference"].mean().values
    bootstrap=rng.choice(delta,size=(10000,len(delta)),replace=True).mean(1)
    results[model]={"difference":float(delta.mean()),"ci95":np.quantile(bootstrap,[.025,.975]).tolist()}
print(json.dumps(results,indent=2))
