
1. Download https://github.com/shrimai/Focused-Attention-Improves-Document-Grounded-Generation/blob/main/patch/generation_utils.py

2. Apply the diff file as patch to the file downloaded above.
```bash
 patch generation_utils.py on_doha.patch 
```

3. Use the new file to cover the original Transformers generation_utils.py. For example: 
```bash
cp patch/generation_utils.py [PATH TO YOUR CONDA ENV]/lib/python3.7/site-packages/transformers/generation_utils.py
```
