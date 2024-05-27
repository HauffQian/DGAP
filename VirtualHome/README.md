


### Execution
Run the program by executing `sh scripts/inference.sh` in the behavior_cloning directory.

Control Flags:
```python
if_gpt = True   # Whether to use the GPT interactive interface for task planning
if_exe_all_action = True  # Whether to execute the complete plan after generation, as opposed to generating and executing step-by-step
```



Before execution, add your API key to gpt_policy.py:
```
api_key = [api-key1, api-key2]
api_key_num = 2
```

### Get the VirtualHome Simulator
The VirtualHome exectuable file we used can be downloaded from [here](https://www.dropbox.com/s/xxfm38fvttlo34m/virtualhome.zip?dl=0). Put it under `./virtualhome`.


Relevant Code Directories
```
inference.sh    Stores some control parameters
gpt_policy.py   File for GPT-related interactive interfaces
interactive_interface_fn   Main process function
/checkpoint/LID-Text/interactive_eval  Stores the log files of the execution results

```

for Virtual