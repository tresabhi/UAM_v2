# UAM_v2

Getting Started (ubuntu):

```
mkdir folder_name
```

```
cd folder_name
```

```
git clone https://github.com/chief-khanman/UAM_v2.git
```

```
cd folder_name/UAM_v2
```

```
conda config --add channels conda-forge
```

```
conda env create -f environment.yml
```

```
conda activate AAM_AMOD
```

You will want to activate this conda environment anytime you want to run somthing.

If you are using VScode set your environment in the bottoom right of your screen to make running files easier

main.py: Runs simulation with basic UAV's

train_single_agent.py: Runs simulation for with one agent and specified number of basic uav's

May be required if this was missed in conda env create:

```
conda install conda-forge::stable-baselines3
```

train_multi_agent.py: All learning agents

May be required if this was missed in conda env create:

```
pip install pettingzoo
```
