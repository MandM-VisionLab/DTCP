# Diversity-Enhanced TCP (DTCP)

<p align="center">
  <img src="assets/dtcp_framework_whiteBG.png" width="900" alt="DTCP Framework"/>
</p>

This repository contains the official code for the paper **Interpretable Decision-Making for End-to-End Autonomous Driving**

## Contents
1. [Setup](#setup)
2. [Dataset](#dataset)
3. [Training](#training)
4. [Evaluation](#evaluation)

## Setup
Install CARLA 0.9.10.1:
```bash
mkdir carla && cd carla
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.10.1.tar.gz
tar -xf CARLA_0.9.10.1.tar.gz
tar -xf AdditionalMaps_0.9.10.1.tar.gz
rm CARLA_0.9.10.1.tar.gz AdditionalMaps_0.9.10.1.tar.gz
cd ..
```

Clone this repository and install dependencies:
```bash
git clone https://github.com/MandM-VisionLab/DTCP.git
cd DTCP
conda env create -f environment.yml --name DTCP
conda activate DTCP
```

Add the repository to your Python path:
```bash
export PYTHONPATH=$PYTHONPATH:PATH_TO_DTCP
```

## Dataset

We use the publicly available TCP dataset. Download it via [HuggingFace](https://huggingface.co/datasets/craigwu/tcp_carla_data) (reassemble the split archive with `cat tcp_carla_data_part_* > tcp_carla_data.zip`) or [Google Drive](https://drive.google.com/file/d/1HZxlSZ_wUVWkNTWMXXcSQxtYdT7GogSm/view?usp=sharing). The total dataset size is approximately 115 GB.


## Training
Set the dataset path in `DTCP/config.py`, then start training:
```bash
python DTCP/train.py --batch_size 32 --logdir /path/to/logdir --gpus num_gpus
```

## Evaluation
First, launch the CARLA server:
```bash
cd CARLA_ROOT
./CarlaUE4.sh --world-port=2000 -opengl
```

Then, set the required paths in `carla_eval.sh`.
Start the evaluation:
```bash
sh carla_eval.sh
```

## Citation
If you use this work, please cite:
If you use this work, please cite:
```bibtex
@inproceedings{mirzaie2025interpretable,
  title={Interpretable decision-making for end-to-end autonomous driving},
  author={Mirzaie, Mona and Rosenhahn, Bodo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={794--804},
  year={2025}
}
```

## Acknowledgements
This work is built upon several repositories:
- [TCP](https://github.com/OpenDriveLab/TCP)
- [Roach](https://github.com/zhejz/carla-roach)
- [CARLA Leaderboard](https://github.com/carla-simulator/leaderboard)
- [Scenario Runner](https://github.com/carla-simulator/scenario_runner)
