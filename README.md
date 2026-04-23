# Diversity-Enhanced TCP (DTCP)

<p align="center">
  <img src="assets/dtcp_framework_whiteBG.png" width="900" alt="DTCP Framework"/>
</p>

This repository contains the official code for the paper **Interpretable Decision-Making for End-to-End Autonomous Driving**


## Dataset

We use the publicly available TCP dataset. Download it via [HuggingFace](https://huggingface.co/datasets/craigwu/tcp_carla_data) (reassemble the split archive with `cat tcp_carla_data_part_* > tcp_carla_data.zip`) or [Google Drive](https://drive.google.com/file/d/1HZxlSZ_wUVWkNTWMXXcSQxtYdT7GogSm/view?usp=sharing). The total dataset size is approximately 115 GB.

## Acknowledgements
This work is built upon several repositories:
- [TCP](https://github.com/OpenDriveLab/TCP)
- [Roach](https://github.com/zhejz/carla-roach)
- [CARLA Leaderboard](https://github.com/carla-simulator/leaderboard)
- [Scenario Runner](https://github.com/carla-simulator/scenario_runner)


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
