# Auditing Fairness under Unobserved Confounding 

This repository is the official implementation of [Auditing Fairness under Unobserved Confounding](https://arxiv.org/abs/2403.14713) (AISTATS 2024),
containing code to implement our bounds and perform semi-synthetic data experiments. If you find this repository useful or use this code in your research, please cite the following paper:

```bibtex
@article{byun2024auditing,
  title={Auditing Fairness under Unobserved Confounding},
  author={Byun, Yewon and Sam, Dylan and Oberst, Michael and Lipton, Zachary C and Wilder, Bryan},
  journal={arXiv preprint arXiv:2403.14713},
  year={2024}
}
```

## Quick Experiments 

To reproduce our results, the following command can be simply used:
``` 
python semi_synth.py
```

## Installation

To install requirements, setup a conda environment using the following commands:
``` 
conda create -n fairness python=3.11 pip 
conda activate fairness
pip install -r requirements.txt
```

## License 

This repository is licensed under the terms of the [MIT License](https://github.com/lasilab/inequity-bounds?tab=MIT-1-ov-file).

## Questions?

For more details, refer to the accompanying paper: [Auditing Fairness under Unobserved Confounding](https://arxiv.org/abs/2403.14713). 
If you have questions, please feel free to reach us at yewonb@cs.cmu.edu or to open an issue.