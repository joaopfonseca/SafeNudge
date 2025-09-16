# SafeNudge

A Python library with the implementation for the algorithms used in
"Safeguarding large language models in real-time with tunable safety-performance
trade-offs", by J. Fonseca, A. Bell and J. Stoyanovich.

`CTG` provides methods to guide model responses based on various criteria,
helping ensure safe, high-quality, and controllable text generation.

## Implemented methods

- **Controlled Text Generation (CTG)**: The SafeNudge implementation.
- **WildGuard Integration (WildguardCTG)**: SafeNudge using the WildGuard classifier
- **Token Masking (TokenMaskingCTG)**: c-FUDGE, as described in the paper

## Installation

A Python distribution of version >= 3.12 is required to run this project.
Earlier Python versions might work in most cases, but they were never tested.


### From Source

```bash
# Clone the repository
git clone https://github.com/joaopfonseca/SafeNudge.git
cd Output-Steering

# Install in development mode
pip install -e .
```

### Using pip

```bash
pip install git+https://github.com/joaopfonseca/SafeNudge.git
```

## Examples

Check the [notebooks directory](https://github.com/joaopfonseca/SafeNudge/tree/main/notebooks) 
for some examples Andrew and I developed while working on SafeNudge and setting
up the experiments!

## Project Structure

```
Output-Steering/
├── ctg/                    # Core library code
├── experiments/            # Experimental code and evaluation
└── notebooks/              # Jupyter notebooks with examples
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@article{fonseca2025safeguarding,
  title={Safeguarding large language models in real-time with tunable safety-performance trade-offs},
  author={Fonseca, Joao and Bell, Andrew and Stoyanovich, Julia},
  journal={arXiv preprint arXiv:2501.02018},
  year={2025}
}
```