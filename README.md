# Primer on SO(3) Action Representations in Deep RL - Website

This repository hosts the project website for **"A Primer on SO(3) Action Representations in Deep Reinforcement Learning"**.

## About the Paper

This work provides a comprehensive primer on leveraging SO(3) rotation representations in deep reinforcement learning for robotic manipulation tasks. The paper addresses the fundamental challenge of how to properly represent 3D rotations as actions in policy networks, examining various parametrizations and their implications for learning efficiency and task performance.

## Website

The website features:
- Overview of SO(3) rotation representations
- Key findings and recommendations
- Interactive visualizations
- Links to paper, code, and citation information

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Visit `http://localhost:5173/so3_primer/` to see the website.

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed development instructions.

## Building for Production

```bash
npm run build
```

The static site will be generated in the `dist/` directory, ready for deployment to GitHub Pages.

## Deploying to GitHub Pages

```bash
npm run deploy
```

This will build the site and push the contents of `dist/` to the `gh-pages` branch, making it available at `https://amacati.github.io/so3_primer/`.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{
    schuck2026primer,
    title={A Primer on {SO(3)} Action Representations in Deep Reinforcement Learning},
    author={Martin Schuck and Sherif Samy and Angela P. Schoellig},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
}
```

## Links

- **Paper**: [ArXiv](https://arxiv.org/abs/2510.11103)
- **Code**: [GitHub Repository](https://github.com/amacati/so3_primer)

## Tech Stack

- **Vite** - Fast build tool and development server
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first styling
- **Three.js** - 3D visualizations for rotation representations
- **D3.js** - Interactive charts and plots
- **KaTeX** - Mathematical notation rendering
