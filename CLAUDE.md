# Project Context

When working with this codebase, prioritize readability over cleverness. Ask clarifying questions before making architectural changes.

## About This Project

A project website for the paper "A Primer On SO(3) Action Representations In Deep Reinforcement Learning" to be presented at ICLR 2026.

## Key Directories

- `public/` - static assets
- `src/` - ts source code for the website
- `index.html` - main HTML file

## Tech Stack

- **Vite** - Fast build tool and development server
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first styling
- **Three.js** - 3D visualizations for rotation representations
- **D3.js** - Interactive charts and plots
- **KaTeX** - Mathematical notation rendering

## Common Commands
```bash
npm install  # install deps
npm run dev  # start development server
npm run build  # build for production
```

## Notes

The website is deployed via GitHub Pages. Only static elements are allowed. Dynamic content must be pre-generated and included in the `public/` directory.
