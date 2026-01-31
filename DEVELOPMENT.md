# SO(3) Primer Website - Development Guide

## Local Development

### Prerequisites
- Node.js 18+ 
- npm

### Setup
```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The development server will start at `http://localhost:5173/so3_primer/`

### Development Features
- **Hot Module Replacement (HMR)**: Changes to code are instantly reflected in the browser
- **Fast Refresh**: Maintains application state during updates
- **TypeScript Support**: Full type checking and IntelliSense
- **Tailwind CSS**: Utility-first CSS with instant recompilation

### Project Structure
```
so3_primer/
├── public/              # Static assets
│   ├── paper.pdf       # Full paper PDF (25MB)
│   └── video/          # Hero video files (MP4, WebM)
├── src/
│   ├── main.ts         # Application entry point
│   └── style.css       # Global styles with Tailwind
├── index.html          # Main HTML file with all sections
├── vite.config.ts      # Vite configuration
├── tailwind.config.js  # Tailwind CSS configuration
└── package.json        # Dependencies and scripts
```

### Adding Content

#### Text Content
Edit `index.html` directly - all sections are defined there with placeholder text.

#### Interactive Visualizations
Add Three.js or D3.js code in:
- `src/visualizations/exploration.ts` - For SO(3) exploration dynamics
- `src/visualizations/entropy.ts` - For entropy comparison plots

#### Hero Video
Place your video files in `public/video/`:
- `hero.mp4` - Primary format
- `hero.webm` - Optimized format

### Building for Production
```bash
# Build static files
npm run build

# Preview production build locally
npm run preview
```

Output will be in the `dist/` directory.

### Deployment to GitHub Pages

#### Option 1: Manual Deployment
```bash
npm run build
git add dist -f
git commit -m "Deploy website"
git subtree push --prefix dist origin gh-pages
```

#### Option 2: GitHub Actions (Recommended)
Create `.github/workflows/deploy.yml` for automatic deployment on push to main.

### Available Scripts
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally

### Troubleshooting

**Port already in use:**
```bash
# Kill existing Vite processes
pkill -f vite
npm run dev
```

**Tailwind styles not applying:**
- Restart the dev server
- Check that class names are in the Tailwind config content paths
- Verify `@import "tailwindcss"` is in `src/style.css`

**Video not playing:**
- Ensure video files are in `public/video/`
- Check video codec compatibility (H.264 for MP4, VP9 for WebM)
- Videos must be accessible at `/video/hero.mp4` and `/video/hero.webm`

**Large files (Paper PDF, Videos):**
- The paper PDF (25MB) is tracked in Git but consider using Git LFS for large binary files
- For production, consider hosting the PDF externally (Google Drive, arXiv) if repo size becomes an issue
- Command to set up Git LFS: `git lfs track "*.pdf" "*.mp4" "*.webm"`

### Tech Stack
- **Vite**: Build tool and dev server
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first styling
- **Three.js**: 3D visualizations for SO(3) representations
- **D3.js**: Data visualizations and charts
- **KaTeX**: Mathematical notation rendering

