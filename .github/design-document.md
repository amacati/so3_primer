# Website design plan

## Structure
The website should be a single page with the following sections:
- The start is a fullscreen video that plays automatically and loops.
- Below the video, the title and links with badges to GitHub, ArXiv, and a pop-up BibTeX citation.
- Then, we have several sections with headings and paragraphs of text.

## Text Sections
1. **A Primer on SO(3) Actions in Deep Reinforcement Learning**
   - Brief introduction to the topic and its importance in robotics.
2. **SO(3) Rotation Representations**
   - Overview of different SO(3) representations (Euler angles, quaternions, rotation matrices, tangent spaces)
   - Pros and cons of each representation.
3. **Key Findings**
    - Summary of the main findings from the paper regarding the effectiveness of different representations in RL tasks
4. **Recommendations**
    - Practical advice based on the research findings for choosing SO(3) representations in RL applications
5. **Explaining the Results**
    - Smoothness of the action space, singularity issues, multi-covers
    - Exploration dynamics in different action spaces. Make an interactive visualization to illustrate this.
    - Entropy regularization. Show two plots comparing low with high entropy in the distribution, and explain that they do not lead to actual diversity in SO(3).
    - Unit rotation-centered policies.
    - Scaling action ranges.
6. **Benchmarks**
    - Description of the benchmark tasks used in the paper
    - Summary of results with links to detailed plots and code

## Tech Stack

### Hosting: GitHub Pages
- Free static site hosting directly from the repository
- Custom domain support
- HTTPS enabled by default
- Perfect for academic project websites

### Frontend Framework: Vite + Vanilla JS/TypeScript
- Fast development and build times
- No heavy framework overhead
- Clean, minimal bundle size
- Easy to maintain and customize

### Interactive Visualizations: Three.js + D3.js
- **Three.js**: For 3D rotation visualizations (SO(3) space exploration)
  - WebGL-based, runs smoothly in all modern browsers
  - Perfect for demonstrating rotation representations and singularities
  - Can create interactive 3D scenes showing exploration dynamics
- **D3.js**: For 2D plots and statistical visualizations
  - Entropy comparison plots
  - Benchmark result charts
  - Interactive data exploration

### Video: Direct HTML5 Video
- Fullscreen autoplay with loop
- Optimized video formats (WebM + MP4 fallback)
- Lazy loading for performance
- No external dependencies required

### Styling: Tailwind CSS
- Utility-first CSS for rapid development
- Responsive design out of the box
- Easy to customize for academic aesthetic
- Small production bundle with purging

### Additional Libraries
- **KaTeX**: For mathematical notation (SO(3) equations)
- **Highlight.js**: For code snippets if needed
- **AOS (Animate On Scroll)**: Subtle scroll animations

### Why This Stack?
1. **GitHub Pages Compatible**: Everything is static and can be deployed directly
2. **No Build-Time Limitations**: All interactive elements run client-side
3. **Performance**: Fast loading, smooth interactions
4. **Maintainability**: Simple stack, easy for others to contribute
5. **Academic Standard**: Clean, professional appearance suitable for research
6. **Future-Proof**: Modern web standards, no deprecated dependencies

### Deployment
- Build with Vite: `npm run build`
- Deploy to `gh-pages` branch or `/docs` folder
- Automatic deployment via GitHub Actions (optional)