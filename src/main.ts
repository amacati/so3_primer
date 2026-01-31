import './style.css'

// BibTeX Modal functionality
const bibtexBtn = document.getElementById('bibtex-btn');
const bibtexModal = document.getElementById('bibtex-modal');
const closeModal = document.getElementById('close-modal');
const copyBibtex = document.getElementById('copy-bibtex');

bibtexBtn?.addEventListener('click', () => {
  bibtexModal?.classList.remove('hidden');
});

closeModal?.addEventListener('click', () => {
  bibtexModal?.classList.add('hidden');
});

bibtexModal?.addEventListener('click', (e) => {
  if (e.target === bibtexModal) {
    bibtexModal.classList.add('hidden');
  }
});

copyBibtex?.addEventListener('click', async () => {
  const bibtexContent = document.getElementById('bibtex-content')?.textContent;
  if (bibtexContent) {
    try {
      await navigator.clipboard.writeText(bibtexContent);
      const btn = copyBibtex as HTMLButtonElement;
      const originalText = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(() => {
        btn.textContent = originalText;
      }, 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }
});

// Smooth scroll for internal links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (this: HTMLAnchorElement, e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href')!);
    target?.scrollIntoView({ behavior: 'smooth' });
  });
});

console.log('SO(3) Primer website loaded');

