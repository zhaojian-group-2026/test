// Render SVG figures to vector PDF (7 in wide) and high-res PNG previews.
// usage: node render.js out/fig1.svg [out/fig2.svg ...]
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const WIDTH_IN = 7.0;

(async () => {
  const files = process.argv.slice(2);
  const browser = await chromium.launch();
  for (const file of files) {
    const svg = fs.readFileSync(file, 'utf8');
    const m = svg.match(/viewBox="0 0 ([\d.]+) ([\d.]+)"/);
    const [w, h] = [parseFloat(m[1]), parseFloat(m[2])];
    const hIn = (WIDTH_IN * h) / w;
    const html = `<!doctype html><html><head><style>
      @page { size: ${WIDTH_IN}in ${hIn}in; margin: 0; }
      html, body { margin: 0; padding: 0; background: #fff; }
      svg { display: block; width: ${WIDTH_IN}in; height: ${hIn}in; }
    </style></head><body>${svg}</body></html>`;
    const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 3 });
    await page.setContent(html, { waitUntil: 'load' });
    const base = file.replace(/\.svg$/, '');
    await page.pdf({ path: base + '.pdf', preferCSSPageSize: true, printBackground: true });
    // PNG: render the svg at its native unit size x3
    const page2 = await browser.newPage({ viewport: { width: Math.ceil(w), height: Math.ceil(h) }, deviceScaleFactor: 3 });
    await page2.setContent(`<!doctype html><html><body style="margin:0">${svg.replace(/width="[\d.]+" height="[\d.]+"/, `width="${w}" height="${h}"`)}</body></html>`);
    await page2.screenshot({ path: base + '.png', clip: { x: 0, y: 0, width: w, height: h } });
    await page.close(); await page2.close();
    console.log('rendered', path.basename(base), `${WIDTH_IN}in x ${hIn.toFixed(2)}in`);
  }
  await browser.close();
})();
