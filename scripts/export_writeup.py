"""Typeset Tell's two-page research brief (summary + methods) from verified run reports."""
import html
import json
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfgen import canvas
from writeup_charts import build_charts

ROOT = Path(__file__).resolve().parents[1]
REPO = 'https://github.com/chanatundev/tell'


def gh(path, label=None):
    """Link to a file or folder in the public repo."""
    kind = 'tree' if path.endswith('/') else 'blob'
    return f'<link href="{REPO}/{kind}/main/{path.rstrip("/")}" color="#334f70">{label or path}</link>'
OUT = ROOT / 'output/pdf/tell-writeup.pdf'
INK = colors.HexColor('#162333')
MUTED = colors.HexColor('#4b5563')
RULE = colors.HexColor('#bac2c9')
WIDTH, HEIGHT, MARGIN, GAP = 612, 792, 43, 21
FULL = WIDTH - 2 * MARGIN
COL = (FULL - GAP) / 2

body = ParagraphStyle('body', fontName='Times-Roman', fontSize=9.7, leading=11.7,
                      textColor=INK, spaceAfter=7)
small = ParagraphStyle('small', parent=body, fontSize=8.2, leading=10.1, textColor=MUTED)
heading = ParagraphStyle('heading', fontName='Helvetica-Bold', fontSize=9.3,
                         leading=12, textColor=INK, spaceAfter=5)


def paragraph(c, text, x, top, width, style=body):
    p = Paragraph(text, style)
    _, height = p.wrap(width, HEIGHT)
    p.drawOn(c, x, top - height)
    return top - height - style.spaceAfter


def section(c, title, text, x, top):
    top = paragraph(c, title, x, top, COL, heading)
    return paragraph(c, text, x, top, COL)


def footer(c, page, pages):
    c.setStrokeColor(INK)
    c.setLineWidth(.65)
    c.line(MARGIN, 45, WIDTH-MARGIN, 45)
    c.setFont('Helvetica-Bold', 8)
    c.setFillColor(INK)
    c.drawString(MARGIN, 31, 'This project is a prototype and is not financial advice.')
    c.setFont('Helvetica', 8)
    c.drawRightString(WIDTH-MARGIN, 31, f'{page} / {pages}')


EQ = ParagraphStyle('eq', parent=body, fontName='Times-Italic', alignment=TA_CENTER, spaceAfter=7)


def methods_page(c):
    c.setFont('Helvetica', 8)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, 753, 'VERITAS AI 2026  /  AI + FINANCE')
    c.drawRightString(WIDTH-MARGIN, 753, 'HOW TELL WAS BUILT AND MEASURED')
    c.setStrokeColor(INK)
    c.setLineWidth(.65)
    c.line(MARGIN, 744, WIDTH-MARGIN, 744)
    title = ParagraphStyle('t2', fontName='Times-Bold', fontSize=16, leading=19, textColor=INK, spaceAfter=10)
    top = paragraph(c, 'Methods', MARGIN, 732, FULL, title)
    left = right = top
    x2 = MARGIN + COL + GAP
    eq = lambda text, x, y: paragraph(c, text, x, y, COL, EQ)

    left = section(c, 'A  DATA AND PREPARATION',
        'V1 trains on real receipts from CORD v2 and Wikimedia Commons and on AI receipts from GPT4o-Receipt. V2 adds real WildReceipt photos and '
        'our own generated receipts. V1 uses an 80/20 split within each source, grouped by store or photographer where available so near-duplicates '
        'stay on one side; V2 keeps WildReceipt\'s own train/test partitions. We also removed every Commons photo whose photographer appears in the test set (319 images; ' + gh('DATA.md') + '). '
        'The 50 test photos cover 30 countries (33 phone photos, 17 scans), and each twin prompt invents its own shop, items and prices.', MARGIN, left)
    left = paragraph(c, 'Every image then goes through the same steps (' + gh('src/normalize.py') + '). It is turned upright, stripped of all metadata, '
        'resized so the long side is 1,024 px, centred on a grey 1,024 \u00d7 1,024 canvas and saved as JPEG at quality 85. The AI images arrive as PNGs '
        'while real photos are already JPEGs, so PNGs first get one JPEG pass (quality 70\u201395) to reduce that format shortcut; differences in earlier compression may remain.', MARGIN, left, COL)
    left = paragraph(c, 'We then checked whether file details still give the label away (' + gh('src/leak_check.py') + '). On the raw files, a logistic '
        'regression on width, height, aspect ratio, file size and JPEG tables separates real from AI with AUROC 0.994. After normalisation, size and '
        'metadata features alone score 0.5.', MARGIN, left, COL)

    left = section(c, 'B  MODEL AND TRAINING',
        'Tell is a supervised binary classifier. We start from EfficientNet-B0 with ImageNet weights (timm), replace the last layer with a single '
        'output and fine-tune every layer (' + gh('src/train.py') + '). For an image with label <i>y</i> (1 = AI, 0 = real) the network outputs a logit '
        '<i>z</i>, and training minimises binary cross-entropy:', MARGIN, left-2)
    left = eq('L = \u2212[ y log \u03c3(z) + (1 \u2212 y) log(1 \u2212 \u03c3(z)) ],&nbsp;&nbsp; \u03c3(z) = 1 / (1 + e<super>\u2212z</super>)', MARGIN, left)
    left = paragraph(c, 'We use AdamW (learning rate 3\u00d710<super>\u22124</super>, weight decay 0.01) with a one-cycle schedule and 15% warm-up, '
        'batch size 8 and three epochs at 512 \u00d7 512 px. Both models trained on an Apple-silicon laptop: about 6 minutes for V1 and 11 for V2.', MARGIN, left, COL)
    left = paragraph(c, 'Augmentation keeps text proportions intact (' + gh('src/tellmodel.py') + '). Each training view is a random square window '
        'covering 50\u2013100% of the short side\u2019s square, with up to \u00b15\u00b0 rotation, JPEG re-compression (quality 60\u201395, 60% of the time), '
        'light blur (25%) and colour jitter. Images are never flipped, because receipts never contain mirrored text. At test time the model scores up to three '
        'square windows that together cover the receipt and averages their logits. After each epoch we compute validation AUROC and keep the best epoch, '
        'the earliest if tied: epoch 1 for V1 (1.000 at every epoch) and epoch 3 for V2.', MARGIN, left, COL)

    right = section(c, 'C  CALIBRATION AND THRESHOLD',
        'Raw network scores tend to be overconfident, so we apply temperature scaling. The reported score is', x2, right)
    right = eq('p = \u03c3(z / T)', x2, right)
    right = paragraph(c, 'where <i>T</i> minimises the negative log-likelihood on validation (<i>T</i> = 1.57 for V1, 0.82 for V2). The flag threshold '
        '<i>t</i> is the smallest value at which no more than 5% of real validation images score higher:', x2, right, COL)
    right = eq('t = min { t : (1/n<sub>real</sub>) \u03a3<sub>i</sub> 1[p<sub>i</sub> &gt; t] \u2264 0.05 }', x2, right)
    right = paragraph(c, 'This gave <i>t</i> = 0.021 for V1 and 0.0024 for V2 (' + gh('frozen/') + '). A photo is flagged when <i>p</i> &gt; <i>t</i>.', x2, right, COL)

    right = section(c, 'D  EVALUATION METRICS',
        'For test pair <i>i</i> with AI score <i>a<sub>i</sub></i> and real score <i>r<sub>i</sub></i>, paired accuracy is', x2, right-2)
    right = eq('PA = (1/N) \u03a3<sub>i</sub> ( 1[a<sub>i</sub> &gt; r<sub>i</sub>] + \u00bd \u00b7 1[a<sub>i</sub> = r<sub>i</sub>] )', x2, right)
    right = paragraph(c, 'so 0.5 is chance. At the frozen threshold, detection rate = TP / <i>n</i><sub>AI</sub> and false-alarm rate = FP / <i>n</i><sub>real</sub>. '
        'Both get Wilson 95% intervals, where <i>q</i> = <i>k</i>/<i>n</i> is the observed rate and <i>z</i> = 1.96:', x2, right, COL)
    right = eq('( q + z\u00b2/2n \u00b1 z \u221a( q(1\u2212q)/n + z\u00b2/4n\u00b2 ) ) / ( 1 + z\u00b2/n )', x2, right)
    right = paragraph(c, 'Paired accuracy gets a percentile bootstrap interval from 10,000 resamples of the 50 pairs. AUROC over all 100 test images was 0.74 '
        'for V1 and 0.90 for V2 (' + gh('src/metrics.py') + ').', x2, right, COL)

    right = section(c, 'E  BASELINES: WHY SIMPLER CHECKS FALL SHORT',
        'We built three simpler detectors a business could use today (' + gh('src/baselines.py') + '). The first flags photos with no camera EXIF data. '
        'Most real photos in our data had none either, and on raw validation files it flagged 274 of 278 real receipts. The second is a logistic regression on '
        'frequency (FFT) and brightness features. It reached AUROC 0.999 on validation fakes from the older generator, then caught only 22 of the 50 newer-generator twins. '
        'The third uses Tesseract OCR for a bookkeeper\u2019s checklist. It flags receipts if the items do not add up to the total, a price is malformed, '
        'or the image nearly duplicates another. It could read both the items and the total on only 29% of validation receipts. '
        'Checks built around older fakes or arithmetic mistakes miss the newer ones.', x2, right-2)
    right = paragraph(c, 'Same 50 pairs for every method (V1 test):', x2, right-1, COL, small)
    rows = [['Method', 'Ranked AI\nhigher', 'Twins\ncaught', 'Real\nflagged'],
            ['Tell V1 (frozen)', '78%', '37/50', '15/50'],
            ['Frequency model', '66%', '22/50', '14/50'],
            ['OCR checklist, lenient', '55%', '11/50', '8/50'],
            ['OCR checklist, strict', '49%', '40/50', '41/50']]
    t = Table(rows, colWidths=[COL*0.43, COL*0.19, COL*0.19, COL*0.19])
    t.setStyle(TableStyle([('FONT', (0, 0), (-1, -1), 'Times-Roman', 9.2), ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 7.2),
                           ('LEADING', (0, 0), (-1, 0), 8.5), ('FONT', (0, 1), (-1, 1), 'Times-Bold', 9.2), ('TEXTCOLOR', (0, 0), (-1, -1), INK),
                           ('ALIGN', (1, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, 0), 'BOTTOM'),
                           ('LINEABOVE', (0, 0), (-1, 0), .6, INK), ('LINEBELOW', (0, 0), (-1, 0), .4, RULE),
                           ('LINEBELOW', (0, -1), (-1, -1), .6, INK), ('LEFTPADDING', (0, 0), (0, -1), 0),
                           ('TOPPADDING', (0, 0), (-1, -1), 1.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5)]))
    _, th = t.wrap(COL, HEIGHT)
    t.drawOn(c, x2, right - th)
    right = paragraph(c, '50% ranking is a coin flip. The strict checklist catches more twins only because it also flags most real receipts (' + gh('results/heldout_v1/heldout_results.md', 'full table') + ').', x2, right - th - 4, COL, small)
    end = min(left, right)
    assert end > 53, f'Methods page overflows: bottom={end:.1f}'
    footer(c, 2, 2)
    return end


def main():
    final = json.loads((ROOT / 'results/post_unseal_v2/report.json').read_text())
    interim = json.loads((ROOT / 'results/post_unseal_interim19/report.json').read_text())
    assert final['evaluation'].startswith('POST-UNSEAL') and final['n_pairs'] == 50
    assert final['metrics']['tp'] == 38 and final['metrics']['fp'] == 5
    epoch_chart, results_chart = build_charts()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    c.setTitle('Tell: Spotting AI-Made Receipt Photos')
    c.setAuthor('Tee to Three')
    c.setSubject('Veritas AI 2026 | AI + Finance | Research brief')

    c.setFont('Helvetica', 8)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, 753, 'VERITAS AI 2026  /  AI + FINANCE')
    c.drawRightString(WIDTH-MARGIN, 753, 'RESEARCH BRIEF  ·  15 NOVEMBER 2026')
    c.setStrokeColor(INK)
    c.setLineWidth(.65)
    c.line(MARGIN, 744, WIDTH-MARGIN, 744)
    title = ParagraphStyle('title', fontName='Times-Bold', fontSize=22, leading=25,
                           alignment=TA_CENTER, textColor=INK, spaceAfter=5)
    top = paragraph(c, 'Tell: Spotting AI-Made<br/>Receipt Photos', MARGIN, 729, FULL, title)
    byline = ParagraphStyle('byline', parent=small, alignment=TA_CENTER, spaceAfter=9)
    top = paragraph(c, 'Tee to Three  |  Research prototype for small-business expense review  |  ' + f'<link href="{REPO}" color="#334f70">github.com/chanatundev/tell</link>' + '<br/>All of this work, V1 and V2 included, was done within the hackathon time limit.', MARGIN, top, FULL, byline)
    abstract = ('<b>Summary.</b> Small businesses often approve expenses from a photo of a receipt, and AI image tools can now fake one in seconds. '
                'Tell is an image classifier that flags receipt photos that look AI-made, prompting the reviewer to ask for the original or a card statement before paying. '
                'We froze our first model (V1) and tested it once on 50 matched pairs of real receipt photos and generated twins with invented details. '
                'V1 caught 37 of 50 twins and flagged 15 of 50 real receipts. With more training data, V2 caught 38 and flagged 5 on the same pairs.')
    top = paragraph(c, abstract, MARGIN, top, FULL, ParagraphStyle('abstract', parent=body, fontSize=10, leading=12.3, spaceAfter=10))
    c.setStrokeColor(RULE)
    c.setLineWidth(.4)
    c.line(MARGIN, top+5, WIDTH-MARGIN, top+5)
    left = right = top - 8
    x2 = MARGIN + COL + GAP

    left = section(c, '1  THE PROBLEM',
        'A fake receipt can list believable items with correct totals, and checking the arithmetic will not catch it. '
        'Tell scores the image itself. Above a fixed threshold it suggests asking for the original receipt or a card statement, '
        'and the reviewer makes the call. Paying a fake claim costs money, and questioning a real one costs goodwill.', MARGIN, left)
    left = section(c, '2  V1: THE ORIGINAL MODEL',
        'V1 is an EfficientNet-B0 image classifier. We started from ImageNet weights and fine-tuned it for three epochs on 512-pixel crops of 1,730 receipts. '
        'Of these, 619 were AI-generated, from the GPT4o-Receipt dataset, and 1,111 were real photos (800 from CORD and 311 from Wikimedia Commons; ' + gh('DATA.md') + '). Another 435 were kept for validation. '
        'We set the threshold to flag no more than 5% of real validation receipts and froze it before testing (' + gh('frozen/') + ').', MARGIN, left-2)
    left = section(c, '3  THE TEST',
        'Each of our 50 test pairs has a real receipt photo from Commons and an AI "twin" from a newer image generator, prompted to look like the same kind of receipt '
        'with invented details. No test image is in training, and photographers named in the test were '
        'removed from training. We froze V1 before opening the test images and scored it once (' + gh('results/heldout_v1/') + '). V1 scored the twin higher than the real photo in 78% of pairs. It caught 37 of the 50 twins (74%) '
        'but also flagged 15 of the 50 real receipts (30%). An OCR checklist that verifies totals did much worse, catching 11 of 50 twins (page 2).', MARGIN, left-2)

    left = section(c, '4  V2: WHAT WE CHANGED',
        'To cut false alarms, V2 changes only the data. We added 1,739 real photos from WildReceipt and 32 new AI receipts, '
        'written separately from the test pairs (32 of 100 attempts passed our checks). The training set grew from 1,730 to 3,022 images, '
        'with the same model and settings. On the same 50 pairs (' + gh('results/post_unseal_v2/') + '), V2 ranked 92% correctly, caught 38 twins (76%) and flagged 5 real receipts (10%).', MARGIN, left-2)

    right = paragraph(c, 'CHARTS', x2, right, COL, heading)
    height = COL * 1.45 / 3.65
    c.drawImage(str(epoch_chart),x2,right-height,width=COL,height=height)
    right -= height+5
    right = paragraph(c, 'Training loss for each epoch (lower is better). Validation AUROC was 0.997 or higher for both models.', x2,right,COL,small)
    height = COL * 1.8 / 3.65
    c.drawImage(str(results_chart),x2,right-height,width=COL,height=height)
    right -= height+5
    right = paragraph(c, 'The bars show how often the AI twin scored higher than its real photo, and how many AI and real images were flagged. '
        'Whiskers show 95% intervals.',x2,right,COL,small)
    right = section(c, '5  LIMITS',
        'We made the 50 test pairs by matching each real photo with an AI twin of the same country and style. The set is still small and synthetic, and it contains no real fraud cases. V1 had never seen these pairs when it was scored. '
        'V2 was built after we saw how V1 did on them, and its choices may suit these particular pairs; scoring it on a fresh set would show whether the improvement '
        'carries over. We also added real and AI data at the same time and cannot tell which helped more.', x2, right-2)
    right = paragraph(c, '<b>Working prototype.</b> A local Gradio demo (' + gh('src/demo.py') + ') takes a receipt photo and shows the score, the threshold and what to do next.',x2,right-3,COL,small)


    end = min(left, right) - 4
    c.setStrokeColor(RULE)
    c.line(MARGIN, end+3, WIDTH-MARGIN, end+3)
    credits_file = ROOT / 'CREDITS.json'
    if credits_file.exists():
        items = json.loads(credits_file.read_text())['compact']
    else:
        items = [dict(label='GPT4o-Receipt', url='https://huggingface.co/datasets/Scam-AI/gpt4o-receipt'),
                 dict(label='CORD',url='https://huggingface.co/datasets/naver-clova-ix/cord-v2'),
                 dict(label='Commons',url='https://commons.wikimedia.org'),
                 dict(label='WildReceipt',url='https://github.com/open-mmlab/mmocr'),
                 dict(label='timm',url='https://github.com/huggingface/pytorch-image-models'),
                 dict(label='PyTorch',url='https://pytorch.org'),dict(label='Gradio',url='https://www.gradio.app'),
                 dict(label='OpenAI',url='https://openai.com'),dict(label='Anthropic',url='https://www.anthropic.com')]
    links = '; '.join(f'<link href="{html.escape(i["url"], quote=True)}" color="#334f70">{html.escape(i["label"])}</link>' for i in items)
    end = paragraph(c, '<b>Linked credits.</b> '+links+'. Full provenance, licences and per-file Commons credits: CREDITS.md / ATTRIBUTION_commons.csv. '
                    'WildReceipt photo licence is unspecified in OpenMMLab metadata; no photo redistribution is claimed.', MARGIN, end-5, FULL, small)
    assert end > 53, f'Content overflows page 1: bottom={end:.1f}'
    footer(c, 1, 2)
    c.showPage()
    end2 = methods_page(c)
    c.save()
    assert len(PdfReader(OUT).pages) == 2
    print(f'{OUT} (two pages; bottoms {end:.1f}pt, {end2:.1f}pt)')


if __name__ == '__main__':
    main()
