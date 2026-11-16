"""Local-only Gradio receipt screening demo. No public tunnel or bundled receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import threading

import gradio as gr
import pandas as pd
import torch
import normalize as N
import tellmodel as TM

ROOT = Path(__file__).resolve().parents[1]

# Paper-and-ink theme: one typeface, flat surfaces, no borders or shadows.
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;700;800&display=swap');
:root, .dark { --tell-paper: #ecebe5; --tell-sheet: #f7f6f2; --tell-ink: #111111; --tell-soft: #4a4a46; }
body, gradio-app, .gradio-container, .main, .wrap, .contain { background: var(--tell-paper) !important; color: var(--tell-ink) !important;
  font-family: 'Archivo', system-ui, sans-serif !important; }
.gradio-container { max-width: 1240px !important; }
.gradio-container * { font-family: 'Archivo', system-ui, sans-serif !important; box-shadow: none !important; }
.block, .form, .panel, .gr-group, fieldset { background: var(--tell-sheet) !important; border: none !important; border-radius: 0 !important; }
#tell-head h1 { font-size: 72px !important; font-weight: 800 !important; letter-spacing: -0.04em; line-height: 1; color: var(--tell-ink) !important; margin: 0 0 8px; }
#tell-head p, .prose, .prose * { color: var(--tell-ink) !important; font-size: 18px !important; }
#tell-head, #tell-note, #tell-model, #tell-foot { background: transparent !important; }
#tell-note p, #tell-model p, #tell-foot p { font-size: 15px !important; color: var(--tell-soft) !important; }
code { background: transparent !important; color: var(--tell-soft) !important; font-family: 'Archivo', sans-serif !important; }
label span, .label-wrap span, span[data-testid="block-info"] { background: transparent !important; color: var(--tell-soft) !important; font-size: 16px !important; font-weight: 500 !important; border: none !important; }
textarea, input { background: var(--tell-paper) !important; color: var(--tell-ink) !important; border: none !important; border-radius: 0 !important; font-size: 22px !important; }
#tell-action textarea { font-size: 30px !important; font-weight: 700 !important; }
#tell-score textarea { font-size: 44px !important; font-weight: 800 !important; letter-spacing: -0.02em; }
button.primary, button.lg.primary { background: var(--tell-ink) !important; color: var(--tell-paper) !important; border: none !important; border-radius: 0 !important; font-size: 20px !important; font-weight: 700 !important; }
.image-container, [data-testid="image"], .upload-container { background: var(--tell-sheet) !important; border: none !important; }
footer { display: none !important; }
"""

def build_demo(run):
    run = Path(run)
    frozen = json.loads((run / 'frozen.json').read_text())
    assert hashlib.sha256((run / 'model.pt').read_bytes()).hexdigest() == frozen['model_sha256'], 'Model hash mismatch'
    assert not frozen['smoke_test'], 'Smoke weights cannot power this demo'
    device = TM.pick_device()
    model = TM.make_model(False).to(device)
    model.load_state_dict(torch.load(run / 'model.pt', map_location=device, weights_only=True))
    model.eval()
    lock = threading.Lock()

    def screen(path):
        if path is None:
            raise gr.Error('Choose a receipt image first.')
        with lock, tempfile.TemporaryDirectory(prefix='tell-') as tmp:
            normalized = Path(tmp) / 'receipt.jpg'
            info = N.normalize_file(Path(path), normalized, frozen.get('pad_square', True))
            df = pd.DataFrame([dict(path=str(normalized), cw=info['cw'], ch=info['ch'])])
            scores, _ = TM.score_files(model, df, '/', frozen['input_size'], device,
                                      frozen['temperature'], batch=1)
            score = float(scores[0])
        flagged = score > frozen['threshold']
        action = ('Ask for the original receipt or card statement before paying.' if flagged else
                  'No AI-image flag. Continue your usual checks; this does not verify the receipt.')
        return action, f'{score:.4f}', f"{frozen['threshold']:.4f}"

    with gr.Blocks(title='Tell — receipt screening', analytics_enabled=False) as app:
        gr.Markdown('# Tell\nScreen a receipt photo for signs of AI generation. A flag is not proof of fraud.', elem_id='tell-head')
        gr.Markdown('**Research prototype.** Original heldout v1: 37/50 twins detected, 15/50 reals flagged. '
                    'Final v2 on the previously opened pairs: 38/50 twins detected, 5/50 reals flagged. V2 is post-unseal development, not fresh heldout. Scores can be overconfident. Do not use this score alone to approve or deny an expense.', elem_id='tell-note')
        gr.Markdown(f"Loaded model: `{run.name}` · SHA-256 `{frozen['model_sha256'][:12]}`. "
                    'V2 evaluations are post-unseal development results, not fresh heldout evidence.', elem_id='tell-model')
        receipt = gr.Image(type='filepath', label='Receipt image', sources=['upload'])
        button = gr.Button('Screen receipt', variant='primary')
        action = gr.Textbox(label='Next step', elem_id='tell-action')
        score = gr.Textbox(label='AI-image model score (not a fraud probability)', elem_id='tell-score')
        threshold = gr.Textbox(label='Validation-selected flag threshold')
        button.click(screen, receipt, [action, score, threshold])
        gr.Markdown('Runs on this Mac. No public sharing link. Uploaded images may be cached by Gradio '
                    'locally. Use redacted copies for demonstrations; no real receipts are bundled.', elem_id='tell-foot')
    return app, screen

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/v2')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    app, _ = build_demo(args.run)
    app.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1', server_port=args.port,
                                                 share=False, inbrowser=False,
                                                 theme=gr.themes.Base(), css=CSS)
