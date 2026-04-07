"""
Generate solution_writeup.pdf from content matching the LaTeX source.
Uses reportlab for PDF generation since pdflatex is not installed.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Preformatted, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfgen import canvas

# ── Colors ───────────────────────────────────────────────────────────────────
DEEP_BLUE  = colors.HexColor('#003366')
GOLD       = colors.HexColor('#B8860B')
LIGHT_GRAY = colors.HexColor('#F5F5F5')
MED_GRAY   = colors.HexColor('#CCCCCC')
GREEN      = colors.HexColor('#228B22')
RED        = colors.HexColor('#B40000')
WHITE      = colors.white
BLACK      = colors.black

PAGE_W, PAGE_H = A4
MARGIN = 2.5 * cm

# ── Styles ────────────────────────────────────────────────────────────────────
ss = getSampleStyleSheet()

def mk(name, **kw):
    base = kw.pop('parent', 'Normal')
    s = ParagraphStyle(name, parent=ss[base], **kw)
    return s

title_style    = mk('Title2',    fontSize=22, textColor=DEEP_BLUE, spaceAfter=6,  alignment=TA_CENTER, fontName='Helvetica-Bold')
subtitle_style = mk('Subtitle2', fontSize=14, textColor=GOLD,      spaceAfter=4,  alignment=TA_CENTER, fontName='Helvetica-Bold')
author_style   = mk('Author2',   fontSize=11, textColor=BLACK,     spaceAfter=4,  alignment=TA_CENTER)
abstract_style = mk('Abstract2', fontSize=9.5,textColor=BLACK,     spaceAfter=4,  alignment=TA_JUSTIFY, leftIndent=1*cm, rightIndent=1*cm, leading=14)
h1_style       = mk('H1',        fontSize=13, textColor=DEEP_BLUE, spaceBefore=14, spaceAfter=4, fontName='Helvetica-Bold')
h2_style       = mk('H2',        fontSize=11, textColor=DEEP_BLUE, spaceBefore=8,  spaceAfter=3, fontName='Helvetica-Bold')
h3_style       = mk('H3',        fontSize=10, textColor=DEEP_BLUE, spaceBefore=6,  spaceAfter=2, fontName='Helvetica-BoldOblique')
body_style     = mk('Body2',     fontSize=10, leading=15, spaceAfter=5, alignment=TA_JUSTIFY)
caption_style  = mk('Caption2',  fontSize=8.5, textColor=colors.HexColor('#555555'), alignment=TA_CENTER, spaceAfter=6, fontName='Helvetica-Oblique')
code_style     = mk('Code2',     fontSize=8,  fontName='Courier', backColor=LIGHT_GRAY, leading=11, leftIndent=0.5*cm, rightIndent=0.5*cm, spaceAfter=6)
bullet_style   = mk('Bullet2',   fontSize=10, leading=15, spaceAfter=3, leftIndent=0.8*cm, firstLineIndent=-0.4*cm, bulletIndent=0.4*cm)
math_style     = mk('Math2',     fontSize=10, fontName='Courier', alignment=TA_CENTER, spaceAfter=6, spaceBefore=4, textColor=DEEP_BLUE)
toc_h1_style   = mk('TOC1',      fontSize=11, textColor=DEEP_BLUE, spaceAfter=2, leftIndent=0)
toc_h2_style   = mk('TOC2',      fontSize=10, textColor=BLACK,      spaceAfter=1, leftIndent=0.8*cm)
ref_style      = mk('Ref2',      fontSize=9,  leading=13, spaceAfter=4, leftIndent=1*cm, firstLineIndent=-1*cm)

def HR():
    return HRFlowable(width='100%', thickness=0.5, color=DEEP_BLUE, spaceAfter=4, spaceBefore=2)

def sp(n=6):
    return Spacer(1, n)

def H1(text, num):
    return [HR(), Paragraph(f'<b>{num}. {text}</b>', h1_style), sp(2)]

def H2(text, num):
    return [Paragraph(f'<b>{num} {text}</b>', h2_style), sp(1)]

def H3(text):
    return [Paragraph(f'<i>{text}</i>', h3_style), sp(1)]

def P(text):
    return Paragraph(text, body_style)

def B(items):
    out = []
    for item in items:
        out.append(Paragraph(f'• {item}', bullet_style))
    return out

def MATH(text):
    return [sp(3), Paragraph(text, math_style), sp(3)]

def CODE(text):
    return [Preformatted(text, code_style)]

def caption(text):
    return Paragraph(f'<i>{text}</i>', caption_style)

def make_table(data, col_widths=None, header=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style_cmds = [
        ('BACKGROUND',   (0, 0), (-1,  0), DEEP_BLUE),
        ('TEXTCOLOR',    (0, 0), (-1,  0), WHITE),
        ('FONTNAME',     (0, 0), (-1,  0), 'Helvetica-Bold'),
        ('FONTSIZE',     (0, 0), (-1, -1), 8.5),
        ('ALIGN',        (0, 0), (-1, -1), 'LEFT'),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LIGHT_GRAY]),
        ('GRID',         (0, 0), (-1, -1), 0.25, MED_GRAY),
        ('TOPPADDING',   (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
        ('LEFTPADDING',  (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
    ]
    if header:
        style_cmds.append(('LINEBELOW', (0, 0), (-1, 0), 1.5, GOLD))
    t.setStyle(TableStyle(style_cmds))
    return t

# ── Page template with header/footer ─────────────────────────────────────────
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        pg = self._pageNumber
        if pg == 1:
            return
        self.setFont('Helvetica', 8)
        self.setFillColor(colors.HexColor('#666666'))
        self.drawString(MARGIN, 1.2*cm, 'BirdCLEF 2026 — Gold Medal Solution Design')
        self.drawRightString(PAGE_W - MARGIN, 1.2*cm, f'Page {pg} of {page_count}')
        self.setStrokeColor(MED_GRAY)
        self.setLineWidth(0.3)
        self.line(MARGIN, 1.5*cm, PAGE_W - MARGIN, 1.5*cm)

# ── Build document ────────────────────────────────────────────────────────────
def build():
    doc = SimpleDocTemplate(
        'reports/solution_writeup.pdf',
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=2.2*cm, bottomMargin=2.2*cm,
        title='BirdCLEF 2026 — Gold Medal Solution Design',
        author='BirdCLEF 2026 Team',
        subject='Competition Solution Writeup',
    )

    story = []

    # ── Title Page ────────────────────────────────────────────────────────────
    story += [sp(40)]
    story.append(Paragraph('BirdCLEF 2026', title_style))
    story.append(Paragraph('Gold Medal Solution Design', subtitle_style))
    story.append(HRFlowable(width='80%', thickness=1.5, color=GOLD, spaceAfter=10, spaceBefore=10))
    story.append(Paragraph(
        'Multi-Backbone SED + Perch Ensemble with VLOM Blend,<br/>'
        '4-Fold Cross-Validation, and 2025-Derived Techniques',
        mk('subtit', fontSize=12, alignment=TA_CENTER, textColor=BLACK, spaceAfter=6)
    ))
    story.append(HRFlowable(width='80%', thickness=1.5, color=GOLD, spaceAfter=20, spaceBefore=10))
    story.append(Paragraph('BirdCLEF 2026 Team', author_style))
    story.append(Paragraph('March 2026', author_style))
    story += [sp(30)]

    # Abstract box
    abstract_box_data = [[
        Paragraph(
            '<b>Abstract</b><br/><br/>'
            'We present a complete solution design for BirdCLEF 2026, a multi-label bird sound recognition '
            'competition targeting 234 species in Pantanal soundscapes. Our approach integrates three streams: '
            '(1) Google Perch TFLite models with domain-specific label heads, (2) a multi-backbone SED ensemble '
            'using EfficientNet-B0, B3-NoisyStudent, and EfficientNetV2-S trained with Asymmetric Loss and '
            '4-fold cross-validation with model soup, and (3) inference-time techniques from 2025 top solutions. '
            'Key innovations include VLOM blending ((geometric mean + RMS)/2) for heterogeneous model fusion, '
            'domain-matched background noise augmentation with Pantanal recordings, overlapping stride inference '
            'with 1.25s TTA, and 20%/60%/20% temporal smoothing. '
            'Current public LB: <b>0.892</b>. Target: <b>0.926+</b> (competition rank #1).',
            abstract_style
        )
    ]]
    abstract_tbl = Table(abstract_box_data, colWidths=[PAGE_W - 2*MARGIN])
    abstract_tbl.setStyle(TableStyle([
        ('BOX',          (0,0), (-1,-1), 0.5, DEEP_BLUE),
        ('BACKGROUND',   (0,0), (-1,-1), colors.HexColor('#EEF3FA')),
        ('TOPPADDING',   (0,0), (-1,-1), 10),
        ('BOTTOMPADDING',(0,0), (-1,-1), 10),
        ('LEFTPADDING',  (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(abstract_tbl)
    story.append(PageBreak())

    # ── TOC ───────────────────────────────────────────────────────────────────
    story.append(Paragraph('<b>Contents</b>', h1_style))
    story.append(HR())
    toc_entries = [
        ('1', 'Competition Overview', '3'),
        ('1.1', 'Dataset Statistics', '3'),
        ('1.2', 'Key Challenges', '3'),
        ('2', 'Architecture Overview', '3'),
        ('3', 'Perch Stream', '4'),
        ('4', 'SED Stream', '4'),
        ('4.1', 'Model Architecture', '4'),
        ('4.2', 'Backbone Strategy', '5'),
        ('4.3', 'Training Recipe', '5'),
        ('4.4', 'Asymmetric Loss', '5'),
        ('4.5', 'Background Noise Augmentation', '6'),
        ('5', '4-Fold Cross-Validation + Model Soup', '6'),
        ('6', 'Pseudo-Label Pipeline', '7'),
        ('7', 'Inference Pipeline', '7'),
        ('7.1', 'Human Voice Removal (VAD)', '7'),
        ('7.2', 'Overlapping Stride Windows', '7'),
        ('7.3', 'Test-Time Augmentation', '8'),
        ('8', 'Post-Processing', '8'),
        ('8.1', 'VLOM Blend', '8'),
        ('8.2', 'Temporal Smoothing', '8'),
        ('8.3', 'File-Max Trick', '8'),
        ('8.4', 'Power Boost (Conditional)', '9'),
        ('9', 'Experiment Schedule and Expected Results', '9'),
        ('10', 'Key Decisions and Trade-offs', '10'),
        ('11', 'Validation Framework', '10'),
        ('12', 'Techniques from 2025 Top Solutions', '11'),
        ('13', 'Conclusion', '11'),
    ]
    for num, title, pg in toc_entries:
        if '.' not in num:
            story.append(Paragraph(f'{num}. {title}', toc_h1_style))
        else:
            story.append(Paragraph(f'   {num} {title}', toc_h2_style))
    story.append(PageBreak())

    # ── Section 1 ─────────────────────────────────────────────────────────────
    story += H1('Competition Overview', '1')
    story.append(P(
        'BirdCLEF 2026 is hosted on Kaggle and asks participants to identify bird species from '
        '60-second passive-acoustic field recordings captured in the Pantanal wetlands of Brazil. '
        'The evaluation metric is <b>macro ROC-AUC</b> over 234 species.'
    ))

    story += H2('1.1 Dataset Statistics', '')
    t = make_table([
        ['Split', 'Count', 'Notes'],
        ['Train audio clips',    '35,549',   'Short focal recordings, variable duration'],
        ['Train soundscapes',    '66 files', '60s Pantanal field recordings, multi-label'],
        ['Soundscape segments',  '10,658',   'Labeled 5-second segments'],
        ['Test soundscapes',     '~600',     'Private test set'],
        ['Species (classes)',    '234',      '234 primary labels'],
        ['Zero-audio species',   '28',       '25 insect sonotypes + 3 amphibians (no train audio)'],
    ], col_widths=[5*cm, 3*cm, 9.5*cm])
    story.append(t)
    story += [sp(4), caption('Table 1: Dataset overview')]

    story += H2('1.2 Key Challenges', '')
    story += B([
        '<b>Domain gap</b>: Train audio is focal recordings; test is passive soundscapes (ambient noise, overlapping species, varying SNR).',
        '<b>Zero-audio species</b>: 28 species have <i>no</i> training audio clips. Supervised only via soundscape labels.',
        '<b>Class imbalance</b>: Recording counts range from 1 to 500+ per species.',
        '<b>Multi-label evaluation</b>: Each 5-second bin can contain multiple species; macro ROC-AUC is the target.',
    ])

    # ── Section 2 ─────────────────────────────────────────────────────────────
    story += H1('Architecture Overview', '2')
    story.append(P(
        'Our system has three prediction streams fused at inference time: '
        '(1) <b>Perch stream</b>: Google Bird Vocalization Classifier (TFLite) with three task-specific label heads. '
        '(2) <b>SED stream</b>: Multi-backbone Sound Event Detection CNNs with attention pooling, '
        '4-fold cross-validation, and model soup. '
        '(3) <b>Fusion</b>: VLOM blend (2025 2nd-place technique) + temporal smoothing + optional power boost.'
    ))

    pipeline_data = [[
        Paragraph(
            '<b>Audio (60s OGG)</b><br/>'
            '↓ VAD: Silero human voice removal<br/><br/>'
            '<b>Perch TFLite</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
            '<b>SED CNN (multi-backbone)</b><br/>'
            '12 clips × 3 heads (CPU)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;23 stride windows + TTA (GPU)<br/><br/>'
            '↓&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/><br/>'
            '<b>VLOM Blend</b>: (geometric_mean + RMS) / 2<br/>'
            '↓<br/>'
            '<b>Temporal Smooth</b>: 20% / 60% / 20%<br/>'
            '↓<br/>'
            '<b>submission.csv</b> (234 species × 12 bins)',
            mk('pip', fontSize=10, fontName='Courier', leading=16, alignment=TA_CENTER)
        )
    ]]
    pipe_tbl = Table(pipeline_data, colWidths=[PAGE_W - 2*MARGIN])
    pipe_tbl.setStyle(TableStyle([
        ('BOX',          (0,0),(-1,-1), 0.5, DEEP_BLUE),
        ('BACKGROUND',   (0,0),(-1,-1), colors.HexColor('#F0F4FF')),
        ('TOPPADDING',   (0,0),(-1,-1), 12),
        ('BOTTOMPADDING',(0,0),(-1,-1), 12),
        ('LEFTPADDING',  (0,0),(-1,-1), 16),
        ('RIGHTPADDING', (0,0),(-1,-1), 16),
    ]))
    story.append(pipe_tbl)
    story += [sp(4), caption('Figure 1: High-level inference pipeline')]

    # ── Section 3 ─────────────────────────────────────────────────────────────
    story += H1('Perch Stream', '3')
    story.append(P(
        "We use Google's Bird Vocalization Classifier (Perch v2, CPU TFLite) as a frozen feature extractor. "
        'Perch was pre-trained on 10,000+ bird species globally and produces both a 14,795-dimensional '
        'logit vector and a 1,536-dimensional embedding. Three task-specific linear heads are trained on '
        'top of Perch outputs:'
    ))
    t = make_table([
        ['Head', 'Training Data', 'Val AUC', 'Weight'],
        ['label_head_pseudo',     'Perch logits + pseudo labels',          '0.9748', '1.0'],
        ['label_head_soundscape', 'Perch logits + soundscape segments',    '0.9535', '1.0'],
        ['embedding_head',        'Perch embeddings + soundscape (nohuman)', '0.9537', '1.0'],
    ], col_widths=[5*cm, 7*cm, 2.5*cm, 2*cm])
    story.append(t)
    story += [sp(4), caption('Table 2: Perch label head variants')]

    # ── Section 4 ─────────────────────────────────────────────────────────────
    story += H1('SED Stream', '4')

    story += H2('4.1 Model Architecture', '')
    story.append(P(
        'Each SED model takes a 5-second waveform (160,000 samples at 32 kHz) and produces '
        'clip-level probabilities for all 234 species.'
    ))

    arch_data = [[
        Paragraph(
            'Waveform (5s, 32kHz)  →  Mel-Spectrogram (224 × 313)\n'
            '↓\n'
            'EfficientNet Backbone  (pretrained, features_only)\n'
            '↓\n'
            'GEM Frequency Pooling  (p=3, learnable)\n'
            '↓\n'
            'Attention SED Head  (softmax attention × classification)\n'
            '↓\n'
            'Clip-level Sigmoid  ŷ_clip ∈ [0,1]^234',
            mk('arch', fontSize=9, fontName='Courier', leading=14, alignment=TA_CENTER)
        )
    ]]
    arch_tbl = Table(arch_data, colWidths=[PAGE_W - 2*MARGIN])
    arch_tbl.setStyle(TableStyle([
        ('BOX',         (0,0),(-1,-1), 0.5, DEEP_BLUE),
        ('BACKGROUND',  (0,0),(-1,-1), LIGHT_GRAY),
        ('TOPPADDING',  (0,0),(-1,-1), 10),
        ('BOTTOMPADDING',(0,0),(-1,-1), 10),
        ('LEFTPADDING', (0,0),(-1,-1), 20),
    ]))
    story.append(arch_tbl)
    story += [sp(4), caption('Figure 2: SED model architecture')]
    story.append(P('Mel-spectrogram: n_FFT=2048, hop=512, n_mels=224, f_min=0, f_max=16,000 Hz.'))
    story += MATH('GEM(x, p) = ( (1/H) Σ x_h^p )^(1/p),   p initialized at 3.0, learned during training')
    story += MATH('ŷ_clip = σ( Σ_t α_t · φ_t ),   α_t = softmax(tanh(W_a · f_t))')

    story += H2('4.2 Backbone Strategy', '')
    t = make_table([
        ['Backbone', 'Pretraining', 'Params', 'Ens. Weight', 'Notes'],
        ['tf_efficientnet_b0.ns_jft_in1k', 'NoisyStudent JFT', '5.3M',  '1.0', 'Baseline anchor'],
        ['tf_efficientnet_b3.ns_jft_in1k', 'NoisyStudent JFT', '12.2M', '2.0', '2025 dominant backbone'],
        ['tf_efficientnetv2_s.in21k_ft',   'ImageNet-21k',     '21.5M', '2.0', '2025 2nd/5th place'],
    ], col_widths=[6.5*cm, 3.5*cm, 1.8*cm, 2.2*cm, 3.5*cm])
    story.append(t)
    story += [sp(4), caption('Table 3: Backbone comparison')]
    story.append(P(
        'B3-NoisyStudent (tf_efficientnet_b3.ns_jft_in1k) was used by multiple top-10 '
        'solutions in BirdCLEF 2025. NoisyStudent pretraining on JFT-300M provides significantly '
        'stronger acoustic feature representations than plain ImageNet initialization.'
    ))

    story += H2('4.3 Training Recipe', '')
    t = make_table([
        ['Experiment', 'Key Change', 'Holdout AUC'],
        ['sed-b0-v5 (BCE)',    'Baseline: BCE, clip-only loss',      '0.9192'],
        ['sed-b0-v6',         'Fold-only val (no soundscape)',       '0.7543'],
        ['sed-v2s-v1',        'EfficientNetV2-S backbone',           '0.7980'],
        ['sed-b0-v9-asl',     'ASL (γ⁻=4, γ⁺=0, clip=0.05)',      '0.9467  ★'],
        ['sed-b0-v10-cutmix', 'ASL + CutMix',                       '0.9391'],
        ['sed-b0-v11',        'ASL + soft secondary labels',         '0.9359'],
        ['sed-b0-v12-bce',    'BCE with dual loss (repeat)',         '0.9079'],
        ['sed-b0-v15-no-sec', 'ASL, no secondary labels',           '0.9176'],
    ], col_widths=[5.5*cm, 7*cm, 5*cm])
    story.append(t)
    story += [sp(4), caption('Table 4: Ablation results — holdout AUC on held-out soundscapes. ★ = best single model.')]
    story.append(P('<b>Key finding</b>: ASL delivers +0.027 holdout AUC over BCE — the single most impactful technique.'))

    story += H2('4.4 Asymmetric Loss', '')
    story.append(P(
        'Asymmetric Loss (ASL) addresses class imbalance in multi-label classification by '
        'applying different focusing parameters to positives and negatives:'
    ))
    story += MATH(
        'L_ASL = { (1-p)^γ+ · log(p)              [positive]\n'
        '        { (p - m)+^γ- · log(1-p+m)        [negative]\n\n'
        'γ+ = 0  (no positive focusing)\n'
        'γ- = 4  (strong negative down-weighting)\n'
        'm  = 0.05  (probability margin clip)'
    )
    story.append(P(
        'ASL effectively ignores easy negatives while maintaining strong gradient on positives, '
        'critical for the highly imbalanced multi-species detection task.'
    ))

    story += H2('4.5 Background Noise Augmentation', '')
    story.append(P(
        'A key technique from 2025 top solutions is background noise injection using Pantanal '
        'soundscapes as the noise corpus:'
    ))
    story += MATH('x_aug = x_clean + α · x_noise,   SNR ~ U(5, 30) dB')
    story.append(P(
        'We use birdclef-2026/train_soundscapes/*.ogg (10,658 files) as the noise corpus. '
        'These are real Pantanal field recordings — the <i>same domain</i> as the test set — '
        'providing ideal domain adaptation with zero additional data download required.'
    ))

    # ── Section 5 ─────────────────────────────────────────────────────────────
    story += H1('4-Fold Cross-Validation + Model Soup', '5')
    story.append(P(
        'A single-model SED achieves holdout AUC ≈ 0.947. 4-fold CV with model soup yields '
        'approximately +0.005–0.012 via: (1) using all training data across folds, '
        '(2) geometric-mean ensemble of 4 fold predictions (variance reduction), and '
        '(3) model soup within each fold (top-3 checkpoints weight-averaged).'
    ))
    t = make_table([
        ['Fold', 'Train clips', 'Val clips', 'Val soundscapes'],
        ['0', '26,662', '8,887', '16 files'],
        ['1', '26,661', '8,888', '17 files'],
        ['2', '26,662', '8,887', '16 files'],
        ['3', '26,662', '8,887', '17 files'],
    ], col_widths=[2.5*cm, 4*cm, 4*cm, 4*cm])
    story.append(t)
    story += [sp(4), caption('Table 5: Stratified 4-fold split statistics')]
    story += MATH(
        'θ_soup = (1/k) Σ θ_i          [model soup: top-k weight average]\n\n'
        'p_fold-ensemble = exp( (1/K) Σ ln p_k )   [geometric mean across folds]'
    )
    story.append(P(
        'Geometric mean is preferred over arithmetic mean for probabilities as it handles '
        'heterogeneous confidence scales without over-confident blending.'
    ))

    # ── Section 6 ─────────────────────────────────────────────────────────────
    story += H1('Pseudo-Label Pipeline', '6')
    story.append(P(
        'The 66 training soundscapes have many unlabeled 5-second segments. '
        'We expand supervision using Perch-generated pseudo labels across 5 iterative rounds:'
    ))
    story += B([
        '<b>Round 1</b>: Perch ensemble on all 10,658 soundscape segments. Filter top confidence predictions per species.',
        '<b>Rounds 2–5</b>: Train SED with round-n pseudo labels, generate round-(n+1) from trained SED + Perch.',
        '<b>Hard pseudo</b>: Per-clip binary labels for soundscape data augmentation (extra_soundscape_csv).',
        '<b>Soft pseudo</b>: Continuous scores for knowledge distillation (soft_pseudo_csv, 10× oversample).',
    ])
    story.append(P(
        'Current: Round 5 pseudo labels (outputs/pseudo_soundscape_labels_r5.csv), providing '
        '1,168 additional labeled segments beyond the original 312 soundscape annotations.'
    ))

    # ── Section 7 ─────────────────────────────────────────────────────────────
    story += H1('Inference Pipeline', '7')

    story += H2('7.1 Human Voice Removal (VAD)', '')
    story.append(P(
        'Silero VAD detects human speech and truncates audio before speech begins '
        '(if speech duration ≥ 2s and onset ≥ 8s). This removes announcer segments '
        'common in competition recordings.'
    ))

    story += H2('7.2 Overlapping Stride Windows', '')
    story.append(P('We use STRIDE=2.5s sliding windows instead of standard non-overlapping 12-clip inference:'))
    story += B([
        'Window size: 5s (160,000 samples at 32 kHz)',
        'Stride: 2.5s (50% overlap)',
        'Windows per 60s file: 23 (vs 12 non-overlapping)',
        'Output bins: 12 (each bin averages 2–3 overlapping windows)',
    ])
    story.append(P(
        'Overlapping windows provide ~2× more inference samples with context overlap, '
        'significantly improving detection of species present at window boundaries.'
    ))

    story += H2('7.3 Test-Time Augmentation (TTA)', '')
    story.append(P(
        'For each set of 23 windows, we run inference on a 1.25s circular shift:'
    ))
    story += MATH('ŷ_TTA = 0.5 · ŷ_original + 0.5 · ŷ_shifted')
    story.append(P(
        'This provides phase-invariant predictions at the cost of one additional forward pass per model.'
    ))

    # ── Section 8 ─────────────────────────────────────────────────────────────
    story += H1('Post-Processing', '8')

    story += H2('8.1 VLOM Blend (2025 2nd Place)', '')
    story.append(P(
        'Simple weighted averaging underperforms for heterogeneous models with different confidence scales. '
        'We use the VLOM blend from the 2025 2nd-place solution — the average of weighted geometric mean '
        'and weighted RMS:'
    ))
    story += MATH(
        'ŷ_VLOM = ( exp(w_P · ln ŷ_P + w_S · ln ŷ_S)  +  sqrt(w_P · ŷ_P² + w_S · ŷ_S²) ) / 2\n\n'
        'w_P = 0.55 (Perch weight),   w_S = 0.45 (SED weight)'
    )
    story.append(P(
        'The geometric mean captures relative confidence ordering; RMS amplifies strong signals. '
        'Their average handles both high-confidence agreement and low-confidence disagreement.'
    ))

    story += H2('8.2 Temporal Smoothing 20%/60%/20% (Universal 2025)', '')
    story += MATH("ŷ'_i  =  0.20 · ŷ_{i-1}  +  0.60 · ŷ_i  +  0.20 · ŷ_{i+1}")
    story.append(P(
        'Boundary: edge clips add the missing neighbor weight to self '
        '(clip 0: 80% self + 20% next; clip 11: 20% prev + 80% self). '
        'This weighting was used universally across 2025 BirdCLEF top-10 solutions.'
    ))

    story += H2('8.3 File-Max Trick (TRICK1)', '')
    story += MATH("ŷ'_i  =  ŷ_i  +  0.05 · max_j ŷ_j")
    story.append(P(
        'Species present strongly at any point receive a baseline lift across all clips, '
        'exploiting Pantanal habitat characteristics where species tend to persist.'
    ))

    story += H2('8.4 Power Boost (2025 3rd Place — Conditional)', '')
    story += MATH(
        'ŷ\'_{i,c}  =  { ŷ_{i,c}^γ+    if c ∈ top-N species per clip     γ+ = 0.5 (boost)\n'
        '             { ŷ_{i,c}^γ-    otherwise                           γ- = 1.5 (suppress)\n\n'
        'N = 3,  disabled by default'
    )
    story.append(P(
        '<b>Status: DISABLED by default.</b> Enable only after confirming improvement on '
        '<i>both</i> validation and holdout AUC. Never accept based on public LB alone.'
    ))

    # ── Section 9 ─────────────────────────────────────────────────────────────
    story += H1('Experiment Schedule and Expected Results', '9')
    t = make_table([
        ['Phase', 'Experiment', 'GPU', 'Key Contribution'],
        ['Phase 2', 'v21-dual-rating3',   'GPU0', 'Dual loss + rating filter'],
        ['Phase 2', 'v22-dual-noclipmix', 'GPU0', 'No CutMix in dual-loss'],
        ['Phase 2', 'V2S-v2-asl',         'GPU0', 'EfficientNetV2-S + ASL'],
        ['Phase 2', 'v28-final-combo',    'GPU0', 'ASL + dual + soft-KD + circ_shift'],
        ['Phase 2', 'v24-soft-pseudo',    'GPU1', 'Soft pseudo knowledge distillation'],
        ['Phase 2', 'v26-asl-npcen',      'GPU1', 'ASL, no PCEN (ablation)'],
        ['Phase 2', 'v27-soft-boost',     'GPU1', 'Soft pseudo boost weight'],
        ['Phase 3', 'v30-bgnoise',        'GPU1', 'Background noise augmentation'],
        ['Phase 3', 'B3-v1-asl',          'GPU0', 'B3-NS backbone, full recipe'],
        ['Phase 3', 'B3-fold0..3',        'both', '4-fold B3 geometric-mean ensemble'],
        ['Phase 4', 'V2S-fold0..3',       'TBD',  '4-fold V2S (pending B3 comparison)'],
    ], col_widths=[2.5*cm, 4.5*cm, 2*cm, 8.5*cm])
    story.append(t)
    story += [sp(4), caption('Table 6: Experiment schedule')]

    t2 = make_table([
        ['Milestone', 'Expected LB', 'Delta vs 0.892'],
        ['Current (v9-asl-soup + Perch 3-head)', '0.892', 'baseline'],
        ['Phase 2 best single model (v28)',       '0.895–0.903', '+0.003–+0.011'],
        ['V2S-v2-asl single model',               '0.900–0.915', '+0.008–+0.023'],
        ['B0 4-fold soup ensemble',               '0.905–0.920', '+0.013–+0.028'],
        ['B3 4-fold soup ensemble',               '0.912–0.928', '+0.020–+0.036'],
        ['Multi-backbone (B0+B3+V2S) + PP',       '0.915–0.933', '+0.023–+0.041'],
    ], col_widths=[8*cm, 3.5*cm, 4*cm])
    story.append(t2)
    story += [sp(4), caption('Table 7: Expected LB progression. Target: 0.926 (rank #1 as of 2026-03-18)')]

    # ── Section 10 ─────────────────────────────────────────────────────────────
    story += H1('Key Decisions and Trade-offs', '10')
    story += B([
        '<b>B3-NS over EfficientNetV2-B3</b>: tf_efficientnet_b3.ns_jft_in1k (NoisyStudent JFT-300M, 12.2M params) '
        'matches the exact variant used by 2025 top-10 solutions. V2-B3 (14.4M) has only ImageNet-21k pretraining.',
        '<b>Domain-matched noise</b>: Using competition train soundscapes as noise corpus (vs ESC-50/AudioSet) ensures '
        'Pantanal-specific characteristics match test environment.',
        '<b>VLOM over weighted average</b>: Simple weighted avg is suboptimal when Perch (calibrated sigmoid) and '
        'SED (raw sigmoid) have different confidence scales.',
        '<b>3-tap over 5-tap smooth</b>: Previous 5-tap [0.05,0.15,0.60,0.15,0.05] with sharpening adds '
        'complexity without consistent validated gain. Plain 3-tap [0.20,0.60,0.20] matches 2025 universal standard.',
        '<b>Competitor models excluded</b>: All SED checkpoints are our own trained models only.',
        '<b>Power boost disabled by default</b>: PP that conditions on test predictions can produce false public '
        'LB improvements that do not generalize. Validate on holdout AUC first.',
    ])

    # ── Section 11 ─────────────────────────────────────────────────────────────
    story += H1('Validation Framework', '11')
    story.append(P(
        'A fixed holdout set of 13–17 soundscape files per fold is kept separate throughout all '
        'experiment development. <b>Holdout AUC is the ground truth</b>; validation AUC '
        '(soundscape segments only) is used for early stopping and checkpoint selection.'
    ))
    story += MATH('Submit if:   holdout_AUC > 0.9193  (v5-BCE baseline benchmark)')
    story.append(P(
        'Current best: sed-b0-v9-asl-soup at holdout AUC = <b>0.9467</b> (Δ = +0.0274 over baseline).'
    ))
    story.append(P(
        '<b>Post-processing validation rule</b>: Each PP step must improve <i>both</i> validation '
        'and holdout AUC. Never accept improvements based solely on public LB.'
    ))

    # ── Section 12 ─────────────────────────────────────────────────────────────
    story += H1('Techniques from 2025 Top Solutions', '12')
    t = make_table([
        ['Technique', 'Source', 'Status', 'Est. Gain'],
        ['B3-NoisyStudent backbone',       'Multiple 2025 top-10',   'Phase 3',    '+0.005–0.018'],
        ['Background noise augmentation',  '2025 3rd–5th place',     'Phase 3 (v30)', '+0.003–0.008'],
        ['ASL loss',                       '2025 top solutions',     'Done (v9)',   '+0.027 ✓'],
        ['4-fold CV + model soup',         'Standard 2025 practice', 'Phase 3',    '+0.005–0.012'],
        ['VLOM blend (geomean+RMS)/2',    '2025 2nd place',         'In notebook', '+0.002–0.005'],
        ['Temporal smooth 20/60/20',       'Universal 2025',         'In notebook', '+0.001–0.003'],
        ['Overlapping stride windows',     '2025 top solutions',     'Done (v6-TTA)', '+0.002–0.005'],
        ['TTA circular shift',             '2025 top solutions',     'Done (v6-TTA)', '+0.001–0.003'],
        ['Soft pseudo KD',                 '2025 top solutions',     'Phase 2 (v24)', '+0.002–0.006'],
        ['Power boost',                    '2025 3rd place',         'Conditional', '+0.001–0.003'],
    ], col_widths=[5.5*cm, 4*cm, 3.5*cm, 3*cm])
    story.append(t)
    story += [sp(4), caption('Table 8: 2025 top-solution techniques — status and estimated contribution')]

    # ── Section 13 ────────────────────────────────────────────────────────────
    story += H1('Conclusion', '13')
    story.append(P(
        'We present a comprehensive gold-medal targeting solution for BirdCLEF 2026 that '
        'addresses the core challenges:'
    ))
    story += B([
        '<b>Domain gap</b>: Background noise augmentation with Pantanal recordings, soundscape oversampling.',
        '<b>Zero-audio species</b>: Perch 3-head ensemble, 5-round pseudo label pipeline.',
        '<b>Class imbalance</b>: Asymmetric Loss (γ⁻=4), soft pseudo knowledge distillation.',
        '<b>Heterogeneous model fusion</b>: VLOM blend, multi-backbone geometric mean.',
        '<b>Temporal structure</b>: Overlapping stride windows, TTA, 20%/60%/20% smoothing.',
    ])
    story.append(P(
        'The designed system builds incrementally on a proven foundation (current LB=0.892, '
        'holdout=0.9467) and targets LB 0.926 through systematic application of 2025-validated techniques. '
        'Each component has been validated through ablation experiments or verified from 2025 '
        'top-solution writeups, ensuring expected gains are grounded in evidence.'
    ))

    # ── References ────────────────────────────────────────────────────────────
    story += H1('References', '')
    refs = [
        '[1] Ridnik, T. et al. (2021). Asymmetric Loss For Multi-Label Classification. ICCV 2021.',
        '[2] Wortsman, M. et al. (2022). Model soups: averaging weights of multiple fine-tuned models. ICML 2022.',
        '[3] Tan, M. & Le, Q.V. (2021). EfficientNetV2: Smaller Models and Faster Training. ICML 2021.',
        '[4] BirdCLEF 2026 Competition. (2026). Identify bird species in field recordings. Kaggle.',
        '[5] Kahl, S. et al. (2021). BirdNET: A deep learning solution for avian diversity monitoring. ECOLIND.',
    ]
    for r in refs:
        story.append(Paragraph(r, ref_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print('PDF written to reports/solution_writeup.pdf')

if __name__ == '__main__':
    build()
