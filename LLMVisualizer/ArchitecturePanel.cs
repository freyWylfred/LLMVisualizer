using System.ComponentModel;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;

namespace LLMVisualizer
{
    public sealed class ArchitecturePanel : Panel
    {
        private LLMModel? _model;
        private float _zoom = 1f;
        private PointF _pan = PointF.Empty;
        private Point _dragStart;
        private bool _dragging;

        private readonly Font _fTitle;
        private readonly Font _fDetail;
        private readonly Font _fSmall;

        // Dynamic logical size computed from layers
        private float _logicalW = 400f;
        private float _logicalH = 200f;

        // ── Layout constants ──────────────────────────────────────────────────
        private const float MARGIN = 30f;
        private const float BW = 130f;        // simple block width
        private const float BH = 70f;         // simple block height
        private const float CTN_W = 200f;     // container width
        private const float CTN_PAD = 10f;    // container inner padding
        private const float FH = 28f;         // container header height
        private const float CHILD_H = 38f;    // child block height
        private const float CHILD_GAP = 6f;   // vertical gap between children
        private const float ARROW_W = 40f;    // horizontal arrow width

        // ── Colors ────────────────────────────────────────────────────────────
        private static readonly Color CEmbed = Color.FromArgb(190, 215, 245);
        private static readonly Color CNorm = Color.FromArgb(215, 195, 240);
        private static readonly Color CAttn = Color.FromArgb(160, 225, 190);
        private static readonly Color CLinear = Color.FromArgb(255, 215, 160);
        private static readonly Color COutput = Color.FromArgb(255, 190, 190);
        private static readonly Color CFrame = Color.FromArgb(234, 234, 244);
        private static readonly Color CConv = Color.FromArgb(255, 240, 180);
        private static readonly Color CActivation = Color.FromArgb(220, 235, 220);
        private static readonly Color CDefault = Color.FromArgb(225, 225, 235);
        private static readonly Color CArrow = Color.FromArgb(75, 75, 100);
        private static readonly Color CText = Color.FromArgb(25, 25, 35);
        private static readonly Color CDetail = Color.FromArgb(65, 65, 85);

        public ArchitecturePanel()
        {
            DoubleBuffered = true;
            BackColor = Color.White;
            _fTitle  = new Font("Segoe UI", 8.5f, FontStyle.Bold);
            _fDetail = new Font("Segoe UI", 7.5f);
            _fSmall  = new Font("Segoe UI", 7f);
        }

        [DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
        public LLMModel? Model
        {
            get => _model;
            set { _model = value; ComputeLogicalSize(); ResetView(); }
        }

        public void ResetView() { _zoom = 1f; _pan = PointF.Empty; Invalidate(); }
        public void ZoomIn()    { _zoom = Math.Min(_zoom * 1.2f, 6f);   Invalidate(); }
        public void ZoomOut()   { _zoom = Math.Max(_zoom / 1.2f, 0.1f); Invalidate(); }

        // ── Layout computation ────────────────────────────────────────────────

        private void ComputeLogicalSize()
        {
            if (_model == null || _model.Layers.Count == 0)
            {
                _logicalW = 400f; _logicalH = 200f; return;
            }
            float totalW = MARGIN;
            float maxH = 0f;
            for (int i = 0; i < _model.Layers.Count; i++)
            {
                var l = _model.Layers[i];
                totalW += LayerWidth(l);
                maxH = Math.Max(maxH, LayerHeight(l));
                if (i < _model.Layers.Count - 1) totalW += ARROW_W;
            }
            totalW += MARGIN;
            _logicalW = totalW;
            _logicalH = MARGIN + maxH + MARGIN;
        }

        private static float LayerWidth(NNLayer layer) => layer.IsContainer ? CTN_W : BW;

        private float LayerHeight(NNLayer layer)
        {
            if (!layer.IsContainer) return BH;
            int n = Math.Max(layer.Children.Count, 1);
            return FH + CTN_PAD + n * CHILD_H + Math.Max(0, n - 1) * CHILD_GAP + CTN_PAD;
        }

        private static Color GetColor(NNLayer layer)
        {
            string t = layer.LayerType;
            string n = layer.Name.ToLowerInvariant();
            if (n.Contains("head") || n.Contains("output") || n.Contains("lm_"))
                return COutput;
            if (t is "Embedding" or "Embed" or "Input" or "InputLayer") return CEmbed;
            if (t is "LayerNorm" or "RMSNorm" or "GroupNorm"
                  or "BatchNorm1d" or "BatchNorm2d" or "BatchNorm3d"
                  or "InstanceNorm1d" or "InstanceNorm2d" or "InstanceNorm3d"
                  or "BatchNormalization" or "LayerNormalization" or "Normalization")
                return CNorm;
            if (t is "MultiheadAttention" or "MultiHeadAttention" or "Attention" or "SelfAttention"
                  or "MultiHeadDotProductAttention"
                  or "TransformerEncoder" or "TransformerDecoder"
                  or "TransformerEncoderLayer" or "TransformerDecoderLayer"
                  or "Transformer"
                  or "LSTM" or "GRU" or "RNN" or "RNNCell" or "LSTMCell" or "GRUCell"
                  or "SimpleRNN" or "Bidirectional")
                return CAttn;
            if (t.StartsWith("Conv", StringComparison.Ordinal)
                || t.StartsWith("SeparableConv", StringComparison.Ordinal)
                || t.StartsWith("DepthwiseConv", StringComparison.Ordinal)
                || t.StartsWith("ConvTranspose", StringComparison.Ordinal)
                || t.Contains("Pool"))
                return CConv;
            if (t is "Linear" or "Dense" or "DenseGeneral" or "LazyLinear" or "Bilinear") return CLinear;
            if (t is "ReLU" or "GELU" or "SiLU" or "Sigmoid" or "Tanh" or "Softmax"
                  or "LeakyReLU" or "PReLU" or "ELU" or "SELU" or "Mish"
                  or "LogSoftmax" or "Softplus" or "Softsign" or "Hardsigmoid" or "Hardswish"
                  or "Dropout" or "Dropout2d" or "Dropout3d" or "AlphaDropout"
                  or "Activation" or "Swish"
                  or "SpatialDropout1D" or "SpatialDropout2D")
                return CActivation;
            return CDefault;
        }

        // ── Mouse ─────────────────────────────────────────────────────────────

        protected override void OnMouseWheel(MouseEventArgs e)
        {
            base.OnMouseWheel(e);
            _zoom = Math.Clamp(_zoom * (e.Delta > 0 ? 1.15f : 1f / 1.15f), 0.1f, 6f);
            Invalidate();
        }

        protected override void OnMouseDown(MouseEventArgs e)
        {
            base.OnMouseDown(e);
            if (e.Button == MouseButtons.Left)
            { _dragging = true; _dragStart = e.Location; Cursor = Cursors.SizeAll; }
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            if (!_dragging) return;
            _pan = new PointF(_pan.X + e.X - _dragStart.X, _pan.Y + e.Y - _dragStart.Y);
            _dragStart = e.Location;
            Invalidate();
        }

        protected override void OnMouseUp(MouseEventArgs e)
        {
            base.OnMouseUp(e);
            _dragging = false;
            Cursor = Cursors.Default;
        }

        // ── Paint ─────────────────────────────────────────────────────────────

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            var g = e.Graphics;
            g.SmoothingMode     = SmoothingMode.AntiAlias;
            g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.ClearTypeGridFit;

            if (_model == null) { DrawPlaceholder(g); return; }

            try
            {
                g.Clear(Color.White);
                float tx = Width  / 2f - _logicalW / 2f * _zoom + _pan.X;
                float ty = Height / 2f - _logicalH / 2f * _zoom + _pan.Y;
                g.TranslateTransform(tx, ty);
                g.ScaleTransform(_zoom, _zoom);
                DrawModel(g, _model);
            }
            catch (Exception ex)
            {
                g.ResetTransform();
                g.Clear(Color.White);
                using var b = new SolidBrush(Color.FromArgb(180, 50, 50));
                g.DrawString($"描画エラー: {ex.Message}", _fDetail, b, 10f, 10f);
            }
        }

        private void DrawPlaceholder(Graphics g)
        {
            g.Clear(Color.FromArgb(245, 245, 250));
            string s1 = "ファイルを開くか、ドラッグ＆ドロップしてください";
            string s2 = "対応形式: Python (.py) / JSON / GGUF";
            using var b1 = new SolidBrush(Color.FromArgb(120, 120, 145));
            using var b2 = new SolidBrush(Color.FromArgb(160, 160, 185));
            var sz1 = g.MeasureString(s1, _fTitle);
            var sz2 = g.MeasureString(s2, _fDetail);
            float cx = Width / 2f, cy = Height / 2f;
            g.DrawString(s1, _fTitle,  b1, cx - sz1.Width / 2f, cy - sz1.Height - 4f);
            g.DrawString(s2, _fDetail, b2, cx - sz2.Width / 2f, cy + 4f);
        }

        // ── Model drawing (dynamic, layer-driven) ─────────────────────────────

        private void DrawModel(Graphics g, LLMModel m)
        {
            var layers = m.Layers;
            if (layers.Count == 0) return;

            float x = MARGIN;
            float midY = _logicalH / 2f;

            for (int i = 0; i < layers.Count; i++)
            {
                var layer = layers[i];
                float w = LayerWidth(layer);
                float h = LayerHeight(layer);
                float y = midY - h / 2f;

                if (layer.IsContainer)
                    DrawContainer(g, layer, x, y, w, h);
                else
                    DrawSimpleBlock(g, layer, x, y, w, h);

                if (i < layers.Count - 1)
                {
                    DrawHArrow(g, x + w, midY, ARROW_W);
                    x += w + ARROW_W;
                }
                else
                {
                    x += w;
                }
            }
        }

        private void DrawSimpleBlock(Graphics g, NNLayer layer, float x, float y, float w, float h)
        {
            var fill = GetColor(layer);
            var r = new RectangleF(x, y, w, h);
            using (var b = new SolidBrush(fill)) RoundFill(g, b, r, 8f);
            using (var p = new Pen(Darken(fill, 0.26f), 1.2f)) RoundDraw(g, p, r, 8f);

            string title = layer.Name;
            string sub   = layer.Activation.Length > 0
                ? $"{layer.LayerType} ({layer.Activation})"
                : layer.LayerType;
            string dim   = layer.DimLabel;

            float lh = _fTitle.GetHeight(g);
            float dh = _fDetail.GetHeight(g);
            bool showSub = sub != title;
            bool showDim = dim.Length > 0;
            float totalH = lh
                + (showSub ? dh + 2f : 0f)
                + (showDim ? dh + 2f : 0f);
            float ty = y + (h - totalH) / 2f;

            using var tb = new SolidBrush(CText);
            using var db = new SolidBrush(CDetail);

            CenterText(g, title, _fTitle, tb, x, ty, w);
            ty += lh + 2f;

            if (showSub)
            {
                CenterText(g, sub, _fDetail, db, x, ty, w);
                ty += dh + 2f;
            }
            if (showDim)
                CenterText(g, dim, _fDetail, db, x, ty, w);
        }

        private void DrawContainer(Graphics g, NNLayer layer, float x, float y, float w, float h)
        {
            var frame = new RectangleF(x, y, w, h);
            using (var fb = new SolidBrush(CFrame)) RoundFill(g, fb, frame, 10f);
            using (var fp = new Pen(Color.FromArgb(170, 170, 200), 1.5f)) RoundDraw(g, fp, frame, 10f);

            string lbl = layer.RepeatCount > 1
                ? $"{layer.Name}: {layer.LayerType} \u00d7{layer.RepeatCount}"
                : $"{layer.Name}: {layer.LayerType}";
            using var tb = new SolidBrush(Color.FromArgb(55, 55, 88));
            var lblSz = g.MeasureString(lbl, _fTitle);
            g.DrawString(lbl, _fTitle, tb, x + (w - lblSz.Width) / 2f, y + (FH - lblSz.Height) / 2f);

            float cy = y + FH + CTN_PAD;
            float cx = x + CTN_PAD;
            float childW = w - CTN_PAD * 2f;

            for (int i = 0; i < layer.Children.Count; i++)
            {
                DrawChildBlock(g, layer.Children[i], cx, cy, childW, CHILD_H);
                if (i < layer.Children.Count - 1)
                    DrawVArrow(g, cx + childW / 2f, cy + CHILD_H, CHILD_GAP);
                cy += CHILD_H + CHILD_GAP;
            }
        }

        private void DrawChildBlock(Graphics g, NNLayer layer, float x, float y, float w, float h)
        {
            var fill = GetColor(layer);
            var r = new RectangleF(x, y, w, h);
            using (var b = new SolidBrush(fill)) RoundFill(g, b, r, 6f);
            using (var p = new Pen(Darken(fill, 0.26f), 1f)) RoundDraw(g, p, r, 6f);

            string title = layer.Activation.Length > 0
                ? $"{layer.Name}: {layer.LayerType} ({layer.Activation})"
                : $"{layer.Name}: {layer.LayerType}";
            string dim   = layer.DimLabel;
            float lh = _fSmall.GetHeight(g);
            float totalH = lh + (dim.Length > 0 ? 2f + lh : 0f);
            float ty = y + (h - totalH) / 2f;

            using var tb = new SolidBrush(CText);
            CenterText(g, title, _fSmall, tb, x, ty, w);

            if (dim.Length > 0)
            {
                ty += lh + 2f;
                using var db = new SolidBrush(CDetail);
                CenterText(g, dim, _fSmall, db, x, ty, w);
            }
        }

        // ── Helpers ───────────────────────────────────────────────────────────

        private static void CenterText(Graphics g, string s, Font f, Brush b, float x, float ty, float w)
        {
            var sz = g.MeasureString(s, f);
            g.DrawString(s, f, b, x + (w - sz.Width) / 2f, ty);
        }

        private void DrawHArrow(Graphics g, float x, float cy, float w)
        {
            using var p = new Pen(CArrow, 1.8f) { CustomEndCap = new AdjustableArrowCap(4f, 5f) };
            g.DrawLine(p, x, cy, x + w - 4f, cy);
        }

        private void DrawVArrow(Graphics g, float cx, float y, float h)
        {
            using var p = new Pen(Color.FromArgb(115, 115, 140), 1.2f)
            { CustomEndCap = new AdjustableArrowCap(3f, 3.5f) };
            g.DrawLine(p, cx, y, cx, y + h - 2f);
        }

        // ── GDI+ helpers ──────────────────────────────────────────────────────

        private static void RoundFill(Graphics g, Brush b, RectangleF r, float rad)
        {
            using var path = RoundPath(r, rad);
            g.FillPath(b, path);
        }

        private static void RoundDraw(Graphics g, Pen p, RectangleF r, float rad)
        {
            using var path = RoundPath(r, rad);
            g.DrawPath(p, path);
        }

        private static GraphicsPath RoundPath(RectangleF r, float rad)
        {
            float d = rad * 2f;
            var path = new GraphicsPath();
            path.AddArc(r.Left,      r.Top,       d, d, 180f, 90f);
            path.AddArc(r.Right - d, r.Top,       d, d, 270f, 90f);
            path.AddArc(r.Right - d, r.Bottom - d, d, d, 0f,   90f);
            path.AddArc(r.Left,      r.Bottom - d, d, d, 90f,  90f);
            path.CloseFigure();
            return path;
        }

        private static Color Darken(Color c, float f) =>
            Color.FromArgb(c.A, (int)(c.R * (1f - f)), (int)(c.G * (1f - f)), (int)(c.B * (1f - f)));

        // ── Export ────────────────────────────────────────────────────────────

        public string? ExportToPng(string path)
        {
            if (_model == null) return "モデルが読み込まれていません。";
            try
            {
                const float scale = 2f;
                int W = Math.Max((int)(_logicalW * scale), 1);
                int H = Math.Max((int)(_logicalH * scale), 1);
                using var bmp = new Bitmap(W, H);
                using var g = Graphics.FromImage(bmp);
                g.SmoothingMode     = SmoothingMode.AntiAlias;
                g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.ClearTypeGridFit;
                g.Clear(Color.White);
                g.ScaleTransform(scale, scale);
                DrawModel(g, _model);
                bmp.Save(path, ImageFormat.Png);
                return null;
            }
            catch (Exception ex)
            {
                return $"PNG 保存エラー: {ex.Message}";
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) { _fTitle.Dispose(); _fDetail.Dispose(); _fSmall.Dispose(); }
            base.Dispose(disposing);
        }
    }
}
