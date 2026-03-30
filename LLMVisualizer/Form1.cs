namespace LLMVisualizer
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            DragEnter += OnDragEnter;
            DragDrop  += OnDragDrop;
        }

        // ── メニューイベント ──────────────────────────────────────────────────

        private void OpenFileMenuItem_Click(object sender, EventArgs e)
        {
            using var dlg = new OpenFileDialog
            {
                Title  = "ファイルを開く",
                Filter = "Python (*.py)|*.py|JSON (*.json)|*.json|GGUF (*.gguf)|*.gguf|すべて (*.*)|*.*"
            };
            if (dlg.ShowDialog() == DialogResult.OK)
                LoadFile(dlg.FileName);
        }

        private void ExportPngMenuItem_Click(object sender, EventArgs e)
        {
            if (architecturePanel.Model == null) return;
            using var dlg = new SaveFileDialog
            {
                Title    = "PNG として保存",
                Filter   = "PNG ファイル (*.png)|*.png",
                FileName = Path.GetFileNameWithoutExtension(architecturePanel.Model.FileName)
                         + "_architecture.png"
            };
            if (dlg.ShowDialog() != DialogResult.OK) return;

            var error = architecturePanel.ExportToPng(dlg.FileName);
            if (error != null)
                MessageBox.Show(error, "エクスポート エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
            else
                statusLabel.Text = $"  PNG 保存完了: {dlg.FileName}";
        }

        private void ExitMenuItem_Click(object sender, EventArgs e) => Close();

        private void ResetZoomMenuItem_Click(object sender, EventArgs e) => architecturePanel.ResetView();
        private void ZoomInMenuItem_Click(object sender, EventArgs e)    => architecturePanel.ZoomIn();
        private void ZoomOutMenuItem_Click(object sender, EventArgs e)   => architecturePanel.ZoomOut();

        // ── ドラッグ＆ドロップ ────────────────────────────────────────────────

        private void OnDragEnter(object? sender, DragEventArgs e)
        {
            if (e.Data?.GetDataPresent(DataFormats.FileDrop) == true)
                e.Effect = DragDropEffects.Copy;
        }

        private void OnDragDrop(object? sender, DragEventArgs e)
        {
            if (e.Data?.GetData(DataFormats.FileDrop) is string[] files && files.Length > 0)
                LoadFile(files[0]);
        }

        // ── ファイル読み込み ──────────────────────────────────────────────────

        private void LoadFile(string path)
        {
            LLMModel? model;
            try
            {
                var ext = Path.GetExtension(path).ToLowerInvariant();
                model = ext switch
                {
                    ".py"   => LLMModel.FromPythonFile(path),
                    ".gguf" => LLMModel.FromGgufFile(path),
                    _       => LLMModel.FromJsonFile(path)
                };
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"ファイルの処理中に予期しないエラーが発生しました。\n\n{ex.GetType().Name}: {ex.Message}\n\n{path}",
                    "エラー",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
                return;
            }

            if (model == null)
            {
                MessageBox.Show(
                    $"ファイルを読み込めませんでした。\n\n{path}",
                    "読み込みエラー",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Warning);
                return;
            }

            if (model.Layers.Count == 0)
            {
                string detail = !string.IsNullOrEmpty(model.LastError)
                    ? model.LastError
                    : "レイヤー情報を抽出できませんでした。";
                MessageBox.Show(
                    $"ファイルを解析しましたが、可視化可能なレイヤーが見つかりませんでした。\n\n{detail}\n\n{path}",
                    "解析結果",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Information);
                return;
            }

            try
            {
                architecturePanel.Model = model;
                UpdateInfoPanel(model);
                statusLabel.Text = $"  {model.FileName}  |  種別: {model.ModelType}"
                                 + $"  |  推定パラメータ数: {model.FormatParameters()}"
                                 + $"  |  レイヤー数: {model.Layers.Count}";
                Text = $"LLM Visualizer — {model.FileName}";
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"モデルの表示中にエラーが発生しました。\n\n{ex.GetType().Name}: {ex.Message}",
                    "表示エラー",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
            }
        }

        // ── 情報パネル更新 ────────────────────────────────────────────────────

        private void UpdateInfoPanel(LLMModel model)
        {
            listViewProps.Items.Clear();

            void Add(string key, string value) =>
                listViewProps.Items.Add(new ListViewItem(new[] { key, value }));

            Add("モデル種別",           model.ModelType);
            Add("語彙数",               model.VocabSize > 0             ? $"{model.VocabSize:N0}"             : "—");
            Add("隠れ次元数",           model.HiddenSize > 0            ? $"{model.HiddenSize:N0}"            : "—");
            Add("レイヤー数",           model.NumHiddenLayers > 0       ? model.NumHiddenLayers.ToString()    : "—");
            Add("アテンションヘッド数",  model.NumAttentionHeads > 0     ? model.NumAttentionHeads.ToString()  : "—");
            if (model.IsGQA)
                Add("KV ヘッド数 (GQA)", model.NumKeyValueHeads.ToString());
            Add("ヘッド次元数",         model.HeadDim > 0               ? model.HeadDim.ToString()            : "—");
            Add("FFN 中間次元数",       model.IntermediateSize > 0      ? $"{model.IntermediateSize:N0}"      : "—");
            Add("最大コンテキスト長",    model.MaxPositionEmbeddings > 0 ? $"{model.MaxPositionEmbeddings:N0}" : "—");
            Add("活性化関数",           !string.IsNullOrEmpty(model.HiddenActivation) ? model.HiddenActivation : "—");
            Add("推定パラメータ数",      model.FormatParameters());
            Add("形式",                 model.SourceFormat);
            Add("レイヤー構成数",       model.Layers.Count.ToString());

            foreach (var layer in model.Layers)
            {
                string info = layer.IsContainer
                    ? $"{layer.LayerType} ({layer.Children.Count}子層" +
                      (layer.RepeatCount > 1 ? $", ×{layer.RepeatCount}" : "") + ")"
                    : $"{layer.LayerType}" +
                      (layer.DimLabel.Length > 0 ? $" [{layer.DimLabel}]" : "");
                Add($"  └ {layer.Name}", info);
            }

            listViewProps.AutoResizeColumn(1, ColumnHeaderAutoResizeStyle.ColumnContent);
        }
    }
}
