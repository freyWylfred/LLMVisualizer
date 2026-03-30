namespace LLMVisualizer
{
    partial class Form1
    {
        private System.ComponentModel.IContainer components = null;

        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
                components.Dispose();
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        private void InitializeComponent()
        {
            menuStrip          = new MenuStrip();
            fileMenuItem       = new ToolStripMenuItem();
            openFileMenuItem   = new ToolStripMenuItem();
            sep1               = new ToolStripSeparator();
            exportPngMenuItem  = new ToolStripMenuItem();
            sep2               = new ToolStripSeparator();
            exitMenuItem       = new ToolStripMenuItem();
            viewMenuItem       = new ToolStripMenuItem();
            resetZoomMenuItem  = new ToolStripMenuItem();
            zoomInMenuItem     = new ToolStripMenuItem();
            zoomOutMenuItem    = new ToolStripMenuItem();
            statusStrip        = new StatusStrip();
            statusLabel        = new ToolStripStatusLabel();
            splitContainer     = new SplitContainer();
            architecturePanel  = new ArchitecturePanel();
            panelInfo          = new Panel();
            labelInfoHeader    = new Label();
            listViewProps      = new ListView();
            colPropName        = new ColumnHeader();
            colPropValue       = new ColumnHeader();

            menuStrip.SuspendLayout();
            statusStrip.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)splitContainer).BeginInit();
            splitContainer.Panel1.SuspendLayout();
            splitContainer.Panel2.SuspendLayout();
            splitContainer.SuspendLayout();
            panelInfo.SuspendLayout();
            SuspendLayout();

            // menuStrip
            menuStrip.Items.AddRange(new ToolStripItem[] { fileMenuItem, viewMenuItem });
            menuStrip.Name = "menuStrip";

            // fileMenuItem
            fileMenuItem.DropDownItems.AddRange(new ToolStripItem[]
            {
                openFileMenuItem, sep1,
                exportPngMenuItem, sep2, exitMenuItem
            });
            fileMenuItem.Name = "fileMenuItem";
            fileMenuItem.Text = "ファイル(&F)";

            // openFileMenuItem
            openFileMenuItem.Name         = "openFileMenuItem";
            openFileMenuItem.ShortcutKeys = Keys.Control | Keys.O;
            openFileMenuItem.Text         = "ファイルを開く(&O)...";
            openFileMenuItem.Click       += OpenFileMenuItem_Click;

            // sep1
            sep1.Name = "sep1";

            // exportPngMenuItem
            exportPngMenuItem.Name   = "exportPngMenuItem";
            exportPngMenuItem.Text   = "PNG として保存(&E)...";
            exportPngMenuItem.Click += ExportPngMenuItem_Click;

            // sep2
            sep2.Name = "sep2";

            // exitMenuItem
            exitMenuItem.Name   = "exitMenuItem";
            exitMenuItem.Text   = "終了(&X)";
            exitMenuItem.Click += ExitMenuItem_Click;

            // viewMenuItem
            viewMenuItem.DropDownItems.AddRange(new ToolStripItem[]
            {
                resetZoomMenuItem, zoomInMenuItem, zoomOutMenuItem
            });
            viewMenuItem.Name = "viewMenuItem";
            viewMenuItem.Text = "表示(&V)";

            // resetZoomMenuItem
            resetZoomMenuItem.Name         = "resetZoomMenuItem";
            resetZoomMenuItem.ShortcutKeys = Keys.Control | Keys.D0;
            resetZoomMenuItem.Text         = "ズームリセット(&R)";
            resetZoomMenuItem.Click       += ResetZoomMenuItem_Click;

            // zoomInMenuItem
            zoomInMenuItem.Name         = "zoomInMenuItem";
            zoomInMenuItem.ShortcutKeys = Keys.Control | Keys.Oemplus;
            zoomInMenuItem.Text         = "拡大(&I)";
            zoomInMenuItem.Click       += ZoomInMenuItem_Click;

            // zoomOutMenuItem
            zoomOutMenuItem.Name         = "zoomOutMenuItem";
            zoomOutMenuItem.ShortcutKeys = Keys.Control | Keys.OemMinus;
            zoomOutMenuItem.Text         = "縮小(&O)";
            zoomOutMenuItem.Click       += ZoomOutMenuItem_Click;

            // statusStrip
            statusStrip.Items.AddRange(new ToolStripItem[] { statusLabel });
            statusStrip.Name = "statusStrip";

            // statusLabel
            statusLabel.Name = "statusLabel";
            statusLabel.Text = "  ファイルを開いてください（メニュー または ドラッグ＆ドロップ）";

            // splitContainer — Horizontal: Panel1=上(アーキテクチャ) / Panel2=下(情報)
            splitContainer.Dock             = DockStyle.Fill;
            splitContainer.Orientation      = Orientation.Horizontal;
            splitContainer.Name             = "splitContainer";
            splitContainer.SplitterDistance = 330;
            splitContainer.Panel1MinSize    = 150;
            splitContainer.Panel2MinSize    = 100;
            splitContainer.Panel1.Controls.Add(architecturePanel);
            splitContainer.Panel2.Controls.Add(panelInfo);

            // architecturePanel
            architecturePanel.Dock      = DockStyle.Fill;
            architecturePanel.Name      = "architecturePanel";
            architecturePanel.BackColor = Color.White;

            // panelInfo
            panelInfo.Dock = DockStyle.Fill;
            panelInfo.Controls.Add(listViewProps);    // Fill — 先に追加
            panelInfo.Controls.Add(labelInfoHeader);  // Top  — 後に追加

            // labelInfoHeader
            labelInfoHeader.AutoSize  = false;
            labelInfoHeader.Dock      = DockStyle.Top;
            labelInfoHeader.Height    = 28;
            labelInfoHeader.Name      = "labelInfoHeader";
            labelInfoHeader.Padding   = new Padding(8, 6, 0, 0);
            labelInfoHeader.Font      = new Font("Segoe UI", 9f, FontStyle.Bold);
            labelInfoHeader.ForeColor = Color.FromArgb(55, 55, 80);
            labelInfoHeader.BackColor = Color.FromArgb(235, 235, 248);
            labelInfoHeader.Text      = "モデル情報";

            // listViewProps
            listViewProps.Columns.AddRange(new ColumnHeader[] { colPropName, colPropValue });
            listViewProps.Dock          = DockStyle.Fill;
            listViewProps.FullRowSelect = true;
            listViewProps.GridLines     = true;
            listViewProps.HeaderStyle   = ColumnHeaderStyle.Nonclickable;
            listViewProps.Name          = "listViewProps";
            listViewProps.View          = View.Details;
            listViewProps.Font          = new Font("Segoe UI", 9f);

            colPropName.Text  = "項目";
            colPropName.Width = 165;
            colPropValue.Text  = "値";
            colPropValue.Width = 200;

            // Form1
            AutoScaleDimensions = new SizeF(7f, 15f);
            AutoScaleMode       = AutoScaleMode.Font;
            ClientSize          = new Size(980, 580);
            Controls.Add(splitContainer);
            Controls.Add(menuStrip);
            Controls.Add(statusStrip);
            MainMenuStrip = menuStrip;
            MinimumSize   = new Size(640, 450);
            Name          = "Form1";
            Text          = "LLM Visualizer";
            AllowDrop     = true;

            menuStrip.ResumeLayout(false);
            menuStrip.PerformLayout();
            statusStrip.ResumeLayout(false);
            statusStrip.PerformLayout();
            splitContainer.Panel1.ResumeLayout(false);
            splitContainer.Panel2.ResumeLayout(false);
            splitContainer.ResumeLayout(false);
            panelInfo.ResumeLayout(false);
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private MenuStrip menuStrip;
        private ToolStripMenuItem fileMenuItem;
        private ToolStripMenuItem openFileMenuItem;
        private ToolStripSeparator sep1;
        private ToolStripMenuItem exportPngMenuItem;
        private ToolStripSeparator sep2;
        private ToolStripMenuItem exitMenuItem;
        private ToolStripMenuItem viewMenuItem;
        private ToolStripMenuItem resetZoomMenuItem;
        private ToolStripMenuItem zoomInMenuItem;
        private ToolStripMenuItem zoomOutMenuItem;
        private StatusStrip statusStrip;
        private ToolStripStatusLabel statusLabel;
        private SplitContainer splitContainer;
        private ArchitecturePanel architecturePanel;
        private Panel panelInfo;
        private Label labelInfoHeader;
        private ListView listViewProps;
        private ColumnHeader colPropName;
        private ColumnHeader colPropValue;
    }
}
