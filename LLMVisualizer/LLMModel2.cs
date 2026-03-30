using System.Text.Json;
using System.Text.RegularExpressions;

namespace LLMVisualizer
{
    // ── ニューラルネットワーク層 ──────────────────────────────────────────

    public class NNLayer
    {
        public string Name { get; set; } = "";
        public string LayerType { get; set; } = "";
        public string RawArgs { get; set; } = "";
        public string Activation { get; set; } = "";
        public List<int> Dims { get; set; } = [];
        public int RepeatCount { get; set; } = 1;
        public List<NNLayer> Children { get; set; } = [];

        public bool IsContainer => Children.Count > 0;

        public string DimLabel => Dims.Count switch
        {
            0 => "",
            1 => $"{Dims[0]:N0}",
            _ => string.Join(" → ", Dims.Select(d => $"{d:N0}"))
        };
    }

    // ── モデル情報 ────────────────────────────────────────────────────────

    public class LLMModel
    {
        public string ModelType { get; set; } = "unknown";
        public int VocabSize { get; set; }
        public int HiddenSize { get; set; }
        public int NumHiddenLayers { get; set; }
        public int NumAttentionHeads { get; set; }
        public int NumKeyValueHeads { get; set; }
        public int IntermediateSize { get; set; }
        public int MaxPositionEmbeddings { get; set; }
        public string HiddenActivation { get; set; } = "";
        public string FileName { get; set; } = "";
        public string FilePath { get; set; } = "";
        public string SourceFormat { get; set; } = "";
        public List<NNLayer> Layers { get; } = [];

        /// <summary>パース失敗時の診断情報</summary>
        public string? LastError { get; set; }

        public int HeadDim => (NumAttentionHeads > 0 && HiddenSize > 0)
            ? HiddenSize / NumAttentionHeads : 0;
        public bool IsGQA => NumKeyValueHeads > 0 && NumKeyValueHeads != NumAttentionHeads;

        // ══════════════════════════════════════════════════════════════════
        //  Python (.py)
        // ══════════════════════════════════════════════════════════════════

        public static LLMModel? FromPythonFile(string path)
        {
            var model = new LLMModel
            {
                FileName = Path.GetFileName(path),
                FilePath = path,
                SourceFormat = "Python"
            };
            try
            {
                var text = File.ReadAllText(path);

                var vars = CollectVariables(text);
                var classes = ParseModuleClasses(text);

                // メインクラス = 他のクラスからサブモジュールとして参照されていないクラス
                ClassBlock? mainClass = null;
                if (classes.Count > 0)
                {
                    var subRefs = new HashSet<string>();
                    foreach (var c in classes)
                        foreach (var other in classes.Where(o => o.Name != c.Name))
                            if (c.InitBody.Contains(other.Name))
                                subRefs.Add(other.Name);
                    mainClass = classes.FirstOrDefault(c => !subRefs.Contains(c.Name)) ?? classes[0];
                }

                string initBody = mainClass?.InitBody ?? text;
                model.ModelType = mainClass?.Name ?? "Model";

                if (mainClass != null)
                    foreach (var (k, v) in ExtractDefaultParams(mainClass.InitSig))
                        vars.TryAdd(k, v);

                foreach (var (name, type, args) in ExtractSelfAssignments(initBody))
                {
                    var layer = BuildLayer(name, type, args, vars, classes);
                    if (layer != null)
                        model.Layers.Add(layer);
                }

                // forward() / __call__() / construct() の呼出順でレイヤーを並べ替え
                if (!string.IsNullOrEmpty(mainClass?.ForwardBody))
                    ReorderByForward(model, mainClass!.ForwardBody);

                // Flax @nn.compact: __call__ 内でレイヤーがインラインで定義されるパターン
                if (model.Layers.Count == 0 && !string.IsNullOrEmpty(mainClass?.ForwardBody))
                    ParseFlaxCompact(mainClass!.ForwardBody, vars, model);

                // TF/Keras Sequential / Functional API fallback
                if (model.Layers.Count == 0)
                    ParseKerasLayers(text, vars, model);

                // フレームワーク自動検出
                if (model.Layers.Count > 0)
                    DetectFramework(text, model);

                FillMetadataFromLayers(model);

                if (model.Layers.Count == 0)
                {
                    model.LastError = classes.Count == 0
                        ? "nn.Module / Model クラスが見つかりませんでした。"
                        : $"クラス '{model.ModelType}' からレイヤーを抽出できませんでした。";
                    return model;
                }
                return model;
            }
            catch (IOException ex)
            {
                model.LastError = $"ファイル読み込みエラー: {ex.Message}";
                return model;
            }
            catch (Exception ex)
            {
                model.LastError = $"Python パース エラー: {ex.GetType().Name}: {ex.Message}";
                return model;
            }
        }

        // ── Python パーサー内部 ───────────────────────────────────────────

        private record ClassBlock(string Name, string InitSig, string InitBody, string ForwardBody);

        private static Dictionary<string, int> CollectVariables(string text)
        {
            var vars = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (Match m in Regex.Matches(text,
                @"^\s*(\w+)\s*=\s*(\d+)\s*(?:#.*)?$", RegexOptions.Multiline))
                vars.TryAdd(m.Groups[1].Value, int.Parse(m.Groups[2].Value));
            foreach (Match m in Regex.Matches(text, @"[""'](\w+)[""']\s*:\s*(\d+)"))
                vars.TryAdd(m.Groups[1].Value, int.Parse(m.Groups[2].Value));
            return vars;
        }

        private static Dictionary<string, int> ExtractDefaultParams(string sig)
        {
            var d = new Dictionary<string, int>();
            foreach (Match m in Regex.Matches(sig, @"(\w+)\s*=\s*(\d+)"))
                d.TryAdd(m.Groups[1].Value, int.Parse(m.Groups[2].Value));
            return d;
        }

        private static List<ClassBlock> ParseModuleClasses(string text)
        {
            var results = new List<ClassBlock>();
            var lines = text.Split('\n');
            for (int i = 0; i < lines.Length; i++)
            {
                var cm = Regex.Match(lines[i], @"^(\s*)class\s+(\w+)\s*\(");
                if (!cm.Success) continue;

                int indent = cm.Groups[1].Value.Length;
                string header = lines[i];
                int j = i;
                while (!header.Contains(':') && j + 1 < lines.Length)
                    header += " " + lines[++j].Trim();
                // PyTorch=Module, TF/Keras=Model, PaddlePaddle=Layer, MindSpore=Cell, Flax=Module, Haiku=Module
                if (!Regex.IsMatch(header, @"(?:Module|Model|Layer|Cell)\b")) continue;

                string className = cm.Groups[2].Value;
                string initSig = "", initBody = "", forwardBody = "";

                for (int k = j + 1; k < lines.Length; k++)
                {
                    string t = lines[k].TrimStart();
                    int ti = lines[k].Length - t.Length;
                    if (t.Length > 0 && ti <= indent && k > j + 1) break;

                    // __init__ (PyTorch, TF, PaddlePaddle, MindSpore, Haiku)
                    if (t.StartsWith("def __init__"))
                    {
                        (initSig, initBody) = ExtractMethodSigAndBody(lines, k, ti);
                    }
                    // setup() (Flax)
                    else if (t.StartsWith("def setup") && string.IsNullOrEmpty(initBody))
                    {
                        (initSig, initBody) = ExtractMethodSigAndBody(lines, k, ti);
                    }
                    // forward() (PyTorch, PaddlePaddle)
                    else if (t.StartsWith("def forward"))
                    {
                        (_, forwardBody) = ExtractMethodSigAndBody(lines, k, ti);
                    }
                    // __call__() (Flax, Haiku)
                    else if (t.StartsWith("def __call__") && string.IsNullOrEmpty(forwardBody))
                    {
                        (_, forwardBody) = ExtractMethodSigAndBody(lines, k, ti);
                    }
                    // construct() (MindSpore)
                    else if (t.StartsWith("def construct") && string.IsNullOrEmpty(forwardBody))
                    {
                        (_, forwardBody) = ExtractMethodSigAndBody(lines, k, ti);
                    }
                }
                results.Add(new ClassBlock(className, initSig, initBody, forwardBody));
            }
            return results;
        }

        /// <summary>メソッドのシグネチャと本体を抽出する共通ヘルパー</summary>
        private static (string sig, string body) ExtractMethodSigAndBody(string[] lines, int k, int defIndent)
        {
            var sb = new System.Text.StringBuilder(lines[k]);
            int pd = lines[k].Count(c => c == '(') - lines[k].Count(c => c == ')');
            int se = k;
            while (pd > 0 && se + 1 < lines.Length)
            {
                se++;
                sb.Append(' ').Append(lines[se].Trim());
                pd += lines[se].Count(c => c == '(') - lines[se].Count(c => c == ')');
            }
            string sig = sb.ToString();

            // コロンが見つからなかった場合はシグネチャのみ返す
            if (!lines[se].TrimEnd().EndsWith(":"))
                return (sig, "");

            var bb = new System.Text.StringBuilder();
            for (int m2 = se + 1; m2 < lines.Length; m2++)
            {
                string bl = lines[m2];
                string bt = bl.TrimStart();
                int bi = bl.Length - bt.Length;
                if (bt.Length == 0) { bb.AppendLine(); continue; }
                if (bi <= defIndent) break;
                bb.AppendLine(bl);
            }
            return (sig, bb.ToString());
        }

        private static List<(string name, string type, string args)> ExtractSelfAssignments(string text)
        {
            var results = new List<(string, string, string)>();
            // PyTorch: nn.XXX  TF/Keras: layers./tf.keras.layers./keras.layers.
            // PaddlePaddle: paddle.nn.  Haiku: hk.  MindSpore: mindspore.nn./ms.nn.
            // Flax: nn. (same as PyTorch prefix)
            foreach (Match m in Regex.Matches(text,
                @"self\.(\w+)\s*=\s*(?:nn\.|layers\.|tf\.keras\.layers\.|keras\.layers\.|paddle\.nn\.|paddle\.|hk\.|mindspore\.nn\.|ms\.nn\.)?([A-Z]\w*)\s*\("))
            {
                int pStart = m.Index + m.Length - 1;
                results.Add((m.Groups[1].Value, m.Groups[2].Value,
                    ExtractBalanced(text, pStart, '(', ')')));
            }
            return results;
        }

        private static string ExtractBalanced(string text, int openPos, char open, char close)
        {
            if (openPos < 0 || openPos >= text.Length) return "";
            int depth = 1, i = openPos + 1;
            while (i < text.Length && depth > 0)
            {
                char c = text[i];
                if (c == open) depth++;
                else if (c == close) depth--;
                else if (c is '"' or '\'')
                {
                    char q = c; i++;
                    while (i < text.Length && text[i] != q) { if (text[i] == '\\') i++; i++; }
                }
                i++;
            }
            int start = openPos + 1;
            int end = Math.Min(i - 1, text.Length);
            return end > start ? text[start..end] : "";
        }

        private static NNLayer? BuildLayer(string name, string type, string rawArgs,
            Dictionary<string, int> vars, List<ClassBlock> classes)
        {
            if (type is "Parameter" or "ParameterList" or "ParameterDict" or "Buffer")
                return null;

            var layer = new NNLayer { Name = name, LayerType = type, RawArgs = rawArgs };

            if (type is "ModuleList" or "LayerList" or "CellList")
            {
                var rangeM = Regex.Match(rawArgs, @"range\s*\(\s*([^)]+)\s*\)");
                if (rangeM.Success)
                    layer.RepeatCount = ResolveInt(rangeM.Groups[1].Value.Trim(), vars);
                var elemM = Regex.Match(rawArgs, @"([A-Z]\w*)\s*\(");
                if (elemM.Success)
                {
                    layer.LayerType = elemM.Groups[1].Value;
                    ExpandSubClass(layer, elemM.Groups[1].Value, vars, classes);
                }
            }
            else if (type == "ModuleDict")
            {
                // nn.ModuleDict({'name': nn.XXX(args), ...})
                foreach (Match m in Regex.Matches(rawArgs,
                    @"['""](\w+)['""]\s*:\s*(?:nn\.)?([A-Z]\w*)\s*\("))
                {
                    string cArgs = ExtractBalanced(rawArgs, m.Index + m.Length - 1, '(', ')');
                    var child = BuildLayer(m.Groups[1].Value, m.Groups[2].Value, cArgs, vars, classes);
                    if (child != null) layer.Children.Add(child);
                }
            }
            else if (type is "TransformerEncoder" or "TransformerDecoder")
            {
                // nn.TransformerEncoder(encoder_layer, num_layers=N)
                var nlM = Regex.Match(rawArgs, @"num_layers\s*=\s*(\w+)");
                if (!nlM.Success)
                {
                    // 2nd positional arg
                    var parts = SplitTopLevel(rawArgs, ',');
                    if (parts.Count >= 2) nlM = Regex.Match(parts[1].Trim(), @"^(\w+)$");
                }
                if (nlM != null && nlM.Success)
                    layer.RepeatCount = ResolveInt(nlM.Groups[1].Value, vars);

                // inline TransformerEncoderLayer/DecoderLayer
                var inlineM = Regex.Match(rawArgs, @"(?:nn\.)?Transformer(?:Encoder|Decoder)Layer\s*\(");
                if (inlineM.Success)
                {
                    string layerArgs = ExtractBalanced(rawArgs, inlineM.Index + inlineM.Length - 1, '(', ')');
                    BuildTransformerLayerChildren(layer, layerArgs, vars);
                }
                else
                {
                    // 1st positional arg may reference a variable defined in __init__
                    var refM = Regex.Match(rawArgs, @"^\s*(?:nn\.)?Transformer(?:Encoder|Decoder)Layer");
                    if (!refM.Success)
                        BuildTransformerLayerChildren(layer, rawArgs, vars);
                }
            }
            else if (type is "TransformerEncoderLayer" or "TransformerDecoderLayer")
            {
                BuildTransformerLayerChildren(layer, rawArgs, vars);
            }
            else if (type == "Sequential")
            {
                // OrderedDict 対応: nn.Sequential(OrderedDict([('name', nn.XXX(args)), ...]))
                bool isOrderedDict = rawArgs.Contains("OrderedDict");
                int idx = 0;
                if (isOrderedDict)
                {
                    foreach (Match m in Regex.Matches(rawArgs,
                        @"\(\s*['""](\w+)['""]\s*,\s*(?:nn\.)?([A-Z]\w*)\s*\("))
                    {
                        string cArgs = ExtractBalanced(rawArgs, m.Index + m.Length - 1, '(', ')');
                        layer.Children.Add(new NNLayer
                        {
                            Name = m.Groups[1].Value,
                            LayerType = m.Groups[2].Value,
                            RawArgs = cArgs,
                            Dims = ExtractDims(cArgs, vars)
                        });
                    }
                }
                else
                {
                    foreach (Match m in Regex.Matches(rawArgs, @"(?:nn\.)?([A-Z]\w*)\s*\("))
                    {
                        string cArgs = ExtractBalanced(rawArgs, m.Index + m.Length - 1, '(', ')');
                        layer.Children.Add(new NNLayer
                        {
                            Name = $"[{idx}]",
                            LayerType = m.Groups[1].Value,
                            RawArgs = cArgs,
                            Dims = ExtractDims(cArgs, vars)
                        });
                        idx++;
                    }
                }
            }
            else
            {
                // Flax のキーワード引数型は専用の次元抽出を使用
                if (type is "Embed" or "DenseGeneral" or "MultiHeadDotProductAttention" or "SelfAttention")
                    layer.Dims = ExtractFlaxDims(type, rawArgs, vars);
                else
                    layer.Dims = ExtractDims(rawArgs, vars);

                // キーワード引数で dims が取れなかった場合 Flax 共通キーワードをフォールバック
                if (layer.Dims.Count == 0 && rawArgs.Contains('='))
                {
                    var flaxDims = ExtractFlaxDims(type, rawArgs, vars);
                    if (flaxDims.Count > 0) layer.Dims = flaxDims;
                }

                ExpandSubClass(layer, type, vars, classes);
            }
            return layer;
        }

        private static void ExpandSubClass(NNLayer parent, string typeName,
            Dictionary<string, int> vars, List<ClassBlock> classes)
        {
            var sub = classes.FirstOrDefault(c => c.Name == typeName);
            if (sub == null) return;
            var subVars = new Dictionary<string, int>(vars);
            foreach (var (k, v) in ExtractDefaultParams(sub.InitSig))
                subVars.TryAdd(k, v);
            foreach (var (sn, st, sa) in ExtractSelfAssignments(sub.InitBody))
            {
                var child = BuildLayer(sn, st, sa, subVars, classes);
                if (child != null) parent.Children.Add(child);
            }
        }

        // ── Flax @nn.compact パーサー ──────────────────────────────────────

        private static void ParseFlaxCompact(string callBody, Dictionary<string, int> vars, LLMModel model)
        {
            // Flax @nn.compact: __call__ 内で nn.Dense(features=128)(x) のようにレイヤーをインライン定義
            // パターン: nn.XXX(args)(prev) or x = nn.XXX(args)(prev)
            int idx = 0;
            foreach (Match m in Regex.Matches(callBody,
                @"(?:nn\.|flax\.linen\.)([A-Z]\w*)\s*\("))
            {
                string layerType = m.Groups[1].Value;
                string args = ExtractBalanced(callBody, m.Index + m.Length - 1, '(', ')');
                string name = ExtractFlaxName(callBody, m.Index) ?? $"layer_{idx}";
                var layer = new NNLayer
                {
                    Name = name,
                    LayerType = NormalizeFlaxType(layerType),
                    RawArgs = args,
                    Dims = ExtractFlaxDims(layerType, args, vars)
                };
                model.Layers.Add(layer);
                idx++;
            }
        }

        private static string? ExtractFlaxName(string text, int matchStart)
        {
            // name='' keyword
            int searchStart = Math.Max(0, matchStart - 200);
            string region = text[searchStart..Math.Min(text.Length, matchStart + 500)];
            var nameM = Regex.Match(region, @"name\s*=\s*['""]([^'""]+)['""]");
            if (nameM.Success) return nameM.Groups[1].Value;
            // x = nn.XXX(...) の左辺変数名
            string before = text[..matchStart];
            var varM = Regex.Match(before, @"(\w+)\s*=\s*$");
            return varM.Success ? varM.Groups[1].Value : null;
        }

        private static string NormalizeFlaxType(string flaxType) => flaxType switch
        {
            "Embed" => "Embedding",
            "DenseGeneral" => "Linear",
            "MultiHeadDotProductAttention" => "MultiheadAttention",
            "SelfAttention" => "MultiheadAttention",
            _ => flaxType
        };

        private static List<int> ExtractFlaxDims(string layerType, string args, Dictionary<string, int> vars)
        {
            var dims = new List<int>();

            // Flax は keyword 引数が主流
            int ResolveKw(string keyword)
            {
                var kv = Regex.Match(args, $@"{keyword}\s*=\s*([^,\)]+)");
                return kv.Success ? ResolveInt(kv.Groups[1].Value.Trim(), vars) : 0;
            }

            if (layerType is "Embed")
            {
                int ne = ResolveKw("num_embeddings");
                int feat = ResolveKw("features");
                if (ne > 0) dims.Add(ne);
                if (feat > 0) dims.Add(feat);
                return dims;
            }
            if (layerType is "Dense" or "DenseGeneral")
            {
                int feat = ResolveKw("features");
                if (feat > 0) dims.Add(feat);
                return dims;
            }
            if (layerType is "MultiHeadDotProductAttention" or "SelfAttention")
            {
                int nh = ResolveKw("num_heads");
                int qkv = ResolveKw("qkv_features");
                if (qkv > 0) dims.Add(qkv);
                if (nh > 0) dims.Add(nh);
                return dims;
            }
            if (layerType is "Conv")
            {
                int feat = ResolveKw("features");
                if (feat > 0) dims.Add(feat);
                return dims;
            }

            // fallback: positional args
            return ExtractDims(args, vars);
        }

        // ── フレームワーク検出 ───────────────────────────────────────────

        private static void DetectFramework(string text, LLMModel model)
        {
            if (Regex.IsMatch(text, @"(?:import\s+flax|from\s+flax|flax\.linen)"))
            { model.SourceFormat = "Python (Flax/JAX)"; return; }
            if (Regex.IsMatch(text, @"(?:import\s+haiku|from\s+haiku|import\s+haiku\s+as\s+hk)"))
            { model.SourceFormat = "Python (Haiku/JAX)"; return; }
            if (Regex.IsMatch(text, @"(?:import\s+paddle|from\s+paddle)"))
            { model.SourceFormat = "Python (PaddlePaddle)"; return; }
            if (Regex.IsMatch(text, @"(?:import\s+mindspore|from\s+mindspore)"))
            { model.SourceFormat = "Python (MindSpore)"; return; }
            if (Regex.IsMatch(text, @"(?:import\s+torch|from\s+torch)"))
            { model.SourceFormat = "Python (PyTorch)"; return; }
            if (Regex.IsMatch(text, @"(?:import\s+tensorflow|from\s+tensorflow|import\s+keras|from\s+keras)"))
            { model.SourceFormat = "Python (TensorFlow/Keras)"; return; }
        }

        // ── PyTorch forward() 解析 ───────────────────────────────────────

        private static void ReorderByForward(LLMModel model, string forwardBody)
        {
            if (string.IsNullOrWhiteSpace(forwardBody)) return;

            // forward 本文から self.xxx の参照順を行単位で抽出
            // (呼出 self.xxx() と属性参照 self.xxx を同一パスで処理し、出現順を維持)
            var callOrder = new List<string>();
            foreach (Match m in Regex.Matches(forwardBody, @"self\.(\w+)"))
            {
                string n = m.Groups[1].Value;
                if (n is "__" or "training") continue;
                if (!callOrder.Contains(n)) callOrder.Add(n);
            }

            if (callOrder.Count == 0) return;

            var dict = new Dictionary<string, NNLayer>();
            foreach (var layer in model.Layers)
                dict.TryAdd(layer.Name, layer);

            var ordered = new List<NNLayer>();
            foreach (var name in callOrder)
            {
                if (dict.TryGetValue(name, out var layer))
                {
                    ordered.Add(layer);
                    dict.Remove(name);
                }
            }
            // forward に出現しなかったレイヤーも末尾に追加
            foreach (var layer in model.Layers)
                if (dict.ContainsKey(layer.Name))
                    ordered.Add(layer);

            model.Layers.Clear();
            model.Layers.AddRange(ordered);

            // F.relu / torch.relu 等の関数的活性化を検出
            DetectFunctionalActivations(model, forwardBody);
        }

        private static void DetectFunctionalActivations(LLMModel model, string forwardBody)
        {
            // パターン: F.relu(self.xxx(...)) / torch.relu(self.xxx(...))
            // あるいは  x = self.xxx(x) \n x = F.relu(x) のような順番
            var lines = forwardBody.Split('\n');
            string? lastLayerName = null;

            foreach (var line in lines)
            {
                var callM = Regex.Match(line, @"self\.(\w+)\s*\(");
                if (callM.Success) lastLayerName = callM.Groups[1].Value;

                // F.relu / F.gelu / F.silu / torch.relu etc.
                var actM = Regex.Match(line,
                    @"(?:F\.|torch\.|torch\.nn\.functional\.)(relu|gelu|silu|elu|leaky_relu|mish|tanh|sigmoid|softmax)\s*\(");
                if (actM.Success && lastLayerName != null)
                {
                    var layer = model.Layers.FirstOrDefault(l => l.Name == lastLayerName);
                    if (layer != null && string.IsNullOrEmpty(layer.Activation))
                        layer.Activation = actM.Groups[1].Value;
                }

                // .relu() / .gelu() インプレース呼出 (self. 経由のレイヤー呼出を除外)
                var inplaceM = Regex.Match(line, @"(?<!self)\.(relu|gelu|silu|tanh|sigmoid)_?\s*\(");
                if (inplaceM.Success && lastLayerName != null)
                {
                    var layer = model.Layers.FirstOrDefault(l => l.Name == lastLayerName);
                    if (layer != null && string.IsNullOrEmpty(layer.Activation))
                        layer.Activation = inplaceM.Groups[1].Value;
                }
            }
        }

        private static void BuildTransformerLayerChildren(NNLayer parent, string rawArgs,
            Dictionary<string, int> vars)
        {
            // d_model (1st positional or keyword)
            int dModel = 0, nHead = 0, dimFF = 0;
            var parts = SplitTopLevel(rawArgs, ',');

            foreach (var p in parts)
            {
                var t = p.Trim();
                var kv = Regex.Match(t, @"(\w+)\s*=\s*(.+)");
                if (kv.Success)
                {
                    string key = kv.Groups[1].Value, val = kv.Groups[2].Value.Trim();
                    if (key == "d_model") dModel = ResolveInt(val, vars);
                    else if (key == "nhead") nHead = ResolveInt(val, vars);
                    else if (key == "dim_feedforward") dimFF = ResolveInt(val, vars);
                }
            }
            // positional fallback: TransformerEncoderLayer(d_model, nhead, dim_feedforward=...)
            if (dModel == 0 && parts.Count >= 1)
            {
                var t = parts[0].Trim();
                if (!t.Contains('=')) dModel = ResolveInt(t, vars);
            }
            if (nHead == 0 && parts.Count >= 2)
            {
                var t = parts[1].Trim();
                if (!t.Contains('=')) nHead = ResolveInt(t, vars);
            }

            if (dModel == 0) dModel = 512;
            if (nHead == 0) nHead = 8;
            if (dimFF == 0) dimFF = dModel * 4;

            parent.Children.Add(new NNLayer
            {
                Name = "self_attn", LayerType = "MultiheadAttention",
                Dims = [dModel, nHead]
            });
            parent.Children.Add(new NNLayer
            {
                Name = "norm1", LayerType = "LayerNorm",
                Dims = [dModel]
            });
            parent.Children.Add(new NNLayer
            {
                Name = "ffn", LayerType = "Linear",
                Dims = [dModel, dimFF, dModel]
            });
            parent.Children.Add(new NNLayer
            {
                Name = "norm2", LayerType = "LayerNorm",
                Dims = [dModel]
            });
        }

        private static List<int> ExtractDims(string args, Dictionary<string, int> vars)
        {
            var dims = new List<int>();
            foreach (var part in SplitTopLevel(args, ','))
            {
                var t = part.Trim();
                if (t.Contains('=')) continue;
                if (int.TryParse(t, out int v)) { dims.Add(v); continue; }
                if (vars.TryGetValue(t, out int vv)) { dims.Add(vv); continue; }
                var dm = Regex.Match(t, @"\w+\[""(\w+)""\]");
                if (dm.Success && vars.TryGetValue(dm.Groups[1].Value, out int dv))
                { dims.Add(dv); continue; }
                var dot = Regex.Match(t, @"\w+\.(\w+)");
                if (dot.Success && vars.TryGetValue(dot.Groups[1].Value, out int dv2))
                { dims.Add(dv2); continue; }
                var mul = Regex.Match(t, @"(\w+)\s*\*\s*(\d+)");
                if (mul.Success)
                {
                    string bName = mul.Groups[1].Value;
                    int factor = int.Parse(mul.Groups[2].Value);
                    if (vars.TryGetValue(bName, out int bv)) { dims.Add(bv * factor); continue; }
                }
            }
            return dims;
        }

        private static List<string> SplitTopLevel(string text, char sep)
        {
            var parts = new List<string>();
            if (string.IsNullOrEmpty(text)) return parts;
            int depth = 0, start = 0;
            for (int i = 0; i < text.Length; i++)
            {
                char c = text[i];
                if (c is '(' or '[' or '{') depth++;
                else if (c is ')' or ']' or '}') { if (depth > 0) depth--; }
                else if (c == sep && depth == 0) { parts.Add(text[start..i]); start = i + 1; }
            }
            if (start < text.Length) parts.Add(text[start..]);
            return parts;
        }

        private static int ResolveInt(string expr, Dictionary<string, int> vars)
        {
            expr = expr.Trim();
            if (int.TryParse(expr, out int v)) return v;
            if (vars.TryGetValue(expr, out int vv)) return vv;
            var dm = Regex.Match(expr, @"\w+\[""(\w+)""\]");
            if (dm.Success && vars.TryGetValue(dm.Groups[1].Value, out int dv)) return dv;
            var dot = Regex.Match(expr, @"\w+\.(\w+)");
            if (dot.Success && vars.TryGetValue(dot.Groups[1].Value, out int dv2)) return dv2;
            return 1;
        }

        // ── TensorFlow / Keras パーサー ───────────────────────────────────────

        private static void ParseKerasLayers(string text, Dictionary<string, int> vars, LLMModel model)
        {
            if (!Regex.IsMatch(text,
                @"(?:import\s+tensorflow|from\s+tensorflow|from\s+keras|import\s+keras|tf\.keras|models\.Sequential|keras\.Sequential)"))
                return;

            int idx = 0;

            // Pattern 1: model.add(layers.XXX(args))
            var addPattern = @"\.add\s*\(\s*(?:layers\.|tf\.keras\.layers\.|keras\.layers\.)?([A-Z]\w*)\s*\(";
            foreach (Match m in Regex.Matches(text, addPattern))
            {
                string layerType = m.Groups[1].Value;
                string args = ExtractBalanced(text, m.Index + m.Length - 1, '(', ')');
                string name = ExtractKerasName(args) ?? $"layer_{idx}";
                model.Layers.Add(new NNLayer
                {
                    Name = name, LayerType = layerType, RawArgs = args,
                    Dims = ExtractTfDims(layerType, args, vars),
                    Activation = ExtractKerasActivation(args)
                });
                idx++;
            }

            // Pattern 2: Sequential([...]) inline
            if (model.Layers.Count == 0)
            {
                var seqM = Regex.Match(text, @"Sequential\s*\(\s*\[");
                if (seqM.Success)
                {
                    int bp = text.IndexOf('[', seqM.Index);
                    string content = ExtractBalanced(text, bp, '[', ']');
                    var lp = @"(?:layers\.|tf\.keras\.layers\.|keras\.layers\.)?([A-Z]\w*)\s*\(";
                    foreach (Match lm in Regex.Matches(content, lp))
                    {
                        string lt = lm.Groups[1].Value;
                        string la = ExtractBalanced(content, lm.Index + lm.Length - 1, '(', ')');
                        string ln = ExtractKerasName(la) ?? $"layer_{idx}";
                        model.Layers.Add(new NNLayer
                        {
                            Name = ln, LayerType = lt, RawArgs = la,
                            Dims = ExtractTfDims(lt, la, vars),
                            Activation = ExtractKerasActivation(la)
                        });
                        idx++;
                    }
                }
            }

            // Pattern 3: Functional API — x = layers.XXX(args)(prev)
            if (model.Layers.Count == 0)
            {
                foreach (Match m in Regex.Matches(text,
                    @"(\w+)\s*=\s*(?:tf\.keras\.|keras\.)?Input\s*\("))
                {
                    string args = ExtractBalanced(text, m.Index + m.Length - 1, '(', ')');
                    model.Layers.Add(new NNLayer
                    {
                        Name = m.Groups[1].Value, LayerType = "Input", RawArgs = args,
                        Dims = ExtractTfDims("Input", args, vars)
                    });
                    idx++;
                }
                foreach (Match m in Regex.Matches(text,
                    @"(\w+)\s*=\s*(?:layers\.|tf\.keras\.layers\.|keras\.layers\.)([A-Z]\w*)\s*\("))
                {
                    string lt = m.Groups[2].Value;
                    string args = ExtractBalanced(text, m.Index + m.Length - 1, '(', ')');
                    model.Layers.Add(new NNLayer
                    {
                        Name = ExtractKerasName(args) ?? m.Groups[1].Value,
                        LayerType = lt, RawArgs = args,
                        Dims = ExtractTfDims(lt, args, vars),
                        Activation = ExtractKerasActivation(args)
                    });
                    idx++;
                }
            }

            if (model.Layers.Count > 0 && model.ModelType is "unknown" or "Model")
                model.ModelType = "Keras";
        }

        private static List<int> ExtractTfDims(string layerType, string args, Dictionary<string, int> vars)
        {
            var dims = new List<int>();

            // Input / InputLayer — extract shape
            if (layerType is "Input" or "InputLayer")
            {
                var sm = Regex.Match(args, @"(?:input_)?shape\s*=\s*\(([^)]+)\)");
                if (!sm.Success) sm = Regex.Match(args, @"^\s*\(([^)]+)\)");
                if (sm.Success)
                    foreach (var p in sm.Groups[1].Value.Split(','))
                    {
                        var t = p.Trim();
                        if (string.IsNullOrEmpty(t)) continue;
                        if (int.TryParse(t, out int v)) { dims.Add(v); continue; }
                        if (vars.TryGetValue(t, out int vv)) { dims.Add(vv); continue; }
                        var mul = Regex.Match(t, @"(\d+)\s*\*\s*(\d+)");
                        if (mul.Success) dims.Add(int.Parse(mul.Groups[1].Value) * int.Parse(mul.Groups[2].Value));
                    }
                return dims;
            }

            // Reshape — target shape tuple
            if (layerType == "Reshape")
            {
                var sm = Regex.Match(args, @"(?:target_shape\s*=\s*)?\(([^)]+)\)");
                if (sm.Success)
                    foreach (var p in sm.Groups[1].Value.Split(','))
                    { var t = p.Trim(); if (int.TryParse(t, out int v)) dims.Add(v); }
                // Also check for input_shape
                var ism = Regex.Match(args, @"input_shape\s*=\s*\(([^)]+)\)");
                if (ism.Success && dims.Count == 0)
                    foreach (var p in ism.Groups[1].Value.Split(','))
                    { var t = p.Trim(); if (int.TryParse(t, out int v)) dims.Add(v); }
                return dims;
            }

            // Conv layers — first arg is filters
            if (layerType.StartsWith("Conv", StringComparison.Ordinal)
                || layerType.StartsWith("SeparableConv", StringComparison.Ordinal)
                || layerType.StartsWith("DepthwiseConv", StringComparison.Ordinal))
            {
                var parts = SplitTopLevel(args, ',');
                if (parts.Count >= 1)
                { var t = parts[0].Trim(); if (int.TryParse(t, out int f)) dims.Add(f);
                  else if (vars.TryGetValue(t, out int fv)) dims.Add(fv); }
                return dims;
            }

            // Pooling — pool size
            if (layerType.Contains("Pooling") || layerType.StartsWith("MaxPool", StringComparison.Ordinal)
                || layerType.StartsWith("AvgPool", StringComparison.Ordinal))
            {
                var tm = Regex.Match(args, @"^\s*\(?(\d+)(?:\s*,\s*(\d+))?\)?");
                if (!tm.Success)
                    tm = Regex.Match(args, @"pool_size\s*=\s*\(?(\d+)(?:\s*,\s*(\d+))?\)?");
                if (tm.Success)
                {
                    dims.Add(int.Parse(tm.Groups[1].Value));
                    if (tm.Groups[2].Success) dims.Add(int.Parse(tm.Groups[2].Value));
                }
                return dims;
            }

            // Flatten, GlobalPooling — no numeric dims
            if (layerType is "Flatten" || layerType.StartsWith("Global", StringComparison.Ordinal))
                return dims;

            // Default: positional numeric args (Dense, LSTM, GRU, Embedding, etc.)
            foreach (var part in SplitTopLevel(args, ','))
            {
                var t = part.Trim();
                if (t.Contains('=')) break;
                if (int.TryParse(t, out int v)) { dims.Add(v); continue; }
                if (vars.TryGetValue(t, out int vv)) { dims.Add(vv); continue; }
                var mul = Regex.Match(t, @"(\d+)\s*\*\s*(\d+)");
                if (mul.Success) { dims.Add(int.Parse(mul.Groups[1].Value) * int.Parse(mul.Groups[2].Value)); continue; }
                break;
            }
            return dims;
        }

        private static string? ExtractKerasName(string args)
        {
            var m = Regex.Match(args, @"name\s*=\s*['""]([^'""]+)['""]");
            return m.Success ? m.Groups[1].Value : null;
        }

        private static string ExtractKerasActivation(string args)
        {
            var m = Regex.Match(args, @"activation\s*=\s*['""](\w+)['""]");
            return m.Success ? m.Groups[1].Value : "";
        }

        private static void FillMetadataFromLayers(LLMModel model)
        {
            foreach (var layer in model.Layers)
            {
                // Embedding (PyTorch/TF) or Embed (Flax/Haiku)
                if (layer.LayerType is "Embedding" or "Embed" && layer.Dims.Count >= 2)
                {
                    if (model.VocabSize == 0) model.VocabSize = layer.Dims[0];
                    if (model.HiddenSize == 0) model.HiddenSize = layer.Dims[1];
                }
                // MultiheadAttention (PyTorch/MindSpore) / MultiHeadAttention (PaddlePaddle/Haiku)
                // MultiHeadDotProductAttention / SelfAttention (Flax)
                if (layer.LayerType is "MultiheadAttention" or "MultiHeadAttention"
                        or "MultiHeadDotProductAttention" or "SelfAttention"
                    && layer.Dims.Count >= 2)
                {
                    if (model.HiddenSize == 0) model.HiddenSize = layer.Dims[0];
                    if (model.NumAttentionHeads == 0) model.NumAttentionHeads = layer.Dims[1];
                }
                // Dense (TF/Flax/MindSpore) / Linear (PyTorch/PaddlePaddle/Haiku)
                if (layer.LayerType is "Dense" or "Linear" or "DenseGeneral"
                    && layer.Dims.Count >= 1 && model.HiddenSize == 0)
                    model.HiddenSize = layer.Dims[0];
                if (layer.IsContainer && layer.RepeatCount > 1)
                    model.NumHiddenLayers = layer.RepeatCount;
            }
            foreach (var child in model.Layers.Where(l => l.IsContainer)
                                              .SelectMany(l => l.Children))
            {
                if (child.LayerType is "Linear" or "Dense" or "DenseGeneral"
                    && child.Dims.Count >= 2
                    && model.HiddenSize > 0 && child.Dims[1] > model.HiddenSize)
                    model.IntermediateSize = child.Dims[1];
                foreach (var gc in child.Children)
                    if (gc.LayerType is "Linear" or "Dense" or "DenseGeneral"
                        && gc.Dims.Count >= 2
                        && model.HiddenSize > 0 && gc.Dims[1] > model.HiddenSize)
                        model.IntermediateSize = gc.Dims[1];
            }
        }

        // ══════════════════════════════════════════════════════════════════
        //  JSON (HuggingFace config.json)  — 既存 + Layers 自動生成
        // ══════════════════════════════════════════════════════════════════

        public static LLMModel? FromJsonFile(string path)
        {
            var model = new LLMModel
            {
                FileName = Path.GetFileName(path),
                FilePath = path,
                SourceFormat = "JSON"
            };
            try
            {
                var json = File.ReadAllText(path);
                using var doc = JsonDocument.Parse(json);
                var root = doc.RootElement;
                model.ExtractFromJson(root);
                model.BuildLayersFromMetadata();
                if (model.Layers.Count == 0)
                    model.LastError = "JSON からレイヤー情報を生成できませんでした（hidden_size 等が不足）。";
                return model;
            }
            catch (JsonException ex)
            {
                model.LastError = $"JSON パース エラー: {ex.Message}";
                return model;
            }
            catch (IOException ex)
            {
                model.LastError = $"ファイル読み込みエラー: {ex.Message}";
                return model;
            }
            catch (Exception ex)
            {
                model.LastError = $"JSON 処理エラー: {ex.GetType().Name}: {ex.Message}";
                return model;
            }
        }

        private void ExtractFromJson(JsonElement root)
        {
            int GetInt(params string[] keys)
            {
                foreach (var k in keys)
                    if (root.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.Number)
                        return v.GetInt32();
                return 0;
            }
            string? GetStr(params string[] keys)
            {
                foreach (var k in keys)
                    if (root.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.String)
                        return v.GetString();
                return null;
            }

            ModelType             = GetStr("model_type") ?? "unknown";
            VocabSize             = GetInt("vocab_size");
            HiddenSize            = GetInt("hidden_size", "n_embd", "d_model");
            NumHiddenLayers       = GetInt("num_hidden_layers", "n_layer", "num_layers");
            NumAttentionHeads     = GetInt("num_attention_heads", "n_head");
            NumKeyValueHeads      = GetInt("num_key_value_heads");
            if (NumKeyValueHeads == 0) NumKeyValueHeads = NumAttentionHeads;
            IntermediateSize      = GetInt("intermediate_size", "ffn_dim", "n_inner");
            MaxPositionEmbeddings = GetInt("max_position_embeddings", "n_positions");
            HiddenActivation      = GetStr("hidden_act", "activation_function") ?? "";
        }

        // ══════════════════════════════════════════════════════════════════
        //  GGUF — 既存 + Layers 自動生成
        // ══════════════════════════════════════════════════════════════════

        public static LLMModel? FromGgufFile(string path)
        {
            var model = new LLMModel
            {
                FileName = Path.GetFileName(path),
                FilePath = path,
                SourceFormat = "GGUF"
            };
            try
            {
                using var fs = File.OpenRead(path);
                using var br = new BinaryReader(fs);
                uint magic = br.ReadUInt32();
                if (magic != 0x46554747)
                {
                    model.LastError = $"GGUF マジックナンバー不正 (0x{magic:X8})。有効な GGUF ファイルではありません。";
                    return model;
                }
                br.ReadUInt32(); br.ReadUInt64();
                var metaCount = br.ReadUInt64();
                for (ulong i = 0; i < metaCount; i++)
                {
                    var key = ReadGgufString(br);
                    var vtype = (GgufVT)br.ReadUInt32();
                    var value = ReadGgufValue(br, vtype);
                    if (value != null) model.ApplyGguf(key, value);
                }
                if (model.NumKeyValueHeads == 0)
                    model.NumKeyValueHeads = model.NumAttentionHeads;
                model.BuildLayersFromMetadata();
                if (model.Layers.Count == 0)
                    model.LastError = "GGUF メタデータからレイヤーを生成できませんでした。";
                return model;
            }
            catch (EndOfStreamException)
            {
                model.LastError = "GGUF ファイルが途中で終了しています（破損の可能性）。";
                return model;
            }
            catch (IOException ex)
            {
                model.LastError = $"ファイル読み込みエラー: {ex.Message}";
                return model;
            }
            catch (Exception ex)
            {
                model.LastError = $"GGUF パース エラー: {ex.GetType().Name}: {ex.Message}";
                return model;
            }
        }

        private void ApplyGguf(string key, string value)
        {
            if (key == "general.architecture") { ModelType = value; return; }
            if (!int.TryParse(value, out int iv)) return;
            if      (key.EndsWith(".vocab_size")              && VocabSize == 0)          VocabSize = iv;
            else if (key.EndsWith(".embedding_length")        && HiddenSize == 0)         HiddenSize = iv;
            else if (key.EndsWith(".block_count")             && NumHiddenLayers == 0)    NumHiddenLayers = iv;
            else if (key.EndsWith(".attention.head_count")    && NumAttentionHeads == 0)  NumAttentionHeads = iv;
            else if (key.EndsWith(".attention.head_count_kv") && NumKeyValueHeads == 0)   NumKeyValueHeads = iv;
            else if (key.EndsWith(".feed_forward_length")     && IntermediateSize == 0)   IntermediateSize = iv;
            else if (key.EndsWith(".context_length")          && MaxPositionEmbeddings == 0) MaxPositionEmbeddings = iv;
        }

        private static string ReadGgufString(BinaryReader br)
            => System.Text.Encoding.UTF8.GetString(br.ReadBytes((int)Math.Min(br.ReadUInt64(), 65536UL)));

        private enum GgufVT : uint
        { U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 }

        private static string? ReadGgufValue(BinaryReader br, GgufVT t) => t switch
        {
            GgufVT.U8   => br.ReadByte().ToString(),
            GgufVT.I8   => br.ReadSByte().ToString(),
            GgufVT.U16  => br.ReadUInt16().ToString(),
            GgufVT.I16  => br.ReadInt16().ToString(),
            GgufVT.U32  => br.ReadUInt32().ToString(),
            GgufVT.I32  => br.ReadInt32().ToString(),
            GgufVT.F32  => br.ReadSingle().ToString("G"),
            GgufVT.BOOL => br.ReadByte() != 0 ? "true" : "false",
            GgufVT.STR  => ReadGgufString(br),
            GgufVT.U64  => br.ReadUInt64().ToString(),
            GgufVT.I64  => br.ReadInt64().ToString(),
            GgufVT.F64  => br.ReadDouble().ToString("G"),
            GgufVT.ARR  => ConsumeGgufArray(br),
            _            => null
        };

        private static string ConsumeGgufArray(BinaryReader br)
        {
            var et = (GgufVT)br.ReadUInt32();
            var n  = br.ReadUInt64();
            for (ulong i = 0; i < n; i++) ReadGgufValue(br, et);
            return $"[{n} items]";
        }

        // ══════════════════════════════════════════════════════════════════
        //  メタデータ → Layers 自動生成 (JSON / GGUF 用)
        // ══════════════════════════════════════════════════════════════════

        private void BuildLayersFromMetadata()
        {
            if (Layers.Count > 0 || HiddenSize == 0) return;
            string normType = ModelType is "gpt2" or "gpt_neo" or "bloom" ? "LayerNorm" : "RMSNorm";

            if (VocabSize > 0)
                Layers.Add(new NNLayer { Name = "tok_emb", LayerType = "Embedding",
                    Dims = [VocabSize, HiddenSize] });

            if (NumHiddenLayers > 0)
            {
                var block = new NNLayer
                {
                    Name = "blocks", LayerType = "TransformerBlock",
                    RepeatCount = NumHiddenLayers
                };
                block.Children.Add(new NNLayer { Name = "norm1", LayerType = normType,
                    Dims = [HiddenSize] });
                block.Children.Add(new NNLayer { Name = "attn", LayerType = "MultiheadAttention",
                    Dims = NumAttentionHeads > 0 ? [HiddenSize, NumAttentionHeads] : [HiddenSize] });
                block.Children.Add(new NNLayer { Name = "norm2", LayerType = normType,
                    Dims = [HiddenSize] });
                block.Children.Add(new NNLayer { Name = "ffn", LayerType = "Linear",
                    Dims = IntermediateSize > 0 ? [HiddenSize, IntermediateSize, HiddenSize] : [HiddenSize] });
                Layers.Add(block);
            }

            Layers.Add(new NNLayer { Name = "final_norm", LayerType = normType,
                Dims = [HiddenSize] });

            if (VocabSize > 0)
                Layers.Add(new NNLayer { Name = "lm_head", LayerType = "Linear",
                    Dims = [HiddenSize, VocabSize] });
        }

        // ══════════════════════════════════════════════════════════════════
        //  パラメータ推定
        // ══════════════════════════════════════════════════════════════════

        public long EstimateParameters()
        {
            if (HiddenSize == 0 || NumHiddenLayers == 0) return 0;
            int hd    = HeadDim > 0 ? HeadDim : 128;
            long embed = (long)VocabSize * HiddenSize;
            long attn  = (long)HiddenSize * NumAttentionHeads * hd
                       + (long)HiddenSize * NumKeyValueHeads  * hd * 2
                       + (long)NumAttentionHeads * hd * HiddenSize;
            long ffn   = (long)HiddenSize * IntermediateSize * 2
                       + (long)IntermediateSize * HiddenSize;
            long norm  = (long)HiddenSize * 2;
            return embed + (long)NumHiddenLayers * (attn + ffn + norm)
                + HiddenSize + (long)VocabSize * HiddenSize;
        }

        public string FormatParameters()
        {
            long p = EstimateParameters();
            return p switch
            {
                0                => "N/A",
                >= 1_000_000_000 => $"{p / 1_000_000_000.0:F1}B",
                >= 1_000_000     => $"{p / 1_000_000.0:F1}M",
                _                => $"{p / 1_000.0:F1}K"
            };
        }
    }
}
