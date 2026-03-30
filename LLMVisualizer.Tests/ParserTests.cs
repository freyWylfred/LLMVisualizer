using LLMVisualizer;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace LLMVisualizer.Tests;

[TestClass]
public class ParserTests
{
    private static string TestDataPath(string file)
        => Path.Combine(AppContext.BaseDirectory, "TestData", file);

    // ═══════════════════════════════════════════════════════════════
    //  PyTorch: 基本 GPT モデル
    // ═══════════════════════════════════════════════════════════════

    [TestMethod]
    public void PyTorch_BasicGPT_ParsesLayers()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_pytorch_basic.py"));
        Assert.IsNotNull(model, $"Parse returned null. Error: {model?.LastError}");
        Assert.IsNull(model.LastError, $"LastError: {model.LastError}");
        Assert.IsTrue(model.Layers.Count > 0, "No layers extracted");
        Assert.AreEqual("SimpleGPT", model.ModelType);
        Assert.AreEqual("Python (PyTorch)", model.SourceFormat);
    }

    [TestMethod]
    public void PyTorch_BasicGPT_ExtractsEmbedding()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_pytorch_basic.py"))!;
        var emb = model.Layers.FirstOrDefault(l => l.Name == "embedding");
        Assert.IsNotNull(emb, "embedding layer not found");
        Assert.AreEqual("Embedding", emb.LayerType);
        Assert.IsTrue(emb.Dims.Count >= 2, $"Expected >=2 dims, got {emb.Dims.Count}");
    }

    [TestMethod]
    public void PyTorch_BasicGPT_ExtractsModuleList()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_pytorch_basic.py"))!;
        var blocks = model.Layers.FirstOrDefault(l => l.Name == "blocks");
        Assert.IsNotNull(blocks, "blocks (ModuleList) not found");
        Assert.IsTrue(blocks.IsContainer, "blocks should be a container");
        Assert.IsTrue(blocks.RepeatCount > 1, $"Expected RepeatCount > 1, got {blocks.RepeatCount}");
    }

    [TestMethod]
    public void PyTorch_BasicGPT_ForwardOrderApplied()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_pytorch_basic.py"))!;
        Assert.IsTrue(model.Layers.Count >= 4, $"Expected >=4 layers, got {model.Layers.Count}");
        // forward() order: embedding → pos_encoding → blocks → ln_f → head
        Assert.AreEqual("embedding", model.Layers[0].Name);
        var lastLayer = model.Layers[^1];
        Assert.AreEqual("head", lastLayer.Name);
    }

    [TestMethod]
    public void PyTorch_BasicGPT_FillsMetadata()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_pytorch_basic.py"))!;
        Assert.AreEqual(30522, model.VocabSize, "VocabSize");
        Assert.AreEqual(768, model.HiddenSize, "HiddenSize");
    }

    // ═══════════════════════════════════════════════════════════════
    //  PyTorch: TransformerEncoder / Decoder
    // ═══════════════════════════════════════════════════════════════

    [TestMethod]
    public void PyTorch_Transformer_ParsesEncoderDecoder()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_pytorch_transformer.py"));
        Assert.IsNotNull(model);
        Assert.IsNull(model.LastError, $"Error: {model.LastError}");

        var enc = model.Layers.FirstOrDefault(l => l.Name == "encoder");
        Assert.IsNotNull(enc, "encoder not found");
        Assert.AreEqual("TransformerEncoder", enc.LayerType);
        Assert.IsTrue(enc.IsContainer, "encoder should have children");
        Assert.AreEqual(6, enc.RepeatCount, "encoder num_layers");

        var dec = model.Layers.FirstOrDefault(l => l.Name == "decoder");
        Assert.IsNotNull(dec, "decoder not found");
        Assert.AreEqual("TransformerDecoder", dec.LayerType);
        Assert.IsTrue(dec.IsContainer, "decoder should have children");
    }

    [TestMethod]
    public void PyTorch_Transformer_HasInternalStructure()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_pytorch_transformer.py"))!;
        var enc = model.Layers.First(l => l.Name == "encoder");
        // TransformerEncoderLayer should produce children: self_attn, norm1, ffn, norm2
        Assert.IsTrue(enc.Children.Count >= 3, $"Expected >=3 children, got {enc.Children.Count}");
        Assert.IsTrue(enc.Children.Any(c => c.LayerType == "MultiheadAttention"), "Missing MultiheadAttention child");
        Assert.IsTrue(enc.Children.Any(c => c.LayerType == "LayerNorm"), "Missing LayerNorm child");
    }

    [TestMethod]
    public void PyTorch_Transformer_ForwardOrder()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_pytorch_transformer.py"))!;
        // forward: embedding → dropout → encoder → (embedding again) → decoder → fc_out
        Assert.AreEqual("embedding", model.Layers[0].Name);
    }

    // ═══════════════════════════════════════════════════════════════
    //  PyTorch: CNN with F.relu
    // ═══════════════════════════════════════════════════════════════

    [TestMethod]
    public void PyTorch_CNN_ParsesConvLayers()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_pytorch_cnn.py"));
        Assert.IsNotNull(model);
        Assert.IsNull(model.LastError, $"Error: {model.LastError}");
        Assert.AreEqual("SimpleCNN", model.ModelType);

        var conv1 = model.Layers.FirstOrDefault(l => l.Name == "conv1");
        Assert.IsNotNull(conv1, "conv1 not found");
        Assert.AreEqual("Conv2d", conv1.LayerType);

        var pool = model.Layers.FirstOrDefault(l => l.Name == "pool");
        Assert.IsNotNull(pool, "MaxPool2d not found");
        Assert.AreEqual("MaxPool2d", pool.LayerType);
    }

    [TestMethod]
    public void PyTorch_CNN_DetectsFunctionalActivation()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_pytorch_cnn.py"))!;
        var conv1 = model.Layers.First(l => l.Name == "conv1");
        // forward() has: x = self.conv1(x) then x = F.relu(x)
        Assert.AreEqual("relu", conv1.Activation, "conv1 should have relu activation from F.relu");
    }

    [TestMethod]
    public void PyTorch_CNN_ExpandsSubClass()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_pytorch_cnn.py"))!;
        var block1 = model.Layers.FirstOrDefault(l => l.Name == "block1");
        Assert.IsNotNull(block1, "block1 not found");
        Assert.IsTrue(block1.IsContainer, "block1 (ResNetBlock) should be a container");
        Assert.IsTrue(block1.Children.Any(c => c.LayerType == "Conv2d"), "ResNetBlock should have Conv2d children");
        Assert.IsTrue(block1.Children.Any(c => c.LayerType == "BatchNorm2d"), "ResNetBlock should have BatchNorm2d children");
    }

    // ═══════════════════════════════════════════════════════════════
    //  PyTorch: Sequential + OrderedDict + ModuleDict
    // ═══════════════════════════════════════════════════════════════

    [TestMethod]
    public void PyTorch_Sequential_ParsesOrderedDict()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_pytorch_sequential.py"));
        Assert.IsNotNull(model);
        Assert.IsNull(model.LastError, $"Error: {model.LastError}");

        var features = model.Layers.FirstOrDefault(l => l.Name == "features");
        Assert.IsNotNull(features, "features (Sequential) not found");
        Assert.IsTrue(features.IsContainer, "features should be a container");
        // OrderedDict keys should be used as child names
        Assert.IsTrue(features.Children.Any(c => c.Name == "fc1"), "OrderedDict child 'fc1' not found");
        Assert.IsTrue(features.Children.Any(c => c.Name == "relu1"), "OrderedDict child 'relu1' not found");
    }

    [TestMethod]
    public void PyTorch_ModuleDict_Parsed()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_pytorch_sequential.py"))!;
        var branches = model.Layers.FirstOrDefault(l => l.Name == "branches");
        Assert.IsNotNull(branches, "branches (ModuleDict) not found");
        Assert.IsTrue(branches.IsContainer, "branches should be a container");
        Assert.IsTrue(branches.Children.Any(c => c.Name == "classifier"), "ModuleDict child 'classifier' not found");
        Assert.IsTrue(branches.Children.Any(c => c.Name == "regressor"), "ModuleDict child 'regressor' not found");
    }

    // ═══════════════════════════════════════════════════════════════
    //  PyTorch: LSTM
    // ═══════════════════════════════════════════════════════════════

    [TestMethod]
    public void PyTorch_LSTM_ParsesLayers()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_pytorch_lstm.py"));
        Assert.IsNotNull(model);
        Assert.IsNull(model.LastError, $"Error: {model.LastError}");
        Assert.AreEqual("LSTMModel", model.ModelType);

        var lstm = model.Layers.FirstOrDefault(l => l.Name == "lstm");
        Assert.IsNotNull(lstm, "LSTM layer not found");
        Assert.AreEqual("LSTM", lstm.LayerType);
        Assert.IsTrue(lstm.Dims.Count >= 2, $"Expected >=2 dims for LSTM, got {lstm.Dims.Count}");
    }

    // ═══════════════════════════════════════════════════════════════
    //  TensorFlow / Keras
    // ═══════════════════════════════════════════════════════════════

    [TestMethod]
    public void TF_Sequential_ParsesLayers()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_tf_sequential.py"));
        Assert.IsNotNull(model);
        Assert.IsNull(model.LastError, $"Error: {model.LastError}");
        Assert.IsTrue(model.Layers.Count > 0, "No layers extracted from TF Sequential");
        Assert.AreEqual("Keras", model.ModelType);
    }

    [TestMethod]
    public void TF_Sequential_ExtractsActivation()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_tf_sequential.py"))!;
        var dense = model.Layers.FirstOrDefault(l => l.LayerType == "Dense");
        Assert.IsNotNull(dense, "Dense layer not found");
        Assert.AreEqual("relu", dense.Activation, "Dense activation should be 'relu'");
    }

    // ═══════════════════════════════════════════════════════════════
    //  エッジケース: 空ファイル
    // ═══════════════════════════════════════════════════════════════

    [TestMethod]
    public void EdgeCase_EmptyFile_ReturnsModelWithError()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_empty.py"));
        Assert.IsNotNull(model, "Should return a model object (not null)");
        Assert.AreEqual(0, model.Layers.Count, "Empty file should have 0 layers");
        Assert.IsFalse(string.IsNullOrEmpty(model.LastError),
            "Should have an error message explaining why no layers were found");
    }

    [TestMethod]
    public void EdgeCase_NoModule_ReturnsModelWithError()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_no_module.py"));
        Assert.IsNotNull(model, "Should return a model object");
        Assert.AreEqual(0, model.Layers.Count, "Non-module file should have 0 layers");
        Assert.IsFalse(string.IsNullOrEmpty(model.LastError),
            "Should have an error explaining no nn.Module found");
    }

    [TestMethod]
    public void EdgeCase_MalformedParens_DoesNotCrash()
    {
        // Should not throw — should either produce partial results or an error message
        var model = LLMModel.FromPythonFile(TestDataPath("test_malformed.py"));
        Assert.IsNotNull(model, "Should not crash on malformed input");
        // Either has layers (partial parse) or a descriptive error
    }

    [TestMethod]
    public void EdgeCase_NonExistentFile_ReturnsError()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("does_not_exist.py"));
        Assert.IsNotNull(model, "Should return model with error, not null");
        Assert.IsFalse(string.IsNullOrEmpty(model.LastError), "Should have IO error");
        Assert.IsTrue(model.LastError!.Contains("ファイル読み込みエラー"),
            $"Expected file IO error, got: {model.LastError}");
    }

    // ═══════════════════════════════════════════════════════════════
    //  JSON エッジケース
    // ═══════════════════════════════════════════════════════════════

    [TestMethod]
    public void JSON_NonExistentFile_ReturnsError()
    {
        var model = LLMModel.FromJsonFile(TestDataPath("nonexistent.json"));
        Assert.IsNotNull(model);
        Assert.IsFalse(string.IsNullOrEmpty(model.LastError));
    }

    [TestMethod]
    public void JSON_InvalidJson_ReturnsError()
    {
        var tempFile = Path.GetTempFileName();
        try
        {
            File.WriteAllText(tempFile, "{ not valid json }}}");
            var model = LLMModel.FromJsonFile(tempFile);
            Assert.IsNotNull(model);
            Assert.IsFalse(string.IsNullOrEmpty(model.LastError),
                "Should report JSON parse error");
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  GGUF エッジケース
    // ═══════════════════════════════════════════════════════════════

    [TestMethod]
    public void GGUF_InvalidMagic_ReturnsError()
    {
        var tempFile = Path.GetTempFileName();
        try
        {
            File.WriteAllBytes(tempFile, [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
            var model = LLMModel.FromGgufFile(tempFile);
            Assert.IsNotNull(model);
            Assert.IsFalse(string.IsNullOrEmpty(model.LastError),
                "Should report invalid GGUF magic");
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [TestMethod]
    public void GGUF_TruncatedFile_ReturnsError()
    {
        var tempFile = Path.GetTempFileName();
        try
        {
            // Write valid magic but truncated content
            using (var bw = new BinaryWriter(File.Create(tempFile)))
            {
                bw.Write(0x46554747u); // GGUF magic
                bw.Write(3u);           // version
                // truncated — missing tensor_count, metadata_kv_count
            }
            var model = LLMModel.FromGgufFile(tempFile);
            Assert.IsNotNull(model);
            Assert.IsFalse(string.IsNullOrEmpty(model.LastError),
                "Should handle truncated GGUF gracefully");
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  内部ユーティリティ: ExtractBalanced / SplitTopLevel
    // ═══════════════════════════════════════════════════════════════

    // ═══════════════════════════════════════════════════════════════
    //  JAX/Flax: setup() パターン
    // ═══════════════════════════════════════════════════════════════

    [TestMethod]
    public void Flax_Setup_ParsesLayers()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_flax_setup.py"));
        Assert.IsNotNull(model, $"Parse returned null. Error: {model?.LastError}");
        Assert.IsNull(model.LastError, $"LastError: {model.LastError}");
        Assert.IsTrue(model.Layers.Count > 0, "No layers extracted");
        Assert.AreEqual("FlaxTransformer", model.ModelType);
    }

    [TestMethod]
    public void Flax_Setup_ExtractsEmbedding()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_flax_setup.py"))!;
        var emb = model.Layers.FirstOrDefault(l => l.Name == "embedding");
        Assert.IsNotNull(emb, "embedding layer not found");
        // Flax の nn.Embed は setup() パスでは正規化されず "Embed" のまま
        Assert.AreEqual("Embed", emb.LayerType);
        Assert.IsTrue(emb.Dims.Count >= 2, $"Expected >=2 dims, got {emb.Dims.Count}");
    }

    [TestMethod]
    public void Flax_Setup_DetectsFramework()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_flax_setup.py"))!;
        Assert.AreEqual("Python (Flax/JAX)", model.SourceFormat);
    }

    [TestMethod]
    public void Flax_Setup_ExtractsHeadLayer()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_flax_setup.py"))!;
        var head = model.Layers.FirstOrDefault(l => l.Name == "head");
        Assert.IsNotNull(head, "head (Dense) layer not found");
        Assert.AreEqual("Dense", head.LayerType);
    }

    // ═══════════════════════════════════════════════════════════════
    //  JAX/Flax: @nn.compact パターン
    // ═══════════════════════════════════════════════════════════════

    [TestMethod]
    public void Flax_Compact_ParsesLayers()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_flax_compact.py"));
        Assert.IsNotNull(model, $"Parse returned null. Error: {model?.LastError}");
        Assert.IsNull(model.LastError, $"LastError: {model.LastError}");
        Assert.IsTrue(model.Layers.Count > 0, $"No layers extracted. LastError: {model.LastError}");
    }

    [TestMethod]
    public void Flax_Compact_ExtractsEmbed()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_flax_compact.py"))!;
        var emb = model.Layers.FirstOrDefault(l => l.LayerType == "Embedding");
        Assert.IsNotNull(emb, "Embed/Embedding layer not found in compact model");
        Assert.IsTrue(emb.Dims.Count >= 1, $"Expected >=1 dims, got {emb.Dims.Count}");
    }

    [TestMethod]
    public void Flax_Compact_ExtractsDenseLayers()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_flax_compact.py"))!;
        var denseLayers = model.Layers.Where(l => l.LayerType == "Dense").ToList();
        Assert.IsTrue(denseLayers.Count >= 2, $"Expected >=2 Dense layers, got {denseLayers.Count}");
    }

    [TestMethod]
    public void Flax_Compact_ExtractsSelfAttention()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_flax_compact.py"))!;
        var attn = model.Layers.FirstOrDefault(l => l.LayerType == "MultiheadAttention");
        Assert.IsNotNull(attn, "SelfAttention (normalized to MultiheadAttention) not found");
    }

    // ═══════════════════════════════════════════════════════════════
    //  JAX/Haiku
    // ═══════════════════════════════════════════════════════════════

    [TestMethod]
    public void Haiku_ParsesLayers()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_haiku.py"));
        Assert.IsNotNull(model, $"Parse returned null. Error: {model?.LastError}");
        Assert.IsNull(model.LastError, $"LastError: {model.LastError}");
        Assert.IsTrue(model.Layers.Count > 0, "No layers extracted");
        Assert.AreEqual("HaikuTransformer", model.ModelType);
    }

    [TestMethod]
    public void Haiku_ExtractsEmbedding()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_haiku.py"))!;
        var emb = model.Layers.FirstOrDefault(l => l.Name == "embedding");
        Assert.IsNotNull(emb, "embedding layer not found");
        Assert.AreEqual("Embed", emb.LayerType);
    }

    [TestMethod]
    public void Haiku_ExtractsLinear()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_haiku.py"))!;
        var linears = model.Layers.Where(l => l.LayerType == "Linear").ToList();
        Assert.IsTrue(linears.Count >= 2, $"Expected >=2 Linear layers, got {linears.Count}");
    }

    [TestMethod]
    public void Haiku_DetectsFramework()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_haiku.py"))!;
        Assert.AreEqual("Python (Haiku/JAX)", model.SourceFormat);
    }

    [TestMethod]
    public void Haiku_CallOrderApplied()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_haiku.py"))!;
        Assert.IsTrue(model.Layers.Count >= 3, $"Expected >=3 layers, got {model.Layers.Count}");
        Assert.AreEqual("embedding", model.Layers[0].Name);
        Assert.AreEqual("head", model.Layers[^1].Name);
    }

    // ═══════════════════════════════════════════════════════════════
    //  PaddlePaddle
    // ═══════════════════════════════════════════════════════════════

    [TestMethod]
    public void Paddle_ParsesLayers()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_paddle.py"));
        Assert.IsNotNull(model, $"Parse returned null. Error: {model?.LastError}");
        Assert.IsNull(model.LastError, $"LastError: {model.LastError}");
        Assert.IsTrue(model.Layers.Count > 0, "No layers extracted");
        Assert.AreEqual("PaddleGPT", model.ModelType);
    }

    [TestMethod]
    public void Paddle_ExtractsEmbedding()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_paddle.py"))!;
        var emb = model.Layers.FirstOrDefault(l => l.Name == "embedding");
        Assert.IsNotNull(emb, "embedding layer not found");
        Assert.AreEqual("Embedding", emb.LayerType);
        Assert.IsTrue(emb.Dims.Count >= 2, $"Expected >=2 dims, got {emb.Dims.Count}");
    }

    [TestMethod]
    public void Paddle_DetectsFramework()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_paddle.py"))!;
        Assert.AreEqual("Python (PaddlePaddle)", model.SourceFormat);
    }

    [TestMethod]
    public void Paddle_ForwardOrderApplied()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_paddle.py"))!;
        Assert.IsTrue(model.Layers.Count >= 3, $"Expected >=3 layers, got {model.Layers.Count}");
        Assert.AreEqual("embedding", model.Layers[0].Name);
        Assert.AreEqual("head", model.Layers[^1].Name);
    }

    [TestMethod]
    public void Paddle_ExpandsSubClass()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_paddle.py"))!;
        var hasAttn = model.Layers.Any(l =>
            l.LayerType == "MultiHeadAttention"
            || (l.IsContainer && l.Children.Any(c => c.LayerType == "MultiHeadAttention")));
        Assert.IsTrue(hasAttn, "Should find MultiHeadAttention layer");
    }

    // ═══════════════════════════════════════════════════════════════
    //  MindSpore
    // ═══════════════════════════════════════════════════════════════

    [TestMethod]
    public void MindSpore_ParsesLayers()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_mindspore.py"));
        Assert.IsNotNull(model, $"Parse returned null. Error: {model?.LastError}");
        Assert.IsNull(model.LastError, $"LastError: {model.LastError}");
        Assert.IsTrue(model.Layers.Count > 0, "No layers extracted");
        Assert.AreEqual("MindSporeTransformer", model.ModelType);
    }

    [TestMethod]
    public void MindSpore_ExtractsEmbedding()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_mindspore.py"))!;
        var emb = model.Layers.FirstOrDefault(l => l.Name == "embedding");
        Assert.IsNotNull(emb, "embedding layer not found");
        Assert.AreEqual("Embedding", emb.LayerType);
        Assert.IsTrue(emb.Dims.Count >= 2, $"Expected >=2 dims, got {emb.Dims.Count}");
    }

    [TestMethod]
    public void MindSpore_DetectsFramework()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_mindspore.py"))!;
        Assert.AreEqual("Python (MindSpore)", model.SourceFormat);
    }

    [TestMethod]
    public void MindSpore_ConstructOrderApplied()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_mindspore.py"))!;
        Assert.IsTrue(model.Layers.Count >= 3, $"Expected >=3 layers, got {model.Layers.Count}");
        Assert.AreEqual("embedding", model.Layers[0].Name);
        Assert.AreEqual("head", model.Layers[^1].Name);
    }

    [TestMethod]
    public void MindSpore_ExpandsSubClass()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_mindspore.py"))!;
        var hasAttn = model.Layers.Any(l =>
            l.LayerType == "MultiheadAttention"
            || (l.IsContainer && l.Children.Any(c => c.LayerType == "MultiheadAttention")));
        Assert.IsTrue(hasAttn, "Should find MultiheadAttention layer");
    }

    [TestMethod]
    public void MindSpore_FillsMetadata()
    {
        var model = LLMModel.FromPythonFile(TestDataPath("test_mindspore.py"))!;
        Assert.AreEqual(32000, model.VocabSize, "VocabSize");
        Assert.AreEqual(768, model.HiddenSize, "HiddenSize");
    }

    // ═══════════════════════════════════════════════════════════════
    //  内部ユーティリティ: FormatParameters
    // ═══════════════════════════════════════════════════════════════

    [TestMethod]
    public void Utility_FormatParameters_HandlesZero()
    {
        var model = new LLMModel();
        Assert.AreEqual("N/A", model.FormatParameters());
    }

    [TestMethod]
    public void Utility_FormatParameters_FormatsCorrectly()
    {
        var model = new LLMModel
        {
            VocabSize = 32000,
            HiddenSize = 4096,
            NumHiddenLayers = 32,
            NumAttentionHeads = 32,
            NumKeyValueHeads = 32,
            IntermediateSize = 11008
        };
        string result = model.FormatParameters();
        Assert.AreNotEqual("N/A", result);
        Assert.IsTrue(result.Contains("B") || result.Contains("M") || result.Contains("K"),
            $"Expected formatted number, got: {result}");
    }
}
