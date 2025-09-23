using Microsoft.ML;
using ML.Service.ImageClassification.Model.Models;

namespace ML.Service.ImageClassification.Model.Services;

public class ImageClassificationPrediction : IImageClassificationPrediction
{
    private readonly MLContext _mlContext;
    private readonly ITransformer _model;
    private readonly PredictionEngine<ImageData, ImageClassificationResponse> _predictionEngine;

    private const string ModelFolder = "TrainedModels";
    private const string ModelFileName = "imageClassifier.zip";

    public ImageClassificationPrediction(
        MLContext mlContext)
    {
        _mlContext = mlContext;
        _model = LoadModel();
        _predictionEngine = CreatePredictionEngine();
    }

    public ImageClassificationResponse ClassifySingleImage(
        ImageClassificationRequest request)
    {
        try
        {
            var prediction = _predictionEngine.Predict(new ImageData());
            return prediction;
        }
        catch (Exception ex)
        {
            throw new Exception($"Error during image classification: {ex.Message}", ex);
        }
    }

    private ITransformer LoadModel()
    {
        var modelPath = Path.Combine(ModelFolder, ModelFileName);

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Model file not found: {modelPath}");
        }

        try
        {
            var model = _mlContext.Model.Load(modelPath, out _);
            return model;
        }
        catch (Exception ex)
        {
            throw new Exception($"Error loading model from {modelPath}: {ex.Message}", ex);
        }
    }

    private PredictionEngine<ImageData, ImageClassificationResponse> CreatePredictionEngine()
    {
        return _mlContext.Model.CreatePredictionEngine<ImageData, ImageClassificationResponse>(_model);
    }

    public void Dispose()
    {
        _predictionEngine.Dispose();
    }
}
