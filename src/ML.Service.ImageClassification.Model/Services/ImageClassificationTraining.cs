using System.Diagnostics.CodeAnalysis;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Microsoft.ML.Vision;
using ML.Service.ImageClassification.Minio.Models;
using ML.Service.ImageClassification.Model.Models;

namespace ML.Service.ImageClassification.Model.Services;

internal sealed class ImageClassificationTraining
{
    private readonly MLContext _mlContext;
    private readonly ILogger<ImageClassificationTraining> _logger;

    private const string DatasetFolder = "Dataset";
    private const string ModelFolder = "TrainedModels";

    public ImageClassificationTraining(
        MLContext mlContext,
        ILogger<ImageClassificationTraining> logger)
    {
        _mlContext = mlContext;
        _logger = logger;
    }

    public void TrainModel()
    {
        try
        {
            _logger.LogInformation("Starting model training process...");

            // 1. Загрузка и разделение данных за один шаг
            var data = _mlContext.Data.LoadFromEnumerable(LoadImageData());
            var split = _mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainTestSplit =
                _mlContext.Data.TrainTestSplit(split.TrainSet, testFraction: 0.125); // 10% от исходных данных

            LogDataCounts(split.TrainSet, trainTestSplit.TestSet, split.TestSet);

            // 2. Создание единого пайплайна
            var pipeline = _mlContext.Transforms
                .Conversion
                .MapValueToKey("LabelAsKey", "Label")
                .Append(_mlContext.Transforms.LoadRawImageBytes("Image", DatasetFolder, "ImagePath"))
                .Append(CreateTrainer(trainTestSplit.TestSet))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // 3. Обучение на всей тренировочной выборке (train + validation)
            _logger.LogInformation("Training started...");
            var model = pipeline.Fit(trainTestSplit.TrainSet);
            _logger.LogInformation("Training finished.");

            // 4. Оценка и сохранение
            EvaluateModel(model, split.TestSet);
            SaveModel(model, trainTestSplit.TrainSet);

            _logger.LogInformation("Model training completed successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error occurred during model training");
            throw;
        }
    }

    private List<ImageData> LoadImageData()
    {
        var datasetFullPath = Path.GetFullPath(DatasetFolder);

        if (!Directory.Exists(datasetFullPath))
        {
            throw new DirectoryNotFoundException($"Dataset folder not found: {datasetFullPath}");
        }

        var imageFiles = Directory.GetFiles(datasetFullPath, "*", SearchOption.AllDirectories)
            .Where(f => Path.GetExtension(f)
                .ToLowerInvariant() is ".jpg" or ".jpeg" or ".png" or ".bmp")
            .ToList();

        if (imageFiles.Count == 0)
        {
            throw new Exception($"No images found in dataset folder: {datasetFullPath}");
        }

        _logger.LogInformation("Found {Count} images in dataset", imageFiles.Count);

        return imageFiles.Select(path => new ImageData
            {
                ImagePath = Path.GetRelativePath(datasetFullPath, path),
                Label = Path.GetRelativePath(datasetFullPath, Path.GetDirectoryName(path)!)
            })
            .OrderBy(_ => Random.Shared.Next()) // Простое перемешивание
            .ToList();
    }

    private void LogDataCounts(
        IDataView trainSet,
        IDataView validationSet,
        IDataView testSet)
    {
        var trainCount = trainSet.GetRowCount();
        var validationCount = validationSet.GetRowCount();
        var testCount = testSet.GetRowCount();

        // If the count is unknown (null), use the Preview method as a fallback.
        _logger.LogInformation("Data counts - Train: {TrainCount}, Validation: {ValCount}, Test: {TestCount}",
            trainCount ?? trainSet.Preview()
                .RowView.Length, validationCount ?? validationSet.Preview()
                .RowView.Length, testCount ?? testSet.Preview()
                .RowView.Length);
    }

    private IEstimator<ITransformer> CreateTrainer(
        IDataView validationSet)
    {
        var options = new ImageClassificationTrainer.Options
        {
            FeatureColumnName = "Image",
            LabelColumnName = "LabelAsKey",
            ValidationSet = validationSet,
            Arch = ImageClassificationTrainer.Architecture.ResnetV250,
            Epoch = 50,
            BatchSize = 32,
            LearningRate = 0.01f,
            WorkspacePath = Path.Combine(ModelFolder, "workspace"),
            TestOnTrainSet = false,
            ReuseTrainSetBottleneckCachedValues = true,
            ReuseValidationSetBottleneckCachedValues = true,
            MetricsCallback = m =>
                _logger.LogInformation("Epoch: {Epoch}, Metrics: {Accuracy}", m.Train.Epoch, m.Train.Accuracy)
        };

        return _mlContext.MulticlassClassification.Trainers.ImageClassification(options);
    }

    private void EvaluateModel(
        ITransformer model,
        IDataView testSet)
    {
        var predictions = model.Transform(testSet);
        var metrics = _mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "LabelAsKey",
            predictedLabelColumnName: "PredictedLabel");

        _logger.LogInformation("Model evaluation results:");
        _logger.LogInformation("MicroAccuracy: {MicroAccuracy:P2}", metrics.MicroAccuracy);
        _logger.LogInformation("MacroAccuracy: {MacroAccuracy:P2}", metrics.MacroAccuracy);
        _logger.LogInformation("LogLoss: {LogLoss:F4}", metrics.LogLoss);
    }

    private void SaveModel(
        ITransformer model,
        IDataView dataView)
    {
        using var memoryStream = new MemoryStream();
        _mlContext.Model.Save(model, dataView.Schema, memoryStream);

        // Сбрасываем позицию потока на начало для чтения
        memoryStream.Position = 0;

        // Создаем MinioModel
        var minioModel = new MinioModel
        {
            Name = "imageClassifier",
            File = memoryStream
        };
    }
}
